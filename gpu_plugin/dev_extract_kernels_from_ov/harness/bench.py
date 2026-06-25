# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Standalone micro-benchmark + correctness harness for the four extracted OpenVINO GPU
GGUF kernels (fc_gguf_opt / fc_gguf_transcode / fc_gguf_prequant / fc_gguf_dp4a).

Each kernel is the VERBATIM OV .cl file (kernels/*.cl), compiled with the OV-faithful JIT
preamble + JIT constants (ov_jit.py) and dispatched with OV's exact work-group geometry
(taken from gguf/fc_gguf_opt.cpp). This means a kernel optimised here drops straight back
into OV with no source changes.

Per case it:
  1. generates a valid random GGUF weight + activation (gguf_ref.py),
  2. runs the kernel once and verifies the result against the NumPy reference decoder,
  3. times N iterations, FLUSHING the GPU L2 (cache_flush.cl) between every timed run so
     the weight is re-read from DRAM each time (otherwise an 18 MiB-resident weight would
     measure L2 bandwidth, not the DRAM bandwidth the decode GEMV is bound by),
  4. computes the roofline occupancy (achieved BW / peak BW, achieved FLOPS / attainable)
     from the B580 capability model (hw.py).

Timing source: OpenCL command-queue profiling events (CL_PROFILING_COMMAND_START/END),
the same clock cliloader's device-timing reports read. Run the whole thing under
`cliloader -dv` (see run.sh) to get the independent cliloader device-time table alongside
these in-process numbers.

Usage:
    python3 bench.py --config ../configs/default.json [--out ../results/report.json]
    python3 bench.py --quick          # tiny built-in sweep, no config needed
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import pyopencl as cl

import ov_jit
import gguf_ref as gr
from hw import HW

KDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels"))

# numpy dtype per activation/output dtype string
_NP = {"f16": np.float16, "f32": np.float32}


# ======================================================================================
# GPU context, kernel build, cache flush, timing
# ======================================================================================
class Gpu:
    def __init__(self, device_index=0, platform_index=0):
        plats = cl.get_platforms()
        self.platform = plats[platform_index]
        devs = self.platform.get_devices()
        self.device = devs[device_index]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self._build_cache = {}
        self._flush_prog = None
        self._flush_buf = None
        self._flush_sink = None
        self._flush_elems = 0

    def build(self, fname, entry_point, consts):
        key = (fname, entry_point)
        if key in self._build_cache:
            return self._build_cache[key]
        with open(os.path.join(KDIR, fname)) as f:
            body = f.read()
        src = ov_jit.build_preamble(entry_point, consts) + "\n" + body
        opts = ["-I", KDIR] + ov_jit.OV_BUILD_OPTIONS
        prog = cl.Program(self.ctx, src).build(options=opts)
        krn = cl.Kernel(prog, entry_point)
        self._build_cache[key] = krn
        return krn

    def setup_flush(self, mib):
        """Allocate the L2-flush scratch (> L2 so it cannot stay resident)."""
        elems = (mib * 1024 * 1024) // 4
        self._flush_elems = elems
        self._flush_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=elems * 4)
        self._flush_sink = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=4096 * 4)
        zeros = np.zeros(elems, np.float32)
        cl.enqueue_copy(self.queue, self._flush_buf, zeros)
        with open(os.path.join(KDIR, "cache_flush.cl")) as f:
            src = f.read()
        self._flush_prog = cl.Program(self.ctx, src).build(options=ov_jit.OV_BUILD_OPTIONS)
        self._flush_kernel = cl.Kernel(self._flush_prog, "cache_flush")
        self.queue.finish()

    def flush_l2(self, sweeps=2):
        k = self._flush_kernel
        gsz = (min(self._flush_elems, 1024 * 1024),)  # plenty of threads to saturate BW
        for _ in range(sweeps):
            k(self.queue, gsz, None, self._flush_buf,
              np.uint32(self._flush_elems), self._flush_sink)
        self.queue.finish()

    def time_event(self, enqueue_fn, iters, warmup, do_flush):
        """Run enqueue_fn(queue)->event `iters` times, flushing L2 before each timed run.

        Returns list of device times in seconds (from CL profiling events).
        enqueue_fn must return the event of the kernel to be timed (the last/primary one).
        """
        for _ in range(warmup):
            ev = enqueue_fn(self.queue)
            self.queue.finish()
        times = []
        for _ in range(iters):
            if do_flush:
                self.flush_l2()
            ev = enqueue_fn(self.queue)
            ev.wait()
            t = (ev.profile.end - ev.profile.start) * 1e-9  # ns -> s
            times.append(t)
        return times


def stats(times):
    return {
        "median_ms": statistics.median(times) * 1e3,
        "min_ms": min(times) * 1e3,
        "mean_ms": statistics.mean(times) * 1e3,
        "max_ms": max(times) * 1e3,
        "iters": len(times),
    }


def rel_l2(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


def cosine(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ======================================================================================
# Per-kernel-group runners
# ======================================================================================
SG = ov_jit.GGUF_GEMV_SG_SIZE


def run_opt(gpu, hw, name, K, N, M, in_dt, out_dt, iters, warmup, do_flush, seed=1234):
    """fc_gguf_opt GEMV: C[M,N] = A[M,K] @ deq(W[N,K])^T."""
    be, bb = gr.GGUF_BLOCK_GEOM[name]
    bpr = K // be
    Wbytes = gr.gen_weight_bytes(name, N, K, seed=seed)          # [N, bpr*bb] uint8
    Wf = gr.dequantize(name, Wbytes)                              # [N,K] f32 reference
    rng = np.random.default_rng(seed + 1)
    A = rng.standard_normal((M, K)).astype(_NP[in_dt]) * 0.5
    # reference: GPU reads activation from the f16/f32 buffer as float -> cast accordingly
    C_ref = (A.astype(np.float32) @ Wf.T).astype(np.float32)

    mf = cl.mem_flags
    d_A = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
    d_W = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Wbytes))
    C = np.zeros((M, N), _NP[out_dt])
    d_C = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=C.nbytes)

    ep, consts = ov_jit.opt_jit(name, K, N, M, in0_dt=in_dt, out_dt=out_dt, dynamic=False)
    krn = gpu.build("fc_gguf_opt.cl", ep, consts)
    gsz = (N * SG, M, 1)
    lsz = (SG, 1, 1)

    def enq(q):
        return krn(q, gsz, lsz, d_A, d_W, d_C)

    # correctness
    enq(gpu.queue).wait()
    cl.enqueue_copy(gpu.queue, C, d_C); gpu.queue.finish()
    err = rel_l2(C, C_ref)
    cos = cosine(C, C_ref)

    times = gpu.time_event(enq, iters, warmup, do_flush)
    st = stats(times)

    weight_bytes = N * bpr * bb
    act_bytes = M * K * np.dtype(_NP[in_dt]).itemsize
    out_bytes = M * N * np.dtype(_NP[out_dt]).itemsize
    bytes_moved = weight_bytes + act_bytes + out_bytes
    flops = 2 * M * N * K
    rl = hw.roofline(flops, bytes_moved, st["median_ms"] * 1e-3, compute_domain="fp32")
    return _record("fc_gguf_opt(GEMV)", name, dict(K=K, N=N, M=M, in_dt=in_dt, out_dt=out_dt),
                   st, rl, dict(rel_l2_err=err, cosine=cos, passed=err < 2e-2),
                   dict(weight_MiB=weight_bytes / 1048576, bytes_moved_MiB=bytes_moved / 1048576))


def _unpack_i4(buf_u8, N, K):
    """Unpack two's-complement i4 nibbles [N,K] from packed bytes [N*K/2]."""
    lo = (buf_u8 & 0x0F).astype(np.int8)
    hi = (buf_u8 >> 4).astype(np.int8)
    q = np.empty(buf_u8.size * 2, np.int8)
    q[0::2] = lo
    q[1::2] = hi
    q = q.astype(np.int32)
    q[q > 7] -= 16   # sign-extend 4-bit
    return q.reshape(N, K)


def run_transcode(gpu, hw, name, K, N, iters, warmup, do_flush, seed=1234):
    """fc_gguf_transcode: GGUF block -> packed i4/i8 weight + f16 per-group scale."""
    be, bb = gr.GGUF_BLOCK_GEOM[name]
    bpr = K // be
    to_i4, qmax = gr.GGUF_TRANSCODE_TARGET[name]
    G = gr.GGUF_REQUANT_GROUP
    ngroups = K // G
    Wbytes = gr.gen_weight_bytes(name, N, K, seed=seed)
    Wf = gr.dequantize(name, Wbytes)                             # [N,K] f32

    mf = cl.mem_flags
    d_W = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Wbytes))
    wq_bytes = (N * K // 2) if to_i4 else (N * K)
    d_WQ = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=wq_bytes)
    SCbuf = np.zeros((ngroups, N), np.float16)
    d_SC = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=SCbuf.nbytes)

    ep, consts = ov_jit.transcode_jit(name, K, N)
    krn = gpu.build("fc_gguf_transcode.cl", ep, consts)
    n_global = ((N + SG - 1) // SG) * SG
    gsz = (n_global, bpr, 1)
    lsz = (SG, 1, 1)

    def enq(q):
        return krn(q, gsz, lsz, d_W, d_WQ, d_SC)

    # correctness: compare reconstruction (q*scale) to the decoded reference within 1 quant step
    enq(gpu.queue).wait()
    WQ = np.zeros(wq_bytes, np.uint8)
    cl.enqueue_copy(gpu.queue, WQ, d_WQ)
    cl.enqueue_copy(gpu.queue, SCbuf, d_SC); gpu.queue.finish()
    if to_i4:
        q = _unpack_i4(WQ, N, K)
    else:
        q = WQ.view(np.int8).astype(np.int32).reshape(N, K)
    scale_gpu = SCbuf.astype(np.float32)                          # [ngroups, N]
    Wf_g = Wf.reshape(N, ngroups, G)
    amax = np.abs(Wf_g).max(axis=2)                               # [N, ngroups]

    # (1) Integer codes must match the reference symmetric quantizer (round(w*qmax/amax),
    #     clamped) -- this is the part the kernel actually computes. We allow +-1 LSB to
    #     absorb round-half-to-even vs round-half-away ties between OpenCL round() and numpy.
    inv = np.where(amax > 0, qmax / amax, 0.0)
    q_np = np.clip(np.round(Wf_g * inv[:, :, None]), -(qmax + 1), qmax).astype(np.int32)
    q_match = float(np.mean(np.abs(q.reshape(N, ngroups, G) - q_np) <= 1))

    # (2) Per-group scale must match the f16-cast of the ideal amax/qmax (the kernel stores
    #     (half)scale; dnnl later reads it as f16 too, so this IS the value OV consumes).
    scale_np_f16 = np.float16(np.where(amax > 0, amax / qmax, 1.0)).astype(np.float32)  # [N,ngroups]
    scale_rel = rel_l2(scale_gpu, scale_np_f16.T)

    # (3) Diagnostic only: reconstruction error normalized by the group's data magnitude
    #     (amax), NOT by the f16 quant step. Near-zero groups (amax ~ f16 subnormal floor)
    #     have a large step-relative error purely from f16 scale rounding while the absolute
    #     error stays ~1 LSB -- the amax-relative figure is the meaningful one.
    recon = q.reshape(N, ngroups, G).astype(np.float32) * scale_gpu.T[:, :, None]
    rel_amax_err = float(np.max(np.abs(recon - Wf_g) / np.maximum(amax[:, :, None], 1e-30)))
    passed = (q_match > 0.999) and (scale_rel < 5e-2)

    times = gpu.time_event(enq, iters, warmup, do_flush)
    st = stats(times)
    bytes_moved = N * bpr * bb + wq_bytes + SCbuf.nbytes
    rl = hw.roofline(flops=N * K, bytes_moved=bytes_moved, seconds=st["median_ms"] * 1e-3,
                     compute_domain="fp32")
    return _record("fc_gguf_transcode", name,
                   dict(K=K, N=N, target=("i4" if to_i4 else "i8"), qmax=qmax),
                   st, rl,
                   dict(int_code_match=q_match, scale_rel_l2=scale_rel,
                        recon_err_rel_amax=rel_amax_err, passed=passed),
                   dict(in_MiB=N * bpr * bb / 1048576, out_MiB=bytes_moved / 1048576))


def run_dp4a(gpu, hw, name, K, N, M, out_dt, iters, warmup, do_flush, seed=1234):
    """fc_gguf_prequant + fc_gguf_dp4a (Q5_K/Q6_K int8-activation decode path)."""
    assert name in ("gguf_q5_k", "gguf_q6_k")
    be, bb = gr.GGUF_BLOCK_GEOM[name]
    bpr = K // be
    G = gr.GGUF_REQUANT_GROUP
    Wbytes = gr.gen_weight_bytes(name, N, K, seed=seed)
    Wf = gr.dequantize(name, Wbytes)                              # [N,K] f32
    rng = np.random.default_rng(seed + 1)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)    # prequant input is f16/f32

    mf = cl.mem_flags
    d_A = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
    d_W = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Wbytes))
    d_Aq = cl.Buffer(gpu.ctx, mf.READ_WRITE, size=M * K)          # int8 [M,K]
    d_Asc = cl.Buffer(gpu.ctx, mf.READ_WRITE, size=M * (K // G) * 4)  # f32 [M,K/32]
    C = np.zeros((M, N), _NP[out_dt])
    d_C = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=C.nbytes)

    pq_ep, pq_consts = ov_jit.prequant_jit(name, K, N, in0_dt="f16")
    pq = gpu.build("fc_gguf_prequant.cl", pq_ep, pq_consts)
    dp_ep, dp_consts, nrow = ov_jit.dp4a_jit(name, K, N, out_dt=out_dt)
    dp = gpu.build("fc_gguf_dp4a.cl", dp_ep, dp_consts)

    pq_g = (K // G, M, 1)
    pq_l = (1, 1, 1)
    row_groups = (N + nrow - 1) // nrow
    dp_g = (row_groups * SG, M, 1)
    dp_l = (SG, 1, 1)

    def enq_pq(q):
        return pq(q, pq_g, pq_l, d_A, d_Aq, d_Asc)

    def enq_full(q):
        enq_pq(q)
        return dp(q, dp_g, dp_l, d_Aq, d_Asc, d_W, d_C)

    # ---- correctness ----
    enq_full(gpu.queue).wait()
    Aq = np.zeros(M * K, np.int8); Asc = np.zeros(M * (K // G), np.float32)
    cl.enqueue_copy(gpu.queue, Aq, d_Aq); cl.enqueue_copy(gpu.queue, Asc, d_Asc)
    cl.enqueue_copy(gpu.queue, C, d_C); gpu.queue.finish()
    Aq = Aq.reshape(M, K); Asc = Asc.reshape(M, K // G)
    # numpy prequant reference (matches fc_gguf_prequant.cl: symmetric absmax/127 per 32)
    Ag = A.astype(np.float32).reshape(M, K // G, G)
    amax = np.abs(Ag).max(axis=2)
    sc_np = (amax / 127.0)
    inv = np.where(amax > 0, 127.0 / amax, 0.0)
    aq_np = np.clip(np.round(Ag * inv[:, :, None]), -127, 127).astype(np.int32)
    pq_q_match = float(np.mean(np.abs(aq_np.reshape(M, K) - Aq.astype(np.int32)) <= 1))
    pq_sc_rel = rel_l2(Asc, sc_np)
    # dp4a reference: int8-quantized activation (dequant) @ decoded weight^T
    Aq_deq = (Aq.astype(np.float32).reshape(M, K // G, G) * Asc[:, :, None]).reshape(M, K)
    C_ref = (Aq_deq @ Wf.T).astype(np.float32)
    err = rel_l2(C, C_ref); cos = cosine(C, C_ref)
    passed = (err < 2e-2) and (pq_q_match > 0.98) and (pq_sc_rel < 1e-2)

    # ---- timing (full path = prequant + dp4a; report both) ----
    times_full = gpu.time_event(enq_full, iters, warmup, do_flush)
    times_dp = gpu.time_event(lambda q: dp(q, dp_g, dp_l, d_Aq, d_Asc, d_W, d_C),
                              iters, warmup, do_flush)
    st_full = stats(times_full)
    st_dp = stats(times_dp)

    weight_bytes = N * bpr * bb
    act_bytes = M * K            # int8 activation read by dp4a
    asc_bytes = M * (K // G) * 4
    out_bytes = M * N * np.dtype(_NP[out_dt]).itemsize
    bytes_moved = weight_bytes + act_bytes + asc_bytes + out_bytes
    flops = 2 * M * N * K
    rl = hw.roofline(flops, bytes_moved, st_dp["median_ms"] * 1e-3, compute_domain="int8")
    rec = _record("fc_gguf_dp4a(+prequant)", name,
                  dict(K=K, N=N, M=M, out_dt=out_dt, NROW=nrow),
                  st_dp, rl,
                  dict(rel_l2_err=err, cosine=cos, prequant_q_match=pq_q_match,
                       prequant_scale_rel=pq_sc_rel, passed=passed),
                  dict(weight_MiB=weight_bytes / 1048576, bytes_moved_MiB=bytes_moved / 1048576))
    rec["timing_full_path"] = st_full
    return rec


def _record(kernel, fmt, params, st, rl, correctness, extra):
    return {
        "kernel": kernel,
        "format": fmt,
        "params": params,
        "timing": st,
        "roofline": rl,
        "correctness": correctness,
        "data": extra,
    }


# ======================================================================================
# Driver
# ======================================================================================
def print_record(r):
    p = r["params"]
    pstr = " ".join("%s=%s" % (k, v) for k, v in p.items())
    c = r["correctness"]
    ok = "PASS" if c.get("passed") else "FAIL"
    rl = r["roofline"]
    print("  [%s] %-26s %-13s | %s" % (ok, r["kernel"], r["format"], pstr))
    print("        time(median)=%.4f ms  BW=%.1f GB/s (%.1f%% of peak)  GFLOPS=%.0f  roofline=%.1f%%  bound=%s"
          % (r["timing"]["median_ms"], rl["achieved_bw_gbps"], rl["bw_util_pct"],
             rl["achieved_gflops"], rl["roofline_pct"], rl["bound_by"]))
    cc = ", ".join("%s=%s" % (k, (("%.4g" % v) if isinstance(v, float) else v))
                   for k, v in c.items() if k != "passed")
    print("        correctness: %s" % cc)


DEFAULT_QUICK = {
    "hw": {},
    "flush_mib": 128,
    "iters": 30,
    "warmup": 5,
    "flush_l2": True,
    "cases": [
        {"kernel": "opt", "format": "gguf_q4_k", "K": 1024, "N": 1024, "M": 1, "in_dt": "f16", "out_dt": "f16"},
        {"kernel": "opt", "format": "gguf_q6_k", "K": 1024, "N": 1024, "M": 1, "in_dt": "f16", "out_dt": "f16"},
        {"kernel": "dp4a", "format": "gguf_q6_k", "K": 1024, "N": 1024, "M": 1, "out_dt": "f16"},
        {"kernel": "transcode", "format": "gguf_q4_k", "K": 1024, "N": 1024},
    ],
}


def run_case(gpu, hw, case, glob):
    kind = case["kernel"]
    name = case["format"]
    iters = case.get("iters", glob["iters"])
    warmup = case.get("warmup", glob["warmup"])
    flush = case.get("flush_l2", glob["flush_l2"])
    seed = case.get("seed", 1234)
    if kind == "opt":
        return run_opt(gpu, hw, name, case["K"], case["N"], case.get("M", 1),
                       case.get("in_dt", "f16"), case.get("out_dt", "f16"),
                       iters, warmup, flush, seed)
    if kind == "transcode":
        return run_transcode(gpu, hw, name, case["K"], case["N"], iters, warmup, flush, seed)
    if kind == "dp4a":
        return run_dp4a(gpu, hw, name, case["K"], case["N"], case.get("M", 1),
                        case.get("out_dt", "f16"), iters, warmup, flush, seed)
    raise ValueError("unknown kernel kind %r" % kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="JSON config (see configs/default.json)")
    ap.add_argument("--out", help="write full JSON report here")
    ap.add_argument("--quick", action="store_true", help="run the tiny built-in sweep")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--platform", type=int, default=0)
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    elif args.quick:
        cfg = DEFAULT_QUICK
    else:
        ap.error("pass --config <file> or --quick")

    glob = {
        "iters": cfg.get("iters", 30),
        "warmup": cfg.get("warmup", 5),
        "flush_l2": cfg.get("flush_l2", True),
    }
    hw = HW(cfg.get("hw"))
    gpu = Gpu(args.device, args.platform)
    gpu.setup_flush(cfg.get("flush_mib", 128))

    print("=" * 100)
    print("Device : %s" % gpu.device.name)
    print("HW peak: FP32 %.2f TFLOPS | INT8(dp4a) %.2f TOPS | BW %.0f GB/s | L2 %.1f MiB"
          % (hw.fp32_tflops, hw.int8_tops, hw.peak_bw_gbps, hw.l2_cache_bytes / 1048576))
    print("Bench  : iters=%d warmup=%d flush_l2=%s (flush buffer %d MiB)"
          % (glob["iters"], glob["warmup"], glob["flush_l2"], cfg.get("flush_mib", 128)))
    print("=" * 100)

    records = []
    for case in cfg["cases"]:
        try:
            r = run_case(gpu, hw, case, glob)
            records.append(r)
            print_record(r)
        except Exception as e:
            print("  [ERR ] %s %s: %s" % (case.get("kernel"), case.get("format"), e))
            import traceback
            traceback.print_exc()

    report = {
        "device": gpu.device.name,
        "hw": hw.summary(),
        "bench": glob,
        "flush_mib": cfg.get("flush_mib", 128),
        "timestamp_unix": int(time.time()),
        "records": records,
    }
    out = args.out or cfg.get("out")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print("\nWrote report -> %s" % out)
    npass = sum(1 for r in records if r["correctness"].get("passed"))
    print("\n%d/%d cases passed correctness." % (npass, len(records)))
    return 0 if npass == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
