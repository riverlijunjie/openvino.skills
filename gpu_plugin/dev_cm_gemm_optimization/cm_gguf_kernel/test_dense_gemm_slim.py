"""
test_dense_gemm_slim.py

Correctness + benchmark + roofline harness for dense slim GEMM CM (C-for-Metal)
kernels:
    gemm_q4k_slim.cm : gemm_q4k_slim
    gemm_q5k_slim.cm : gemm_q5k_slim
    gemm_q6k_slim.cm : gemm_q6k_slim

Converted from sycl_gguf_kernel/gemm_q{4,5,6}k_L1*slim*.cpp. These kernels
handle small token batches (token_len <= 16) using the "flat shuffled" weight
layout from shuffle_q{4,5,6}k() -- same layout as the SYCL slim GEMM kernels
and the OpenCL reference port (/mnt/river/ovmx/ocl_gguf_kernel/). Each CM
thread corresponds 1:1 to one OpenCL work-item (row-group h, K-block hh); the
per-row Q4/5/6_K sub-block dot products are vectorised with vector<float,16>
SIMD arithmetic instead of OpenCL's scalar per-work-item loop.

Input activations: fp16 [token_len * input_len]
Output:            fp32 [token_len * output_len]

Weight layout:
  Q4K: pqs || psl  (prepared with shuffle_q4k)
  Q5K: pqs || pqh || psl  (prepared with shuffle_q5k)
  Q6K: pql || pqh || ps || pd  (prepared with shuffle_q6k)

Dispatch:
  global (threads) = (output_len/OPG * nbpr,)   [1D]
  local  (threads) = (nbpr,)
  OPG = 16, nbpr = input_len // 256

Usage:
    python test_dense_gemm_slim.py [--device 0] [--iters 50] [--warmup 5]
        [--flush_mib 256] [--peak_bw 456] [--peak_fp16 116000] [--verbose]
        [--no-bench]
"""

import os
import statistics
import argparse

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    raise SystemExit("pyopencl is required: pip install pyopencl")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPG = 16


def cl_src(name):
    return os.path.join(SCRIPT_DIR, name)


_FLUSH_SRC = r"""
#include <cm/cm.h>
#define FLUSH_VEC 16
extern "C" _GENX_MAIN_ void cache_flush(
    float* buf  [[type("svmptr_t")]],
    uint   n,
    float* sink [[type("svmptr_t")]])
{
    uint local_size = cm_local_size(0);
    uint gid    = cm_group_id(0) * local_size + cm_local_id(0);
    uint stride = cm_group_count(0) * local_size;

    vector<float, FLUSH_VEC> acc = 0.0f;
    for (uint i = gid * FLUSH_VEC; i + FLUSH_VEC <= n; i += stride * FLUSH_VEC) {
        vector<float, FLUSH_VEC> v = cm_ptr_load<float, FLUSH_VEC>((float*)buf, i * (uint)sizeof(float));
        acc += v;
    }
    float s = cm_sum<float>(acc);
    if (gid < 4096u) {
        float prev = sink[gid];
        sink[gid] = prev + s;
    }
}
"""


class HW:
    def __init__(self, peak_bw_gbps=456.0, fp16_gflops=116000.0):
        self.peak_bw_gbps = peak_bw_gbps
        self.fp16_gflops = fp16_gflops

    def roofline(self, flops, bytes_moved, seconds):
        ag = flops / seconds / 1e9
        ab = bytes_moved / seconds / 1e9
        ai = flops / max(bytes_moved, 1)
        # Roofline lower-bound latency: separately compute how long the
        # kernel would take running at peak compute throughput alone, and
        # how long at peak memory bandwidth alone. Compute and memory traffic
        # can at best fully overlap, so the kernel can never finish faster
        # than whichever of the two takes LONGER -- that larger latency is
        # the achievable roofline baseline, and the resource that produced it
        # is the actual bottleneck ("bound_by").
        # CM kernels use FP16 operands/arithmetic; BMG peak is 116 TFLOPS.
        t_compute_s = flops / max(self.fp16_gflops * 1e9, 1e-12)
        t_memory_s  = bytes_moved / max(self.peak_bw_gbps * 1e9, 1e-12)
        roofline_s  = max(t_compute_s, t_memory_s)
        bound = "compute" if t_compute_s >= t_memory_s else "memory"
        roof = flops / max(roofline_s, 1e-12) / 1e9
        return dict(achieved_gflops=ag, achieved_bw_gbps=ab,
                    arith_intensity=ai, roofline_gflops=roof,
                    compute_bound_ms=t_compute_s * 1e3,
                    memory_bound_ms=t_memory_s * 1e3,
                    roofline_ms=roofline_s * 1e3,
                    roofline_pct=roofline_s / max(seconds, 1e-12) * 100.0,
                    bw_util_pct=ab / max(self.peak_bw_gbps, 1e-12) * 100.0,
                    bound_by=bound)


class Gpu:
    def __init__(self, device_idx=0):
        devices = [d for p in cl.get_platforms() for d in p.get_devices()]
        if not devices:
            raise RuntimeError("No OpenCL devices found")
        self.device = devices[device_idx % len(devices)]
        print(f"Device [{device_idx}]: {self.device.name}")
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx,
                                     properties=cl.command_queue_properties.PROFILING_ENABLE)
        self._prog_cache = {}
        self._flush_krn = self._flush_buf = self._flush_sink = None
        self._flush_n = 0

    def build(self, src_file, build_opts=""):
        key = (src_file, build_opts)
        if key not in self._prog_cache:
            with open(src_file) as f:
                src = f.read()
            self._prog_cache[key] = cl.Program(self.ctx, src).build(options=build_opts)
        return self._prog_cache[key]

    def _build_src(self, src, key, build_opts=""):
        if key not in self._prog_cache:
            self._prog_cache[key] = cl.Program(self.ctx, src).build(options=build_opts)
        return self._prog_cache[key]

    def setup_flush(self, mib=256):
        n = mib * 1024 * 1024 // 4
        self._flush_n = n
        mf = cl.mem_flags
        self._flush_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                    hostbuf=np.zeros(n, dtype=np.float32))
        self._flush_sink = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=4096 * 4)
        self._flush_krn = cl.Kernel(self._build_src(_FLUSH_SRC, "_flush", build_opts="-cmc"), "cache_flush")
        self.queue.finish()

    def flush_l3(self, sweeps=2):
        if self._flush_krn is None:
            return
        gsz = (min(self._flush_n, 1 << 20),)
        for _ in range(sweeps):
            self._flush_krn(self.queue, gsz, None,
                            self._flush_buf, np.uint32(self._flush_n), self._flush_sink)
        self.queue.finish()

    def time_kernel(self, enq_fn, iters=50, warmup=5, do_flush=True):
        for _ in range(warmup):
            enq_fn(self.queue); self.queue.finish()
        times = []
        for _ in range(iters):
            if do_flush: self.flush_l3()
            ev = enq_fn(self.queue); ev.wait()
            times.append((ev.profile.end - ev.profile.start) * 1e-9)
        return times


def stats(ts):
    return dict(mean_ms=statistics.mean(ts)*1e3,
                median_ms=statistics.median(ts)*1e3,
                min_ms=min(ts)*1e3, max_ms=max(ts)*1e3, iters=len(ts))


def _cbuf(ctx, arr):
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(arr))


def check_close(label, ref, got, rtol=1e-2, atol=1e-3, verbose=False):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    max_abs = float(np.max(np.abs(ref - got)))
    ref_sc = float(np.max(np.abs(ref)))
    rel = max_abs / (ref_sc + 1e-9)
    ok = bool(np.allclose(ref, got, rtol=rtol, atol=max(atol, ref_sc * 1e-6)))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}  max_abs={max_abs:.3e}  rel={rel:.3e}")
    if verbose and not ok:
        wi = int(np.argmax(np.abs(ref - got)))
        print(f"         worst idx={wi}  ref={ref.ravel()[wi]:.6f}  got={got.ravel()[wi]:.6f}")
    return ok


def _print_rl(rl, moved_bytes):
    print(f"    BW={rl['achieved_bw_gbps']:.1f} GB/s ({rl['bw_util_pct']:.1f}% peak)  "
          f"AI={rl['arith_intensity']:.2f}  roofline={rl['roofline_pct']:.1f}%  "
          f"bound={rl['bound_by']}  FP16-GFLOPS={rl['achieved_gflops']:.1f}")
    print(f"    roofline latency: compute={rl['compute_bound_ms']:.3f} ms  "
          f"memory={rl['memory_bound_ms']:.3f} ms  ->  bound={rl['roofline_ms']:.3f} ms")
    print(f"    moved={moved_bytes/1e6:.1f} MB")


# ---------------------------------------------------------------------------
# GGUF block generators
# ---------------------------------------------------------------------------
def gen_q4k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    d  = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc = rng.integers(0, 64, (n, 12), dtype=np.uint8)
    qs = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qs], axis=1).ravel()


def gen_q5k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    d  = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    dm = np.float16(rng.uniform(0.001, 0.05, n)).view(np.uint8).reshape(n, 2)
    sc = rng.integers(0, 64, (n, 12), dtype=np.uint8)
    qh = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    qs = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    return np.concatenate([d, dm, sc, qh, qs], axis=1).ravel()


def gen_q6k_blocks(n, seed=1234):
    rng = np.random.default_rng(seed)
    ql = rng.integers(0, 256, (n, 128), dtype=np.uint8)
    qh = rng.integers(0, 256, (n, 64), dtype=np.uint8)
    sc = rng.integers(-8, 8, (n, 16), dtype=np.int8).view(np.uint8)
    d  = np.float16(rng.uniform(0.001, 0.10, n)).view(np.uint8).reshape(n, 2)
    return np.concatenate([ql, qh, sc, d], axis=1).ravel()


# ---------------------------------------------------------------------------
# Shuffle functions: raw GGUF -> flat separated-section layout
# ---------------------------------------------------------------------------
def shuffle_q4k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 144)
    d_b, dm_b = blk[:, 0:2], blk[:, 2:4]
    sc = blk[:, 4:16]; qs = blk[:, 16:144].reshape(NB, 4, 32)
    low = qs & 0x0F; high = (qs >> 4) & 0x0F
    rw = np.empty((NB, 4, 64), dtype=np.uint8)
    rw[:,:,0:32]=low; rw[:,:,32:64]=high; rw=rw.reshape(NB,256)
    (sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11)=(sc[:,i] for i in range(12))
    rs=np.empty((NB,8),dtype=np.uint8); rm=np.empty((NB,8),dtype=np.uint8)
    rs[:,0]=sc0&63; rs[:,1]=sc1&63; rs[:,2]=sc2&63; rs[:,3]=sc3&63
    rs[:,4]=(sc8&0x0F)|((sc0>>2)&0x30); rs[:,5]=(sc9&0x0F)|((sc1>>2)&0x30)
    rs[:,6]=(sc10&0x0F)|((sc2>>2)&0x30); rs[:,7]=(sc11&0x0F)|((sc3>>2)&0x30)
    rm[:,0]=sc4&63; rm[:,1]=sc5&63; rm[:,2]=sc6&63; rm[:,3]=sc7&63
    rm[:,4]=((sc8>>4)&0x0F)|((sc4>>2)&0x30); rm[:,5]=((sc9>>4)&0x0F)|((sc5>>2)&0x30)
    rm[:,6]=((sc10>>4)&0x0F)|((sc6>>2)&0x30); rm[:,7]=((sc11>>4)&0x0F)|((sc7>>2)&0x30)
    rw8=rw.reshape(NB,8,32)
    opqs=((rw8[:,:,0:16]&0x0F)|((rw8[:,:,16:32]&0x0F)<<4)).reshape(NB,128).astype(np.uint8)
    opsl=np.zeros((NB,16),dtype=np.uint8)
    opsl[:,0]=(rs[:,0]&0x0F)|((rs[:,1]&0x0F)<<4); opsl[:,1]=(rs[:,2]&0x0F)|((rs[:,3]&0x0F)<<4)
    opsl[:,2]=(rs[:,4]&0x0F)|((rs[:,5]&0x0F)<<4); opsl[:,3]=(rs[:,6]&0x0F)|((rs[:,7]&0x0F)<<4)
    opsl[:,4]=(rm[:,0]&0x0F)|((rm[:,1]&0x0F)<<4); opsl[:,5]=(rm[:,2]&0x0F)|((rm[:,3]&0x0F)<<4)
    opsl[:,6]=(rm[:,4]&0x0F)|((rm[:,5]&0x0F)<<4); opsl[:,7]=(rm[:,6]&0x0F)|((rm[:,7]&0x0F)<<4)
    opsl[:,8]=((rs[:,0]&0x30)>>4)|((rs[:,1]&0x30)>>2)|(rs[:,2]&0x30)|((rs[:,3]&0x30)<<2)
    opsl[:,9]=((rs[:,4]&0x30)>>4)|((rs[:,5]&0x30)>>2)|(rs[:,6]&0x30)|((rs[:,7]&0x30)<<2)
    opsl[:,10]=((rm[:,0]&0x30)>>4)|((rm[:,1]&0x30)>>2)|(rm[:,2]&0x30)|((rm[:,3]&0x30)<<2)
    opsl[:,11]=((rm[:,4]&0x30)>>4)|((rm[:,5]&0x30)>>2)|(rm[:,6]&0x30)|((rm[:,7]&0x30)<<2)
    opsl[:,12:14]=d_b; opsl[:,14:16]=dm_b
    return opqs.reshape(-1), opsl.reshape(-1)


def shuffle_q5k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 176)
    d_b,dm_b=blk[:,0:2],blk[:,2:4]; sc=blk[:,4:16]; qh_r=blk[:,16:48]; qs_r=blk[:,48:176].reshape(NB,4,32)
    g_arr=np.arange(8,dtype=np.uint8)
    rh=((qh_r[:,None,:]>>g_arr[None,:,None])&1).astype(np.uint8)
    low=qs_r&0x0F; high=(qs_r>>4)&0x0F
    rw=np.empty((NB,8,32),dtype=np.uint8); rw[:,0::2,:]=low; rw[:,1::2,:]=high
    (sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11)=(sc[:,i] for i in range(12))
    rs=np.empty((NB,8),dtype=np.uint8); rm=np.empty((NB,8),dtype=np.uint8)
    rs[:,0]=sc0&63; rs[:,1]=sc1&63; rs[:,2]=sc2&63; rs[:,3]=sc3&63
    rs[:,4]=(sc8&0x0F)|((sc0>>2)&0x30); rs[:,5]=(sc9&0x0F)|((sc1>>2)&0x30)
    rs[:,6]=(sc10&0x0F)|((sc2>>2)&0x30); rs[:,7]=(sc11&0x0F)|((sc3>>2)&0x30)
    rm[:,0]=sc4&63; rm[:,1]=sc5&63; rm[:,2]=sc6&63; rm[:,3]=sc7&63
    rm[:,4]=((sc8>>4)&0x0F)|((sc4>>2)&0x30); rm[:,5]=((sc9>>4)&0x0F)|((sc5>>2)&0x30)
    rm[:,6]=((sc10>>4)&0x0F)|((sc6>>2)&0x30); rm[:,7]=((sc11>>4)&0x0F)|((sc7>>2)&0x30)
    rw8=rw.reshape(NB,8,32)
    opqs=((rw8[:,:,0:16]&0x0F)|((rw8[:,:,16:32]&0x0F)<<4)).reshape(NB,128).astype(np.uint8)
    opqh=np.zeros((NB,8,4),dtype=np.uint8)
    for s in range(8): opqh|=(rh[:,:,s*4:s*4+4]<<s).astype(np.uint8)
    opqh=opqh.reshape(NB,32)
    opsl=np.zeros((NB,16),dtype=np.uint8)
    opsl[:,0]=(rs[:,0]&0x0F)|((rs[:,1]&0x0F)<<4); opsl[:,1]=(rs[:,2]&0x0F)|((rs[:,3]&0x0F)<<4)
    opsl[:,2]=(rs[:,4]&0x0F)|((rs[:,5]&0x0F)<<4); opsl[:,3]=(rs[:,6]&0x0F)|((rs[:,7]&0x0F)<<4)
    opsl[:,4]=(rm[:,0]&0x0F)|((rm[:,1]&0x0F)<<4); opsl[:,5]=(rm[:,2]&0x0F)|((rm[:,3]&0x0F)<<4)
    opsl[:,6]=(rm[:,4]&0x0F)|((rm[:,5]&0x0F)<<4); opsl[:,7]=(rm[:,6]&0x0F)|((rm[:,7]&0x0F)<<4)
    opsl[:,8]=((rs[:,0]&0x30)>>4)|((rs[:,1]&0x30)>>2)|(rs[:,2]&0x30)|((rs[:,3]&0x30)<<2)
    opsl[:,9]=((rs[:,4]&0x30)>>4)|((rs[:,5]&0x30)>>2)|(rs[:,6]&0x30)|((rs[:,7]&0x30)<<2)
    opsl[:,10]=((rm[:,0]&0x30)>>4)|((rm[:,1]&0x30)>>2)|(rm[:,2]&0x30)|((rm[:,3]&0x30)<<2)
    opsl[:,11]=((rm[:,4]&0x30)>>4)|((rm[:,5]&0x30)>>2)|(rm[:,6]&0x30)|((rm[:,7]&0x30)<<2)
    opsl[:,12:14]=d_b; opsl[:,14:16]=dm_b
    return opqs.reshape(-1), opqh.reshape(-1), opsl.reshape(-1)


def shuffle_q6k(raw, K, N):
    NB = N * K // 256
    blk = raw.reshape(NB, 210)
    ql_r=blk[:,0:128].reshape(NB,2,64); qh_r=blk[:,128:192].reshape(NB,2,32)
    sc_r=blk[:,192:208]; d_b=blk[:,208:210]
    low=ql_r&0x0F; high=(ql_r>>4)&0x0F
    rw=np.empty((NB,2,128),dtype=np.uint8)
    rw[:,:,0:64]=low; rw[:,:,64:128]=high
    rw[:,:,0:32]=(rw[:,:,0:32]+((qh_r&0x03)<<4)).astype(np.uint8)
    rw[:,:,32:64]=(rw[:,:,32:64]+((qh_r&0x0C)<<2)).astype(np.uint8)
    rw[:,:,64:96]=(rw[:,:,64:96]+(qh_r&0x30)).astype(np.uint8)
    rw[:,:,96:128]=(rw[:,:,96:128]+((qh_r&0xC0)>>2)).astype(np.uint8)
    rw_flat=rw.reshape(NB,256); rw8=rw_flat.reshape(NB,8,32)
    opql=((rw8[:,:,0:16]&0x0F)|((rw8[:,:,16:32]&0x0F)<<4)).reshape(NB,128).astype(np.uint8)
    rw16=rw_flat.reshape(NB,16,16)
    a=rw16[:,:,0:4]&0x30; b=rw16[:,:,4:8]&0x30; c=rw16[:,:,8:12]&0x30; e=rw16[:,:,12:16]&0x30
    opqh=((a>>4)|(b>>2)|c|(e<<2)).reshape(NB,64).astype(np.uint8)
    return opql.reshape(-1), opqh.reshape(-1), sc_r.reshape(-1).copy(), d_b.reshape(-1).copy()


# ---------------------------------------------------------------------------
# Reference dequantize -> W[N, K] float32
# ---------------------------------------------------------------------------
def deq_q4k(pqs, psl, K, N):
    nprow=K//256; NB=N*nprow
    qs_b=pqs.reshape(NB,8,16); sl_b=psl.reshape(NB,16)
    sli=sl_b[:,0].astype(np.uint32)|(sl_b[:,1].astype(np.uint32)<<8)|(sl_b[:,2].astype(np.uint32)<<16)|(sl_b[:,3].astype(np.uint32)<<24)
    mli=sl_b[:,4].astype(np.uint32)|(sl_b[:,5].astype(np.uint32)<<8)|(sl_b[:,6].astype(np.uint32)<<16)|(sl_b[:,7].astype(np.uint32)<<24)
    shi=sl_b[:,8].astype(np.uint32)|(sl_b[:,9].astype(np.uint32)<<8)
    mhi=sl_b[:,10].astype(np.uint32)|(sl_b[:,11].astype(np.uint32)<<8)
    d=sl_b[:,12:14].copy().view(np.float16).reshape(NB).astype(np.float32)
    dmin=sl_b[:,14:16].copy().view(np.float16).reshape(NB).astype(np.float32)
    j=np.arange(8,dtype=np.uint32)
    sq=((sli[:,None]>>(j*4))&0xF)|(((shi[:,None]>>(j*2))&3)<<4)
    mq=((mli[:,None]>>(j*4))&0xF)|(((mhi[:,None]>>(j*2))&3)<<4)
    sc=sq.astype(np.float32)*d[:,None]; mv=mq.astype(np.float32)*dmin[:,None]
    lo=(qs_b&0x0F).astype(np.float32); hi=((qs_b>>4)&0x0F).astype(np.float32)
    W=np.empty((NB,8,32),dtype=np.float32)
    W[:,:,0:16]=lo*sc[:,:,None]-mv[:,:,None]; W[:,:,16:32]=hi*sc[:,:,None]-mv[:,:,None]
    return W.reshape(N,nprow,256).reshape(N,K)


def deq_q5k(pqs, pqh, psl, K, N):
    nprow=K//256; NB=N*nprow
    qs_b=pqs.reshape(NB,8,16); qh_b=pqh.reshape(NB,8,4); sl_b=psl.reshape(NB,16)
    sli=sl_b[:,0].astype(np.uint32)|(sl_b[:,1].astype(np.uint32)<<8)|(sl_b[:,2].astype(np.uint32)<<16)|(sl_b[:,3].astype(np.uint32)<<24)
    mli=sl_b[:,4].astype(np.uint32)|(sl_b[:,5].astype(np.uint32)<<8)|(sl_b[:,6].astype(np.uint32)<<16)|(sl_b[:,7].astype(np.uint32)<<24)
    shi=sl_b[:,8].astype(np.uint32)|(sl_b[:,9].astype(np.uint32)<<8)
    mhi=sl_b[:,10].astype(np.uint32)|(sl_b[:,11].astype(np.uint32)<<8)
    d=sl_b[:,12:14].copy().view(np.float16).reshape(NB).astype(np.float32)
    dmin=sl_b[:,14:16].copy().view(np.float16).reshape(NB).astype(np.float32)
    j=np.arange(8,dtype=np.uint32)
    sq=((sli[:,None]>>(j*4))&0xF)|(((shi[:,None]>>(j*2))&3)<<4)
    mq=((mli[:,None]>>(j*4))&0xF)|(((mhi[:,None]>>(j*2))&3)<<4)
    sc=sq.astype(np.float32)*d[:,None]; mv=mq.astype(np.float32)*dmin[:,None]
    rh=np.zeros((NB,8,32),dtype=np.uint8)
    for s in range(8): rh[:,:,4*s:4*s+4]=(qh_b>>s)&1
    lo_b=rh[:,:,0:16].astype(np.float32); hi_b=rh[:,:,16:32].astype(np.float32)
    w0=(qs_b&0x0F).astype(np.float32)+lo_b*16.0; w1=((qs_b>>4)&0x0F).astype(np.float32)+hi_b*16.0
    W=np.empty((NB,8,32),dtype=np.float32)
    W[:,:,0:16]=w0*sc[:,:,None]-mv[:,:,None]; W[:,:,16:32]=w1*sc[:,:,None]-mv[:,:,None]
    return W.reshape(N,nprow,256).reshape(N,K)


def deq_q6k(pql, pqh, ps_u8, pd_buf, K, N):
    nprow=K//256; NB=N*nprow
    ql_b=pql.reshape(NB,8,16); qh_b=pqh.reshape(NB,8,8)
    s_b=ps_u8.reshape(NB,16).view(np.int8)
    d=pd_buf.reshape(NB,2).copy().view(np.float16).reshape(NB).astype(np.float32)
    clo=d[:,None]*s_b[:,0::2].astype(np.float32); chi=d[:,None]*s_b[:,1::2].astype(np.float32)
    hlo=(qh_b[:,:,0].astype(np.uint32)|(qh_b[:,:,1].astype(np.uint32)<<8)|
         (qh_b[:,:,2].astype(np.uint32)<<16)|(qh_b[:,:,3].astype(np.uint32)<<24))
    hhi=(qh_b[:,:,4].astype(np.uint32)|(qh_b[:,:,5].astype(np.uint32)<<8)|
         (qh_b[:,:,6].astype(np.uint32)<<16)|(qh_b[:,:,7].astype(np.uint32)<<24))
    SH=np.array([0,8,16,24,2,10,18,26,4,12,20,28,6,14,22,30],dtype=np.uint32)
    w0=(ql_b&0x0F).astype(np.float32)+(((hlo[:,:,None]>>SH[None,None,:])&3).astype(np.float32)*16.0)
    w0=(w0-32.0)*clo[:,:,None]
    w1=((ql_b>>4)&0x0F).astype(np.float32)+(((hhi[:,:,None]>>SH[None,None,:])&3).astype(np.float32)*16.0)
    w1=(w1-32.0)*chi[:,:,None]
    W=np.empty((NB,8,32),dtype=np.float32); W[:,:,0:16]=w0; W[:,:,16:32]=w1
    return W.reshape(N,nprow,256).reshape(N,K)


# ---------------------------------------------------------------------------
# Build SG-transposed weight buffers (same layout as gemv kernels)
# ---------------------------------------------------------------------------
def build_q4k_flat(K, N, seed=1234):
    from test_dense_gemv import build_q4k_weight
    return build_q4k_weight(K, N, seed)


def build_q5k_flat(K, N, seed=1234):
    from test_dense_gemv import build_q5k_weight
    return build_q5k_weight(K, N, seed)


def build_q6k_flat(K, N, seed=1234):
    from test_dense_gemv import build_q6k_weight
    return build_q6k_weight(K, N, seed)


# ===========================================================================
# Test runner
# ===========================================================================
def test_gemm_slim(gpu, hw, quant,
                   K=2048, N=2048, token_len=8,
                   seed=42, verbose=False,
                   iters=50, warmup=5, do_bench=True):
    """
    Test gemm_q{4,5,6}k_slim CM kernel (v3: unified layout).
    y = W * X  where W: [N, K] quantized (SG-transposed), X: [token_len, K] fp16
    Output: [token_len, N] fp32
    Dispatch: global (threads) = (N/OPG * nbpr, token_len)  [2D]
              local  (threads) = (nbpr, 1)
    """
    label = f"gemm_{quant}_slim  N={N} K={K} token_len={token_len}"
    print(f"\n  [{label}]")
    assert N % OPG == 0, "N must be a multiple of OPG=16"
    assert K % 256 == 0, "K must be a multiple of 256"

    nbpr = K // 256

    if quant == "q4k":
        flat, ref_W, wbytes = build_q4k_flat(K, N, seed)
        cm_file = "gemm_q4k_slim.cm"
        kernel_name = "gemm_q4k_slim"
    elif quant == "q5k":
        flat, ref_W, wbytes = build_q5k_flat(K, N, seed)
        cm_file = "gemm_q5k_slim.cm"
        kernel_name = "gemm_q5k_slim"
    elif quant == "q6k":
        flat, ref_W, wbytes = build_q6k_flat(K, N, seed)
        cm_file = "gemm_q6k_slim.cm"
        kernel_name = "gemm_q6k_slim"
    else:
        raise ValueError(f"Unknown quant: {quant}")

    rng = np.random.default_rng(seed + 1)
    x_f32 = rng.standard_normal((token_len, K)).astype(np.float32)
    x_in = x_f32.astype(np.float16)

    # Reference: fp16 activations + fp16-rounded dequantized weights
    # (matching the kernel's computation precision)
    x_ref = x_in.astype(np.float64)
    if quant == "q6k":
        # q6k_slim uses cm_dpas (fp16); round weight reference too
        w_ref = ref_W.astype(np.float16).astype(np.float64)
    else:
        # q4k/q5k slim use scalar dot product with fp16 inputs; weight is fp32
        w_ref = ref_W.astype(np.float64)
    ref = (w_ref @ x_ref.T).astype(np.float32).T

    # Build & run GPU kernel (CM front-end: '-cmc')
    prog = gpu.build(cl_src(cm_file), "-cmc")
    krn = cl.Kernel(prog, kernel_name)

    x_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(x_in))
    w_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=flat)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=token_len * N * 4)

    krn.set_args(x_b, w_b, out_b,
                 np.uint32(token_len), np.uint32(K), np.uint32(N))

    # Dispatch: 2D, global=(N/OPG*nbpr, token_len), local=(nbpr,1)
    ngroups = N // OPG
    gsize = (ngroups * nbpr, token_len)
    lsize = (nbpr, 1)
    print(f"    gsize={gsize}  lsize={lsize}  ngroups={ngroups}  nbpr={nbpr}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue); ev.wait()
    got = np.empty(token_len * N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, N)

    ok = check_close(kernel_name, ref, got, verbose=verbose)

    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        flops = token_len * 2 * N * K
        moved = wbytes + token_len * K * 2 + token_len * N * 4  # fp16 inputs
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms")
        _print_rl(rl, moved)
    return ok, timing, rl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--peak_bw", type=float, default=456.0)
    ap.add_argument("--peak_fp16", type=float, default=116000.0,
                    help="FP16 peak compute throughput in GFLOPS (BMG: 116000)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    hw = HW(peak_bw_gbps=args.peak_bw, fp16_gflops=args.peak_fp16)
    gpu = Gpu(args.device)
    if not args.no_bench:
        gpu.setup_flush(args.flush_mib)

    kw = dict(verbose=args.verbose, iters=args.iters, warmup=args.warmup,
              do_bench=not args.no_bench)

    SEP = "=" * 80
    print(SEP)
    print("  Dense slim GEMM CM kernels: gemm_q4k_slim / gemm_q5k_slim / gemm_q6k_slim")
    print(SEP)

    results = []
    shapes = [
        # (quant, K, N, token_len)
        ("q4k", 2048, 2048,    1),
        ("q4k", 2048, 2048,   16),
        ("q4k", 2048, 2048, 1024),
        ("q4k", 4096, 4096,    1),
        ("q4k", 4096, 4096, 1024),
        ("q5k", 2048, 2048,    1),
        ("q5k", 2048, 2048, 1024),
        ("q5k", 4096, 4096,    1),
        ("q6k", 2048, 2048,    1),
        ("q6k", 2048, 2048,   17),   # non-multiple-of-8 tail tile (dpas path)
        ("q6k", 4096, 4096, 1024),
    ]
    for quant, K, N, token_len in shapes:
        ok, timing, rl = test_gemm_slim(gpu, hw, quant, K=K, N=N,
                                        token_len=token_len, **kw)
        results.append((quant, K, N, token_len, ok))

    print(f"\n{SEP}")
    print("  Summary:")
    for quant, K, N, token_len, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] gemm_{quant}_slim  N={N} K={K} token_len={token_len}")
    all_ok = all(r[-1] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
