"""
test_moe_gemv.py

Correctness + benchmark + roofline harness for MoE GEMV CM (C-for-Metal)
kernels:
    moe_gemv_q4k_sg.cm : group_up_gate_q4k_sg, down_merge_q4k_sg
    moe_gemv_q5k_sg.cm : group_up_gate_q5k_sg, down_merge_q5k_sg
    moe_gemv_q6k_sg.cm : group_up_gate_q6k_sg, down_merge_q6k_sg
    moe_gemv_q80_sg.cm : shared_gate_up_q8_0,  shared_down_merge_q8_0

Converted from sycl_gguf_kernel/gemv_q{4,5,6}k_L1*.cpp (MoE up/gate/down
paths). These CM kernels use the same SG-transposed per-expert weight layout
as the OpenCL reference port (/mnt/river/ovmx/ocl_gguf_kernel/) and as
q4k_moe_gemv/test_moe_gemv_sg_kernels.py, but have NO sub-group/lane
dimension: each CM thread computes all OPG=16 output rows of a row-group
directly via vector<float,16> SIMD arithmetic. Built with the '-cmc' OpenCL
build option (Intel NEO's CM front-end).

MoE-FFN shapes (Qwen3-35B-A3B style):
    gate_up : token_len=1, K=2048, output_len=512,  256 experts, top-8 active
    down    : token_len=1, K=512,  output_len=2048, 256 experts, top-8 active

Usage:
    python test_moe_gemv.py [--device 0] [--iters 50] [--warmup 5]
        [--flush_mib 128] [--peak_bw 456] [--peak_fp16 116000]
        [--verbose] [--no-bench]
"""

import os
import statistics
import argparse
import sys

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

    def setup_flush(self, mib=128):
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


def _print_rl(rl, moved):
    print(f"    BW={rl['achieved_bw_gbps']:.1f} GB/s ({rl['bw_util_pct']:.1f}% peak)  "
          f"AI={rl['arith_intensity']:.2f}  roofline={rl['roofline_pct']:.1f}%  "
          f"bound={rl['bound_by']}  FP16-GFLOPS={rl['achieved_gflops']:.1f}")
    print(f"    roofline latency: compute={rl['compute_bound_ms']:.3f} ms  "
          f"memory={rl['memory_bound_ms']:.3f} ms  ->  bound={rl['roofline_ms']:.3f} ms")
    print(f"    moved={moved/1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Shared data generators and packing helpers (identical to test_dense_gemv.py)
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


def shuffle_q4k(raw, K, N):
    NB=N*K//256; blk=raw.reshape(NB,144)
    d_b,dm_b=blk[:,0:2],blk[:,2:4]; sc=blk[:,4:16]; qs=blk[:,16:144].reshape(NB,4,32)
    low=qs&0x0F; high=(qs>>4)&0x0F
    rw=np.empty((NB,4,64),dtype=np.uint8); rw[:,:,0:32]=low; rw[:,:,32:64]=high; rw=rw.reshape(NB,256)
    (sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11)=(sc[:,i] for i in range(12))
    rs=np.empty((NB,8),dtype=np.uint8); rm=np.empty((NB,8),dtype=np.uint8)
    rs[:,0]=sc0&63;rs[:,1]=sc1&63;rs[:,2]=sc2&63;rs[:,3]=sc3&63
    rs[:,4]=(sc8&0x0F)|((sc0>>2)&0x30);rs[:,5]=(sc9&0x0F)|((sc1>>2)&0x30)
    rs[:,6]=(sc10&0x0F)|((sc2>>2)&0x30);rs[:,7]=(sc11&0x0F)|((sc3>>2)&0x30)
    rm[:,0]=sc4&63;rm[:,1]=sc5&63;rm[:,2]=sc6&63;rm[:,3]=sc7&63
    rm[:,4]=((sc8>>4)&0x0F)|((sc4>>2)&0x30);rm[:,5]=((sc9>>4)&0x0F)|((sc5>>2)&0x30)
    rm[:,6]=((sc10>>4)&0x0F)|((sc6>>2)&0x30);rm[:,7]=((sc11>>4)&0x0F)|((sc7>>2)&0x30)
    rw8=rw.reshape(NB,8,32)
    opqs=((rw8[:,:,0:16]&0x0F)|((rw8[:,:,16:32]&0x0F)<<4)).reshape(NB,128).astype(np.uint8)
    opsl=np.zeros((NB,16),dtype=np.uint8)
    opsl[:,0]=(rs[:,0]&0x0F)|((rs[:,1]&0x0F)<<4);opsl[:,1]=(rs[:,2]&0x0F)|((rs[:,3]&0x0F)<<4)
    opsl[:,2]=(rs[:,4]&0x0F)|((rs[:,5]&0x0F)<<4);opsl[:,3]=(rs[:,6]&0x0F)|((rs[:,7]&0x0F)<<4)
    opsl[:,4]=(rm[:,0]&0x0F)|((rm[:,1]&0x0F)<<4);opsl[:,5]=(rm[:,2]&0x0F)|((rm[:,3]&0x0F)<<4)
    opsl[:,6]=(rm[:,4]&0x0F)|((rm[:,5]&0x0F)<<4);opsl[:,7]=(rm[:,6]&0x0F)|((rm[:,7]&0x0F)<<4)
    opsl[:,8]=((rs[:,0]&0x30)>>4)|((rs[:,1]&0x30)>>2)|(rs[:,2]&0x30)|((rs[:,3]&0x30)<<2)
    opsl[:,9]=((rs[:,4]&0x30)>>4)|((rs[:,5]&0x30)>>2)|(rs[:,6]&0x30)|((rs[:,7]&0x30)<<2)
    opsl[:,10]=((rm[:,0]&0x30)>>4)|((rm[:,1]&0x30)>>2)|(rm[:,2]&0x30)|((rm[:,3]&0x30)<<2)
    opsl[:,11]=((rm[:,4]&0x30)>>4)|((rm[:,5]&0x30)>>2)|(rm[:,6]&0x30)|((rm[:,7]&0x30)<<2)
    opsl[:,12:14]=d_b;opsl[:,14:16]=dm_b
    return opqs.reshape(-1), opsl.reshape(-1)


def shuffle_q5k(raw, K, N):
    NB=N*K//256; blk=raw.reshape(NB,176)
    d_b,dm_b=blk[:,0:2],blk[:,2:4]; sc=blk[:,4:16]; qh_r=blk[:,16:48]; qs_r=blk[:,48:176].reshape(NB,4,32)
    g_arr=np.arange(8,dtype=np.uint8)
    rh=((qh_r[:,None,:]>>g_arr[None,:,None])&1).astype(np.uint8)
    low=qs_r&0x0F; high=(qs_r>>4)&0x0F
    rw=np.empty((NB,8,32),dtype=np.uint8); rw[:,0::2,:]=low; rw[:,1::2,:]=high
    (sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9,sc10,sc11)=(sc[:,i] for i in range(12))
    rs=np.empty((NB,8),dtype=np.uint8); rm=np.empty((NB,8),dtype=np.uint8)
    rs[:,0]=sc0&63;rs[:,1]=sc1&63;rs[:,2]=sc2&63;rs[:,3]=sc3&63
    rs[:,4]=(sc8&0x0F)|((sc0>>2)&0x30);rs[:,5]=(sc9&0x0F)|((sc1>>2)&0x30)
    rs[:,6]=(sc10&0x0F)|((sc2>>2)&0x30);rs[:,7]=(sc11&0x0F)|((sc3>>2)&0x30)
    rm[:,0]=sc4&63;rm[:,1]=sc5&63;rm[:,2]=sc6&63;rm[:,3]=sc7&63
    rm[:,4]=((sc8>>4)&0x0F)|((sc4>>2)&0x30);rm[:,5]=((sc9>>4)&0x0F)|((sc5>>2)&0x30)
    rm[:,6]=((sc10>>4)&0x0F)|((sc6>>2)&0x30);rm[:,7]=((sc11>>4)&0x0F)|((sc7>>2)&0x30)
    rw8=rw.reshape(NB,8,32)
    opqs=((rw8[:,:,0:16]&0x0F)|((rw8[:,:,16:32]&0x0F)<<4)).reshape(NB,128).astype(np.uint8)
    opqh=np.zeros((NB,8,4),dtype=np.uint8)
    for s in range(8): opqh|=(rh[:,:,s*4:s*4+4]<<s).astype(np.uint8)
    opqh=opqh.reshape(NB,32)
    opsl=np.zeros((NB,16),dtype=np.uint8)
    opsl[:,0]=(rs[:,0]&0x0F)|((rs[:,1]&0x0F)<<4);opsl[:,1]=(rs[:,2]&0x0F)|((rs[:,3]&0x0F)<<4)
    opsl[:,2]=(rs[:,4]&0x0F)|((rs[:,5]&0x0F)<<4);opsl[:,3]=(rs[:,6]&0x0F)|((rs[:,7]&0x0F)<<4)
    opsl[:,4]=(rm[:,0]&0x0F)|((rm[:,1]&0x0F)<<4);opsl[:,5]=(rm[:,2]&0x0F)|((rm[:,3]&0x0F)<<4)
    opsl[:,6]=(rm[:,4]&0x0F)|((rm[:,5]&0x0F)<<4);opsl[:,7]=(rm[:,6]&0x0F)|((rm[:,7]&0x0F)<<4)
    opsl[:,8]=((rs[:,0]&0x30)>>4)|((rs[:,1]&0x30)>>2)|(rs[:,2]&0x30)|((rs[:,3]&0x30)<<2)
    opsl[:,9]=((rs[:,4]&0x30)>>4)|((rs[:,5]&0x30)>>2)|(rs[:,6]&0x30)|((rs[:,7]&0x30)<<2)
    opsl[:,10]=((rm[:,0]&0x30)>>4)|((rm[:,1]&0x30)>>2)|(rm[:,2]&0x30)|((rm[:,3]&0x30)<<2)
    opsl[:,11]=((rm[:,4]&0x30)>>4)|((rm[:,5]&0x30)>>2)|(rm[:,6]&0x30)|((rm[:,7]&0x30)<<2)
    opsl[:,12:14]=d_b;opsl[:,14:16]=dm_b
    return opqs.reshape(-1), opqh.reshape(-1), opsl.reshape(-1)


def shuffle_q6k(raw, K, N):
    NB=N*K//256; blk=raw.reshape(NB,210)
    ql_r=blk[:,0:128].reshape(NB,2,64); qh_r=blk[:,128:192].reshape(NB,2,32)
    sc_r=blk[:,192:208]; d_b=blk[:,208:210]
    low=ql_r&0x0F; high=(ql_r>>4)&0x0F
    rw=np.empty((NB,2,128),dtype=np.uint8); rw[:,:,0:64]=low; rw[:,:,64:128]=high
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


def _ascon(a): return np.ascontiguousarray(a)


def pack_qk_pqs_sg(pqs, K, N, opg=OPG):
    nbpr=K//256; nrg=N//opg
    return _ascon(pqs.reshape(nrg,opg,nbpr,8,4,4).transpose(0,2,3,4,1,5)).reshape(-1).astype(np.uint8)


def pack_qk_psl_sg(psl, K, N, opg=OPG):
    nbpr=K//256; nrg=N//opg
    p=_ascon(psl.reshape(nrg,opg,nbpr,16).transpose(0,2,1,3))
    sl=_ascon(p[...,0:4]).reshape(nrg,nbpr,opg*4); ml=_ascon(p[...,4:8]).reshape(nrg,nbpr,opg*4)
    sh=_ascon(p[...,8:10]).reshape(nrg,nbpr,opg*2); mh=_ascon(p[...,10:12]).reshape(nrg,nbpr,opg*2)
    d=_ascon(p[...,12:14]).reshape(nrg,nbpr,opg*2); dm=_ascon(p[...,14:16]).reshape(nrg,nbpr,opg*2)
    return np.concatenate([sl,ml,sh,mh,d,dm],axis=2).reshape(-1).astype(np.uint8)


def pack_q5k_pqh_sg(pqh, K, N, opg=OPG):
    nbpr=K//256; nrg=N//opg
    return _ascon(pqh.reshape(nrg,opg,nbpr,8,4).transpose(0,2,3,1,4)).reshape(-1).astype(np.uint8)


def pack_q6k_pqh_sg(pqh, K, N, opg=OPG):
    nbpr=K//256; nrg=N//opg
    return _ascon(pqh.reshape(nrg,opg,nbpr,8,2,4).transpose(0,2,3,4,1,5)).reshape(-1).astype(np.uint8)


def pack_q6k_ps_sg(ps, K, N, opg=OPG):
    nbpr=K//256; nrg=N//opg
    return _ascon(ps.reshape(nrg,opg,nbpr,16).transpose(0,2,3,1)).reshape(-1).astype(np.uint8)


def pack_q6k_pd_sg(pd, K, N, opg=OPG):
    nbpr=K//256; nrg=N//opg
    return _ascon(pd.reshape(nrg,opg,nbpr,2).transpose(0,2,1,3)).reshape(-1).astype(np.uint8)


def silu(x):
    x=np.asarray(x,dtype=np.float64); pos=x>=0; sig=np.empty_like(x)
    sig[pos]=1.0/(1.0+np.exp(-x[pos])); ex=np.exp(x[~pos]); sig[~pos]=ex/(1.0+ex)
    return (x*sig).astype(np.float32)


# ---------------------------------------------------------------------------
# Expert pool builders (minimal, single expert for quick smoke test)
# ---------------------------------------------------------------------------
def build_q4k_single_expert(K, N, seed=1234):
    nprow=N*K//256; raw=gen_q4k_blocks(nprow,seed)
    pqs,psl=shuffle_q4k(raw,K,N); ref_W=deq_q4k(pqs,psl,K,N)
    pqs_T=pack_qk_pqs_sg(pqs,K,N); psl_T=pack_qk_psl_sg(psl,K,N)
    return np.concatenate([pqs_T,psl_T]), ref_W, nprow*144


def build_q5k_single_expert(K, N, seed=1234):
    nprow=N*K//256; raw=gen_q5k_blocks(nprow,seed)
    pqs,pqh,psl=shuffle_q5k(raw,K,N); ref_W=deq_q5k(pqs,pqh,psl,K,N)
    pqs_T=pack_qk_pqs_sg(pqs,K,N); pqh_T=pack_q5k_pqh_sg(pqh,K,N); psl_T=pack_qk_psl_sg(psl,K,N)
    return np.concatenate([pqs_T,pqh_T,psl_T]), ref_W, nprow*176


def build_q6k_single_expert(K, N, seed=1234):
    nprow=N*K//256; raw=gen_q6k_blocks(nprow,seed)
    pql,pqh,ps,pd=shuffle_q6k(raw,K,N); ref_W=deq_q6k(pql,pqh,ps,pd,K,N)
    pql_T=pack_qk_pqs_sg(pql,K,N); pqh_T=pack_q6k_pqh_sg(pqh,K,N)
    ps_T=pack_q6k_ps_sg(ps,K,N); pd_T=pack_q6k_pd_sg(pd,K,N)
    return np.concatenate([pql_T,pqh_T,ps_T,pd_T]), ref_W, nprow*210


def _pow2_floor(x):
    p=1
    while p*2<=x: p*=2
    return p


def choose_ksplit(base_groups, split_space, target=2048, maxk=8):
    if base_groups >= 512: return 1
    ks=max(1,round(target/max(base_groups,1)))
    ks=min(ks,split_space,maxk)
    return _pow2_floor(ks)


_CM_SRC = dict(q4k="moe_gemv_q4k_sg.cm", q5k="moe_gemv_q5k_sg.cm", q6k="moe_gemv_q6k_sg.cm")
_BUILD_FN = dict(q4k=build_q4k_single_expert, q5k=build_q5k_single_expert, q6k=build_q6k_single_expert)
_BPB = dict(q4k=144, q5k=176, q6k=210)


def test_group_up_gate(gpu, hw, quant, K=2048, output_len=512,
                       token_len=1, seed=1, verbose=False,
                       iters=50, warmup=5, do_bench=True):
    """Single-expert group_up_gate (no routing index, index=0 only)."""
    label = f"group_up_gate_{quant}_sg  K={K} N={output_len} token_len={token_len} [single expert]"
    print(f"\n  [{label}]")
    assert output_len % OPG == 0

    ups_flat,  ups_ref,  expert_size = _BUILD_FN[quant](K, output_len, seed)
    gates_flat, gates_ref, _         = _BUILD_FN[quant](K, output_len, seed + 999)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((token_len, K)).astype(np.float16)
    x_f32 = x.astype(np.float32)

    # Reference (single active expert = index 0)
    output_num = 1
    aligned_output_num = OPG
    ref = np.zeros((token_len, output_num, output_len), dtype=np.float32)
    for t in range(token_len):
        up = ups_ref @ x_f32[t]
        gate = gates_ref @ x_f32[t]
        ref[t, 0] = silu(gate) * up

    ksplit = choose_ksplit((output_len // OPG) * token_len, K // 256)
    prog = gpu.build(cl_src(_CM_SRC[quant]), f"-cmc -DKSPLIT={ksplit}")
    krn = cl.Kernel(prog, f"group_up_gate_{quant}_sg")

    x_b = _cbuf(gpu.ctx, x)
    ups_b = _cbuf(gpu.ctx, ups_flat)
    gates_b = _cbuf(gpu.ctx, gates_flat)
    # Pass an all-zero index buffer (kernel also accepts NULL -> uses v as index)
    idx_buf = np.zeros(token_len * aligned_output_num, dtype=np.uint32)  # index[*]=0
    idx_b = _cbuf(gpu.ctx, idx_buf)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY,
                      size=token_len * output_num * output_len * 4)

    krn.set_args(x_b, ups_b, gates_b, idx_b, out_b,
                 np.uint32(output_num), np.uint32(aligned_output_num),
                 np.uint32(token_len), np.uint32(K), np.uint32(output_len))

    # CM has no sub-group lane dimension: one thread computes OPG=16 rows.
    gsize = (output_len // OPG, ksplit, output_num * token_len)
    lsize = (1, ksplit, 1)
    print(f"    KSPLIT={ksplit}  gsize={gsize}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue); ev.wait()
    got = np.empty(token_len * output_num * output_len, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, output_num, output_len)

    ok = check_close(f"group_up_gate_{quant}_sg", ref, got, verbose=verbose)

    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup)
        timing = stats(ts)
        flops = token_len * output_num * 2 * (2 * output_len * K)
        moved = token_len * output_num * 2 * expert_size + token_len * K * 2 + token_len * output_num * output_len * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  min={timing['min_ms']:.3f} ms")
        _print_rl(rl, moved)
    return ok, timing, rl


def test_down_merge(gpu, hw, quant, K=512, output_len=2048,
                    token_len=1, seed=2, verbose=False,
                    iters=50, warmup=5, do_bench=True):
    """Single-expert down_merge."""
    label = f"down_merge_{quant}_sg  K={K} N={output_len} token_len={token_len} [single expert]"
    print(f"\n  [{label}]")
    assert output_len % OPG == 0

    downs_flat, downs_ref, expert_size = _BUILD_FN[quant](K, output_len, seed)

    rng = np.random.default_rng(seed)
    input_num = 1; aligned_input_num = OPG
    idx = np.zeros(token_len * aligned_input_num, dtype=np.uint32)
    eff = rng.uniform(0.1, 1.0, token_len * aligned_input_num).astype(np.float32)
    inputs = rng.standard_normal((token_len, input_num, K)).astype(np.float16)
    inputs_f32 = inputs.astype(np.float32)

    ref = np.zeros((token_len, output_len), dtype=np.float32)
    for t in range(token_len):
        ref[t] = float(eff[t * aligned_input_num]) * (downs_ref @ inputs_f32[t, 0])

    ksplit = choose_ksplit((output_len // OPG) * token_len, input_num * (K // 256))
    prog = gpu.build(cl_src(_CM_SRC[quant]), f"-cmc -DKSPLIT={ksplit}")
    krn = cl.Kernel(prog, f"down_merge_{quant}_sg")

    in_b = _cbuf(gpu.ctx, inputs)
    downs_b = _cbuf(gpu.ctx, downs_flat)
    idx_b = _cbuf(gpu.ctx, idx)
    eff_b = _cbuf(gpu.ctx, eff)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_WRITE, size=token_len * output_len * 4)

    krn.set_args(in_b, downs_b, idx_b, eff_b, out_b,
                 np.uint32(input_num), np.uint32(aligned_input_num),
                 np.uint32(token_len), np.uint32(K), np.uint32(output_len),
                 np.int32(0))

    gsize = (output_len // OPG, ksplit, token_len)
    lsize = (1, ksplit, 1)

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue); ev.wait()
    got = np.empty(token_len * output_len, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, output_len)

    ok = check_close(f"down_merge_{quant}_sg", ref, got, verbose=verbose)

    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup)
        timing = stats(ts)
        flops = token_len * input_num * 2 * output_len * K
        moved = token_len * input_num * expert_size + token_len * input_num * K * 2 + token_len * output_len * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  min={timing['min_ms']:.3f} ms")
        _print_rl(rl, moved)
    return ok, timing, rl


# ---------------------------------------------------------------------------
# Q8_0 shared-expert tests (moe_gemv_q80_sg.cm)
# ---------------------------------------------------------------------------
def pack_q80_pqs_sg(pqs, K, N, opg=OPG):
    nbpr = K // 256; nrg = N // opg
    src = pqs.reshape(nrg, opg, nbpr, 8, 8, 4)
    return _ascon(src.transpose(0, 2, 3, 4, 1, 5)).reshape(-1).astype(np.uint8)


def pack_q80_pd_sg(pd, K, N, opg=OPG):
    nbpr = K // 256; nrg = N // opg
    src = pd.reshape(nrg, opg, nbpr, 8, 2)
    return _ascon(src.transpose(0, 2, 3, 1, 4)).reshape(-1).astype(np.uint8)


def gen_q80_row_major(N, K, seed=1234):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((N, K)).astype(np.float32) * 0.05)
    nblk = K // 32
    Wb = W.reshape(N, nblk, 32)
    amax = np.max(np.abs(Wb), axis=2)
    d = (amax / 127.0).astype(np.float32)
    d_safe = np.where(d == 0.0, 1.0, d)
    q = np.round(Wb / d_safe[:, :, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    ref_W = (q.astype(np.float32) * d16.astype(np.float32)[:, :, None]).reshape(N, K)
    q_bytes = q.view(np.uint8).reshape(N, K)
    d_bytes = d16.view(np.uint8).reshape(N, nblk * 2)
    return ref_W, q_bytes, d_bytes


def build_q80_shared_weight(N, K, seed=1234):
    assert K % 256 == 0 and N % OPG == 0
    ref_W, q_bytes, d_bytes = gen_q80_row_major(N, K, seed=seed)
    pqs_T = pack_q80_pqs_sg(q_bytes.reshape(-1), K, N)
    pd_T = pack_q80_pd_sg(d_bytes.reshape(-1), K, N)
    flat = np.concatenate([pqs_T, pd_T])
    return flat, ref_W


def test_shared_gate_up_q80(gpu, hw, hidden_size=2048, intermediate_size=512,
                            token_len=1, seed=3, verbose=False,
                            iters=50, warmup=5, do_bench=True):
    label = f"shared_gate_up_q8_0  hidden={hidden_size} inter={intermediate_size} token_len={token_len}"
    print(f"\n  [{label}]")
    gate_flat, gate_ref = build_q80_shared_weight(intermediate_size, hidden_size, seed)
    up_flat, up_ref = build_q80_shared_weight(intermediate_size, hidden_size, seed + 777)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((token_len, hidden_size)).astype(np.float16)
    x_f32 = x.astype(np.float32)
    ref = np.zeros((token_len, intermediate_size), dtype=np.float32)
    for t in range(token_len):
        gate = gate_ref @ x_f32[t]
        up = up_ref @ x_f32[t]
        ref[t] = silu(gate) * up

    prog = gpu.build(cl_src("moe_gemv_q80_sg.cm"), "-cmc")
    krn = cl.Kernel(prog, "shared_gate_up_q8_0")

    x_b = _cbuf(gpu.ctx, x)
    gate_b = _cbuf(gpu.ctx, gate_flat)
    up_b = _cbuf(gpu.ctx, up_flat)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=token_len * intermediate_size * 2)

    krn.set_args(gate_b, up_b, x_b, out_b, np.uint32(hidden_size), np.uint32(intermediate_size))
    gsize = (intermediate_size // OPG, 1, token_len)
    lsize = (1, 1, 1)

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue); ev.wait()
    got_h = np.empty(token_len * intermediate_size, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, got_h, out_b)
    gpu.queue.finish()
    got = got_h.astype(np.float32).reshape(token_len, intermediate_size)

    ok = check_close("shared_gate_up_q8_0", ref, got, rtol=2e-2, atol=2e-2, verbose=verbose)
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup)
        timing = stats(ts)
        print(f"    mean={timing['mean_ms']:.3f} ms  min={timing['min_ms']:.3f} ms")
    return ok


def test_shared_down_merge_q80(gpu, hw, hidden_size=2048, intermediate_size=512,
                               token_len=1, seed=4, verbose=False,
                               iters=50, warmup=5, do_bench=True):
    label = f"shared_down_merge_q8_0  hidden={hidden_size} inter={intermediate_size} token_len={token_len}"
    print(f"\n  [{label}]")
    down_flat, down_ref = build_q80_shared_weight(hidden_size, intermediate_size, seed)

    rng = np.random.default_rng(seed)
    gate_up = rng.standard_normal((token_len, intermediate_size)).astype(np.float16)
    gate_up_f32 = gate_up.astype(np.float32)
    shared_gate = rng.uniform(0.1, 1.0, token_len).astype(np.float16)
    shared_gate_f32 = shared_gate.astype(np.float32)

    ref = np.zeros((token_len, hidden_size), dtype=np.float32)
    for t in range(token_len):
        ref[t] = shared_gate_f32[t] * (down_ref @ gate_up_f32[t])

    prog = gpu.build(cl_src("moe_gemv_q80_sg.cm"), "-cmc")
    krn = cl.Kernel(prog, "shared_down_merge_q8_0")

    gu_b = _cbuf(gpu.ctx, gate_up)
    down_b = _cbuf(gpu.ctx, down_flat)
    sg_b = _cbuf(gpu.ctx, shared_gate)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=token_len * hidden_size * 2)

    krn.set_args(gu_b, down_b, sg_b, out_b,
                 np.uint32(intermediate_size), np.uint32(hidden_size), np.int32(0))
    gsize = (hidden_size // OPG, 1, token_len)
    lsize = (1, 1, 1)

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue); ev.wait()
    got_h = np.empty(token_len * hidden_size, dtype=np.float16)
    cl.enqueue_copy(gpu.queue, got_h, out_b)
    gpu.queue.finish()
    got = got_h.astype(np.float32).reshape(token_len, hidden_size)

    ok = check_close("shared_down_merge_q8_0", ref, got, rtol=2e-2, atol=2e-2, verbose=verbose)
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup)
        timing = stats(ts)
        print(f"    mean={timing['mean_ms']:.3f} ms  min={timing['min_ms']:.3f} ms")
    return ok


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
    print("  MoE GEMV CM kernels (cm_gguf_kernel/): q4k / q5k / q6k / q8_0-shared")
    print(SEP)

    results = []
    for quant in ["q4k", "q5k", "q6k"]:
        ok1, _, _ = test_group_up_gate(gpu, hw, quant,
                                        K=2048, output_len=512, token_len=1, **kw)
        ok2, _, _ = test_down_merge(gpu, hw, quant,
                                     K=512, output_len=2048, token_len=1, **kw)
        results.append((quant, "group_up_gate", ok1))
        results.append((quant, "down_merge", ok2))

    ok3 = test_shared_gate_up_q80(gpu, hw, **kw)
    ok4 = test_shared_down_merge_q80(gpu, hw, **kw)
    results.append(("q8_0", "shared_gate_up", ok3))
    results.append(("q8_0", "shared_down_merge", ok4))

    print(f"\n{SEP}")
    print("  Summary:")
    for quant, op, ok in results:
        print(f"    [{'PASS' if ok else 'FAIL'}] {op}_{quant}_sg")
    all_ok = all(r[-1] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
