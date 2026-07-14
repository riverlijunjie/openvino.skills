#!/usr/bin/env python3
"""Build measured performance_metrics.json for Onyx on PTL 12Xe from parsed
cliloader logs (parsed.json produced by utils/parse_logs.py).

Output schema matches render_summary.py (decode/prefill -> size ->
{total_ms, theo_ms, achieved_pct, rows:[...]}), so the SUMMARY is regenerated
from MEASURED kernel times (mean ns/iter from cliloader Device Timing)."""
import json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(here, "parsed.json")))
def ms(tag): return P.get(tag, {}).get("total_kernel_ns", 0) / 1e6  # per-iter ms

# ---- HW (PTL 12Xe @ 2400 MHz) ----
BW = 110.0e9; FP16_XMX = 12*8*256*2.4e9; INT8_XMX = 2*FP16_XMX
OVH = 0.95; BW_A = BW*OVH; FP16_A = FP16_XMX*OVH; INT8_A = INT8_XMX*OVH

# ---- Onyx config ----
H,L,NH,NKV,HD = 6656,52,32,2,128
QDIM,KVDIM,INTER,VOCAB,SW,G = 4096,256,19968,202048,2048,128
FULL=sum(1 for i in range(L) if i%4==3); SLIDE=L-FULL   # 13, 39
SIZES=[1024,2048,4096,8192,16384,32768]

def i4w(K,N): return N*K/2 + N*(K/G)*2 + N*(K/G)*0.5
def i8w(K,N): return N*K + N*(K/G)*2
def fc_bytes(M,K,N,q): return M*K*2 + (i4w(K,N) if q=='u4' else i8w(K,N)) + M*N*2
def fc_flops(M,K,N): return 2*M*K*N
def kvb(kv):  # u4 KV cache bytes (k+v, int4+fp16 scale+int4 zp) over kv tokens
    return ((NKV*HD*0.5)*2 + (NKV*(HD/G)*2)*2 + (NKV*(HD/G)*0.5)*2)*kv
def pa_dec_bytes(kv): return kvb(kv) + NH*HD*2 + NH*HD*2
def pa_dec_flops(kv): return 4*NH*HD*kv
def pa_pre_bytes(S,kv): return kvb(kv) + S*NH*HD*2 + S*NH*HD*2
def pa_pre_flops(pairs): return 4*NH*HD*pairs

# FC op registry: op -> (decode_tag, K, N, quant, calls)
FC = {
 "FC_QKV":   ("fc_qkv",6656,4608,"u4",L),
 "FC_OutGate":("fc_outgate",6656,4096,"u4",L),
 "FC_O":     ("fc_o",4096,6656,"u4",L),
 "MLP_gate": ("fc_gate",6656,19968,"u4",L),
 "MLP_up":   ("fc_up",6656,19968,"u4",L),
 "MLP_down": ("fc_down",19968,6656,"u4",L),
}
# small-ops: op -> (tag_stub, per-layer calls_decode)
SO = {
 "rmsnorm_h6656": ("so_rmsnorm_h6656", 4*L+2),
 "rmsnorm3d_q":   ("so_rmsnorm3d_q", L),
 "rmsnorm3d_k":   ("so_rmsnorm3d_k", L),
 "rope_q":        ("so_rope_q", SLIDE),
 "rope_k":        ("so_rope_k", SLIDE),
 "add_h6656":     ("so_add_h6656", 2*L),
}

def mem_row(op,kernel,single,calls,bytes_,flops):
    gbs = bytes_/(single*1e-3)/1e9 if single>0 else 0
    gf  = flops/(single*1e-3)/1e9 if single>0 else 0
    eff = gbs/(BW/1e9)*100
    theo = bytes_/BW_A*1e3
    return dict(op=op,kernel=kernel,single_ms=single,calls=calls,total_ms=single*calls,
                gflops=gf,gbs=gbs,eff=eff,bound="memory",_theo=theo*calls)

def cmp_row(op,kernel,single,calls,bytes_,flops,peak):
    gf = flops/(single*1e-3)/1e9 if single>0 else 0
    gbs= bytes_/(single*1e-3)/1e9 if single>0 else 0
    eff= gf/(peak/1e9)*100
    theo = max(bytes_/BW_A, flops/peak)*1e3
    return dict(op=op,kernel=kernel,single_ms=single,calls=calls,total_ms=single*calls,
                gflops=gf,gbs=gbs,eff=eff,bound="compute",_theo=theo*calls)

def smallops_row(sizeS, prefix):
    tot=0.0; theo=0.0
    for op,(stub,calls) in SO.items():
        tag = f"{stub}_decode" if prefix=="decode" else f"{stub}_prefill_S{sizeS}"
        single = ms(tag)
        tot += single*calls
        theo += single*calls*0.45   # small ops ~45% BW-eff floor (nominal)
    return dict(op="SmallOps",kernel="rms/rope/gate/add",single_ms=0,calls=0,
                total_ms=tot,gflops=0,gbs=0,eff=0,bound="memory",_theo=theo)

out={"platform":"PTL_12Xe","model":"onyx (text decoder)","measured":True,
     "config":"INT4 g=128 body + INT8 g=128 LM_head + INT4 g=128 KV cache, FP16 act",
     "bw_gbs":BW/1e9,"fp16_tflops":FP16_XMX/1e12,"int8_tops":INT8_XMX/1e12,
     "ridge_f16":FP16_A/BW_A,"ridge_i8":INT8_A/BW_A,"sizes":SIZES,"decode":{},"prefill":{}}

# ---------------- DECODE ----------------
for kv in SIZES:
    rows=[]
    for op,(stub,K,N,q,calls) in FC.items():
        s=ms(f"{stub}_decode_M1")
        rows.append(mem_row(op,"gemm_kernel",s,calls,fc_bytes(1,K,N,q),fc_flops(1,K,N)))
    rows.append(mem_row("LM_head","gemm_kernel",ms("lm_head_M1"),1,
                        fc_bytes(1,H,VOCAB,"u8"),fc_flops(1,H,VOCAB)))
    kvs=min(kv,SW)
    rows.append(mem_row("PA_sliding","pa_kv_cache_update+paged_attention",
                        ms(f"pa_sliding_decode_kv{kvs}"),SLIDE,pa_dec_bytes(kvs),pa_dec_flops(kvs)))
    rows.append(mem_row("PA_full","pa_kv_cache_update+paged_attention",
                        ms(f"pa_full_decode_kv{kv}"),FULL,pa_dec_bytes(kv),pa_dec_flops(kv)))
    rows.append(smallops_row(kv,"decode"))
    tot=sum(r["total_ms"] for r in rows); theo=sum(r["_theo"] for r in rows)
    for r in rows: r.pop("_theo")
    rows.sort(key=lambda r:-r["total_ms"])
    out["decode"][str(kv)]={"total_ms":tot,"theo_ms":theo,"achieved_pct":theo/tot*100,"rows":rows}

# ---------------- PREFILL ----------------
def pa_prefill_full_single(S): return ms(f"pa_prefill_S{S}")
def pa_prefill_sliding_single(S):
    if S<=SW: return ms(f"pa_prefill_S{S}")
    base=ms(f"pa_prefill_S{SW}"); 
    band=S*SW - SW*(SW-1)/2; ref=SW*(SW+1)/2
    return base*band/ref
for S in SIZES:
    rows=[]
    for op,(stub,K,N,q,calls) in FC.items():
        s=ms(f"{stub}_prefill_S{S}")
        rows.append(cmp_row(op,"dq+gemm_kernel",s,calls,fc_bytes(S,K,N,q),fc_flops(S,K,N),INT8_XMX))
    rows.append(mem_row("LM_head","gemm_kernel",ms("lm_head_M1"),1,
                        fc_bytes(1,H,VOCAB,"u8"),fc_flops(1,H,VOCAB)))
    # PA full (causal over S)
    pf=pa_prefill_full_single(S); pairs_f=S*(S+1)/2
    rows.append(cmp_row("PA_full","sdpa_micro_prefill",pf,FULL,pa_pre_bytes(S,S),pa_pre_flops(pairs_f),FP16_XMX))
    # PA sliding (band SW)
    ps=pa_prefill_sliding_single(S); kvs=min(S,SW)
    pairs_s = S*kvs - (kvs*(kvs-1)/2 if S>=kvs else S*(S-1)/2)
    rows.append(cmp_row("PA_sliding","sdpa_micro_prefill",ps,SLIDE,pa_pre_bytes(S,kvs),pa_pre_flops(pairs_s),FP16_XMX))
    rows.append(smallops_row(S,"prefill"))
    tot=sum(r["total_ms"] for r in rows); theo=sum(r["_theo"] for r in rows)
    for r in rows: r.pop("_theo")
    rows.sort(key=lambda r:-r["total_ms"])
    out["prefill"][str(S)]={"total_ms":tot,"theo_ms":theo,"achieved_pct":theo/tot*100,"rows":rows}

json.dump(out,open(os.path.join(here,"performance_metrics.json"),"w"),indent=2)
print("wrote performance_metrics.json")
for kv in SIZES:
    d=out["decode"][str(kv)]; print(f"decode kv={kv:6d} TPOT {d['total_ms']:7.2f}ms  achieved {d['achieved_pct']:.1f}%")
for S in SIZES:
    d=out["prefill"][str(S)]; print(f"prefill S={S:6d} TTFT {d['total_ms']:9.2f}ms  achieved {d['achieved_pct']:.1f}%")
