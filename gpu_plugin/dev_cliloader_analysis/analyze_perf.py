import re
from collections import defaultdict

with open('/home/ov2022/workspace/remote_debug/openvino/perf_dGPU.log', 'r') as f:
    lines = f.readlines()

total_time_ns = 24799431731
data_start = None
for i, l in enumerate(lines):
    if 'Function Name' in l and 'Calls' in l:
        data_start = i + 1
        break

records = []
for l in lines[data_start:]:
    l = l.strip()
    if not l or 'CLIntercept' in l or 'shutdown' in l:
        continue
    parts = l.rsplit(',', 6)
    if len(parts) < 7: continue
    name = parts[0].strip()
    try:
        calls = int(parts[1].strip())
        time_ns = int(parts[2].strip())
        avg_ns = int(parts[4].strip())
        min_ns = int(parts[5].strip())
        max_ns = int(parts[6].strip())
    except: continue
    records.append((name, calls, time_ns, avg_ns, min_ns, max_ns))

out = []

# == HtoD ==
htod = sorted([r for r in records if 'HtoD' in r[0] and 'clEnqueueMemcpy' in r[0]], key=lambda x: -x[2])
total_htod = sum(r[2] for r in htod)
out.append(f"=== HtoD TRANSFER: {total_htod/1e6:.1f} ms ({total_htod/total_time_ns*100:.2f}%) ===")
for r in htod:
    size_m = re.search(r'(\d+) bytes', r[0])
    sb = int(size_m.group(1)) if size_m else 0
    tx = sb * r[1]
    bw = tx / (r[2] / 1e9) / (1024**3) if r[2] > 0 else 0
    out.append(f"  {sb/1e6:>8.1f}MB x{r[1]:>4} = {tx/1e9:>6.2f}GB  {r[2]/1e6:>8.1f}ms  BW={bw:>6.1f}GB/s")

# == MOE (only named kernels, skip small MtoH-like entries) ==
moe = sorted([r for r in records if 'moe_3gemm' in r[0] and r[2] > 5000000], key=lambda x: -x[2])
total_moe_all = sum(r[2] for r in records if 'moe_3gemm' in r[0])
out.append(f"\n=== MOE KERNELS: {total_moe_all/1e6:.1f} ms ({total_moe_all/total_time_ns*100:.2f}%) ===")
out.append(f"Top MOE kernels (>5ms):")
for r in moe:
    moe_type = "unknown"
    for kw in ['mlp_down','mlp_gate_up','fuse_softmax_topk','shared_expert_gate',
               'permute_input','permute_output','router_prefill_permute','router_token_permute']:
        if kw in r[0]: moe_type = kw; break
    gws_m = re.search(r'GWS\[\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*\]', r[0])
    gws_str = f"[{gws_m.group(1)}x{gws_m.group(2)}x{gws_m.group(3)}]" if gws_m else "[?]"
    out.append(f"  [{moe_type:<25}] x{r[1]:>5}  {r[2]/1e6:>8.1f}ms  avg={r[3]/1e3:>8.1f}us  GWS={gws_str}")

# == Attention ==
attn = sorted([r for r in records if 'paged_attention' in r[0] or 'sdpa_micro' in r[0]], key=lambda x: -x[2])
total_attn = sum(r[2] for r in attn)
out.append(f"\n=== ATTENTION: {total_attn/1e6:.1f} ms ({total_attn/total_time_ns*100:.2f}%) ===")
for r in attn:
    atype = "PA_decode" if ('paged_attention' in r[0] and 'finalization' not in r[0]) else \
            "PA_final" if 'finalization' in r[0] else "SDPA_prefill"
    out.append(f"  [{atype:<15}] x{r[1]:>5}  {r[2]/1e6:>8.1f}ms  avg={r[3]/1e3:>8.1f}us")

# == Other ==
out.append(f"\n=== OTHER KERNELS ===")
for cat_name, pat in [('RMS','rms_gpu'),('ROPE','rope'),('DynQuant','dynamic_quantize'),
                       ('Reorder','reorder_data'),('Activation','activation_ref'),('Concat','concatenation')]:
    recs = [r for r in records if pat in r[0]]
    t = sum(r[2] for r in recs)
    c = sum(r[1] for r in recs)
    out.append(f"  {cat_name:<12}: {t/1e6:>8.1f}ms ({t/total_time_ns*100:.3f}%)  x{c}")

dtoh = [r for r in records if 'DtoH' in r[0]]
mtoh = [r for r in records if 'MtoH' in r[0]]
out.append(f"  DtoH_Copy  : {sum(r[2] for r in dtoh)/1e6:>8.1f}ms")
out.append(f"  MtoH_Copy  : {sum(r[2] for r in mtoh)/1e6:>8.1f}ms")

# == MOE breakdown by sub-kernel type (aggregated) ==
out.append(f"\n=== MOE SUB-KERNEL BREAKDOWN ===")
moe_all = [r for r in records if 'moe_3gemm' in r[0]]
moe_types = defaultdict(lambda: {'time':0, 'calls':0})
for r in moe_all:
    mtype = "other_moe"
    for kw in ['mlp_down','mlp_gate_up','fuse_softmax_topk','shared_expert_gate',
               'permute_input','permute_output','router_prefill_permute','router_token_permute']:
        if kw in r[0]: mtype = kw; break
    moe_types[mtype]['time'] += r[2]
    moe_types[mtype]['calls'] += r[1]
for mt, v in sorted(moe_types.items(), key=lambda x: -x[1]['time']):
    out.append(f"  {mt:<25}: {v['time']/1e6:>8.1f}ms  x{v['calls']:>6}")

print('\n'.join(out))
with open('/tmp/analysis_final.txt', 'w') as f:
    f.write('\n'.join(out))
print("Done")
