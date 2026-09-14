"""Optimization analysis for qsa_sparse_attention_dpas.

Two things it measures on the remote BMG GPU:

1. Tunable sensitivity: sweeps QSA_ATT_WORKERS (head-dim split) x QSA_HEADS_PER_TILE
   (heads sharing a dpas tile) and times the kernel, to check whether the values the
   plugin auto-picks are actually optimal.

2. A traffic model that separates the two amplification sources the sparse dpas tile
   pays, from the synthesized selection:
     * union compute amplification  - a tile of TT tokens attends the UNION of their
       top-k sets, so every token computes over blocks it did not select;
     * head-group KV redundancy     - (Hq/HT) head-groups each re-load the KV of their
       shared kv_head, i.e. (Hq/HT)/Hkv redundant reads.
   It prints the minimal ("useful") vs the actually-moved KV bytes and the resulting
   bandwidth-roofline, next to the measured time, so the gap is attributable.

    CM_FE_DIR=/mnt/river python profile_q3_dpas.py --sizes 2048 --iters 20
"""
from __future__ import annotations

import argparse
import numpy as np

import qsa_config
import qsa_harness as H

REG_N, REG_K, REG_M, WIDE = 16, 16, 8, 32
F16 = 2


def valid_workers(cfg):
    out = []
    Dh, Dv = cfg.k_head_size, cfg.v_head_size
    for w in (1, 2, 4, 8):
        if Dh % (w * REG_K) or Dv % (w * REG_N):
            continue
        dh_w, dv_w = Dh // w, Dv // w
        if dh_w % WIDE or dv_w % WIDE:
            continue
        fits = dh_w * REG_N * 2 + REG_N * dv_w * 4 + REG_K * (dh_w + dv_w) * 2 <= 12 * 1024
        out.append((w, fits))
    return out


def valid_ht(cfg):
    return [ht for ht in (2, 4, 8, 16) if cfg.n_rep % ht == 0 and REG_N % ht == 0]


def traffic_model(cfg, batch, sel, cnt, ht):
    """Return dict of KV-byte terms (bytes) for heads_per_tile=ht."""
    Dh, Dv, r, Hq, Hkv = (cfg.k_head_size, cfg.v_head_size, cfg.compress_ratio,
                          cfg.heads_num, cfg.kv_heads_num)
    tt = REG_N // ht
    per_blk = r * (Dh + Dv) * F16
    n_head_groups = Hq // ht

    # union blocks per sub-tile of TT tokens (+ its always-visible tail steps)
    union_blocks = 0
    useful_block_selects = 0          # sum over tokens of cnt (per-token useful)
    nt = batch.num_tokens
    for s0 in range(0, nt, tt):
        toks = range(s0, min(s0 + tt, nt))
        u = set()
        for t in toks:
            k = int(cnt[t])
            u.update(int(b) for b in sel[t, :k] if b >= 0)
            useful_block_selects += k
        union_blocks += len(u)

    # KV bytes actually moved: every head-group reloads its sub-tiles' union blocks.
    actual = n_head_groups * union_blocks * per_blk
    # Minimal: each kv_head loads each needed (union) block once.
    useful_union = Hkv * union_blocks * per_blk
    # Fully-minimal per-token (no tile grouping): each kv_head loads each token's blocks.
    useful_token = Hkv * useful_block_selects * per_blk
    return dict(union_blocks=union_blocks, actual=actual, useful_union=useful_union,
                useful_token=useful_token,
                head_group_redundancy=n_head_groups / Hkv,
                union_compute_amp=union_blocks * tt / max(1, useful_block_selects))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048])
    ap.add_argument("--pa-block", type=int, default=16)
    ap.add_argument("--block-topk", type=int, default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-flush", action="store_true")
    args = ap.parse_args()
    from clops import cl

    cfg = qsa_config.load(pa_block_size=args.pa_block, block_topk=args.block_topk)
    Hq, Hkv, Dh, Dv = cfg.heads_num, cfg.kv_heads_num, cfg.k_head_size, cfg.v_head_size
    gate = cfg.has_output_gate
    qhs = 2 * Dh if gate else Dh
    gd = Dh if gate else 0
    q_pitch = Hq * qhs
    flusher = None if args.no_flush else H.CacheFlusher()

    print(f"config: Hq={Hq} Hkv={Hkv} n_rep={cfg.n_rep} Dh={Dh} r={cfg.compress_ratio} "
          f"block_topk={cfg.block_topk}")
    print(f"valid workers (w, fits<=12KB): {valid_workers(cfg)}   valid heads_per_tile: {valid_ht(cfg)}")

    for n in args.sizes:
        batch = H.make_mode_batch("prefill", n, cfg.pa_block_size)
        tile_map, num_tiles = H.make_score_tile_map(batch)
        rng = np.random.default_rng(0)
        query = H.rand_f16(rng, batch.num_tokens, q_pitch)
        kc = H.rand_f16(rng, batch.num_blocks, Hkv, cfg.pa_block_size, Dh)
        vc = H.rand_f16(rng, batch.num_blocks, Hkv, cfg.pa_block_size, Dv)
        sel, cnt = H.make_selection(batch, cfg.compress_ratio, cfg.block_topk)
        t_q = cl.tensor(query)
        t_kc, t_vc = cl.tensor(kc.reshape(-1)), cl.tensor(vc.reshape(-1))
        t_past, t_sub = cl.tensor(batch.past_lens), cl.tensor(batch.subsequence_begins)
        t_bidx, t_bib = cl.tensor(batch.block_indices), cl.tensor(batch.block_indices_begins)
        t_sel, t_cnt, t_tile = cl.tensor(sel), cl.tensor(cnt), cl.tensor(tile_map)
        t_out = cl.tensor(np.zeros((batch.num_tokens, Hq * Dv), np.float16))

        print(f"\n=== prefill q={n}  tiles={num_tiles} ===")
        for ht in valid_ht(cfg):
            tm = traffic_model(cfg, batch, sel, cnt, ht)
            rl_ms = tm["actual"] / H.PEAK_BW * 1e3
            rl_min = tm["useful_union"] / H.PEAK_BW * 1e3
            for w, fits in valid_workers(cfg):
                if not fits:
                    continue
                jit = cfg.jit("qsa_sparse_attention_dpas", QSA_ATT_WORKERS=w, QSA_HEADS_PER_TILE=ht)
                kern = H.build_kernel("qsa_sparse_attention_dpas.cm", jit)
                GWS = [(Hq // ht) * w, num_tiles * ht, 1]
                LWS = [w, 1, 1]

                def enqueue():
                    kern.enqueue("qsa_sparse_attention_dpas", GWS, LWS, t_q, t_kc, t_vc,
                                 t_past, t_sub, t_bidx, t_bib, t_sel, t_cnt, t_tile, t_out,
                                 num_tiles, 0, q_pitch, qhs, gd)

                stats = H.bench(enqueue, iters=args.iters, warmup=8, flusher=flusher)
                eff = 100.0 * rl_ms / stats["mean"]
                print(f"  HT={ht} W={w}: mean={stats['mean']:8.4f} ms | "
                      f"actualKV={tm['actual']/1e6:8.1f}MB min={tm['useful_union']/1e6:7.1f}MB "
                      f"| bw-roofline(actual)={rl_ms:7.4f} ms eff={eff:5.1f}% "
                      f"| KVredund={tm['head_group_redundancy']:.1f}x unionAmp={tm['union_compute_amp']:.2f}x")


if __name__ == "__main__":
    main()
