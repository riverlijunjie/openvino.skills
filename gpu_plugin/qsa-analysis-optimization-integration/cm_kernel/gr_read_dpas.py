"""Explicit opt-in multi-dispatch DPAS pipeline for gr_read; the single-kernel scalar
gr_read.cm stays the default. Caller owns the scratch tensors and enqueues them on the
same in-order queue (mirrors qsa_prepare_dpas.py's PrepareDPAS).

Pipeline (4-5 kernel dispatches, each a separate clEnqueueNDRangeKernel so the
producer/consumer ordering across global-memory round-trips is a safe in-order-queue
dependency instead of an intra-dispatch race - see gr_rbar_dpas.cm):
  1. gr_rbar_dpas      : Rbar[tokens,CD]              (branch RMSNorm)
  2. gr_down_proj_dpas : T[tokens,L]      = SiLU(Rbar @ W_down^T / C)     (dpas)
  3. gr_up_proj_dpas   : gate[tokens,CD]  = sigmoid(T @ W_up^T)           (dpas)
  4. gr_writegate_dpas : write_gate[tokens,C] = 2*sigmoid(Rbar @ W_inject^T / C) (dpas, optional)
  5. gr_combine        : mixed[tokens,D]  = (1/C) * sum_c gate*Rbar       (elementwise)
"""
from __future__ import annotations

import qsa_harness as H

REG_M = 8
REG_N = 16
REG_K = 16


class GRReadDPAS:
    def __init__(self, C: int, D: int, L: int, eps: float, norm_gain_plus_one: bool,
                produce_write_gate: bool, *, wg: int = 16, combine_tile: int = 64):
        CD = C * D
        if C != 4 or C % 2 != 0:
            raise ValueError("Unsupported GR DPAS shape: write-gate packing requires C==4")
        if CD < 32 or CD % 32 != 0:
            raise ValueError("Unsupported GR DPAS shape: C*D must be >=32 and a multiple of 32")
        if L < 32 or L % 32 != 0:
            raise ValueError("Unsupported GR DPAS shape: GR_LOWRANK must be >=32 and a multiple of 32")
        if L % REG_M != 0 or CD % REG_M != 0:
            raise ValueError("Unsupported GR DPAS shape: GR_LOWRANK and C*D must be multiples of 8")
        if D % combine_tile != 0:
            raise ValueError(f"GR_HIDDEN ({D}) must be a multiple of combine_tile ({combine_tile})")

        self.C, self.D, self.L, self.CD = C, D, L, CD
        self.produce_write_gate = produce_write_gate
        self.wg = wg
        self.combine_tile = combine_tile

        self.rbar_kern = H.build_kernel("gr_rbar_dpas.cm", dict(
            KERNEL_NAME="gr_rbar_dpas", GR_BRANCHES=C, GR_HIDDEN=D, GR_EPS=eps,
            GR_NORM_GAIN_PLUS_ONE=1 if norm_gain_plus_one else 0, GR_WG_SIZE=wg))
        self.down_kern = H.build_kernel("gr_down_proj_dpas.cm", dict(
            KERNEL_NAME="gr_down_proj_dpas", GR_BRANCHES=C, GR_HIDDEN=D, GR_LOWRANK=L))
        self.up_kern = H.build_kernel("gr_up_proj_dpas.cm", dict(
            KERNEL_NAME="gr_up_proj_dpas", GR_BRANCHES=C, GR_HIDDEN=D, GR_LOWRANK=L))
        if produce_write_gate:
            self.wgate_kern = H.build_kernel("gr_writegate_dpas.cm", dict(
                KERNEL_NAME="gr_writegate_dpas", GR_BRANCHES=C, GR_HIDDEN=D))
        self.combine_kern = H.build_kernel("gr_combine.cm", dict(
            KERNEL_NAME="gr_combine", GR_BRANCHES=C, GR_HIDDEN=D, GR_COMBINE_TILE=combine_tile))

    def enqueue(self, rows: int, residual, gain, w_down, w_up, w_inject,
               rbar_scratch, t_scratch, gate_scratch, mixed_out, write_gate_out=None):
        if rows == 0:
            return
        C, D, L, CD = self.C, self.D, self.L, self.CD
        wg = self.wg
        n_tiles = (rows + REG_N - 1) // REG_N

        self.rbar_kern.enqueue("gr_rbar_dpas", [wg, rows, 1], [wg, 1, 1],
                               residual, gain, rbar_scratch, rows)
        self.down_kern.enqueue("gr_down_proj_dpas", [L // REG_M, n_tiles, 1], [1, 1, 1],
                               rbar_scratch, w_down, t_scratch, rows)
        self.up_kern.enqueue("gr_up_proj_dpas", [CD // REG_M, n_tiles, 1], [1, 1, 1],
                             t_scratch, w_up, gate_scratch, rows)
        if self.produce_write_gate:
            self.wgate_kern.enqueue("gr_writegate_dpas", [1, n_tiles, 1], [1, 1, 1],
                                    rbar_scratch, w_inject, write_gate_out, rows)
        d_tiles = (D + self.combine_tile - 1) // self.combine_tile
        self.combine_kern.enqueue("gr_combine", [d_tiles, rows, 1], [1, 1, 1],
                                  rbar_scratch, gate_scratch, mixed_out, rows)
