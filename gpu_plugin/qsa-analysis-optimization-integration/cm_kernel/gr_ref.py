"""FP32/FP64 numpy reference for the Gated Residual (hyper-connection) kernels.

Mirrors `gated_residual_gpu_test.cpp`'s `reference_rbar` / `reference_read` and the
WRITE combine `R'[c,d] = R[c,d] + S[c]*Y[d]`. Accumulations use float64 (the gtest
reference uses `double`); inputs/outputs are still exchanged as float16 with the
kernel so results are compared at the kernel's own precision (see the 2e-2 / 1e-2
absolute tolerances used by the C++ unit tests and the standalone test here).
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rbar_ref(residual: np.ndarray, gain: np.ndarray, C: int, D: int, eps: float) -> np.ndarray:
    """Branch-wise grouped RMSNorm. residual: [rows, C*D] f32, gain: [C*D] f32."""
    rows = residual.shape[0]
    r = residual.reshape(rows, C, D).astype(np.float64)
    g = gain.reshape(C, D).astype(np.float64)
    sumsq = np.sum(r * r, axis=2) / D  # [rows, C]
    inv_rms = 1.0 / np.sqrt(sumsq + eps)
    rbar = r * inv_rms[:, :, None] * g[None, :, :]
    return rbar.reshape(rows, C * D)


def gr_read_ref(residual: np.ndarray, gain: np.ndarray, w_down: np.ndarray, w_up: np.ndarray,
                w_inject: np.ndarray | None, C: int, D: int, L: int, eps: float):
    """residual/gain/w_down[L,CD]/w_up[CD,L]/w_inject[C,CD] f32 (already the
    plain checkpoint values; caller applies GR_NORM_GAIN_PLUS_ONE beforehand if
    that JIT flag is set). Returns (mixed[rows,D], write_gate[rows,C] or None), f64.
    """
    rows = residual.shape[0]
    CD = C * D
    rbar = rbar_ref(residual, gain, C, D, eps).astype(np.float64)  # [rows, CD]
    inv_c = 1.0 / C

    logits_down = rbar @ w_down.astype(np.float64).T  # [rows, L]
    t = (logits_down * inv_c)
    t = t * sigmoid(t)  # SiLU

    logits_up = t @ w_up.astype(np.float64).T  # [rows, CD]
    gate = sigmoid(logits_up).reshape(rows, C, D)
    mixed = (gate * rbar.reshape(rows, C, D)).sum(axis=1) * inv_c  # [rows, D]

    write_gate = None
    if w_inject is not None:
        logits_inj = rbar @ w_inject.astype(np.float64).T  # [rows, C]
        write_gate = 2.0 * sigmoid(logits_inj * inv_c)
    return mixed, write_gate


def gr_write_ref(residual: np.ndarray, y: np.ndarray, gate: np.ndarray, C: int, D: int) -> np.ndarray:
    """residual: [rows, C*D], y: [rows, D], gate: [rows, C] -> residual_out [rows, C*D]."""
    rows = residual.shape[0]
    r = residual.reshape(rows, C, D).astype(np.float64)
    out = r + gate[:, :, None].astype(np.float64) * y[:, None, :].astype(np.float64)
    return out.reshape(rows, C * D)
