"""Cavity-locality split of the cost diagonal (Opus Idea 2, graded H_c).

E = Σ_S h_S Z_S  (Walsh / FWHT of the unsat-count vector).
  E_in : strings whose support lies in one 3-bit cavity, plus the DC term
  E_x  : strings that touch ≥2 cavities

Two classes only — do NOT split by exact span 1/2/3/4. DC rides with E_in
so E_in + E_x = E exactly; when γ_in = γ_x = γ this is γ·E up to amax shift.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from explore_OCT29.lib.packing import K_BITS, n_cavities
from explore_OCT29.lib.sat_gen import clause_count


def fwht_np(a: np.ndarray) -> np.ndarray:
    """Unnormalized FWHT along the last axis (length 2^n). Self-inverse up to 1/2^n."""
    out = np.array(a, dtype=np.float64, copy=True)
    n_last = out.shape[-1]
    if n_last < 1 or (n_last & (n_last - 1)):
        raise ValueError(f"last dim {n_last} is not a power of 2")
    h = 1
    sh = out.shape[:-1]
    while h < n_last:
        out = out.reshape(sh + (n_last // (2 * h), 2, h))
        x = out[..., 0, :].copy()
        y = out[..., 1, :].copy()
        out[..., 0, :] = x + y
        out[..., 1, :] = x - y
        out = out.reshape(sh + (n_last,))
        h *= 2
    return out


def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized FWHT along the last axis. Self-inverse up to 1/2^n."""
    n_last = int(x.shape[-1])
    if n_last < 1 or (n_last & (n_last - 1)):
        raise ValueError(f"last dim {n_last} is not a power of 2")
    lead = x.shape[:-1]
    out = x.to(dtype=torch.float64).contiguous()
    h = 1
    while h < n_last:
        out = out.reshape(*lead, n_last // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = torch.stack((a + b, a - b), dim=-2)
        out = out.reshape(*lead, n_last)
        h *= 2
    return out


@lru_cache(maxsize=16)
def cavity_pc(n: int) -> np.ndarray:
    """Number of distinct 3-bit cavities touched by Walsh index s (var0=MSB)."""
    C = n_cavities(n)
    s = np.arange(1 << n, dtype=np.int64)
    occ = np.zeros(1 << n, dtype=np.int64)
    pos = n
    for c in range(C):
        pos -= K_BITS
        occ |= (((s >> pos) & ((1 << K_BITS) - 1)) != 0).astype(np.int64) << c
    pc = np.zeros_like(occ)
    for c in range(C):
        pc += (occ >> c) & 1
    return pc


@lru_cache(maxsize=16)
def masks_in_x(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over Walsh index: in-cavity (pc≤1, includes DC) vs across (pc≥2)."""
    pc = cavity_pc(n)
    return pc <= 1, pc >= 2


def split_E_np(E: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Split last-axis length-2^n cost into (E_in, E_x). E_in+E_x = E."""
    dim = 1 << n
    if E.shape[-1] != dim:
        raise ValueError(f"E last dim {E.shape[-1]} != 2^{n}={dim}")
    W = fwht_np(E)
    mask_in, mask_x = masks_in_x(n)
    Ein = fwht_np(W * mask_in.astype(np.float64)) / float(dim)
    Ex = fwht_np(W * mask_x.astype(np.float64)) / float(dim)
    return Ein.astype(np.float32), Ex.astype(np.float32)


def split_E_torch(E: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    dim = 1 << n
    if int(E.shape[-1]) != dim:
        raise ValueError(f"E last dim {tuple(E.shape)} != 2^{n}")
    W = fwht_torch(E)
    mask_in, mask_x = masks_in_x(n)
    min_t = torch.as_tensor(mask_in, device=E.device)
    mx_t = torch.as_tensor(mask_x, device=E.device)
    Ein = fwht_torch(W * min_t.to(W.dtype)) / float(dim)
    Ex = fwht_torch(W * mx_t.to(W.dtype)) / float(dim)
    return Ein.to(dtype=torch.float32), Ex.to(dtype=torch.float32)


def walsh_energy_fracs(E: np.ndarray, n: int) -> dict:
    """Mean Walsh-energy fractions over a batch E (B, 2^n) or a single vector.

    frac_x / frac_in exclude DC (global phase). frac_span[k] is exact k-cavity
    support, logged only — not a trainable split.
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 1:
        E = E[None, :]
    W = fwht_np(E)
    pc = cavity_pc(n)
    w2 = W * W
    dc = w2[:, 0]
    tot = w2[:, 1:].sum(axis=1)  # exclude DC
    tot = np.maximum(tot, 1e-30)
    frac_in = (w2[:, pc == 1].sum(axis=1) / tot).mean() if np.any(pc == 1) else 0.0
    frac_x = (w2[:, pc >= 2].sum(axis=1) / tot).mean() if np.any(pc >= 2) else 0.0
    span = {}
    C = n_cavities(n)
    for k in range(1, C + 1):
        m = pc == k
        span[k] = float((w2[:, m].sum(axis=1) / tot).mean()) if np.any(m) else 0.0
    return {
        "frac_in": float(frac_in),
        "frac_x": float(frac_x),
        "frac_dc_over_all": float((dc / np.maximum(w2.sum(axis=1), 1e-30)).mean()),
        "frac_span": {str(k): v for k, v in span.items()},
        "B": int(E.shape[0]),
        "n": int(n),
        "C": int(C),
        "m": int(clause_count(n)),
        "n_classes": 2,
        "label": "graded_H_c",
    }
