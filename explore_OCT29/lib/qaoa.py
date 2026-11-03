"""Ideal qubit QAOA used as the Nov_3rd / beat-QAOA reference.

Angle convention (qaoa_ref), matching ``Nov_3rd/qaoa_init.py`` and
``explore_OCT29/lib/blockmix.py``:

  |+⟩^{⊗n} → ∏_ℓ [ exp(-i γ_ℓ Cn) · ⊗_j exp(-i β_ℓ X_j / 2) ]
  Cn = E · (n / m),  m = round(9.93 n)
  state index: var0 = MSB (same layout as ``sat_gen.fast_energies``)

``qaoa_psucc`` is differentiable. Success is the probability mass on E == 0.
"""
from __future__ import annotations

import numpy as np
import torch

from .sat_gen import clause_count as _clause_count


def clause_count(n: int) -> int:
    return int(_clause_count(n))


def inv_p_from_psucc(ps) -> np.ndarray:
    """1 / p_succ, floored so empty success mass stays finite."""
    if torch.is_tensor(ps):
        ps = ps.detach().cpu().numpy()
    arr = np.asarray(ps, dtype=np.float64)
    return 1.0 / np.clip(arr, 1e-12, None)


def summarize_inv(invs) -> dict:
    a = np.asarray(invs, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {
            "median_inv_p": float("nan"),
            "B90_inv_p": float("nan"),
            "mean_inv_p": float("nan"),
            "frac_inv_le_2": float("nan"),
            "B": 0,
        }
    return {
        "median_inv_p": float(np.median(a)),
        "B90_inv_p": float(np.quantile(a, 0.90)),
        "mean_inv_p": float(np.mean(a)),
        "frac_inv_le_2": float(np.mean(a <= 2.0)),
        "B": int(a.size),
    }


def _as_1d(x, name: str, device, dtype) -> torch.Tensor:
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    t = t.to(device=device, dtype=dtype).reshape(-1)
    if t.numel() < 1:
        raise ValueError(f"{name} is empty")
    return t


def _apply_mixer_x_half(psi: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    """⊗_j exp(-i β X_j / 2). psi: (B, 2^n), var0 = MSB."""
    c = torch.cos(0.5 * beta).to(dtype=psi.dtype)
    s = (-1j * torch.sin(0.5 * beta)).to(dtype=psi.dtype)
    Bsz = psi.shape[0]
    t = psi
    n = int(n)
    for q in range(n):
        x = t.reshape(Bsz, 1 << (n - 1 - q), 2, 1 << q)
        t0 = x[:, :, 0, :]
        t1 = x[:, :, 1, :]
        y0 = c * t0 + s * t1
        y1 = s * t0 + c * t1
        t = torch.stack((y0, y1), dim=2).reshape(Bsz, 1 << n)
    return t


def qaoa_psucc(
    betas,
    gammas,
    E: torch.Tensor,
    n: int,
    m: int | None = None,
) -> torch.Tensor:
    """Success probability of ideal QAOA. E: (B, 2^n) or (2^n,). Returns (B,)."""
    if not torch.is_tensor(E):
        E = torch.as_tensor(E)
    if E.ndim == 1:
        E = E.reshape(1, -1)
    if E.ndim != 2:
        raise ValueError(f"E must be (B, 2^n), got {tuple(E.shape)}")
    Bsz, dim = E.shape
    n = int(n)
    if dim != (1 << n):
        raise ValueError(f"E width {dim} != 2^{n}")
    if m is None:
        m = clause_count(n)
    device = E.device
    betas_t = _as_1d(betas, "betas", device, torch.float32)
    gammas_t = _as_1d(gammas, "gammas", device, torch.float32)
    if betas_t.numel() != gammas_t.numel():
        raise ValueError(
            f"betas/gammas length {betas_t.numel()} != {gammas_t.numel()}"
        )
    Cn = E.to(dtype=torch.float32) * (float(n) / float(m))
    psi = torch.full(
        (Bsz, dim),
        1.0 / (float(dim) ** 0.5),
        dtype=torch.complex64,
        device=device,
    )
    for ell in range(int(betas_t.numel())):
        g = gammas_t[ell].to(dtype=torch.complex64)
        psi = psi * torch.exp(-1j * g * Cn.to(dtype=torch.complex64))
        psi = _apply_mixer_x_half(psi, betas_t[ell], n)
    probs = psi.real.square() + psi.imag.square()
    sat = (E == 0).to(dtype=probs.dtype)
    return (probs * sat).sum(dim=-1)
