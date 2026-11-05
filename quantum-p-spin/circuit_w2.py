"""Wave2 U_c recipes: structured cost channels + free SNAP (beat QAOA 0.438).

Recipes (all prefer free SNAP when *_snap):
  signj_mag_snap   : phase ∑ γ_{ℓ,t}·sign(J_t)·ZZZ_t   (mag in γ); warm≈γ_ℓ|J|
  soft_tanh_snap   : terms = -s tanh(c J_t) ZZZ; γ full (L,T)
  soft_norm_snap   : terms = -s (J/rms) ZZZ; γ full
  sched_aw_snap    : γ_{ℓ,t}=α_ℓ·w_t on xj terms (shared w)
  sched_aeps_snap  : γ_{ℓ,t}=α_ℓ·(1+ε_t) on xj (=factor1p; longer train)
  mag_bucket_snap  : K |J|-quantile buckets → K channels × L + SNAP
  two_scale_snap   : H_lo (|J|≤med) + H_hi (|J|>med); (γ_ℓ, δ_ℓ) + SNAP
  poly3_snap       : a_ℓ H_c + b_ℓ H_{J²} + c_ℓ H_{|J|}; 3L + SNAP (novel)
  vert_bucket_snap : bucket by vertex-strength of triple; K ch × L + SNAP
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from circuit_snap import B_SHELL, make_ansatz
from circuit_fourier import shell_phi_from_betas
from circuit_perzz import (
    n_triples,
    zzz_bitstrings,
    terms_on_grid,
    forward_state_perzz,
    fidelity_perzz,
)
from pspin import scale_p3, all_triples

__all__ = [
    "B_SHELL",
    "make_ansatz",
    "n_triples",
    "terms_on_grid",
    "forward_state_perzz",
    "fidelity_perzz",
    "build_channels",
    "expand_gamma_w2",
    "init_w2",
    "n_params_w2",
    "uses_free_snap",
    "needs_channels",
    "W2_RECIPES",
]

W2_RECIPES = (
    "signj_mag_snap",
    "soft_tanh_snap",
    "soft_norm_snap",
    "sched_aw_snap",
    "sched_aeps_snap",
    "mag_bucket_snap",
    "two_scale_snap",
    "poly3_snap",
    "vert_bucket_snap",
)


def uses_free_snap(recipe: str) -> bool:
    return "_snap" in recipe


def needs_channels(recipe: str) -> bool:
    """True if terms are multi-channel (not per-triple γ @ terms)."""
    base = _base(recipe)
    return base in ("mag_bucket", "two_scale", "poly3", "vert_bucket")


def _base(recipe: str) -> str:
    base = recipe
    for suf in ("_snap_warm", "_snap_adam", "_snap", "_warm", "_adam", "_big"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def _vertex_strength(n: int, J: np.ndarray) -> np.ndarray:
    """Per-vertex strength = sum |J| over triples containing v."""
    triples = all_triples(n)
    s = np.zeros(n, dtype=np.float64)
    for (i, j, k), jv in zip(triples, J):
        a = abs(float(jv))
        s[i] += a
        s[j] += a
        s[k] += a
    return s


def build_channels(n: int, J: np.ndarray, recipe: str, n_buckets: int = 4) -> np.ndarray:
    """Return (C, 2**n) cost channels for this instance/recipe."""
    zzz = zzz_bitstrings(n)  # (T, dim)
    s = scale_p3(n)
    J = J.astype(np.float64)
    T = J.shape[0]
    base = _base(recipe)

    if base == "signj_mag":
        # -s * sign(J) * ZZZ  (magnitude learned in γ)
        sgn = np.sign(J)
        sgn[sgn == 0.0] = 1.0
        return ((-s * sgn)[:, None] * zzz)

    if base == "soft_tanh":
        soft = np.tanh(1.5 * J)
        return ((-s * soft)[:, None] * zzz)

    if base == "soft_norm":
        rms = float(np.sqrt(np.mean(J * J)) + 1e-12)
        soft = J / rms
        return ((-s * soft)[:, None] * zzz)

    if base in ("sched_aw", "sched_aeps"):
        # standard xj terms; γ factorization elsewhere
        return ((-s * J)[:, None] * zzz)

    if base == "mag_bucket":
        # K quantile buckets of |J|; channel = sum_{t in b} -s J_t ZZZ
        K = n_buckets
        absJ = np.abs(J)
        # quantile edges
        qs = np.linspace(0, 100, K + 1)
        edges = np.percentile(absJ, qs)
        edges[0] = -1.0  # include min
        edges[-1] = edges[-1] + 1.0
        dim = 1 << n
        out = np.zeros((K, dim), dtype=np.float64)
        for b in range(K):
            mask = (absJ >= edges[b]) & (absJ < edges[b + 1])
            if not np.any(mask):
                continue
            out[b] = ((-s * J[mask])[:, None] * zzz[mask]).sum(axis=0)
        return out

    if base == "two_scale":
        med = float(np.median(np.abs(J)))
        lo = np.abs(J) <= med
        hi = ~lo
        dim = 1 << n
        out = np.zeros((2, dim), dtype=np.float64)
        if np.any(lo):
            out[0] = ((-s * J[lo])[:, None] * zzz[lo]).sum(axis=0)
        if np.any(hi):
            out[1] = ((-s * J[hi])[:, None] * zzz[hi]).sum(axis=0)
        return out

    if base == "poly3":
        # H_c, H_{J^2}, H_{|J|}
        dim = 1 << n
        out = np.zeros((3, dim), dtype=np.float64)
        out[0] = ((-s * J)[:, None] * zzz).sum(axis=0)
        out[1] = ((-s * (J * J))[:, None] * zzz).sum(axis=0)
        out[2] = ((-s * np.abs(J))[:, None] * zzz).sum(axis=0)
        return out

    if base == "vert_bucket":
        # bucket triples by mean vertex-strength of (i,j,k)
        K = n_buckets
        triples = all_triples(n)
        vs = _vertex_strength(n, J)
        score = np.array([vs[i] + vs[j] + vs[k] for i, j, k in triples], dtype=np.float64)
        qs = np.linspace(0, 100, K + 1)
        edges = np.percentile(score, qs)
        edges[0] -= 1.0
        edges[-1] += 1.0
        dim = 1 << n
        out = np.zeros((K, dim), dtype=np.float64)
        for b in range(K):
            mask = (score >= edges[b]) & (score < edges[b + 1])
            if not np.any(mask):
                continue
            out[b] = ((-s * J[mask])[:, None] * zzz[mask]).sum(axis=0)
        return out

    raise ValueError(recipe)


def n_channels(recipe: str, T: int, n_buckets: int = 4) -> int:
    base = _base(recipe)
    if base in ("signj_mag", "soft_tanh", "soft_norm", "sched_aw", "sched_aeps"):
        return T
    if base in ("mag_bucket", "vert_bucket"):
        return n_buckets
    if base == "two_scale":
        return 2
    if base == "poly3":
        return 3
    raise ValueError(recipe)


def expand_gamma_w2(recipe: str, raw: dict, L: int, C: int) -> torch.Tensor:
    """Return γ of shape (L, C)."""
    base = _base(recipe)
    if base in ("signj_mag", "soft_tanh", "soft_norm"):
        g = raw["gamma_full"]
        if tuple(g.shape) != (L, C):
            raise ValueError(f"gamma_full {tuple(g.shape)} != {(L, C)}")
        return g
    if base == "sched_aw":
        return raw["alpha"].reshape(L, 1) * raw["w"].reshape(1, C)
    if base == "sched_aeps":
        return raw["alpha"].reshape(L, 1) * (1.0 + raw["eps"].reshape(1, C))
    if base in ("mag_bucket", "vert_bucket", "two_scale", "poly3"):
        g = raw["gamma_ch"]
        if tuple(g.shape) != (L, C):
            raise ValueError(f"gamma_ch {tuple(g.shape)} != {(L, C)}")
        return g
    raise ValueError(recipe)


def n_params_w2(recipe: str, L: int, T: int, n_buckets: int = 4) -> int:
    snap = 4 * L if uses_free_snap(recipe) else L
    base = _base(recipe)
    C = n_channels(recipe, T, n_buckets=n_buckets)
    if base in ("signj_mag", "soft_tanh", "soft_norm"):
        return L * C + snap
    if base == "sched_aw":
        return L + C + snap  # alpha + w
    if base == "sched_aeps":
        return L + C + snap
    if base in ("mag_bucket", "vert_bucket", "two_scale", "poly3"):
        return L * C + snap
    raise ValueError(recipe)


def init_w2(recipe: str, L: int, T: int, seed: int = 0, n_buckets: int = 4) -> dict:
    torch.manual_seed(seed)
    out: dict = {}
    base = _base(recipe)
    C = n_channels(recipe, T, n_buckets=n_buckets)

    if base in ("signj_mag", "soft_tanh", "soft_norm"):
        # tiny ±γ; warm path overwrites for signj
        out["gamma_full"] = 0.01 * torch.randn(L, C)
    elif base == "sched_aw":
        out["alpha"] = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
        out["w"] = torch.ones(C) + 0.01 * torch.randn(C)
    elif base == "sched_aeps":
        out["alpha"] = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
        out["eps"] = 0.01 * torch.randn(C)
    elif base == "two_scale":
        # start near QAOA: γ on full-ish, δ small
        out["gamma_ch"] = torch.zeros(L, 2)
        out["gamma_ch"][:, 0] = 0.01 + 0.001 * torch.randn(L)
        out["gamma_ch"][:, 1] = 0.01 + 0.001 * torch.randn(L)
    elif base == "poly3":
        out["gamma_ch"] = torch.zeros(L, 3)
        out["gamma_ch"][:, 0] = 0.01 + 0.001 * torch.randn(L)  # a ≈ γ_QAOA
        out["gamma_ch"][:, 1] = 0.001 * torch.randn(L)
        out["gamma_ch"][:, 2] = 0.001 * torch.randn(L)
    elif base in ("mag_bucket", "vert_bucket"):
        out["gamma_ch"] = 0.01 + 0.01 * torch.randn(L, C)
    else:
        raise ValueError(recipe)

    if uses_free_snap(recipe):
        pcs = torch.arange(B_SHELL, dtype=torch.float32)
        phi = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
        out["phi"] = phi + 0.001 * torch.randn(L, B_SHELL)
    else:
        out["beta"] = torch.full((L,), 0.01) + 0.001 * torch.randn(L)
    return out


def warm_from_qaoa(recipe: str, raw: dict, L: int, J_ref: np.ndarray | None, ckpt: dict) -> dict:
    """Warm-start γ / phi from a QAOA (fourier) checkpoint."""
    g = ckpt["gamma"].float().reshape(L)
    base = _base(recipe)
    if base == "signj_mag" and J_ref is not None:
        # γ_{ℓ,t} ≈ γ_ℓ * |J_t| recovers γ_ℓ*(-s J ZZZ)
        absJ = torch.as_tensor(np.abs(J_ref), dtype=torch.float32)
        raw["gamma_full"] = g.reshape(L, 1) * absJ.reshape(1, -1) + 0.005 * torch.randn(L, absJ.numel())
    elif base in ("soft_tanh", "soft_norm") and "gamma_full" in raw:
        raw["gamma_full"] = g.reshape(L, 1) + 0.01 * torch.randn_like(raw["gamma_full"])
    elif base == "sched_aw":
        raw["alpha"] = g.clone() + 0.001 * torch.randn(L)
        raw["w"] = torch.ones_like(raw["w"]) + 0.01 * torch.randn_like(raw["w"])
    elif base == "sched_aeps":
        raw["alpha"] = g.clone() + 0.001 * torch.randn(L)
        raw["eps"] = 0.01 * torch.randn_like(raw["eps"])
    elif base == "two_scale":
        raw["gamma_ch"] = torch.stack([g, g], dim=1) + 0.005 * torch.randn(L, 2)
    elif base == "poly3":
        raw["gamma_ch"] = torch.zeros(L, 3)
        raw["gamma_ch"][:, 0] = g + 0.001 * torch.randn(L)
        raw["gamma_ch"][:, 1] = 0.001 * torch.randn(L)
        raw["gamma_ch"][:, 2] = 0.001 * torch.randn(L)
    elif base in ("mag_bucket", "vert_bucket"):
        C = raw["gamma_ch"].shape[1]
        raw["gamma_ch"] = g.reshape(L, 1).expand(L, C).contiguous() + 0.01 * torch.randn(L, C)

    if "phi" in raw:
        if "beta" in ckpt:
            raw["phi"] = shell_phi_from_betas(ckpt["beta"].float()) + 0.001 * torch.randn_like(raw["phi"])
        elif "phi" in ckpt:
            raw["phi"] = ckpt["phi"].float()[:L, :B_SHELL] + 0.001 * torch.randn_like(raw["phi"])
    return raw
