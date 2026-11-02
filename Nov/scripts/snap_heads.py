#!/usr/bin/env python3
"""SNAP angle parameterization heads for Nov1st-final.

Baseline modes (Zach t2307):
  shell     — θ_k = φ_{pc(k)}           shape (…, 4)
  free      — θ_0..7 free               shape (…, 8)
  bits      — θ_k = Σ_b α_b bit_b(k)    shape (…, 3)
  fourier   — low-rank Fourier on Fock  shape (…, 1+2M)  M=2 → 5
  proj      — θ_0=β, else 0             shape (…,) / (…, 1)
  x_pop     — θ_k = β · pc(k)           shape (…,) / (…, 1)

SNAP-easiness / restriction ladder (Zach t2320):
  const       — θ_k = α                         1  (sanity kill)
  kerr        — θ_k = χ·k(k-1)/2 + ω·k          2  (native Kerr + linear)
  gauss       — θ_k = α + β·pc + δ·pc²          3  (quadratic / "Gaussian" on popcount)
  poly2_n     — θ_k = α + β·k + δ·k²            3  (quadratic on Fock index)
  poly3_pc    — θ_k = α+β·pc+δ·pc²+ε·pc³        4
  poly3_n     — θ_k = α+β·k+δ·k²+ε·k³           4
  few_m       — θ_0..M-1 free, else 0           M  (default M=3)
  gauss_plus1 — gauss + ρ·1_{pc=3}              4  (hybrid residual on top shell)

shared=False → per-layer leading dim L; shared=True → one vector for all layers.
"""
from __future__ import annotations

import math
from typing import Any

import torch

NU = 8  # computational Fock support

# Original bakeoff set (do not reorder — angle_explore defaults)
SNAP_MODES = ("shell", "free", "bits", "fourier", "proj", "x_pop")

# Restriction ladder, most-restricted → freer (snap_easy_explore)
SNAP_EASY_MODES = (
    "const",
    "kerr",
    "gauss",
    "poly2_n",
    "few_m",
    "gauss_plus1",
    "poly3_pc",
    "poly3_n",
    "shell",  # reference 4/L
    "free",   # upper bound 8/L
)


def snap_dim(mode: str, fourier_M: int = 2, few_M: int = 3) -> int:
    if mode == "shell":
        return 4
    if mode == "free":
        return 8
    if mode == "bits":
        return 3
    if mode == "fourier":
        return 1 + 2 * fourier_M  # DC + (cos,sin)*M
    if mode in ("proj", "x_pop", "const"):
        return 1
    if mode == "kerr":
        return 2  # χ, ω
    if mode in ("gauss", "poly2_n"):
        return 3  # α, β, δ
    if mode in ("poly3_pc", "poly3_n", "gauss_plus1"):
        return 4
    if mode == "few_m":
        return int(few_M)
    raise ValueError(mode)


def init_snap_params(
    mode: str,
    L: int,
    *,
    shared: bool,
    fourier_M: int = 2,
    few_M: int = 3,
    device=None,
) -> torch.Tensor:
    d = snap_dim(mode, fourier_M, few_M)
    if shared:
        t = torch.zeros(d, dtype=torch.float32, device=device)
    else:
        t = torch.zeros(L, d, dtype=torch.float32, device=device)
    if mode == "shell":
        # small breakup of shells
        if shared:
            t = torch.linspace(0.0, 0.3, d)
        else:
            t = torch.linspace(0.0, 0.3, d).unsqueeze(0).repeat(L, 1)
    elif mode == "x_pop":
        t = t + 0.4
    elif mode == "proj":
        t = t + 0.5
    elif mode == "const":
        t = t + 0.1
    elif mode == "kerr":
        # mild Kerr + small linear
        if shared:
            t = torch.tensor([0.15, 0.05], dtype=torch.float32, device=device)
        else:
            t = torch.tensor([0.15, 0.05], dtype=torch.float32, device=device).unsqueeze(0).repeat(L, 1)
    elif mode in ("gauss", "poly2_n"):
        # α≈0, β≈0.3, δ≈0.05
        base = torch.tensor([0.0, 0.3, 0.05], dtype=torch.float32, device=device)
        t = base if shared else base.unsqueeze(0).repeat(L, 1)
    elif mode in ("poly3_pc", "poly3_n"):
        base = torch.tensor([0.0, 0.3, 0.05, 0.01], dtype=torch.float32, device=device)
        t = base if shared else base.unsqueeze(0).repeat(L, 1)
    elif mode == "gauss_plus1":
        base = torch.tensor([0.0, 0.3, 0.05, 0.0], dtype=torch.float32, device=device)
        t = base if shared else base.unsqueeze(0).repeat(L, 1)
    elif mode == "few_m":
        if shared:
            t = torch.linspace(0.0, 0.4, d)
        else:
            t = torch.linspace(0.0, 0.4, d).unsqueeze(0).repeat(L, 1)
    return t.float()


def expand_theta(
    mode: str,
    vec: torch.Tensor,
    popcount: torch.Tensor,
    *,
    fourier_M: int = 2,
    few_M: int = 3,
) -> torch.Tensor:
    """Map parameter vector → length-8 Fock phases θ_k."""
    vec = vec.to(dtype=torch.float32).reshape(-1)
    device = vec.device
    pc = popcount.to(device=device).to(torch.float32)
    ks = torch.arange(NU, device=device, dtype=torch.float32)

    if mode == "shell":
        ph = vec.reshape(4)
        return ph[popcount.to(device=device).long()]
    if mode == "free":
        return vec.reshape(8)
    if mode == "bits":
        a = vec.reshape(3)
        th = torch.zeros(NU, dtype=torch.float32, device=device)
        for b in range(3):
            th = th + a[b] * ((ks.long() >> b) & 1).to(torch.float32)
        return th
    if mode == "fourier":
        M = fourier_M
        expect = 1 + 2 * M
        assert vec.numel() == expect, (vec.shape, expect)
        c0 = vec[0]
        th = torch.zeros(NU, dtype=torch.float32, device=device) + c0
        for m in range(1, M + 1):
            a = vec[1 + 2 * (m - 1)]
            b = vec[2 + 2 * (m - 1)]
            ang = 2.0 * math.pi * m * ks / float(NU)
            th = th + a * torch.cos(ang) + b * torch.sin(ang)
        return th
    if mode == "proj":
        th = torch.zeros(NU, dtype=torch.float32, device=device)
        th[0] = vec.reshape(-1)[0]
        return th
    if mode == "x_pop":
        beta = vec.reshape(-1)[0]
        return beta * pc
    if mode == "const":
        return torch.zeros(NU, dtype=torch.float32, device=device) + vec.reshape(-1)[0]
    if mode == "kerr":
        chi, omega = vec.reshape(2)
        return chi * ks * (ks - 1.0) * 0.5 + omega * ks
    if mode == "gauss":
        a, b, d = vec.reshape(3)
        return a + b * pc + d * pc * pc
    if mode == "poly2_n":
        a, b, d = vec.reshape(3)
        return a + b * ks + d * ks * ks
    if mode == "poly3_pc":
        a, b, d, e = vec.reshape(4)
        return a + b * pc + d * pc * pc + e * pc * pc * pc
    if mode == "poly3_n":
        a, b, d, e = vec.reshape(4)
        return a + b * ks + d * ks * ks + e * ks * ks * ks
    if mode == "few_m":
        M = int(few_M)
        assert vec.numel() == M, (vec.shape, M)
        th = torch.zeros(NU, dtype=torch.float32, device=device)
        th[:M] = vec.reshape(M)
        return th
    if mode == "gauss_plus1":
        a, b, d, rho = vec.reshape(4)
        return a + b * pc + d * pc * pc + rho * (pc >= 2.999).to(torch.float32)
    raise ValueError(mode)


def layer_vec(params: torch.Tensor, ell: int, shared: bool) -> torch.Tensor:
    if shared:
        return params
    return params[ell]


def pad_snap(
    params: torch.Tensor | None,
    L_new: int,
    mode: str,
    shared: bool,
    fourier_M: int = 2,
    few_M: int = 3,
):
    """Pad / truncate snap params for init transfer."""
    if params is None:
        return init_snap_params(
            mode, L_new, shared=shared, fourier_M=fourier_M, few_M=few_M
        )
    p = params.detach().clone().float()
    if shared:
        d = snap_dim(mode, fourier_M, few_M)
        if p.numel() == d:
            return p.reshape(d)
        return init_snap_params(
            mode, L_new, shared=True, fourier_M=fourier_M, few_M=few_M
        )
    # per-layer (L, d)
    if p.dim() == 1:
        p = p.unsqueeze(0)
    L_old, d = p.shape
    if L_new == L_old:
        return p
    if L_new < L_old:
        return p[:L_new].clone()
    out = torch.zeros(L_new, d, dtype=torch.float32)
    out[:L_old] = p
    if L_old >= 1:
        out[L_old:] = p[-1]
    return out


def snap_tag(mode: str, shared: bool, fourier_M: int = 2, few_M: int = 3) -> str:
    sh = "shared" if shared else "perL"
    if mode == "fourier":
        return f"snap_{mode}M{fourier_M}_{sh}"
    if mode == "few_m":
        return f"snap_{mode}M{few_M}_{sh}"
    return f"snap_{mode}_{sh}"


def describe_heads() -> list[dict[str, Any]]:
    rows = []
    for mode in SNAP_MODES:
        for shared in (False, True):
            rows.append(
                {
                    "mode": mode,
                    "shared": shared,
                    "dim": snap_dim(mode),
                    "tag": snap_tag(mode, shared),
                }
            )
    return rows


def describe_easy_heads(few_M: int = 3) -> list[dict[str, Any]]:
    rows = []
    for mode in SNAP_EASY_MODES:
        for shared in (False, True):
            rows.append(
                {
                    "mode": mode,
                    "shared": shared,
                    "dim": snap_dim(mode, few_M=few_M),
                    "tag": snap_tag(mode, shared, few_M=few_M),
                    "params_per_layer": snap_dim(mode, few_M=few_M),
                }
            )
    return rows


def fit_shell_to_mode(
    shell_phi: torch.Tensor,
    mode: str,
    *,
    few_M: int = 3,
) -> torch.Tensor:
    """Least-squares project shell φ_pc (L,4) or (4,) onto a restricted head.

    For modes that are not a linear projection of shell, falls back to init +
    best-effort LS on the 4 popcount samples.
    """
    phi = shell_phi.detach().float().cpu()
    if phi.dim() == 1:
        phi = phi.unsqueeze(0)
    L, d_shell = phi.shape
    assert d_shell == 4, phi.shape
    pcs = torch.arange(4, dtype=torch.float32)  # shells 0..3

    def _fit_poly_pc(deg: int) -> torch.Tensor:
        # Vandermonde on pc=0..3
        cols = [pcs ** p for p in range(deg + 1)]
        A = torch.stack(cols, dim=1)  # (4, deg+1)
        out = []
        for ell in range(L):
            y = phi[ell]
            # lstsq
            sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.reshape(-1)
            out.append(sol)
        return torch.stack(out, dim=0)

    if mode == "const":
        # best constant = mean of shells
        return phi.mean(dim=1, keepdim=True)
    if mode == "gauss":
        return _fit_poly_pc(2)  # α,β,δ
    if mode == "poly3_pc":
        return _fit_poly_pc(3)
    if mode == "gauss_plus1":
        # fit gauss on pc=0,1,2; residual = φ3 - gauss(3)
        g = _fit_poly_pc(2)
        a, b, d = g[:, 0], g[:, 1], g[:, 2]
        pred3 = a + b * 3.0 + d * 9.0
        rho = phi[:, 3] - pred3
        return torch.stack([a, b, d, rho], dim=1)
    if mode == "x_pop":
        # θ = β·pc → fit β from shells (ignore α)
        # min_β Σ_pc (β pc - φ_pc)^2
        num = (pcs * phi).sum(dim=1)
        den = (pcs * pcs).sum().clamp(min=1e-8)
        beta = (num / den).unsqueeze(1)
        return beta
    if mode == "shell":
        return phi.clone()
    if mode == "few_m":
        # Map shell→Fock then take first M; approximate via expand
        # θ_k ≈ φ[pc(k)]; take first M Fock phases
        pc_of_k = torch.tensor([bin(k).count("1") for k in range(8)], dtype=torch.long)
        out = []
        for ell in range(L):
            th = phi[ell][pc_of_k]
            out.append(th[:few_M].clone())
        return torch.stack(out, dim=0)
    if mode in ("kerr", "poly2_n", "poly3_n"):
        # Fit on Fock index using shell-expanded phases as targets
        pc_of_k = torch.tensor([bin(k).count("1") for k in range(8)], dtype=torch.long)
        ks = torch.arange(8, dtype=torch.float32)
        out = []
        for ell in range(L):
            y = phi[ell][pc_of_k]
            if mode == "kerr":
                # χ·k(k-1)/2 + ω·k
                A = torch.stack([ks * (ks - 1.0) * 0.5, ks], dim=1)
            elif mode == "poly2_n":
                A = torch.stack([torch.ones(8), ks, ks * ks], dim=1)
            else:
                A = torch.stack([torch.ones(8), ks, ks * ks, ks ** 3], dim=1)
            sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.reshape(-1)
            out.append(sol)
        return torch.stack(out, dim=0)
    if mode == "free":
        pc_of_k = torch.tensor([bin(k).count("1") for k in range(8)], dtype=torch.long)
        out = []
        for ell in range(L):
            out.append(phi[ell][pc_of_k].clone())
        return torch.stack(out, dim=0)
    # fallback zeros of correct dim
    return init_snap_params(mode, L, shared=False, few_M=few_M)


def project_shared(params_per_L: torch.Tensor) -> torch.Tensor:
    """Average per-layer params → shared vector."""
    p = params_per_L.detach().float()
    if p.dim() == 1:
        return p.clone()
    return p.mean(dim=0)
