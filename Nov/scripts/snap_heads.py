#!/usr/bin/env python3
"""SNAP angle parameterization heads for Nov1st-final.

Modes (Zach t2307 add):
  shell     — θ_k = φ_{pc(k)}           shape (…, 4)
  free      — θ_0..7 free               shape (…, 8)
  bits      — θ_k = Σ_b α_b bit_b(k)    shape (…, 3)
  fourier   — low-rank Fourier on Fock  shape (…, 1+2M)  M=2 → 5
  proj      — θ_0=β, else 0             shape (…,) / (…, 1)
  x_pop     — θ_k = β · pc(k)           shape (…,) / (…, 1)

shared=False → per-layer leading dim L; shared=True → one vector for all layers.
"""
from __future__ import annotations

import math
from typing import Any

import torch

NU = 8  # computational Fock support

SNAP_MODES = ("shell", "free", "bits", "fourier", "proj", "x_pop")


def snap_dim(mode: str, fourier_M: int = 2) -> int:
    if mode == "shell":
        return 4
    if mode == "free":
        return 8
    if mode == "bits":
        return 3
    if mode == "fourier":
        return 1 + 2 * fourier_M  # DC + (cos,sin)*M
    if mode in ("proj", "x_pop"):
        return 1
    raise ValueError(mode)


def init_snap_params(
    mode: str,
    L: int,
    *,
    shared: bool,
    fourier_M: int = 2,
    device=None,
) -> torch.Tensor:
    d = snap_dim(mode, fourier_M)
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
    return t.float()


def expand_theta(
    mode: str,
    vec: torch.Tensor,
    popcount: torch.Tensor,
    *,
    fourier_M: int = 2,
) -> torch.Tensor:
    """Map parameter vector → length-8 Fock phases θ_k."""
    vec = vec.to(dtype=torch.float32).reshape(-1)
    device = vec.device
    pc = popcount.to(device=device)
    if mode == "shell":
        ph = vec.reshape(4)
        return ph[pc]
    if mode == "free":
        return vec.reshape(8)
    if mode == "bits":
        a = vec.reshape(3)
        ks = torch.arange(NU, device=device)
        th = torch.zeros(NU, dtype=torch.float32, device=device)
        for b in range(3):
            th = th + a[b] * ((ks >> b) & 1).to(torch.float32)
        return th
    if mode == "fourier":
        # θ_k = c0 + Σ_m [a_m cos(2π m k / 8) + b_m sin(2π m k / 8)]
        M = fourier_M
        expect = 1 + 2 * M
        assert vec.numel() == expect, (vec.shape, expect)
        c0 = vec[0]
        ks = torch.arange(NU, device=device, dtype=torch.float32)
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
        return beta * pc.to(torch.float32)
    raise ValueError(mode)


def layer_vec(params: torch.Tensor, ell: int, shared: bool) -> torch.Tensor:
    if shared:
        return params
    return params[ell]


def pad_snap(params: torch.Tensor | None, L_new: int, mode: str, shared: bool, fourier_M: int = 2):
    """Pad / truncate snap params for init transfer."""
    if params is None:
        return init_snap_params(mode, L_new, shared=shared, fourier_M=fourier_M)
    p = params.detach().clone().float()
    if shared:
        # shared: just return clone (independent of L)
        d = snap_dim(mode, fourier_M)
        if p.numel() == d:
            return p.reshape(d)
        return init_snap_params(mode, L_new, shared=True, fourier_M=fourier_M)
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


def snap_tag(mode: str, shared: bool, fourier_M: int = 2) -> str:
    sh = "shared" if shared else "perL"
    if mode == "fourier":
        return f"snap_{mode}M{fourier_M}_{sh}"
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
