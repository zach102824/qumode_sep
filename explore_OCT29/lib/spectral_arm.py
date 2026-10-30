"""Spectral Arm C helpers — W_H frame + load Arm C shell params.

Self-contained under explore_OCT29. Used by probe6 follow-ons (probe8+).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .gray_grid import ungray
from .packing import NU
from .sat_gen import fast_energies

PKG = Path(__file__).resolve().parents[1]
ARM_C_PATH = PKG / "runs" / "probe6_spectral_mixer" / "arm_C_shell_params.pt"
ARM_C_RESULTS = PKG / "runs" / "probe6_spectral_mixer" / "results.json"


def popcount(x: int) -> int:
    return int(bin(int(x)).count("1"))


def build_W_H(nf: int = 16, dtype=torch.complex64) -> torch.Tensor:
    """Ideal W_H = Π_gray · H^{⊗3} on levels 0..7; identity pad on 8..nf-1."""
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / math.sqrt(2.0)
    H3 = H1
    for _ in range(2):
        H3 = torch.kron(H3, H1)
    Pi = torch.zeros(8, 8, dtype=torch.float64)
    for B in range(8):
        f = ungray(B)
        Pi[f, B] = 1.0
    WH8 = Pi @ H3
    W = torch.eye(nf, dtype=torch.float64)
    W[:8, :8] = WH8
    return W.to(dtype=dtype)


def load_arm_C(path: Path | None = None) -> dict:
    """Load Arm C shell checkpoint: gamma (L,), phi (L,4), mode='shell'."""
    p = Path(path) if path is not None else ARM_C_PATH
    if not p.exists():
        raise FileNotFoundError(f"Arm C params missing: {p}")
    st = torch.load(p, map_location="cpu", weights_only=False)
    gamma = st["gamma"].detach().cpu().to(dtype=torch.float32).reshape(-1)
    phi = st["phi"].detach().cpu().to(dtype=torch.float32)
    if phi.ndim != 2 or phi.shape[1] != 4:
        raise ValueError(f"Arm C phi shape {tuple(phi.shape)} expected (L,4)")
    if gamma.numel() != phi.shape[0]:
        raise ValueError(f"gamma L={gamma.numel()} != phi L={phi.shape[0]}")
    return {
        "gamma": gamma,
        "phi": phi,
        "mode": st.get("mode", "shell"),
        "L": int(gamma.numel()),
        "path": str(p),
    }


def clause_span(v) -> int:
    """Number of distinct cavities touched by a 4-var clause (v//3)."""
    return len({int(x) // 3 for x in v})


def energies_by_span(n: int, cls_v, cls_s) -> Dict[int, np.ndarray]:
    """Unsat-count vectors E^(s) for clause spans s∈{2,3,4}. Shape (2^n,) each.

    Sum_s E^(s) == total E (verified). Empty span → zeros.
    """
    by: Dict[int, list] = {2: [], 3: [], 4: []}
    for v, s in zip(cls_v, cls_s):
        sp = clause_span(v)
        if sp in by:
            by[sp].append((list(map(int, v)), list(map(int, s))))
    out: Dict[int, np.ndarray] = {}
    dim = 1 << int(n)
    for sp, cl in by.items():
        if cl:
            out[sp] = fast_energies(int(n), cl).astype(np.float32)
        else:
            out[sp] = np.zeros(dim, dtype=np.float32)
    return out


def stack_span_E(n: int, cls_v, cls_s) -> np.ndarray:
    """Stack E^(2), E^(3), E^(4) → shape (3, 2^n)."""
    es = energies_by_span(n, cls_v, cls_s)
    return np.stack([es[2], es[3], es[4]], axis=0)


def warm_gamma_span(gamma: torch.Tensor, n_spans: int = 3) -> torch.Tensor:
    """Broadcast Arm C / champ γ_ℓ → γ_ℓ^(s) for s=0..n_spans-1. Shape (L, n_spans)."""
    g = gamma.detach().cpu().to(dtype=torch.float32).reshape(-1)
    return g[:, None].expand(g.numel(), n_spans).contiguous().clone()


def arm_C_yardstick() -> dict:
    """Test + val Med numbers from probe6 RESULTS (for CURRENT_BEST / gate)."""
    return {
        "name": "probe6 Arm C free shells W_H·SNAP(φ_pc)·W_H†",
        "L": 16,
        "params": str(ARM_C_PATH),
        "val": {
            9: {"Med": 1.429, "B90": 2.562, "leak": 0.0, "L_times_Med": 22.9, "B": 24},
            12: {"Med": 2.162, "B90": 7.316, "leak": 0.0, "L_times_Med": 34.6, "B": 24},
            15: {"Med": 2.255, "B90": 5.271, "leak": 0.0, "L_times_Med": 36.1, "B": 16},
            18: {"Med": 5.306, "B90": 36.004, "leak": 0.0, "L_times_Med": 84.9, "B": 16},
        },
        "test": {
            9: {"Med": 1.336, "B90": 2.092, "leak": 0.0, "L_times_Med": 21.4, "B": 40},
            12: {"Med": 2.175, "B90": 10.266, "leak": 0.0, "L_times_Med": 34.8, "B": 40},
            15: {"Med": 4.495, "B90": 16.540, "leak": 0.0, "L_times_Med": 71.9, "B": 40},
            18: {"Med": 3.764, "B90": 11.832, "leak": 0.0, "L_times_Med": 60.2, "B": 16},
        },
        "b_test_12_15_18_Med": 1.096,
        "promote_vs_A": {"ratio_n15": 1.41, "ratio_n18": 1.70},
        "note": "probe6 val n15 used B=16; val2 gate wants B=40 — re-score on full val when gating.",
    }
