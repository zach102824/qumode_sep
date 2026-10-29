"""Factor-graph GNN heads — step 0 shared mixer.

Emits shared SNAP+D (10) + γ_ℓ (1) per layer = 11, independent of C.
Plus shared α₀ (2,) size-normalized. No head_q / per-cavity mixers.
Freedom via clause weights / local εΔ comes in later ladder steps.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import NU, n_cavities


@dataclass
class GNNArch:
    H: int = 128
    rounds: int = 4
    agg: str = "mean"  # mean|sum|max
    dropout: float = 0.0
    layernorm: bool = True
    residual: bool = True
    degree_features: bool = True
    lr: float = 3e-4
    wd: float = 1e-4
    ema: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "GNNArch":
        base = GNNArch()
        for k, v in d.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    def n_params_estimate(self, n: int = 9, m: int = 89, L: int = 5) -> int:
        """Rough count: backbone + shared heads (independent of C)."""
        H = self.H
        # heads: alpha0=2, A=L*2, TH=L*NU, gamma=L  →  2 + L*(2+NU+1)
        head_out = 2 + L * (2 + NU + 1)
        return H * H * self.rounds * 4 + H * head_out

    def mixer_params_per_layer(self) -> int:
        """10 shared mixer (TH:NU + A:2) + 1 γ = 11."""
        return NU + 2 + 1


class FactorGNN(nn.Module):
    """Bipartite var↔clause factor-graph GNN → shared mixer + γ_ℓ heads.

    Weight shapes do **not** depend on n or C, so a checkpoint trained at
    n=9 can zero-shot load into an n=12 FactorGNN (same arch, L).
    """

    def __init__(self, arch: GNNArch, n: int, m: int, L: int, seed: int = 0):
        super().__init__()
        self.arch = arch
        self.n = int(n)
        self.m = int(m)
        self.L = int(L)
        self.C = n_cavities(n)
        H = arch.H
        in_dim = 4 if arch.degree_features else 2
        g = torch.Generator()
        g.manual_seed(int(seed))

        def _lin(din, dout, scale=1.0):
            layer = nn.Linear(din, dout, bias=False)
            with torch.no_grad():
                layer.weight.normal_(0.0, scale / max(din, 1) ** 0.5, generator=g)
            return layer

        self.W_in = _lin(in_dim, H)
        self.W_vc = nn.ModuleList([_lin(H, H) for _ in range(arch.rounds)])
        self.W_cv = nn.ModuleList([_lin(H, H) for _ in range(arch.rounds)])
        self.ln_c = nn.ModuleList(
            [nn.LayerNorm(H) if arch.layernorm else nn.Identity() for _ in range(arch.rounds)]
        )
        self.ln_v = nn.ModuleList(
            [nn.LayerNorm(H) if arch.layernorm else nn.Identity() for _ in range(arch.rounds)]
        )
        self.drop = nn.Dropout(arch.dropout) if arch.dropout > 0 else nn.Identity()

        # Shared heads — independent of C
        self.head_alpha0 = nn.Linear(H, 2)          # shared α₀
        self.head_A = nn.Linear(H, L * 2)           # shared D per layer
        self.head_TH = nn.Linear(H, L * NU)         # shared SNAP per layer
        self.head_g = nn.Linear(H, L)               # γ_ℓ per layer
        with torch.no_grad():
            for head, sc in (
                (self.head_alpha0, 0.3),
                (self.head_A, 0.3),
                (self.head_TH, 0.1),
                (self.head_g, 0.1),
            ):
                head.weight.normal_(0.0, sc / max(H, 1) ** 0.5, generator=g)
                head.bias.zero_()

    def _agg(self, msgs: torch.Tensor, dim: int) -> torch.Tensor:
        if self.arch.agg == "sum":
            return msgs.sum(dim=dim)
        if self.arch.agg == "max":
            return msgs.max(dim=dim).values
        return msgs.mean(dim=dim)

    def forward_graph(
        self,
        cls_v: torch.Tensor,
        cls_s: torch.Tensor,
    ) -> dict:
        """cls_v, cls_s: (m, 4). Returns shared-mixer ansatz param dict."""
        n, m, H, L = self.n, self.m, self.arch.H, self.L
        device = cls_v.device
        dtype = next(self.parameters()).dtype
        cls_v = cls_v.long().to(device)
        cls_s = cls_s.to(device=device, dtype=dtype)

        pos = torch.zeros(n, device=device, dtype=dtype)
        neg = torch.zeros(n, device=device, dtype=dtype)
        flat_v = cls_v.reshape(-1)
        flat_s = cls_s.reshape(-1)
        pos.index_add_(0, flat_v, (flat_s > 0).to(dtype))
        neg.index_add_(0, flat_v, (flat_s <= 0).to(dtype))
        deg = torch.clamp(pos + neg, min=1.0)
        scale = 2.0 * m / max(n, 1)
        if self.arch.degree_features:
            fv = torch.stack(
                [torch.ones(n, device=device, dtype=dtype), pos / scale, neg / scale, deg / scale],
                dim=-1,
            )
        else:
            fv = torch.stack(
                [torch.ones(n, device=device, dtype=dtype), deg / scale],
                dim=-1,
            )
        hv = F.relu(self.W_in(fv))
        hc = torch.zeros(m, H, device=device, dtype=dtype)

        for r in range(self.arch.rounds):
            msgs = hv[cls_v]  # (m, 4, H)
            pooled = self._agg(msgs, dim=1)
            new = F.relu(self.W_vc[r](pooled))
            new = self.ln_c[r](new)
            new = self.drop(new)
            if self.arch.residual and r > 0:
                hc = hc + new
            else:
                hc = new

            msgs_c = hc.unsqueeze(1).expand(-1, 4, -1).reshape(-1, H)
            idx = cls_v.reshape(-1)
            acc = torch.zeros(n, H, device=device, dtype=dtype)
            acc.index_add_(0, idx, msgs_c)
            cnt = torch.zeros(n, device=device, dtype=dtype)
            cnt.index_add_(0, idx, torch.ones_like(idx, dtype=dtype))
            pooled_v = acc / cnt.clamp(min=1.0).unsqueeze(-1)
            new_v = F.relu(self.W_cv[r](pooled_v))
            new_v = self.ln_v[r](new_v)
            new_v = self.drop(new_v)
            hv = hv + new_v if self.arch.residual else new_v

        g = hv.mean(dim=0)
        alpha0 = self.head_alpha0(g)            # (2,)
        A = self.head_A(g).reshape(L, 2)        # (L, 2)
        TH = self.head_TH(g).reshape(L, NU)     # (L, NU)
        gamma = self.head_g(g)                  # (L,)
        # size-normalized displacement scale (transfer-friendly)
        N = float(2 ** n)
        scale_d = (64.0 / N) ** 0.5
        alpha0 = alpha0 * scale_d
        A = A * scale_d
        return {
            "alpha0": alpha0,
            "A": A,
            "TH": TH,
            "gamma": gamma,
        }

    def forward(self, cls_v, cls_s):
        return self.forward_graph(cls_v, cls_s)


BASELINE_ARCH = GNNArch(H=128, rounds=4, agg="mean", degree_features=True)

# Evolve search space — arch / train only. No head_q, no per-cavity mixer flags.
SEARCH_SPACE = {
    "H": [64, 128, 192, 256],
    "rounds": [2, 4, 6, 8],
    "agg": ["mean", "sum", "max"],
    "dropout": [0.0, 0.1],
    "layernorm": [True, False],
    "residual": [True, False],
    "degree_features": [True, False],
    "lr": [1e-4, 3e-4, 1e-3],
    "wd": [0.0, 1e-4],
}


def mutate_arch(parent: GNNArch, rng: np.random.Generator, n_flips: int = 2) -> GNNArch:
    child = GNNArch.from_dict(parent.to_dict())
    axes = list(SEARCH_SPACE.keys())
    rng.shuffle(axes)
    for ax in axes[:n_flips]:
        choices = [c for c in SEARCH_SPACE[ax] if c != getattr(child, ax)]
        if choices:
            setattr(child, ax, choices[int(rng.integers(len(choices)))])
    return child


def random_arch(rng: np.random.Generator) -> GNNArch:
    d = {k: SEARCH_SPACE[k][int(rng.integers(len(SEARCH_SPACE[k])))] for k in SEARCH_SPACE}
    return GNNArch.from_dict(d)
