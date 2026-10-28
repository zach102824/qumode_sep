"""Searchable factor-graph GNN (numpy) predicting ansatz heads.

Baseline / evolve parent: H=128, rounds=4, agg=mean, degree features.
Outputs size-normalized mixer params for MultiCavityAnsatz.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

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
        """Rough count for tie-break."""
        H = self.H
        C = n_cavities(n)
        # message MLPs + heads
        return (
            H * H * self.rounds * 4
            + H * (C * L * 2 + C * L * NU + C * 2 + L + 1)
        )


def _init_W(rng: np.random.Generator, din: int, dout: int, scale: float = 1.0) -> np.ndarray:
    return (rng.normal(0, scale / np.sqrt(max(din, 1)), size=(din, dout))).astype(np.float64)


def _relu(x):
    return np.maximum(x, 0.0)


def _layernorm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class FactorGNN:
    """Lightweight numpy GNN over bipartite var↔clause factor graph."""

    def __init__(self, arch: GNNArch, n: int, m: int, L: int, seed: int = 0):
        self.arch = arch
        self.n = n
        self.m = m
        self.L = L
        self.C = n_cavities(n)
        self.rng = np.random.default_rng(seed)
        H = arch.H
        in_dim = 4 if arch.degree_features else 2
        self.W_in = _init_W(self.rng, in_dim, H)
        self.W_vc = [_init_W(self.rng, H, H) for _ in range(arch.rounds)]
        self.W_cv = [_init_W(self.rng, H, H) for _ in range(arch.rounds)]
        # heads
        self.W_alpha0 = _init_W(self.rng, H, self.C * 2, 0.3)
        self.W_A = _init_W(self.rng, H, self.C * L * 2, 0.3)
        self.W_TH = _init_W(self.rng, H, self.C * L * NU, 0.1)
        self.W_q = _init_W(self.rng, H, L, 0.1)
        self.W_g = _init_W(self.rng, H, 1, 0.1)
        self.b_alpha0 = np.zeros(self.C * 2)
        self.b_A = np.zeros(self.C * L * 2)
        self.b_TH = np.zeros(self.C * L * NU)
        self.b_q = np.zeros(L)
        self.b_g = np.zeros(1)

    def parameters(self) -> Dict[str, np.ndarray]:
        d = {
            "W_in": self.W_in,
            "W_alpha0": self.W_alpha0,
            "W_A": self.W_A,
            "W_TH": self.W_TH,
            "W_q": self.W_q,
            "W_g": self.W_g,
            "b_alpha0": self.b_alpha0,
            "b_A": self.b_A,
            "b_TH": self.b_TH,
            "b_q": self.b_q,
            "b_g": self.b_g,
        }
        for i, W in enumerate(self.W_vc):
            d[f"W_vc_{i}"] = W
        for i, W in enumerate(self.W_cv):
            d[f"W_cv_{i}"] = W
        return d

    def load_parameters(self, d: Dict[str, np.ndarray]):
        for k, v in d.items():
            if k.startswith("W_vc_"):
                self.W_vc[int(k.split("_")[-1])] = np.asarray(v)
            elif k.startswith("W_cv_"):
                self.W_cv[int(k.split("_")[-1])] = np.asarray(v)
            elif hasattr(self, k):
                setattr(self, k, np.asarray(v))

    def _agg(self, msgs: np.ndarray, axis: int) -> np.ndarray:
        if self.arch.agg == "sum":
            return msgs.sum(axis=axis)
        if self.arch.agg == "max":
            return msgs.max(axis=axis)
        return msgs.mean(axis=axis)

    def forward_graph(self, cls_v: np.ndarray, cls_s: np.ndarray) -> dict:
        """cls_v, cls_s: (m, 4). Returns ansatz param dict for one formula."""
        n, m, H, L, C = self.n, self.m, self.arch.H, self.L, self.C
        # var features: [1, pos_deg_norm, neg_deg_norm, total_deg_norm]
        pos = np.zeros(n)
        neg = np.zeros(n)
        for c in range(m):
            for j in range(4):
                v = int(cls_v[c, j])
                if cls_s[c, j] > 0:
                    pos[v] += 1
                else:
                    neg[v] += 1
        deg = np.maximum(pos + neg, 1.0)
        scale = 2.0 * m / max(n, 1)
        if self.arch.degree_features:
            fv = np.stack([np.ones(n), pos / scale, neg / scale, deg / scale], axis=-1)
        else:
            fv = np.stack([np.ones(n), deg / scale], axis=-1)
        hv = _relu(fv @ self.W_in)  # (n, H)
        hc = np.zeros((m, H))
        for r in range(self.arch.rounds):
            # var -> clause
            for c in range(m):
                vs = [int(cls_v[c, j]) for j in range(4)]
                msgs = hv[vs]  # (4, H)
                pooled = self._agg(msgs[None], axis=1)[0]
                new = _relu(pooled @ self.W_vc[r])
                if self.arch.layernorm:
                    new = _layernorm(new)
                if self.arch.residual and hc[c].shape == new.shape:
                    hc[c] = hc[c] + new if r > 0 else new
                else:
                    hc[c] = new
            # clause -> var
            acc = np.zeros((n, H))
            cnt = np.zeros(n)
            for c in range(m):
                for j in range(4):
                    v = int(cls_v[c, j])
                    acc[v] += hc[c]
                    cnt[v] += 1
            pooled_v = acc / np.maximum(cnt[:, None], 1.0)
            new_v = _relu(pooled_v @ self.W_cv[r])
            if self.arch.layernorm:
                new_v = _layernorm(new_v)
            hv = hv + new_v if self.arch.residual else new_v

        g = hv.mean(axis=0)  # graph readout
        alpha0 = (g @ self.W_alpha0 + self.b_alpha0).reshape(C, 2)
        A = (g @ self.W_A + self.b_A).reshape(C, L, 2)
        TH = (g @ self.W_TH + self.b_TH).reshape(C, L, NU)
        q_logits = g @ self.W_q + self.b_q
        gamma = float((g @ self.W_g + self.b_g)[0])
        # size-normalize displacements by sqrt(N)
        N = float(2 ** n)
        alpha0 = alpha0 / np.sqrt(N) * np.sqrt(64.0)  # ref n=6
        A = A / np.sqrt(N) * np.sqrt(64.0)
        return {
            "alpha0": alpha0,
            "A": A,
            "TH": TH,
            "q_logits": q_logits,
            "gamma": gamma,
        }


BASELINE_ARCH = GNNArch(H=128, rounds=4, agg="mean", degree_features=True)


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
