"""Qumode-native gate-sequence ansatz (explore track).

NOT locked forever to the QAOA-like sandwich. Motifs are ordered lists of
native ops: SNAP, D (displacement), COST (diag Φ=H_cost). Prep is a separate
prefix. Params are shared across cavities (size-independent fixed-angle).

Ban: free U_phase^q / dense nonlocal phase heads that only emulate qubit QAOA
without a native payoff. COST stays exact H_cost (freedom 0) unless a later
ladder stage opens clause weights / εΔ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .ansatz import MultiCavityAnsatz, AnsatzConfig, _soft_clip_alpha
from .packing import NU


# --- motif vocabulary -------------------------------------------------------
OPS = ("SNAP", "D", "COST")
PREP_OPS = ("SNAP", "D")


@dataclass(frozen=True)
class Motif:
    """Circuit structure: prep ops once, then layer ops repeated L times."""

    name: str
    prep: Tuple[str, ...]
    layer: Tuple[str, ...]
    L: int = 5

    def validate(self) -> None:
        for o in self.prep:
            if o not in PREP_OPS:
                raise ValueError(f"prep op {o} not in {PREP_OPS}")
        for o in self.layer:
            if o not in OPS:
                raise ValueError(f"layer op {o} not in {OPS}")
        if "COST" not in self.layer:
            raise ValueError("layer must include COST (diag H_cost)")
        if not any(o in ("SNAP", "D") for o in self.layer):
            raise ValueError("layer needs at least one SNAP or D mixer op")
        if self.L < 1 or self.L > 16:
            raise ValueError(f"L={self.L} out of range")

    def to_dict(self) -> dict:
        return {"name": self.name, "prep": list(self.prep), "layer": list(self.layer), "L": self.L}

    @staticmethod
    def from_dict(d: dict) -> "Motif":
        m = Motif(
            name=str(d.get("name", "custom")),
            prep=tuple(d.get("prep", ["D"])),
            layer=tuple(d["layer"]),
            L=int(d.get("L", 5)),
        )
        m.validate()
        return m

    def param_layout(self) -> dict:
        """Counts of learnable blocks (shared across cavities)."""
        n_snap_prep = sum(1 for o in self.prep if o == "SNAP")
        n_d_prep = sum(1 for o in self.prep if o == "D")
        n_snap = sum(1 for o in self.layer if o == "SNAP")
        n_d = sum(1 for o in self.layer if o == "D")
        n_cost = sum(1 for o in self.layer if o == "COST")
        return {
            "snap_prep": n_snap_prep,  # (n_snap_prep, NU)
            "d_prep": n_d_prep,        # (n_d_prep, 2)
            "snap": n_snap,            # (L, n_snap, NU)
            "d": n_d,                  # (L, n_d, 2)
            "cost": n_cost,            # (L, n_cost)
        }

    def n_params(self) -> int:
        lay = self.param_layout()
        return (
            lay["snap_prep"] * NU
            + lay["d_prep"] * 2
            + self.L * lay["snap"] * NU
            + self.L * lay["d"] * 2
            + self.L * lay["cost"]
        )


# Seed motifs — qumode-native, not "QAOA with SNAP+D labels"
SEED_MOTIFS: List[Motif] = [
    Motif("baseline_snap_d_cost", prep=("D",), layer=("SNAP", "D", "COST"), L=5),
    Motif("snap_d_snap_cost", prep=("D",), layer=("SNAP", "D", "SNAP", "COST"), L=5),  # Gray-X-like sandwich
    Motif("d_snap_cost", prep=("D",), layer=("D", "SNAP", "COST"), L=5),
    Motif("snap_cost_d", prep=("D",), layer=("SNAP", "COST", "D"), L=5),
    Motif("cost_snap_d", prep=("D",), layer=("COST", "SNAP", "D"), L=5),
    Motif("prep_snap_d__snap_d_cost", prep=("SNAP", "D"), layer=("SNAP", "D", "COST"), L=5),
    Motif("snap_d_d_cost", prep=("D",), layer=("SNAP", "D", "D", "COST"), L=5),
    Motif("snap_snap_d_cost", prep=("D",), layer=("SNAP", "SNAP", "D", "COST"), L=5),
    Motif("d_cost_snap", prep=("D",), layer=("D", "COST", "SNAP"), L=5),
    Motif("snap_d_cost_L4", prep=("D",), layer=("SNAP", "D", "COST"), L=4),
    Motif("snap_d_snap_cost_L4", prep=("D",), layer=("SNAP", "D", "SNAP", "COST"), L=4),
    Motif("snap_d_snap_cost_L6", prep=("D",), layer=("SNAP", "D", "SNAP", "COST"), L=6),
]


def init_params(motif: Motif, seed: int = 0, device=None, scale: float = 0.3) -> dict:
    motif.validate()
    rng = np.random.default_rng(seed)
    device = device or torch.device("cpu")
    lay = motif.param_layout()
    L = motif.L
    out = {}
    if lay["snap_prep"]:
        out["TH_prep"] = torch.as_tensor(
            rng.normal(0, scale, size=(lay["snap_prep"], NU)), dtype=torch.float32, device=device
        )
    if lay["d_prep"]:
        out["A_prep"] = torch.as_tensor(
            rng.normal(0, scale, size=(lay["d_prep"], 2)), dtype=torch.float32, device=device
        )
    if lay["snap"]:
        out["TH"] = torch.as_tensor(
            rng.normal(0, scale, size=(L, lay["snap"], NU)), dtype=torch.float32, device=device
        )
    if lay["d"]:
        out["A"] = torch.as_tensor(
            rng.normal(0, scale, size=(L, lay["d"], 2)), dtype=torch.float32, device=device
        )
    if lay["cost"]:
        out["gamma"] = torch.as_tensor(
            rng.normal(0, scale, size=(L, lay["cost"])), dtype=torch.float32, device=device
        )
    for v in out.values():
        v.requires_grad_(True)
    return out


class MultiCavitySeqAnsatz(MultiCavityAnsatz):
    """Same Fock simulator; forward driven by a Motif + shared params."""

    def __init__(self, cfg: AnsatzConfig, motif: Motif, device=None, dtype=torch.complex64):
        # cfg.L must match motif.L for buffer sizes used elsewhere; set it
        cfg = AnsatzConfig(n=cfg.n, L=motif.L, nf=cfg.nf, amax=cfg.amax, mode=cfg.mode)
        super().__init__(cfg, device=device, dtype=dtype)
        self.motif = motif
        motif.validate()

    def _snap_mat(self, th: torch.Tensor) -> torch.Tensor:
        """th: (NU,) → nf×nf diagonal SNAP (0 phase on leak levels)."""
        ph = torch.zeros(self.nf, dtype=self.rdtype, device=self.lam.device)
        ph = ph.clone()
        ph[:NU] = th.to(self.rdtype)
        return torch.diag(torch.exp(-1j * ph.to(self.cdtype)))

    def _apply_all_cav(self, psi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        for c in range(self.C):
            psi = self._apply_local(psi, M, c)
        return psi

    def _apply_cost(self, psi: torch.Tensor, Cn: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        phase = torch.exp(-1j * gamma.to(self.cdtype) * Cn.to(self.cdtype))
        sl = tuple(slice(0, NU) for _ in range(self.C))
        psi_new = psi.clone()
        psi_new[sl] = psi[sl] * phase
        return psi_new

    def forward_motif(self, E: torch.Tensor, params: dict) -> dict:
        C, nf = self.C, self.nf
        psi = self.vacuum()
        Cn = self.cost_Cn(E)

        # prep
        i_snap = 0
        i_d = 0
        for op in self.motif.prep:
            if op == "SNAP":
                psi = self._apply_all_cav(psi, self._snap_mat(params["TH_prep"][i_snap]))
                i_snap += 1
            elif op == "D":
                a = params["A_prep"][i_d]
                psi = self._apply_all_cav(psi, self._D(a[0], a[1]))
                i_d += 1

        for ell in range(self.motif.L):
            i_snap = i_d = i_cost = 0
            for op in self.motif.layer:
                if op == "SNAP":
                    psi = self._apply_all_cav(psi, self._snap_mat(params["TH"][ell, i_snap]))
                    i_snap += 1
                elif op == "D":
                    a = params["A"][ell, i_d]
                    psi = self._apply_all_cav(psi, self._D(a[0], a[1]))
                    i_d += 1
                elif op == "COST":
                    g = params["gamma"][ell, i_cost]
                    psi = self._apply_cost(psi, Cn, g)
                    i_cost += 1

        probs = psi.real.square() + psi.imag.square()
        comp_sl = tuple(slice(0, NU) for _ in range(C))
        p_comp = probs[comp_sl]
        p_leak = (1.0 - p_comp.sum()).clamp(min=0.0)
        E_grid = E.to(device=self.lam.device)[self.grid_to_x]
        succ = p_comp[E_grid == 0].sum()
        p_succ = succ.clamp(min=0.0)
        inv_p = 1.0 / p_succ.clamp(min=1e-12)
        return {
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": inv_p,
            "norm": probs.sum(),
            "psi": psi,
            "freedom_step": 0,
            "motif": self.motif.name,
        }


def mutate_motif(parent: Motif, rng: np.random.Generator) -> Motif:
    """Structural mutation favoring native-efficient motifs (not free phase heads)."""
    parent.validate()
    prep = list(parent.prep)
    layer = list(parent.layer)
    L = parent.L
    kind = int(rng.integers(0, 7))
    if kind == 0 and len(layer) >= 2:
        i, j = rng.choice(len(layer), size=2, replace=False)
        layer[i], layer[j] = layer[j], layer[i]
    elif kind == 1:
        # insert a mixer op (SNAP or D) before COST if present
        op = str(rng.choice(["SNAP", "D"]))
        if "COST" in layer:
            idx = layer.index("COST")
            layer.insert(idx, op)
        else:
            layer.append(op)
    elif kind == 2 and len(layer) > 2:
        # drop a non-COST mixer if ≥1 mixer remains
        idxs = [i for i, o in enumerate(layer) if o != "COST"]
        if len(idxs) > 1:
            del layer[int(rng.choice(idxs))]
    elif kind == 3:
        # flip prep between D / SNAP+D / D+SNAP
        prep = list(rng.choice([
            ("D",),
            ("SNAP", "D"),
            ("D", "SNAP"),
            ("SNAP", "D", "SNAP"),
        ]))
    elif kind == 4:
        L = int(np.clip(L + int(rng.choice([-1, 1])), 3, 8))
    elif kind == 5:
        # replace layer with a SNAP-D-SNAP sandwich (Gray-X-like)
        layer = ["SNAP", "D", "SNAP", "COST"]
    else:
        # duplicate a mixer op
        idxs = [i for i, o in enumerate(layer) if o in ("SNAP", "D")]
        if idxs:
            i = int(rng.choice(idxs))
            layer.insert(i, layer[i])
    # sanitize
    if "COST" not in layer:
        layer.append("COST")
    if not any(o in ("SNAP", "D") for o in layer):
        layer.insert(0, "SNAP")
    # cap length
    while len(layer) > 6:
        # drop last non-COST
        for i in range(len(layer) - 1, -1, -1):
            if layer[i] != "COST":
                del layer[i]
                break
        else:
            break
    name = f"mut_{'-'.join(prep)}__{'-'.join(layer)}_L{L}"
    m = Motif(name=name[:80], prep=tuple(prep), layer=tuple(layer), L=L)
    m.validate()
    return m
