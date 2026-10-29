"""Seq motifs with compiled atomic blocks (Opus probe 5).

Vocabulary additions beyond raw SNAP/D/COST:
  W, WDAG     — frozen flat-prep unitary / adjoint (from W.pt)
  FLAT_PREP   — alias for W (prep only)
  SNAP0       — phase only Fock |0⟩ (scalar β)
  PROJ        — atomic W·SNAP0(β)·W† (one β)

Projector seed: prep=(W,), layer=(COST, PROJ), L=4
  ≡ |0⟩→W→[U_Φ(γ) → W·SNAP0(β)·W†]^L
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .ansatz import AnsatzConfig, MultiCavityAnsatz, _soft_clip_alpha
from .drivers import CompiledW
from .packing import NU

PKG = Path(__file__).resolve().parents[1]
DEFAULT_W = PKG / "runs" / "driver_bakeoff" / "W.pt"

# Extended vocab
RAW_OPS = ("SNAP", "D", "COST")
ATOM_OPS = ("W", "WDAG", "FLAT_PREP", "SNAP0", "PROJ")
ALL_LAYER_OPS = RAW_OPS + ATOM_OPS
ALL_PREP_OPS = ("SNAP", "D", "W", "FLAT_PREP")


@dataclass(frozen=True)
class WMotif:
    name: str
    prep: Tuple[str, ...]
    layer: Tuple[str, ...]
    L: int = 4

    def validate(self) -> None:
        for o in self.prep:
            if o not in ALL_PREP_OPS:
                raise ValueError(f"prep op {o} not in {ALL_PREP_OPS}")
        for o in self.layer:
            if o not in ALL_LAYER_OPS:
                raise ValueError(f"layer op {o} not in {ALL_LAYER_OPS}")
        if "COST" not in self.layer:
            raise ValueError("layer must include COST")
        has_mixer = any(o in ("SNAP", "D", "SNAP0", "PROJ", "W", "WDAG") for o in self.layer)
        if not has_mixer:
            raise ValueError("layer needs a mixer op")
        if self.L < 1 or self.L > 16:
            raise ValueError(f"L={self.L}")

    def to_dict(self) -> dict:
        return {"name": self.name, "prep": list(self.prep), "layer": list(self.layer), "L": self.L}

    @staticmethod
    def from_dict(d: dict) -> "WMotif":
        m = WMotif(
            name=str(d.get("name", "custom")),
            prep=tuple(d.get("prep", ["W"])),
            layer=tuple(d["layer"]),
            L=int(d.get("L", 4)),
        )
        m.validate()
        return m

    def param_layout(self) -> dict:
        n_snap_prep = sum(1 for o in self.prep if o == "SNAP")
        n_d_prep = sum(1 for o in self.prep if o == "D")
        # W/FLAT_PREP in prep: 0 params
        n_snap = sum(1 for o in self.layer if o == "SNAP")
        n_d = sum(1 for o in self.layer if o == "D")
        n_cost = sum(1 for o in self.layer if o == "COST")
        n_snap0 = sum(1 for o in self.layer if o == "SNAP0")
        n_proj = sum(1 for o in self.layer if o == "PROJ")
        return {
            "snap_prep": n_snap_prep,
            "d_prep": n_d_prep,
            "snap": n_snap,
            "d": n_d,
            "cost": n_cost,
            "snap0": n_snap0,
            "proj": n_proj,
        }

    def n_params(self) -> int:
        lay = self.param_layout()
        return (
            lay["snap_prep"] * NU
            + lay["d_prep"] * 2
            + self.L * lay["snap"] * NU
            + self.L * lay["d"] * 2
            + self.L * lay["cost"]
            + self.L * lay["snap0"]
            + self.L * lay["proj"]
        )


# Seed pool — projector family first, then hybrids
SEED_W_MOTIFS: List[WMotif] = [
    WMotif("proj_cost_proj", prep=("W",), layer=("COST", "PROJ"), L=4),
    WMotif("proj_cost_proj_L6", prep=("W",), layer=("COST", "PROJ"), L=6),
    WMotif("proj_cost_w_snap0_wdag", prep=("W",), layer=("COST", "W", "SNAP0", "WDAG"), L=4),
    WMotif("flat_cost_proj", prep=("FLAT_PREP",), layer=("COST", "PROJ"), L=4),
    WMotif("proj_cost_proj_cost", prep=("W",), layer=("COST", "PROJ", "COST"), L=4),  # double cost
    WMotif("proj_proj_cost", prep=("W",), layer=("PROJ", "COST"), L=4),  # mixer-first
    WMotif("proj_cost_snap0", prep=("W",), layer=("COST", "SNAP0"), L=4),  # weak: SNAP0 in flat basis
    WMotif("hybrid_cost_proj_d", prep=("W",), layer=("COST", "PROJ", "D"), L=4),
    WMotif("hybrid_cost_snap_d_proj", prep=("W",), layer=("COST", "SNAP", "D", "PROJ"), L=4),
    WMotif("raw_cost_snap_d", prep=("D",), layer=("COST", "SNAP", "D"), L=4),  # old baseline control
    WMotif("proj_cost_proj_L8", prep=("W",), layer=("COST", "PROJ"), L=8),
]


def init_w_params(
    motif: WMotif,
    seed: int = 0,
    device=None,
    scale: float = 0.2,
    warm_gamma=None,
    warm_beta=None,
) -> dict:
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
        g = torch.as_tensor(rng.normal(0, scale, size=(L, lay["cost"])), dtype=torch.float32, device=device)
        if warm_gamma is not None:
            wg = torch.as_tensor(warm_gamma, dtype=torch.float32, device=device)
            for j in range(lay["cost"]):
                n = min(L, wg.numel())
                g[:n, j] = wg[:n]
        out["gamma"] = g
    if lay["snap0"]:
        b = torch.as_tensor(rng.normal(0, scale, size=(L, lay["snap0"])), dtype=torch.float32, device=device)
        if warm_beta is not None:
            wb = torch.as_tensor(warm_beta, dtype=torch.float32, device=device)
            for j in range(lay["snap0"]):
                n = min(L, wb.numel())
                b[:n, j] = wb[:n]
        out["beta_snap0"] = b
    if lay["proj"]:
        b = torch.as_tensor(rng.normal(0, scale, size=(L, lay["proj"])), dtype=torch.float32, device=device)
        if warm_beta is not None:
            wb = torch.as_tensor(warm_beta, dtype=torch.float32, device=device)
            for j in range(lay["proj"]):
                n = min(L, wb.numel())
                b[:n, j] = wb[:n]
        out["beta_proj"] = b
    for v in out.values():
        v.requires_grad_(True)
    return out


class MultiCavityWSeqAnsatz(MultiCavityAnsatz):
    """Fock simulator driven by WMotif + frozen W."""

    def __init__(
        self,
        cfg: AnsatzConfig,
        motif: WMotif,
        W: torch.Tensor,
        device=None,
        dtype=torch.complex64,
    ):
        cfg = AnsatzConfig(n=cfg.n, L=motif.L, nf=cfg.nf, amax=cfg.amax, mode=cfg.mode)
        super().__init__(cfg, device=device, dtype=dtype)
        self.motif = motif
        motif.validate()
        Wc = W.detach().to(device=self.lam.device, dtype=self.cdtype)
        if Wc.shape[0] != self.nf:
            # pad/crop
            nf = self.nf
            Wfull = torch.eye(nf, dtype=self.cdtype, device=self.lam.device)
            m = min(nf, Wc.shape[0])
            Wfull[:m, :m] = Wc[:m, :m]
            Wc = Wfull
        self.register_buffer("Wmat", Wc)
        self.register_buffer("Wdag", Wc.conj().T.contiguous())

    def _snap_mat(self, th: torch.Tensor) -> torch.Tensor:
        ph = torch.zeros(self.nf, dtype=self.rdtype, device=self.lam.device)
        ph = ph.clone()
        ph[:NU] = th.to(self.rdtype)
        return torch.diag(torch.exp(-1j * ph.to(self.cdtype)))

    def _snap0_mat(self, beta: torch.Tensor) -> torch.Tensor:
        ph = torch.zeros(self.nf, dtype=self.rdtype, device=self.lam.device)
        ph = ph.clone()
        ph[0] = beta.to(self.rdtype)
        return torch.diag(torch.exp(-1j * ph.to(self.cdtype)))

    def _proj_mat(self, beta: torch.Tensor) -> torch.Tensor:
        return self.Wmat @ self._snap0_mat(beta) @ self.Wdag

    def _apply_all(self, psi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        for c in range(self.C):
            psi = self._apply_local(psi, M, c)
        return psi

    def _apply_cost(self, psi: torch.Tensor, Cn: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        phase = torch.exp(-1j * gamma.to(self.cdtype) * Cn.to(self.cdtype))
        sl = tuple(slice(0, NU) for _ in range(self.C))
        psi_new = psi.clone()
        psi_new[sl] = psi[sl] * phase
        return psi_new

    def forward_wmotif(self, E: torch.Tensor, params: dict) -> dict:
        psi = self.vacuum()
        Cn = self.cost_Cn(E)

        i_snap = i_d = 0
        for op in self.motif.prep:
            if op == "SNAP":
                psi = self._apply_all(psi, self._snap_mat(params["TH_prep"][i_snap]))
                i_snap += 1
            elif op == "D":
                a = params["A_prep"][i_d]
                psi = self._apply_all(psi, self._D(a[0], a[1]))
                i_d += 1
            elif op in ("W", "FLAT_PREP"):
                psi = self._apply_all(psi, self.Wmat)

        for ell in range(self.motif.L):
            i_snap = i_d = i_cost = i_s0 = i_proj = 0
            for op in self.motif.layer:
                if op == "SNAP":
                    psi = self._apply_all(psi, self._snap_mat(params["TH"][ell, i_snap]))
                    i_snap += 1
                elif op == "D":
                    a = params["A"][ell, i_d]
                    psi = self._apply_all(psi, self._D(a[0], a[1]))
                    i_d += 1
                elif op == "COST":
                    psi = self._apply_cost(psi, Cn, params["gamma"][ell, i_cost])
                    i_cost += 1
                elif op == "W" or op == "FLAT_PREP":
                    psi = self._apply_all(psi, self.Wmat)
                elif op == "WDAG":
                    psi = self._apply_all(psi, self.Wdag)
                elif op == "SNAP0":
                    psi = self._apply_all(psi, self._snap0_mat(params["beta_snap0"][ell, i_s0]))
                    i_s0 += 1
                elif op == "PROJ":
                    psi = self._apply_all(psi, self._proj_mat(params["beta_proj"][ell, i_proj]))
                    i_proj += 1

        probs = psi.real.square() + psi.imag.square()
        comp_sl = tuple(slice(0, NU) for _ in range(self.C))
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
            "motif": self.motif.name,
        }


def mutate_w_motif(parent: WMotif, rng: np.random.Generator) -> WMotif:
    """Mutations favoring compiled atoms + projector compositions."""
    parent.validate()
    prep = list(parent.prep)
    layer = list(parent.layer)
    L = parent.L
    kind = int(rng.integers(0, 9))
    if kind == 0 and len(layer) >= 2:
        i, j = rng.choice(len(layer), size=2, replace=False)
        layer[i], layer[j] = layer[j], layer[i]
    elif kind == 1:
        op = str(rng.choice(["PROJ", "SNAP0", "SNAP", "D", "W", "WDAG"]))
        if "COST" in layer:
            layer.insert(layer.index("COST"), op)
        else:
            layer.append(op)
    elif kind == 2 and len(layer) > 2:
        idxs = [i for i, o in enumerate(layer) if o != "COST"]
        if len(idxs) > 1:
            del layer[int(rng.choice(idxs))]
    elif kind == 3:
        _preps = [("W",), ("FLAT_PREP",), ("D",), ("W", "D"), ("D", "W")]
        prep = list(_preps[int(rng.integers(0, len(_preps)))])
    elif kind == 4:
        L = int(np.clip(L + int(rng.choice([-1, 1, 2])), 3, 10))
    elif kind == 5:
        # reset to pure projector
        layer = ["COST", "PROJ"]
        prep = ["W"]
    elif kind == 6:
        layer = ["COST", "W", "SNAP0", "WDAG"]
        prep = ["W"]
    elif kind == 7:
        # add PROJ after COST
        if "PROJ" not in layer:
            if "COST" in layer:
                layer.insert(layer.index("COST") + 1, "PROJ")
            else:
                layer += ["COST", "PROJ"]
    else:
        idxs = [i for i, o in enumerate(layer) if o in ("SNAP", "D", "PROJ", "SNAP0")]
        if idxs:
            i = int(rng.choice(idxs))
            layer.insert(i, layer[i])
    if "COST" not in layer:
        layer.insert(0, "COST")
    if not any(o in ("SNAP", "D", "SNAP0", "PROJ", "W", "WDAG") for o in layer):
        layer.append("PROJ")
    while len(layer) > 8:
        for i in range(len(layer) - 1, -1, -1):
            if layer[i] != "COST":
                del layer[i]
                break
        else:
            break
    name = f"mut_{'-'.join(prep)}__{'-'.join(layer)}_L{L}"
    m = WMotif(name=name[:90], prep=tuple(prep), layer=tuple(layer), L=L)
    m.validate()
    return m


def load_W(path: Path | None = None, nf: int = 16, device=None) -> torch.Tensor:
    path = Path(path) if path else DEFAULT_W
    cw = CompiledW.load(path, device=device)
    W = cw.W
    if W.shape[0] != nf:
        Wfull = torch.eye(nf, dtype=W.dtype)
        m = min(nf, W.shape[0])
        Wfull[:m, :m] = W[:m, :m]
        W = Wfull
    return W.to(device) if device is not None else W
