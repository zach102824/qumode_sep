"""Driver bake-off arms (Opus probe 1, two-arm cut).

Arms
----
1. projector : |0⟩ → W → [U_Φ(γ_ℓ) → W · SNAP₀(β_ℓ) · W†]^L
               W compiles flat prep (W|0⟩=|flat₈⟩); SNAP₀ phases Fock |0⟩ only.
2. snap_d    : seq_best-style motif COST→SNAP→D (or D prep + SNAP→D→COST),
               shared fixed-angle SNAP+D params.

No GNN. Ideal Gray-X ceiling dropped (Zach steer 2026-10-29).
Self-contained: only explore_OCT29.lib imports.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .ansatz import AnsatzConfig, MultiCavityAnsatz, _soft_clip_alpha
from .gray_grid import build_grid_to_x
from .packing import K_BITS, NU, n_cavities
from .seq_ansatz import Motif, MultiCavitySeqAnsatz, init_params
from .train_loop import DATA, RUNS, _split_arrays, load_pack

PKG = Path(__file__).resolve().parents[1]
BAKE_RUNS = PKG / "runs" / "driver_bakeoff"


def _S(med: float, b90: float) -> float:
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def fit_b(ns: List[int], vals: List[float]) -> dict:
    """Least-squares fit cost ≈ C · b^n  on log: log cost = log C + n log b."""
    ns_a = np.asarray(ns, dtype=np.float64)
    vs = np.asarray(vals, dtype=np.float64)
    mask = np.isfinite(vs) & (vs > 0)
    if mask.sum() < 2:
        return {"b": float("nan"), "C": float("nan"), "ns": ns, "vals": vals}
    x = ns_a[mask]
    y = np.log(vs[mask])
    # y = a + b_coeff * x  with b = exp(b_coeff), C = exp(a)
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    logC, logb = float(coef[0]), float(coef[1])
    return {
        "b": float(math.exp(logb)),
        "C": float(math.exp(logC)),
        "logb": logb,
        "ns": [int(n) for n in x],
        "vals": [float(v) for v in vs[mask]],
    }


def linear_ramp(L: int, Delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """γ_ℓ = Δ·ℓ/L, β_ℓ = Δ(1−ℓ/L) with ℓ = 1..L (annealing schedule)."""
    ell = np.arange(1, L + 1, dtype=np.float64)
    gamma = Delta * (ell / L)
    beta = Delta * (1.0 - ell / L)
    return gamma.astype(np.float32), beta.astype(np.float32)


# ---------------------------------------------------------------------------
# W compile: flat prep via SNAP+D pairs
# ---------------------------------------------------------------------------


def _displacement_mat(re: torch.Tensor, im: torch.Tensor, lam, V, nv, amax: float, cdtype, rdtype):
    re = re.to(dtype=rdtype)
    im = im.to(dtype=rdtype)
    _, _, r, ph = _soft_clip_alpha(re, im, amax)
    phase_n = torch.exp(1j * ph.to(cdtype) * nv.to(cdtype))
    W = phase_n[:, None] * V
    eigs = torch.exp(-1j * r.to(cdtype) * lam.to(cdtype))
    return (W * eigs[None, :]) @ W.conj().T


def _snap_mat(th: torch.Tensor, nf: int, NU_: int, device, cdtype, rdtype):
    ph = torch.zeros(nf, dtype=rdtype, device=device)
    ph = ph.clone()
    ph[:NU_] = th.to(rdtype)
    return torch.diag(torch.exp(-1j * ph.to(cdtype)))


@dataclass
class CompiledW:
    """Frozen single-cavity flat-prep unitary W (nf×nf) and fidelity metrics."""

    W: torch.Tensor  # (nf, nf) complex
    nf: int
    n_pairs: int
    fidelity: float
    leak: float
    A: np.ndarray  # (n_pairs, 2)
    TH: np.ndarray  # (n_pairs, NU)
    amax: float = 1.8

    def to_dict(self) -> dict:
        return {
            "nf": self.nf,
            "n_pairs": self.n_pairs,
            "fidelity": self.fidelity,
            "leak": self.leak,
            "A": self.A.tolist(),
            "TH": self.TH.tolist(),
            "amax": self.amax,
        }

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "W": self.W.detach().cpu(),
                "meta": self.to_dict(),
            },
            path,
        )

    @staticmethod
    def load(path: Path, device=None) -> "CompiledW":
        path = Path(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        meta = blob["meta"]
        W = blob["W"]
        if device is not None:
            W = W.to(device)
        return CompiledW(
            W=W,
            nf=int(meta["nf"]),
            n_pairs=int(meta["n_pairs"]),
            fidelity=float(meta["fidelity"]),
            leak=float(meta["leak"]),
            A=np.asarray(meta["A"], dtype=np.float64),
            TH=np.asarray(meta["TH"], dtype=np.float64),
            amax=float(meta.get("amax", 1.8)),
        )


def compile_flat_prep(
    nf: int = 16,
    n_pairs: int = 3,
    steps: int = 2500,
    lr: float = 5e-3,
    amax: float = 1.8,
    seed: int = 0,
    device=None,
    target_F: float = 0.999,
    leak_pen: float = 10.0,
    log_fn=None,
) -> CompiledW:
    """Optimize SNAP+D chain so W|0⟩ ≈ |flat₈⟩ on levels 0..NU-1, leak < 1e-3."""
    device = device or torch.device("cpu")
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), "[W]", *a, flush=True))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cdtype = torch.complex64
    rdtype = torch.float32

    a = torch.diag(torch.sqrt(torch.arange(1, nf, dtype=rdtype)), 1)
    X = 1j * (a.T.contiguous() - a)
    lam_np, V_np = np.linalg.eigh(X.numpy().astype(np.complex128))
    lam = torch.as_tensor(lam_np.real, dtype=rdtype, device=device)
    V = torch.as_tensor(V_np, dtype=cdtype, device=device)
    nv = torch.arange(nf, dtype=rdtype, device=device)

    A = torch.nn.Parameter(
        torch.as_tensor(rng.normal(0, 0.4, size=(n_pairs, 2)), dtype=rdtype, device=device)
    )
    TH = torch.nn.Parameter(
        torch.as_tensor(rng.normal(0, 0.4, size=(n_pairs, NU)), dtype=rdtype, device=device)
    )
    # warm: first D toward coherent that overlaps flat
    with torch.no_grad():
        A[0, 0] = 1.2
        A[0, 1] = 0.0

    target = torch.zeros(nf, dtype=cdtype, device=device)
    target[:NU] = 1.0 / math.sqrt(NU)

    opt = torch.optim.Adam([A, TH], lr=lr)
    best = None
    best_loss = float("inf")

    def apply_W(psi0: torch.Tensor) -> torch.Tensor:
        psi = psi0
        for i in range(n_pairs):
            D = _displacement_mat(A[i, 0], A[i, 1], lam, V, nv, amax, cdtype, rdtype)
            psi = D @ psi
            S = _snap_mat(TH[i], nf, NU, device, cdtype, rdtype)
            psi = S @ psi
        return psi

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        psi0 = torch.zeros(nf, dtype=cdtype, device=device)
        psi0[0] = 1.0 + 0.0j
        psi = apply_W(psi0)
        # fidelity to flat (computational); leak = mass outside 0..NU-1
        probs = (psi.real.square() + psi.imag.square())
        p_comp = probs[:NU].sum()
        leak = (1.0 - p_comp).clamp(min=0.0)
        ov = (target.conj() * psi).sum()
        F = (ov.real.square() + ov.imag.square()).clamp(min=1e-12)
        loss = (1.0 - F) + leak_pen * leak
        # also nudge phases so flat is equal-amp (already in F)
        loss.backward()
        opt.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best = {
                "F": float(F.detach()),
                "leak": float(leak.detach()),
                "A": A.detach().cpu().clone(),
                "TH": TH.detach().cpu().clone(),
                "psi": psi.detach().cpu().clone(),
            }
        if step % 250 == 0 or step + 1 == steps:
            log(f"step {step}: F={float(F.detach()):.6f} leak={float(leak.detach()):.3e} loss={float(loss.detach()):.4e}")
        if best is not None and best["F"] >= target_F and best["leak"] < 1e-3:
            log(f"early stop at step {step}: F={best['F']:.6f} leak={best['leak']:.3e}")
            break

    assert best is not None
    # rebuild W matrix by applying to each basis vector
    A_b = best["A"].to(device)
    TH_b = best["TH"].to(device)
    # temporarily assign
    with torch.no_grad():
        A.data.copy_(A_b)
        TH.data.copy_(TH_b)
        cols = []
        for k in range(nf):
            e = torch.zeros(nf, dtype=cdtype, device=device)
            e[k] = 1.0 + 0.0j
            cols.append(apply_W(e))
        Wmat = torch.stack(cols, dim=1)  # W|k⟩ = col k

    cw = CompiledW(
        W=Wmat.detach().cpu(),
        nf=nf,
        n_pairs=n_pairs,
        fidelity=best["F"],
        leak=best["leak"],
        A=best["A"].numpy(),
        TH=best["TH"].numpy(),
        amax=amax,
    )
    log(f"compiled W: F={cw.fidelity:.6f} leak={cw.leak:.3e} nf={nf} pairs={n_pairs}")
    return cw


# ---------------------------------------------------------------------------
# Shared Fock simulator extensions for projector arm
# ---------------------------------------------------------------------------


class ProjectorAnsatz(MultiCavityAnsatz):
    """|0⟩ → W → [U_Φ(γ) → W SNAP₀(β) W†]^L with frozen W, 2 params/layer."""

    def __init__(self, cfg: AnsatzConfig, W: torch.Tensor, device=None, dtype=torch.complex64):
        super().__init__(cfg, device=device, dtype=dtype)
        # W is (nf, nf); truncate/pad if needed
        nf = self.nf
        Wc = W.detach().to(device=self.lam.device, dtype=self.cdtype)
        if Wc.shape[0] != nf:
            raise ValueError(f"W nf={Wc.shape[0]} != sim nf={nf}")
        self.register_buffer("Wmat", Wc)
        self.register_buffer("Wdag", Wc.conj().T.contiguous())

    def _snap0(self, beta: torch.Tensor) -> torch.Tensor:
        """Phase only Fock |0⟩ → exp(-i β)."""
        ph = torch.zeros(self.nf, dtype=self.rdtype, device=self.lam.device)
        ph = ph.clone()
        ph[0] = beta.to(self.rdtype)
        return torch.diag(torch.exp(-1j * ph.to(self.cdtype)))

    def _apply_all(self, psi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        for c in range(self.C):
            psi = self._apply_local(psi, M, c)
        return psi

    def forward_proj(self, E: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> dict:
        """gamma, beta: (L,) shared across cavities."""
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        beta = beta.to(dtype=self.rdtype, device=self.lam.device).reshape(L)

        psi = self.vacuum()
        # prep: W on every cavity
        psi = self._apply_all(psi, self.Wmat)

        Cn = self.cost_Cn(E)
        for ell in range(L):
            # U_Φ(γ)
            phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Cn.to(self.cdtype))
            sl = tuple(slice(0, NU) for _ in range(self.C))
            psi_new = psi.clone()
            psi_new[sl] = psi[sl] * phase
            psi = psi_new
            # mixer: W · SNAP₀(β) · W†
            mixer = self.Wmat @ self._snap0(beta[ell]) @ self.Wdag
            psi = self._apply_all(psi, mixer)

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
            "arm": "projector",
        }


# ---------------------------------------------------------------------------
# Eval / train helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_projector(
    n: int,
    W: CompiledW,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    L: int = 4,
    split: str = "val",
    max_count: int = 24,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
    device=None,
) -> dict:
    device = device or torch.device("cpu")
    nf = int(nf or W.nf)
    pack = load_pack(n, data_dir=data_dir)
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"n={n} {split} missing E")
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = ProjectorAnsatz(cfg, W.W.to(device), device=device).to(device)
    invs, leaks = [], []
    t0 = time.time()
    for b in range(len(seeds)):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_proj(Eb, gamma, beta)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "arm": "projector",
        "n": n,
        "split": split,
        "B": int(len(invs)),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0,
        "L": L,
        "nf": nf,
    }


@torch.no_grad()
def eval_snap_d(
    n: int,
    motif: Motif,
    params: dict,
    split: str = "val",
    max_count: int = 24,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
    device=None,
) -> dict:
    device = device or torch.device("cpu")
    pack = load_pack(n, data_dir=data_dir)
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"n={n} {split} missing E")
    cfg = AnsatzConfig(n=n, L=motif.L, nf=nf)
    sim = MultiCavitySeqAnsatz(cfg, motif, device=device).to(device)
    p = {k: v.detach().to(device) for k, v in params.items()}
    invs, leaks = [], []
    t0 = time.time()
    for b in range(len(seeds)):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_motif(Eb, p)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "arm": "snap_d",
        "n": n,
        "split": split,
        "B": int(len(invs)),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0,
        "L": motif.L,
        "motif": motif.name,
        "nf": nf or cfg.nf,
    }


def _instance_loss(out: dict, leakpen: float = 0.1) -> torch.Tensor:
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


def train_projector_lbfgs(
    W: CompiledW,
    L: int = 4,
    n_train: int = 9,
    max_train: int = 48,
    max_val: int = 24,
    steps: int = 60,
    lr: float = 0.25,
    Delta0: float = 1.0,
    seed: int = 0,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
    device=None,
    leakpen: float = 0.1,
    log_fn=None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """L-BFGS (torch) on (γ, β) from linear-ramp init; train on n9 only."""
    device = device or torch.device("cpu")
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), "[proj]", *a, flush=True))
    torch.manual_seed(seed)
    nf = int(nf or W.nf)
    g0, b0 = linear_ramp(L, Delta0)
    gamma = torch.nn.Parameter(torch.as_tensor(g0, dtype=torch.float32, device=device))
    beta = torch.nn.Parameter(torch.as_tensor(b0, dtype=torch.float32, device=device))

    pack = load_pack(n_train, data_dir=data_dir)
    seeds, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
    cfg = AnsatzConfig(n=n_train, L=L, nf=nf)
    sim = ProjectorAnsatz(cfg, W.W.to(device), device=device).to(device)

    opt = torch.optim.LBFGS(
        [gamma, beta],
        lr=lr,
        max_iter=4,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    rng = np.random.default_rng(seed)
    best = {"loss": float("inf"), "gamma": None, "beta": None, "val": None}

    def closure():
        opt.zero_grad(set_to_none=True)
        bs = min(8, len(seeds))
        idx = rng.choice(len(seeds), size=bs, replace=False)
        losses = []
        for b in idx:
            Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
            out = sim.forward_proj(Eb, gamma, beta)
            losses.append(_instance_loss(out, leakpen=leakpen))
        loss = torch.stack(losses).mean()
        loss.backward()
        return loss

    for step in range(steps):
        loss = opt.step(closure)
        # val
        with torch.no_grad():
            m = eval_projector(
                n_train, W, gamma.detach(), beta.detach(), L=L,
                split="val", max_count=max_val, nf=nf, data_dir=data_dir, device=device,
            )
        sc = m["S"]
        if sc < best["loss"]:
            best = {
                "loss": sc,
                "gamma": gamma.detach().cpu().clone(),
                "beta": beta.detach().cpu().clone(),
                "val": m,
                "step": step,
                "train_loss": float(loss.detach()) if torch.is_tensor(loss) else float(loss),
            }
            log(
                f"step {step}: NEW best S={sc:.4f} Med={m['median_inv_p']:.2f}/"
                f"B90={m['B90_inv_p']:.2f} leak={m['mean_leak']:.3e} "
                f"γ={gamma.detach().cpu().numpy().round(3).tolist()} "
                f"β={beta.detach().cpu().numpy().round(3).tolist()}"
            )
        elif step % 5 == 0:
            log(f"step {step}: S={sc:.4f} stagnant")

    return best["gamma"], best["beta"], best


def train_snap_d_lbfgs(
    motif: Motif,
    n_train: int = 9,
    max_train: int = 48,
    max_val: int = 24,
    steps: int = 80,
    lr: float = 0.15,
    seed: int = 0,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
    device=None,
    leakpen: float = 0.1,
    Delta0: float = 1.0,
    log_fn=None,
) -> Tuple[dict, dict]:
    """Init SNAP+D from ramp-ish γ/small SNAP; L-BFGS/Adam hybrid on n9."""
    device = device or torch.device("cpu")
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), "[snap_d]", *a, flush=True))
    torch.manual_seed(seed)
    params = init_params(motif, seed=seed, device=device, scale=0.15)
    # warm gamma from linear ramp; leave SNAP/D small
    g0, _ = linear_ramp(motif.L, Delta0)
    with torch.no_grad():
        if "gamma" in params:
            # gamma shape (L, n_cost)
            for j in range(params["gamma"].shape[1]):
                params["gamma"][:, j] = torch.as_tensor(g0, dtype=torch.float32, device=device)

    pack = load_pack(n_train, data_dir=data_dir)
    seeds, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
    cfg = AnsatzConfig(n=n_train, L=motif.L, nf=nf)
    sim = MultiCavitySeqAnsatz(cfg, motif, device=device).to(device)

    # Adam is more reliable for SNAP phases than LBFGS; use Adam then short LBFGS
    opt = torch.optim.Adam(list(params.values()), lr=3e-3)
    rng = np.random.default_rng(seed)
    best = {"score": float("inf"), "params": None, "val": None, "step": -1}

    adam_steps = max(steps - 20, steps // 2)
    for step in range(steps):
        if step == adam_steps:
            opt = torch.optim.LBFGS(
                list(params.values()), lr=lr, max_iter=3, history_size=15, line_search_fn="strong_wolfe"
            )
            log(f"switch to LBFGS at step {step}")

        if isinstance(opt, torch.optim.LBFGS):
            def closure():
                opt.zero_grad(set_to_none=True)
                bs = min(6, len(seeds))
                idx = rng.choice(len(seeds), size=bs, replace=False)
                losses = []
                for b in idx:
                    Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
                    out = sim.forward_motif(Eb, params)
                    losses.append(_instance_loss(out, leakpen=leakpen))
                loss = torch.stack(losses).mean()
                loss.backward()
                return loss
            loss = opt.step(closure)
        else:
            opt.zero_grad(set_to_none=True)
            bs = min(8, len(seeds))
            idx = rng.choice(len(seeds), size=bs, replace=False)
            losses = []
            for b in idx:
                Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
                out = sim.forward_motif(Eb, params)
                losses.append(_instance_loss(out, leakpen=leakpen))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), 5.0)
            opt.step()

        do_val = (step % 5 == 0) or (step + 1 == steps) or step < 3
        if not do_val:
            continue
        m = eval_snap_d(
            n_train, motif, params, split="val", max_count=max_val,
            nf=nf, data_dir=data_dir, device=device,
        )
        if m["S"] < best["score"]:
            best = {
                "score": m["S"],
                "params": {k: v.detach().cpu().clone() for k, v in params.items()},
                "val": m,
                "step": step,
                "train_loss": float(loss.detach()) if torch.is_tensor(loss) else float(loss),
            }
            log(
                f"step {step}: NEW best S={m['S']:.4f} Med={m['median_inv_p']:.2f}/"
                f"B90={m['B90_inv_p']:.2f} leak={m['mean_leak']:.3e}"
            )

    if best["params"] is None:
        best["params"] = {k: v.detach().cpu().clone() for k, v in params.items()}
        best["val"] = eval_snap_d(
            n_train, motif, params, split="val", max_count=max_val,
            nf=nf, data_dir=data_dir, device=device,
        )
    return best["params"], best


def scan_delta_projector(
    W: CompiledW,
    L: int = 4,
    n: int = 9,
    Deltas: Optional[List[float]] = None,
    max_count: int = 16,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
    device=None,
    log_fn=None,
) -> dict:
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), "[scan]", *a, flush=True))
    Deltas = Deltas or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    rows = []
    best = None
    for D in Deltas:
        g, b = linear_ramp(L, D)
        gamma = torch.as_tensor(g)
        beta = torch.as_tensor(b)
        m = eval_projector(
            n, W, gamma, beta, L=L, split="val", max_count=max_count,
            nf=nf, data_dir=data_dir, device=device,
        )
        m["Delta"] = float(D)
        rows.append(m)
        log(f"Δ={D:.2f}: Med={m['median_inv_p']:.2f} B90={m['B90_inv_p']:.2f} S={m['S']:.3f} leak={m['mean_leak']:.3e}")
        if best is None or m["S"] < best["S"]:
            best = m
    return {"rows": rows, "best": best}


def scan_delta_snap_d(
    motif: Motif,
    n: int = 9,
    Deltas: Optional[List[float]] = None,
    max_count: int = 16,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
    device=None,
    seed: int = 0,
    log_fn=None,
) -> dict:
    """Ramp only on γ; SNAP/D from small fixed init (β-like via SNAP scale)."""
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), "[scan]", *a, flush=True))
    Deltas = Deltas or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    rows = []
    best = None
    for D in Deltas:
        params = init_params(motif, seed=seed, device=device, scale=0.05)
        g0, b0 = linear_ramp(motif.L, D)
        with torch.no_grad():
            if "gamma" in params:
                for j in range(params["gamma"].shape[1]):
                    params["gamma"][:, j] = torch.as_tensor(g0, dtype=torch.float32, device=device or "cpu")
            # put ramp β into SNAP level-0 phase as a crude mixer knob
            if "TH" in params:
                for ell in range(motif.L):
                    params["TH"][ell, 0, 0] = float(b0[ell])
            if "A" in params:
                for ell in range(motif.L):
                    params["A"][ell, 0, 0] = 0.3 * float(b0[ell])
        m = eval_snap_d(
            n, motif, params, split="val", max_count=max_count,
            nf=nf, data_dir=data_dir, device=device,
        )
        m["Delta"] = float(D)
        rows.append(m)
        log(f"snap_d Δ={D:.2f}: Med={m['median_inv_p']:.2f} B90={m['B90_inv_p']:.2f} S={m['S']:.3f}")
        if best is None or m["S"] < best["S"]:
            best = {**m, "params": {k: v.detach().cpu().clone() for k, v in params.items()}}
    return {"rows": rows, "best": best}
