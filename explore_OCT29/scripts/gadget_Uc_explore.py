#!/usr/bin/env python3
"""Gadget-Uc explore: X-gadget cost (clause parity / fixed stencil) + Arm C shell mixer.

BinH W=H^{⊗3}. Cost is NOT diag(E_unsat) for the GRAPE story:
  - gadget-clause: U_c = ∏_c exp(-iγ P_c), P_c=(∏ s_i)∏ X_{v_i}
    (sim: Z-parity diag via W† e^{-iγλ} W equivalence; verified vs Pauli product)
  - fixed-stencil: U_c = ∏_j exp(-iγ X_j)  (fully instance-blind)

Train {9,12} L=16; eval val/test + scale1000. Writes runs/gadget_Uc/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.ansatz import AnsatzConfig, MultiCavityAnsatz  # noqa: E402
from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.packing import K_BITS, NU, n_cavities  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import (  # noqa: E402
    build_W_BinH,
    build_grid_to_x_binary,
)
from explore_OCT29.scripts.probe6_spectral_mixer import (  # noqa: E402
    SpectralAnsatz,
    _loss,
    _S,
)

OUT = _PKG / "runs" / "gadget_Uc"
ARM_C_PARAMS = _PKG / "runs" / "probe6_spectral_mixer" / "arm_C_shell_params.pt"
CHAMP_PARAMS = _PKG / "runs" / "probe5_seq_vocab" / "champ_params.pt"
SCALE_DIR = _PKG / "data" / "scale1000"

# Arm C B=1000 yardstick (promote gate)
ARM_C_B1000 = {
    9: {"Med": 1.441, "B90": 3.314},
    12: {"Med": 1.991, "B90": 6.252},
    15: {"Med": 2.921, "B90": 10.959},
    18: {"Med": 4.573, "B90": 27.810},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[gadget]", *a, flush=True)


# ---------------------------------------------------------------------------
# Pauli / parity helpers (binary Fock packing)
# ---------------------------------------------------------------------------

def _pauli_x_bit(j: int, nf: int = 8) -> torch.Tensor:
    """X on binary bit j of Fock 0..7; identity pad above 7."""
    X = torch.zeros(nf, nf, dtype=torch.complex64)
    for f in range(NU):
        fp = f ^ (1 << j)
        X[fp, f] = 1.0 + 0.0j
    for f in range(NU, nf):
        X[f, f] = 1.0 + 0.0j
    return X


def clause_parity_lambda_vec(n: int, cls_v, cls_s) -> np.ndarray:
    """λ(z) = ∑_c (∏_i s_i) (-1)^{∑ z_{v_i}}  for z=0..2^n-1 (var0=MSB).

    Eigenvalues of ∑_c P_c^Z with P_c^Z = (∏ s_i) ∏ Z_{v_i}.
    Under BinH, W† exp(-iγ λ·diag) W = ∏ exp(-iγ P_c^X).
    """
    dim = 1 << n
    lam = np.zeros(dim, dtype=np.float64)
    # bit of var v in index x (var0=MSB): (x >> (n-1-v)) & 1
    for vrow, srow in zip(cls_v, cls_s):
        vs = [int(x) for x in vrow]
        ss = [int(x) for x in srow]
        sign0 = 1.0
        for s in ss:
            sign0 *= float(s)
        # contrib[x] = sign0 * ∏ (-1)^{bit_v} = sign0 * (-1)^{popcount of selected bits}
        # vectorized: start ones, for each var multiply by ±1 on half-spaces
        contrib = np.full(dim, sign0, dtype=np.float64)
        for vv in vs:
            shift = n - 1 - vv
            # bit=1 → multiply -1
            idx = np.arange(dim)
            bits = (idx >> shift) & 1
            contrib *= np.where(bits == 1, -1.0, 1.0)
        lam += contrib
    return lam.astype(np.float32)


def verify_gadget_equivalence(n_bits: int = 3, gamma: float = 0.37) -> dict:
    """One-cavity: ∏ exp(-iγ X…) via Pauli vs W† diag(e^{-iγλ}) W."""
    nf = 8
    W = build_W_BinH(nf=nf, dtype=torch.complex128).numpy()[:8, :8]
    # single XXX clause on bits 0,1,2 with signs +1,+1,+1
    Xs = [_pauli_x_bit(j, nf=8).numpy().astype(np.complex128)[:8, :8] for j in range(3)]
    P = Xs[0] @ Xs[1] @ Xs[2]
    U_pauli = np.cos(gamma) * np.eye(8) - 1j * np.sin(gamma) * P
    # λ for ZZZ: (-1)^{b0+b1+b2}
    lam = np.zeros(8, dtype=np.float64)
    for f in range(8):
        pop = bin(f).count("1")
        lam[f] = (-1.0) ** pop
    U_diag = np.diag(np.exp(-1j * gamma * lam))
    U_via = W.conj().T @ U_diag @ W
    err = float(np.linalg.norm(U_pauli - U_via, ord="fro"))
    # also multi single-X product vs stencil
    U_st = np.eye(8, dtype=np.complex128)
    for j in range(3):
        U_st = (np.cos(gamma) * np.eye(8) - 1j * np.sin(gamma) * Xs[j]) @ U_st
    # W† exp(-iγ(Z0+Z1+Z2)) W
    lam1 = np.zeros(8)
    for f in range(8):
        lam1[f] = sum((1.0 if ((f >> j) & 1) == 0 else -1.0) for j in range(3))
    U_st_via = W.conj().T @ np.diag(np.exp(-1j * gamma * lam1)) @ W
    err_st = float(np.linalg.norm(U_st - U_st_via, ord="fro"))
    out = {
        "XXX_pauli_vs_Wdiag": {"fro": err, "pass": err < 1e-10},
        "stencil_pauli_vs_Wdiag": {"fro": err_st, "pass": err_st < 1e-10},
        "gamma": gamma,
    }
    log("verify XXX fro", f"{err:.3e}", "stencil fro", f"{err_st:.3e}")
    return out


# ---------------------------------------------------------------------------
# Ansatz with cost modes
# ---------------------------------------------------------------------------

class GadgetAnsatz(SpectralAnsatz):
    """Arm C sandwich with selectable cost: unsat | parity | stencil_x."""

    def __init__(self, cfg: AnsatzConfig, W: torch.Tensor, device=None, dtype=torch.complex64):
        super().__init__(cfg, W, device=device, dtype=dtype)
        g = build_grid_to_x_binary(cfg.n)
        self.register_buffer(
            "grid_to_x_bin", torch.as_tensor(g, dtype=torch.long, device=self.lam.device)
        )
        # also overwrite Gray grid_to_x → binary for success scoring in binary packing
        self.register_buffer(
            "grid_to_x", torch.as_tensor(g, dtype=torch.long, device=self.lam.device)
        )
        # local X unitaries builders cached as buffers list
        Xs = torch.stack([_pauli_x_bit(j, nf=self.nf) for j in range(K_BITS)])
        self.register_buffer("Xbits", Xs.to(dtype=self.cdtype))

    def _parity_Cn(self, lam_vec: torch.Tensor) -> torch.Tensor:
        n = self.cfg.n
        m = int(round(9.93 * n))
        lam = lam_vec.to(device=self.lam.device, dtype=self.rdtype)
        grid = lam[self.grid_to_x_bin]
        Cn = grid * (n / m)
        Cn = Cn - Cn.amax()
        return Cn

    def _apply_stencil_x(self, psi: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """∏_j exp(-iγ X_j) via per-cavity product of bit-X gadgets."""
        g = gamma.to(dtype=self.rdtype)
        c = torch.cos(g).to(self.cdtype)
        s = torch.sin(g).to(self.cdtype)
        # per bit j: U_j = c I - i s X_j  on one cavity
        eye = torch.eye(self.nf, dtype=self.cdtype, device=self.lam.device)
        Ucav = eye.clone()
        for j in range(K_BITS):
            Uj = c * eye - 1j * s * self.Xbits[j]
            Ucav = Uj @ Ucav
        for cav in range(self.C):
            psi = self._apply_local(psi, Ucav, cav)
        return psi

    def forward_gadget(
        self,
        E: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
        cost_mode: str,
        lam_vec: torch.Tensor | None = None,
    ) -> dict:
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)

        if cost_mode == "unsat":
            Cn = self.cost_Cn(E)  # uses grid_to_x (=binary here)
        elif cost_mode == "parity":
            if lam_vec is None:
                raise ValueError("parity mode needs lam_vec")
            Cn = self._parity_Cn(lam_vec)
        elif cost_mode == "stencil_x":
            Cn = None
        else:
            raise ValueError(cost_mode)

        for ell in range(L):
            if cost_mode == "stencil_x":
                psi = self._apply_stencil_x(psi, gamma[ell])
            else:
                phase = torch.exp(-1j * gamma[ell].to(self.cdtype) * Cn.to(self.cdtype))
                sl = tuple(slice(0, NU) for _ in range(self.C))
                psi_new = psi.clone()
                psi_new[sl] = psi[sl] * phase
                psi = psi_new
            ph = phi[ell].to(self.rdtype).reshape(4)
            th = ph[self.popcount]
            mixer = self.Wmat @ self._snap_theta(th) @ self.Wdag
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
            "arm": cost_mode,
        }


def make_sim(n, W, L, nf, device):
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    return GadgetAnsatz(cfg, W, device=device).to(device)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_arm(
    n, W, gamma, phi, L, nf, device, cost_mode,
    split="val", max_count=24, data_dir=None, progress_every=0,
):
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        # scale1000 packs may only have scale1000_* or test_*
        if f"scale1000_seeds" in pack:
            key = "scale1000"
        elif f"test_seeds" in pack:
            key = "test"
        else:
            raise RuntimeError(f"n={n}: no split {split} in pack")
    seeds, cls_v, cls_s, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(seeds) == 0:
        return None
    sim = make_sim(n, W, L, nf, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(seeds)
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        lam = None
        if cost_mode == "parity":
            lam = torch.as_tensor(
                clause_parity_lambda_vec(n, cls_v[b], cls_s[b]),
                dtype=torch.float32,
                device=device,
            )
        out = sim.forward_gadget(Eb, gamma, phi, cost_mode, lam_vec=lam)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if progress_every and ((b + 1) % progress_every == 0 or b + 1 == B):
            rate = (b + 1) / max(time.time() - t0, 1e-9)
            log(
                f"eval {cost_mode} n{n} {b+1}/{B} Med≈{np.median(invs):.3f} "
                f"rate={rate:.2f}/s"
            )
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n,
        "split": key,
        "B": len(invs),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "frac_inv_le_2": float(np.mean(invs_a <= 2.0)),
        "wall_s": time.time() - t0,
        "L": L,
        "L_times_Med": L * med,
        "cost_mode": cost_mode,
    }


def train_arm(
    cost_mode, W, L, nf, device, steps=100, max_train=32, max_val=12, seed=0,
    gamma0=None, phi0=None, lr=2e-3, train_ns=(9, 12),
):
    train_ns = tuple(int(x) for x in train_ns)
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(
        gamma0.detach().clone() if gamma0 is not None else torch.linspace(0.3, 1.0, L)
    )
    if phi0 is not None:
        phi = torch.nn.Parameter(phi0.detach().clone())
    else:
        phi = torch.nn.Parameter(torch.zeros(L, 4))
    params = [gamma, phi]

    # preload E + clauses
    packs = {}
    for n in train_ns:
        pack = load_pack(n)
        seeds, cls_v, cls_s, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            seeds, cls_v, cls_s, _, E = _split_arrays(pack, "val", max_count=max_train)
        entry = {
            "E": torch.as_tensor(E, dtype=torch.float32, device=device),
            "cls_v": cls_v,
            "cls_s": cls_s,
        }
        if cost_mode == "parity":
            lams = [
                clause_parity_lambda_vec(n, cls_v[b], cls_s[b]) for b in range(len(E))
            ]
            entry["lam"] = torch.as_tensor(np.stack(lams), dtype=torch.float32, device=device)
        packs[n] = entry

    opt = torch.optim.Adam(params, lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}
    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    n_lo, n_hi = min(train_ns), max(train_ns)

    for step in range(steps):
        opt.zero_grad()
        losses = []
        for n, P in packs.items():
            idx = int(rng.integers(0, len(P["E"])))
            lam = None if cost_mode != "parity" else P["lam"][idx]
            out = sims[n].forward_gadget(
                P["E"][idx], gamma, phi, cost_mode, lam_vec=lam
            )
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(params, lr=0.2, max_iter=4, history_size=8)

                def closure():
                    opt_l.zero_grad()
                    ls = []
                    for n, P in packs.items():
                        for j in range(min(8, len(P["E"]))):
                            lam = None if cost_mode != "parity" else P["lam"][j]
                            o = sims[n].forward_gadget(
                                P["E"][j], gamma, phi, cost_mode, lam_vec=lam
                            )
                            ls.append(_loss(o))
                    lt = torch.stack(ls).mean()
                    lt.backward()
                    return lt

                try:
                    opt_l.step(closure)
                except Exception as e:
                    log("LBFGS skip", e)

            vals = {}
            for n in train_ns:
                vals[n] = eval_arm(
                    n, W, gamma.detach(), phi.detach(), L, nf, device, cost_mode,
                    split="val", max_count=max_val,
                )
            sc = 0.4 * vals[n_lo]["S"] + 0.6 * vals[n_hi]["S"]
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": vals,
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "mode": "shell",
                        "cost_mode": cost_mode,
                        "W": "BinH",
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                bits = " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in train_ns
                )
                log(f"train {cost_mode} step {step}: Scomb={sc:.3f}", bits)
    return best


def ladder_eval(W, gamma, phi, L, nf, device, cost_mode, counts=None):
    counts = counts or {
        ("val", 9): 24, ("val", 12): 24, ("val", 15): 16, ("val", 18): 16,
        ("test", 9): 40, ("test", 12): 40, ("test", 15): 40, ("test", 18): 16,
    }
    out = {}
    for split in ("val", "test"):
        out[split] = {}
        for n in (9, 12, 15, 18):
            m = eval_arm(
                n, W, gamma, phi, L, nf, device, cost_mode,
                split=split, max_count=counts.get((split, n), 16),
            )
            if m:
                out[split][n] = m
                log(
                    f"ladder {cost_mode} {split} n{n}: "
                    f"Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} "
                    f"leak={m['mean_leak']:.4f} B={m['B']}"
                )
    return out


def scale1000_eval(W, gamma, phi, L, nf, device, cost_mode, ns, max_count=1000):
    out = {}
    if not SCALE_DIR.exists():
        log("scale1000 dir missing; skip")
        return out
    for n in ns:
        m = eval_arm(
            n, W, gamma, phi, L, nf, device, cost_mode,
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            progress_every=50 if n >= 15 else 100,
        )
        if m:
            out[n] = m
            log(
                f"scale1000 {cost_mode} n{n}: "
                f"Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} B={m['B']}"
            )
    return out


def fit_b_ns(ladder_split, ns=(12, 15, 18)):
    ns_u, meds, b90s = [], [], []
    for n in ns:
        m = ladder_split.get(n)
        if m:
            ns_u.append(n)
            meds.append(m["median_inv_p"])
            b90s.append(m["B90_inv_p"])
    if len(ns_u) < 2:
        return None
    return {"Med": fit_b(ns_u, meds), "B90": fit_b(ns_u, b90s), "ns": ns_u}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


def write_results_md(results, path: Path):
    lines = [
        "# Gadget-Uc explore — X-gadget cost + Arm C shell mixer (BinH)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "See `docs/PLAN_gadget_Uc.md`. Cost is **not** unsat diag(E) for GRAPE;",
        "gadget-clause uses parity Pauli gadgets; stencil is fully instance-blind.",
        "",
        "## Operator equivalence checks",
        "",
    ]
    v = results.get("verify", {})
    for k, block in v.items():
        if isinstance(block, dict) and "fro" in block:
            lines.append(
                f"- **{k}**: fro={block['fro']:.3e} pass={block['pass']}"
            )
    lines.append("")

    ref = ARM_C_B1000
    lines += [
        "## Arm C B=1000 reference (promote gate)",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        lines.append(f"| {n} | {ref[n]['Med']:.3f} | {ref[n]['B90']:.3f} |")
    lines.append("")

    for arm in ("parity", "stencil_x"):
        block = results.get(arm)
        if not block:
            continue
        lines += [f"## Arm `{arm}`", ""]
        if "train_best_step" in block:
            lines.append(
                f"Train best step={block['train_best_step']} Scomb={block['train_score']:.4f}"
            )
            lines.append("")
        lad = block.get("ladder", {})
        for split in ("val", "test"):
            if split not in lad:
                continue
            lines += [f"### {split}", "", "| n | Med | B90 | leak | L×Med | B |",
                      "|---|-----|-----|------|-------|---|"]
            for n in (9, 12, 15, 18):
                m = lad[split].get(str(n)) or lad[split].get(n)
                if not m:
                    continue
                lines.append(
                    f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
                    f"{m['mean_leak']:.4f} | {m['L_times_Med']:.1f} | {m['B']} |"
                )
            lines.append("")
        sc = block.get("scale1000", {})
        if sc:
            lines += [
                "### scale1000",
                "",
                "| n | Med | B90 | frac≤2 | B | vs ArmC Med |",
                "|---|-----|-----|--------|---|-------------|",
            ]
            for n in (9, 12, 15, 18):
                m = sc.get(str(n)) or sc.get(n)
                if not m:
                    continue
                ratio = ref[n]["Med"] / max(m["median_inv_p"], 1e-12)
                lines.append(
                    f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
                    f"{m.get('frac_inv_le_2', float('nan')):.3f} | {m['B']} | "
                    f"×{ratio:.3f} |"
                )
            lines.append("")
        bfit = block.get("b_test_12_15_18")
        if bfit:
            lines.append(f"- b_test_12_15_18: {bfit}")
            lines.append("")

    # head-to-head table
    lines += [
        "## Head-to-head (test Med / B90)",
        "",
        "| n | parity Med/B90 | stencil Med/B90 | ArmC B1000 Med/B90 |",
        "|---|----------------|-----------------|--------------------|",
    ]
    for n in (9, 12, 15, 18):
        def gb(arm, n):
            lad = (results.get(arm) or {}).get("ladder", {}).get("test", {})
            m = lad.get(str(n)) or lad.get(n)
            if not m:
                # prefer scale1000 if present
                sc = (results.get(arm) or {}).get("scale1000", {})
                m = sc.get(str(n)) or sc.get(n)
            return m
        p, s = gb("parity", n), gb("stencil_x", n)
        ps = f"{p['median_inv_p']:.3f}/{p['B90_inv_p']:.3f}" if p else "—"
        ss = f"{s['median_inv_p']:.3f}/{s['B90_inv_p']:.3f}" if s else "—"
        rs = f"{ref[n]['Med']:.3f}/{ref[n]['B90']:.3f}"
        lines.append(f"| {n} | {ps} | {ss} | {rs} |")
    lines.append("")

    prom = results.get("promote", {})
    lines += ["## Promote", "", f"```{prom}```", ""]
    path.write_text("\n".join(lines) + "\n")
    log("wrote", path)


def load_warm(L):
    if ARM_C_PARAMS.exists():
        st = torch.load(ARM_C_PARAMS, map_location="cpu", weights_only=False)
        g = st["gamma"].detach().cpu().to(torch.float32).reshape(-1)
        p = st["phi"].detach().cpu().to(torch.float32)
        log("warm from Arm C", ARM_C_PARAMS)
        if g.numel() != L:
            g = torch.linspace(0.3, 1.0, L)
        if p.shape[0] != L:
            p = torch.zeros(L, 4)
        return g, p
    if CHAMP_PARAMS.exists():
        st = torch.load(CHAMP_PARAMS, map_location="cpu", weights_only=False)
        g = st["gamma"].reshape(L).to(torch.float32)
        p = torch.zeros(L, 4)
        p[:, 0] = st["beta_proj"].reshape(L).to(torch.float32)
        log("warm from champ PROJ")
        return g, p
    return torch.linspace(0.3, 1.0, L), torch.zeros(L, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--scale_ns", type=str, default="9,12,15")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--skip_scale", action="store_true")
    ap.add_argument("--arms", type=str, default="parity,stencil_x")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L, nf = args.L, args.nf
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    log("verify operator equivalence")
    verify = verify_gadget_equivalence()
    results = {
        "L": L,
        "nf": nf,
        "seed": args.seed,
        "steps": args.steps,
        "verify": verify,
        "armC_B1000_ref": ARM_C_B1000,
        "plan": str(_PKG / "docs" / "PLAN_gadget_Uc.md"),
    }

    W = build_W_BinH(nf=nf)
    g0, p0 = load_warm(L)

    for arm in arms:
        log(f"=== train arm {arm} ===")
        best = train_arm(
            arm, W, L, nf, device, steps=args.steps,
            max_train=args.max_train, max_val=args.max_val, seed=args.seed + hash(arm) % 100,
            gamma0=g0, phi0=p0, train_ns=(9, 12),
        )
        st = best["state"]
        torch.save(st, OUT / f"{arm}_params.pt")
        log(f"ladder {arm}")
        lad = ladder_eval(W, st["gamma"], st["phi"], L, nf, device, arm)
        block = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "train_val": best["val"],
            "ladder": lad,
            "b_test_12_15_18": fit_b_ns(lad.get("test", {})),
        }
        if not args.skip_scale:
            scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]
            log(f"scale1000 {arm} ns={scale_ns} B={args.scale_B}")
            block["scale1000"] = scale1000_eval(
                W, st["gamma"], st["phi"], L, nf, device, arm,
                ns=scale_ns, max_count=args.scale_B,
            )
            block["b_scale"] = fit_b_ns(block["scale1000"], ns=tuple(scale_ns))
        results[arm] = block
        # incremental dump
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_results_md(results, OUT / "RESULTS.md")

    # promote: parity scale/test Med within 1.15× of Arm C on n15 (soft)
    promote = {"ok": False, "reason": ""}
    try:
        src = results.get("parity", {})
        sc = src.get("scale1000") or {}
        lad_t = (src.get("ladder") or {}).get("test") or {}
        ratios = {}
        for n in (9, 12, 15, 18):
            m = (sc.get(str(n)) or sc.get(n) or lad_t.get(str(n)) or lad_t.get(n))
            if m:
                ratios[n] = ARM_C_B1000[n]["Med"] / max(m["median_inv_p"], 1e-12)
        promote["ratios_Med_ArmC_over_parity"] = ratios
        # competitive = all available ratios >= 1/1.15 ≈ 0.87 (not much worse)
        if ratios:
            promote["ok"] = all(r >= 0.87 for r in ratios.values()) and (
                ratios.get(15, 0) >= 0.87 or ratios.get(12, 0) >= 0.9
            )
            promote["reason"] = "parity Med within ~15% of Arm C on reported ns"
        else:
            promote["reason"] = "no parity metrics"
    except Exception as e:
        promote = {"ok": False, "error": str(e)}
    results["promote"] = promote
    log("promote", promote)

    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_results_md(results, OUT / "RESULTS.md")
    log("DONE", OUT)


if __name__ == "__main__":
    main()
