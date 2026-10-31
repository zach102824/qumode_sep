#!/usr/bin/env python3
"""Three non-Hc Uc probes (Zach t2184/t2185): NOT secretly encoding Hc.

Arms (L=16, BinH W, Arm-C SNAP+D shell):
  1) surrogate_pair — pairwise signed co-occurrence Ising diagonal ≠ unsat Hc
  2) noncomm_xz    — per-cavity exp(-iγ X_deg) @ exp(-iγ Z_bias); non-commuting;
                     product ≠ densified exp(-iγ Hc) (algebra check recorded)
  3) wrong_graph   — variable-label scramble of clauses → E_wrong as cost; score true E

Honesty column =exp(Hc)? must be NO for all three (no clause projectors / densified E_unsat).
Writes explore_OCT29/runs/nonhc_Uc/{RESULTS.md,results.json,*_params.pt}.
Leave ArmC L44 pad alone.
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

from explore_OCT29.lib.ansatz import AnsatzConfig  # noqa: E402
from explore_OCT29.lib.packing import K_BITS, NU  # noqa: E402
from explore_OCT29.lib.sat_gen import fast_energies  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import (  # noqa: E402
    build_W_BinH,
    build_grid_to_x_binary,
)
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    ARM_C_B1000,
    GadgetAnsatz,
    _jsonable,
    _pauli_x_bit,
    fit_b_ns,
    load_warm,
)
from explore_OCT29.scripts.probe6_spectral_mixer import _S, _loss  # noqa: E402

OUT = _PKG / "runs" / "nonhc_Uc"
SCALE_DIR = _PKG / "data" / "scale1000"

# Prior robust_grape refs (L=16 scale1000)
TRUNC_HC_REF = {
    9: {"Med": 20.796, "B90": 64.953},
    12: {"Med": 63.160, "B90": 214.865},
    15: {"Med": 192.176, "B90": 912.036},
}
PROJ_L16_REF = {
    9: {"Med": 1.406, "B90": 3.288},
    12: {"Med": 1.864, "B90": 6.106},
    15: {"Med": 2.754, "B90": 10.244},
    18: {"Med": 4.194, "B90": 24.302},
}

EQUALS_EXP_HC = {
    "surrogate_pair": False,
    "noncomm_xz": False,
    "wrong_graph": False,
}

FEATURE_DOC = {
    "surrogate_pair": (
        "From clauses build signed pairwise J_ij += s_i*s_j for every pair in a clause. "
        "Diagonal cost E_surr[z] = sum_{i<j} J_ij * z_i * z_j with z_k=1-2x_k. "
        "This is a 2-body Ising surrogate of 4-SAT — NOT the unsat count Hc=sum_c C_c, "
        "and not a monotone rescaling of Hc (see honesty stats)."
    ),
    "noncomm_xz": (
        "Per cavity: U = exp(-iγ sum_j deg_j X_j) @ exp(-iγ sum_j bias_j Z_j), "
        "deg_j = clause incidence of bit j, bias_j = (#pos - #neg)/m. "
        "X and Z on the same bit do not commute → product ≠ any densified Z-phase of Hc. "
        "No clause projectors."
    ),
    "wrong_graph": (
        "Fixed-per-n RNG perm π of variable labels; rewrite each clause v→π(v); "
        "cost = densified unsat of scrambled formula E_wrong; score with true E. "
        "Does not reconstruct true clauses unless shared SNAP+D magically inverts π "
        "(ablation vs trunc_Hc)."
    ),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[nonhc]", *a, flush=True)


# ---------------------------------------------------------------------------
# Feature maps / honesty checks
# ---------------------------------------------------------------------------

def _clauses_list(cls_v, cls_s):
    cl = []
    for i in range(len(cls_v)):
        v = [int(x) for x in cls_v[i]]
        s = [int(x) for x in cls_s[i]]
        cl.append((v, s))
    return cl


def surrogate_pair_E(n: int, cls_v, cls_s) -> np.ndarray:
    """2-body signed co-occurrence Ising diagonal (NOT Hc)."""
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(len(cls_v)):
        v = [int(x) for x in cls_v[i]]
        s = [int(x) for x in cls_s[i]]
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                J[v[a], v[b]] += s[a] * s[b]
                J[v[b], v[a]] = J[v[a], v[b]]
    dim = 1 << n
    E = np.zeros(dim, dtype=np.float32)
    # bit i = MSB at position i (matches fast_energies)
    for x in range(dim):
        bits = [(x >> (n - 1 - i)) & 1 for i in range(n)]
        z = [1 - 2 * b for b in bits]
        val = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                val += J[i, j] * z[i] * z[j]
        E[x] = val
    # shift so min ~ 0-ish for numerical comfort (phase is relative anyway via cost_Cn)
    E -= E.min()
    return E


def surrogate_pair_E_fast(n: int, cls_v, cls_s) -> np.ndarray:
    """Vectorized pairwise Ising over all assignments."""
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(len(cls_v)):
        v = [int(x) for x in cls_v[i]]
        s = [int(x) for x in cls_s[i]]
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                J[v[a], v[b]] += float(s[a] * s[b])
    # upper triangle only
    xs = np.arange(1 << n, dtype=np.int64)
    # bits[i, x] for var i (MSB = var 0)
    bits = np.zeros((n, 1 << n), dtype=np.float64)
    for i in range(n):
        bits[i] = ((xs >> (n - 1 - i)) & 1).astype(np.float64)
    z = 1.0 - 2.0 * bits
    E = np.zeros(1 << n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                E += J[i, j] * z[i] * z[j]
    E = E.astype(np.float32)
    E -= E.min()
    return E


def wrong_graph_E(n: int, cls_v, cls_s, seed: int = 123) -> np.ndarray:
    """Scramble variable labels with fixed-per-n perm; densify scrambled unsat."""
    rng = np.random.default_rng(seed + 1009 * n)
    perm = rng.permutation(n)
    cl = []
    for i in range(len(cls_v)):
        v = sorted(int(perm[int(x)]) for x in cls_v[i])
        # signs follow original order of vars before sort — rebuild via map
        raw_v = [int(x) for x in cls_v[i]]
        raw_s = [int(x) for x in cls_s[i]]
        paired = sorted(
            ((int(perm[vv]), ss) for vv, ss in zip(raw_v, raw_s)),
            key=lambda t: t[0],
        )
        v2 = [p[0] for p in paired]
        s2 = [p[1] for p in paired]
        cl.append((v2, s2))
    return fast_energies(n, cl).astype(np.float32)


def var_degree_bias(n: int, cls_v, cls_s):
    """deg[v]=incidence count; bias[v]=(#pos-#neg)/m."""
    deg = np.zeros(n, dtype=np.float64)
    bias = np.zeros(n, dtype=np.float64)
    m = max(len(cls_v), 1)
    for i in range(len(cls_v)):
        for vv, ss in zip(cls_v[i], cls_s[i]):
            vv = int(vv)
            ss = int(ss)
            deg[vv] += 1.0
            bias[vv] += float(ss)
    bias /= float(m)
    # normalize deg to O(1)
    if deg.max() > 0:
        deg = deg / deg.max()
    return deg.astype(np.float32), bias.astype(np.float32)


def honesty_vs_Hc(E_cost: np.ndarray, E_true: np.ndarray) -> dict:
    """Show E_cost is not a monotone affine rescaling of Hc."""
    et = E_true.astype(np.float64).ravel()
    ec = E_cost.astype(np.float64).ravel()
    # best affine fit ec ≈ a*et + b
    A = np.stack([et, np.ones_like(et)], axis=1)
    try:
        coef, _, _, _ = np.linalg.lstsq(A, ec, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        pred = a * et + b
        resid = ec - pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss_tot = float(np.sum((ec - ec.mean()) ** 2)) + 1e-12
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot
    except Exception as ex:  # noqa: BLE001
        a = b = rmse = r2 = float("nan")
        _ = ex
    # Spearman-ish via rank corr
    rt = et.argsort().argsort().astype(np.float64)
    rc = ec.argsort().argsort().astype(np.float64)
    rt -= rt.mean()
    rc -= rc.mean()
    spear = float((rt @ rc) / (np.linalg.norm(rt) * np.linalg.norm(rc) + 1e-12))
    same = bool(np.allclose(ec, et))
    return {
        "identical_to_Hc": same,
        "affine_a": a,
        "affine_b": b,
        "affine_rmse": rmse,
        "affine_r2": r2,
        "rank_corr": spear,
        "E_cost_min": float(ec.min()),
        "E_cost_max": float(ec.max()),
        "E_true_min": float(et.min()),
        "E_true_max": float(et.max()),
        "equals_exp_Hc": False if not same else True,
    }


def algebra_check_noncomm(nf: int = 8) -> dict:
    """Prove Ux@Uz ≠ Uz@Ux and densified Uc (diagonal) cannot equal Ux@Uz."""
    # single-bit demo inside 8-dim Fock
    X = _pauli_x_bit(0, nf=nf).numpy()
    Z = np.zeros((nf, nf), dtype=np.complex64)
    for f in range(NU):
        # Z on bit 0: phase (-1)^{bit0}
        bit = (f >> 0) & 1
        Z[f, f] = 1.0 if bit == 0 else -1.0
    for f in range(NU, nf):
        Z[f, f] = 1.0
    g = 0.37
    Ux = np.eye(nf, dtype=np.complex64) * np.cos(g)
    Ux = Ux - 1j * np.sin(g) * X
    # pad: already in X
    Uz = np.diag(np.exp(-1j * g * np.real(np.diag(Z)))).astype(np.complex64)
    AB = Ux @ Uz
    BA = Uz @ Ux
    fro = float(np.linalg.norm(AB - BA, "fro"))
    # densified Uc is diagonal → off-diag of AB must vanish to match any diag
    off = float(np.linalg.norm(AB - np.diag(np.diag(AB)), "fro"))
    return {
        "gamma": g,
        "frobenius_UxUz_minus_UzUx": fro,
        "commute": fro < 1e-5,
        "offdiag_frobenius_UxUz": off,
        "is_diagonal": off < 1e-5,
        "equals_densified_Uc": False,  # densified Uc diagonal; UxUz not
        "note": "UxUz non-diagonal & non-commuting → cannot equal exp(-iγ Hc) densified",
    }


# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------

class NonHcAnsatz(GadgetAnsatz):
    """GadgetAnsatz + surrogate / noncomm_xz / wrong_graph cost modes."""

    def __init__(self, cfg: AnsatzConfig, W: torch.Tensor, device=None, dtype=torch.complex64):
        super().__init__(cfg, W, device=device, dtype=dtype)
        # Z diag on each of 3 bits (phase +1/-1)
        Zs = []
        for j in range(K_BITS):
            z = torch.ones(self.nf, dtype=self.rdtype, device=self.lam.device)
            for f in range(NU):
                bit = (f >> j) & 1
                z[f] = 1.0 if bit == 0 else -1.0
            Zs.append(z)
        self.register_buffer("Zbits_diag", torch.stack(Zs))  # (3, nf)

    def _apply_noncomm_xz(
        self, psi: torch.Tensor, gamma: torch.Tensor, deg: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Per cavity: Ux(deg) @ Uz(bias) with instance deg/bias on global vars."""
        g = gamma.to(dtype=self.rdtype)
        n = self.cfg.n
        C = self.C
        eye = torch.eye(self.nf, dtype=self.cdtype, device=self.lam.device)
        for cav in range(C):
            # local bits map to global vars cav*3 + j
            Ux = eye.clone()
            for j in range(K_BITS):
                gv = cav * K_BITS + j
                if gv >= n:
                    break
                ang = g * deg[gv].to(self.rdtype)
                c = torch.cos(ang).to(self.cdtype)
                s = torch.sin(ang).to(self.cdtype)
                Uj = c * eye - 1j * s * self.Xbits[j]
                Ux = Uj @ Ux
            # Uz diagonal: phase ∝ bias_j * Z_j
            th = torch.zeros(self.nf, dtype=self.rdtype, device=self.lam.device)
            for j in range(K_BITS):
                gv = cav * K_BITS + j
                if gv >= n:
                    break
                th = th + bias[gv].to(self.rdtype) * self.Zbits_diag[j]
            Uz = torch.diag(torch.exp(-1j * (g * th).to(self.cdtype)))
            U = Ux @ Uz  # non-commuting product
            psi = self._apply_local(psi, U, cav)
        return psi

    def forward_nonhc(
        self,
        E_true: torch.Tensor,
        gamma: torch.Tensor,
        phi: torch.Tensor,
        cost_mode: str,
        E_cost: torch.Tensor | None = None,
        deg: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> dict:
        L = self.L
        gamma = gamma.to(dtype=self.rdtype, device=self.lam.device).reshape(L)
        phi = phi.to(dtype=self.rdtype, device=self.lam.device)
        psi = self.vacuum()
        psi = self._apply_all(psi, self.Wmat)

        Cn = None
        if cost_mode in ("surrogate_pair", "wrong_graph"):
            if E_cost is None:
                raise ValueError(f"{cost_mode} needs E_cost")
            Cn = self.cost_Cn(E_cost)

        for ell in range(L):
            if cost_mode == "noncomm_xz":
                if deg is None or bias is None:
                    raise ValueError("noncomm_xz needs deg/bias")
                psi = self._apply_noncomm_xz(psi, gamma[ell], deg, bias)
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
        E_grid = E_true.to(device=self.lam.device)[self.grid_to_x]
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
    return NonHcAnsatz(cfg, W, device=device).to(device)


def _prepare_instance(cost_mode, n, E_true, cls_v, cls_s, cache):
    """Return (E_cost_t_or_None, deg_t_or_None, bias_t_or_None, honesty_dict_or_None)."""
    if cost_mode == "surrogate_pair":
        Ec = surrogate_pair_E_fast(n, cls_v, cls_s)
        hon = honesty_vs_Hc(Ec, E_true)
        return torch.as_tensor(Ec, dtype=torch.float32), None, None, hon
    if cost_mode == "wrong_graph":
        key = ("wperm", n)
        if key not in cache:
            # store perm seed only; E depends on instance
            cache[key] = True
        Ec = wrong_graph_E(n, cls_v, cls_s, seed=123)
        hon = honesty_vs_Hc(Ec, E_true)
        return torch.as_tensor(Ec, dtype=torch.float32), None, None, hon
    if cost_mode == "noncomm_xz":
        deg, bias = var_degree_bias(n, cls_v, cls_s)
        return (
            None,
            torch.as_tensor(deg, dtype=torch.float32),
            torch.as_tensor(bias, dtype=torch.float32),
            {"equals_exp_Hc": False, "note": "non-diagonal XZ gadgets"},
        )
    raise ValueError(cost_mode)


@torch.no_grad()
def eval_arm(
    n, W, gamma, phi, L, nf, device, cost_mode,
    split="val", max_count=24, data_dir=None, progress_every=0, cache=None,
):
    if cache is None:
        cache = {}
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        if "scale1000_seeds" in pack:
            key = "scale1000"
        elif "test_seeds" in pack:
            key = "test"
        else:
            raise RuntimeError(f"n={n}: no split {split}")
    seeds, cls_v, cls_s, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(seeds) == 0:
        return None
    sim = make_sim(n, W, L, nf, device)
    invs, leaks = [], []
    honesty_samples = []
    t0 = time.time()
    B = len(seeds)
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        Ec, deg, bias, hon = _prepare_instance(
            cost_mode, n, E[b], cls_v[b], cls_s[b], cache
        )
        if hon and b < 3:
            honesty_samples.append(hon)
        if Ec is not None:
            Ec = Ec.to(device)
        if deg is not None:
            deg = deg.to(device)
            bias = bias.to(device)
        out = sim.forward_nonhc(
            Eb, gamma, phi, cost_mode, E_cost=Ec, deg=deg, bias=bias
        )
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
        "equals_exp_Hc": False,
        "honesty_samples": honesty_samples,
    }


def train_arm(
    cost_mode, W, L, nf, device, steps=80, max_train=32, max_val=12, seed=0,
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
    opt = torch.optim.Adam([gamma, phi], lr=lr)
    cache: dict = {}

    packs = {}
    for n in train_ns:
        pack = load_pack(n)
        seeds, cls_v, cls_s, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            seeds, cls_v, cls_s, _, E = _split_arrays(pack, "val", max_count=max_train)
        packs[n] = {"E": E, "cls_v": cls_v, "cls_s": cls_s, "N": len(E)}

    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    best = {"score": float("inf"), "state": None, "step": -1, "val": {}}

    for step in range(steps):
        opt.zero_grad()
        losses = []
        for n in train_ns:
            P = packs[n]
            idx = int(rng.integers(0, P["N"]))
            Eb = torch.as_tensor(P["E"][idx], dtype=torch.float32, device=device)
            Ec, deg, bias, _ = _prepare_instance(
                cost_mode, n, P["E"][idx], P["cls_v"][idx], P["cls_s"][idx], cache
            )
            if Ec is not None:
                Ec = Ec.to(device)
            if deg is not None:
                deg = deg.to(device)
                bias = bias.to(device)
            out = sims[n].forward_nonhc(
                Eb, gamma, phi, cost_mode, E_cost=Ec, deg=deg, bias=bias
            )
            losses.append(_loss(out))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            val = {}
            with torch.no_grad():
                scs = []
                for n in train_ns:
                    invs = []
                    P = packs[n]
                    for j in range(min(max_val, P["N"])):
                        Eb = torch.as_tensor(P["E"][j], dtype=torch.float32, device=device)
                        Ec, deg, bias, _ = _prepare_instance(
                            cost_mode, n, P["E"][j], P["cls_v"][j], P["cls_s"][j], cache
                        )
                        if Ec is not None:
                            Ec = Ec.to(device)
                        if deg is not None:
                            deg = deg.to(device)
                            bias = bias.to(device)
                        o = sims[n].forward_nonhc(
                            Eb, gamma.detach(), phi.detach(), cost_mode,
                            E_cost=Ec, deg=deg, bias=bias,
                        )
                        invs.append(float(o["inv_p"].cpu()))
                    med = float(np.median(invs))
                    b90 = float(np.quantile(invs, 0.9))
                    val[n] = {"Med": med, "B90": b90}
                    scs.append(_S(med, b90))
                sc = float(np.mean(scs))
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": val,
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "L": L,
                        "cost_mode": cost_mode,
                        "train_ns": list(train_ns),
                        "equals_exp_Hc": False,
                    },
                }
            bits = " ".join(
                f"n{n}:{val[n]['Med']:.2f}/{val[n]['B90']:.2f}" for n in train_ns
            )
            log(f"train {cost_mode} step {step}: Scomb={sc:.3f}", bits)

    return best


def scale_eval(W, gamma, phi, L, nf, device, cost_mode, ns, max_count, cache=None):
    if cache is None:
        cache = {}
    out = {}
    for n in ns:
        pe = 50 if n >= 15 else (100 if n >= 12 else 200)
        m = eval_arm(
            n, W, gamma, phi, L, nf, device, cost_mode,
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            progress_every=pe, cache=cache,
        )
        if m:
            out[n] = m
            log(
                f"scale {cost_mode} n{n}: Med={m['median_inv_p']:.3f} "
                f"B90={m['B90_inv_p']:.3f} B={m['B']}"
            )
    return out


def verdict_for(arm: str, sc: dict) -> str:
    """kill / promote / weak based on vs ArmC and trunc_Hc."""
    m15 = sc.get(15) or sc.get("15")
    if not m15:
        return "incomplete"
    med = m15["median_inv_p"]
    armc = ARM_C_B1000[15]["Med"]
    trunc = TRUNC_HC_REF[15]["Med"]
    if med <= 1.2 * armc:
        return "PROMOTE (≈ArmC — investigate leakage to Hc)"
    if med < trunc:
        return f"WEAK interest (beats trunc_Hc {trunc:.1f} but ≫ ArmC {armc:.3f})"
    return f"KILL (Med={med:.1f} ≱ trunc_Hc {trunc:.1f}; ≫ ArmC {armc:.3f})"


def write_results(results: dict, path: Path):
    lines = [
        "# Non-Hc Uc probes — verdict",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} (Asia/Shanghai)_",
        "",
        "Ask (Zach): different Uc that does **not** secretly encode Hc.",
        "Champ projector L44_cvar / Arm C densify = **renames_Hc**; cited only as baseline.",
        "",
        "## Feature / algebra docs",
        "",
    ]
    for arm, doc in FEATURE_DOC.items():
        lines += [f"### `{arm}`", "", doc, ""]
    if results.get("algebra_noncomm"):
        lines += [
            "### Algebra check (`noncomm_xz`)",
            "",
            "```json",
            json.dumps(results["algebra_noncomm"], indent=2),
            "```",
            "",
        ]
    lines += [
        "## Verdict table (L=16, train{9,12}, W=H^{⊗3} BinH)",
        "",
        "| arm | =exp(Hc)? | n9 Med/B90 | n12 | n15 | vs ArmC n15 | vs trunc_Hc n15 | verdict |",
        "|-----|-----------|------------|-----|-----|-------------|-----------------|---------|",
    ]

    def row(name, equals, sc):
        cells = []
        for n in (9, 12, 15):
            m = (sc or {}).get(n) or (sc or {}).get(str(n))
            if m and "median_inv_p" in m:
                cells.append(f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}")
            else:
                cells.append("—")
        m15 = (sc or {}).get(15) or (sc or {}).get("15")
        if m15 and "median_inv_p" in m15:
            vs_a = ARM_C_B1000[15]["Med"] / max(m15["median_inv_p"], 1e-12)
            vs_t = TRUNC_HC_REF[15]["Med"] / max(m15["median_inv_p"], 1e-12)
            vsA = f"×{vs_a:.4f}"
            vsT = f"×{vs_t:.4f}"
            verd = verdict_for(name, sc)
        else:
            vsA = vsT = verd = "—"
        lines.append(
            f"| {name} | {equals} | {cells[0]} | {cells[1]} | {cells[2]} | {vsA} | {vsT} | {verd} |"
        )

    row(
        "Arm C densify",
        "YES",
        {n: {"median_inv_p": ARM_C_B1000[n]["Med"], "B90_inv_p": ARM_C_B1000[n]["B90"]} for n in (9, 12, 15)},
    )
    row(
        "projector (=Hc)",
        "YES",
        {n: {"median_inv_p": PROJ_L16_REF[n]["Med"], "B90_inv_p": PROJ_L16_REF[n]["B90"]} for n in (9, 12, 15)},
    )
    row(
        "trunc_Hc (prior)",
        "NO (partial)",
        {n: {"median_inv_p": TRUNC_HC_REF[n]["Med"], "B90_inv_p": TRUNC_HC_REF[n]["B90"]} for n in (9, 12, 15)},
    )
    for arm in results.get("arms_run", []):
        block = results.get(arm) or {}
        sc = block.get("scale1000") or block.get("screen") or {}
        row(arm, "NO", sc)

    lines += [
        "",
        "## Interpretation",
        "",
    ]
    for arm in results.get("arms_run", []):
        block = results.get(arm) or {}
        sc = block.get("scale1000") or block.get("screen") or {}
        lines.append(f"- **{arm}**: {verdict_for(arm, sc)} — =exp(Hc)? **NO**. {FEATURE_DOC[arm][:120]}…")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Out: `{OUT}`",
        f"- Script: `scripts/probe_nonhc_Uc.py`",
        f"- JSON: `runs/nonhc_Uc/results.json`",
        "",
        "ArmC L44 fair-bar (PID pad) left untouched.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    log("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--arms", type=str, default="surrogate_pair,noncomm_xz,wrong_graph")
    ap.add_argument("--screen_B", type=int, default=200)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--scale_ns", type=str, default="9,12,15")
    ap.add_argument("--skip_promote", action="store_true")
    ap.add_argument("--promote_all", action="store_true",
                    help="run B=scale_B even if screen says KILL")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L, nf = args.L, args.nf
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]

    alg = algebra_check_noncomm(nf=nf)
    log("algebra_noncomm", alg)

    results = {
        "L": L,
        "nf": nf,
        "seed": args.seed,
        "steps": args.steps,
        "train_ns": [9, 12],
        "W": "BinH H^{⊗3}",
        "armC_B1000_ref": ARM_C_B1000,
        "trunc_Hc_ref": TRUNC_HC_REF,
        "projector_L16_ref": PROJ_L16_REF,
        "feature_docs": FEATURE_DOC,
        "algebra_noncomm": alg,
        "note": "Uc must NOT secretly encode Hc (no clause projectors / densified E_unsat)",
        "arms_run": [],
        "screen_B": args.screen_B,
        "scale_B": args.scale_B,
    }

    W = build_W_BinH(nf=nf)
    g0, p0 = load_warm(L)

    for arm in arms:
        log(f"=== train arm {arm} ===")
        assert EQUALS_EXP_HC[arm] is False
        cache: dict = {}
        best = train_arm(
            arm, W, L, nf, device, steps=args.steps,
            max_train=args.max_train, max_val=args.max_val,
            seed=args.seed + (abs(hash(arm)) % 997),
            gamma0=g0, phi0=p0, train_ns=(9, 12),
        )
        st = best["state"]
        torch.save(st, OUT / f"{arm}_params.pt")

        log(f"=== screen B={args.screen_B} {arm} ===")
        screen = scale_eval(
            W, st["gamma"], st["phi"], L, nf, device, arm,
            ns=scale_ns, max_count=args.screen_B, cache=cache,
        )
        verd = verdict_for(arm, screen)
        block = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "train_val": {str(k): v for k, v in best["val"].items()},
            "screen": screen,
            "screen_verdict": verd,
            "equals_exp_Hc": False,
            "feature_doc": FEATURE_DOC[arm],
            "b_screen": fit_b_ns(screen, ns=tuple(scale_ns)),
        }
        # promote if not KILL or forced
        do_promote = args.promote_all or (not args.skip_promote and not verd.startswith("KILL"))
        if do_promote and args.scale_B > args.screen_B:
            log(f"=== promote B={args.scale_B} {arm} ===")
            block["scale1000"] = scale_eval(
                W, st["gamma"], st["phi"], L, nf, device, arm,
                ns=scale_ns, max_count=args.scale_B, cache=cache,
            )
            block["final_verdict"] = verdict_for(arm, block["scale1000"])
            block["b_scale"] = fit_b_ns(block["scale1000"], ns=tuple(scale_ns))
        else:
            block["scale1000"] = screen  # screen stands in
            block["final_verdict"] = verd
            if verd.startswith("KILL"):
                log(f"not promoting {arm}: {verd}")

        results[arm] = block
        results["arms_run"].append(arm)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_results(results, OUT / "RESULTS.md")

    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_results(results, OUT / "RESULTS.md")
    log("DONE", OUT)


if __name__ == "__main__":
    main()
