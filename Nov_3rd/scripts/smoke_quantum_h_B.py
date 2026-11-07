#!/usr/bin/env python3
"""Smoke B — sparse quantum p=3 TFIM on the locked Nov_3rd gate sequence.

Hamiltonian (SHORTLIST family B, stoquastic E1 sign):
    H = H_c - λ Σ_i X_i,   λ > 0
    H_c = -s^{-1} Σ_{(i,j,k)∈T} J_ijk Z_i Z_j Z_k
    s = sqrt(|T| / n),   J_ijk ∈ {±1}

T is a cross-cavity 3-uniform hypergraph.
  reuse  — one cross-cavity triple per Max-E 4-SAT clause (scale1000 generator,
           base 54e9); first-wins unique; J = product of the three literal signs.
  regen  — same |T|, independent uniform cross-cavity triples and independent signs.
  sparse — cap |T| = 2n so supports cannot saturate C(n,3) at n∈{6,9}.

Gates (not modified): U_c^X = W exp(-i γ H_c) W, W = H^{⊗3}, then bare
Hamming-shell SNAP. power_p: φ_k = c1 k + a_n c2 k^2 + a_n c3 k^3,
a_n = (8/(n-1))^p. Fourier QAOA: ideal-qubit per-layer (β, γ), paper init
(β, γ) = (0.01, −0.01), cost exp(-i γ Cn) with the same Cn as the SNAP pulse,
mixer exp(-i β X/2). Objective is the physical ⟨H⟩, read out from the Z
histogram of H_c plus single-qubit X marginals. No YY/XX correlators.

Metric: ε = (⟨H⟩ − E_0) / n. BEAT iff Med(ε_SNAP) ≤ 0.98 × Med(ε_QAOA) on both
n=6 and n=9. Does not write qaoa_angles/, settings.py, or train69 ladders.

The opus note opus_reply_20261104_2150_hamiltonian_families.md was not in the
tree (origin/main at the explore-doc commit). Rank / p-spin locks are taken
from Nov_3rd/runs/quantum_h_explore/{PLAN,SHORTLIST,SMOKE}.md. Rank2
intra-cavity XX/XXX is not the main path (it needs XX correlators; fails
readout lock R).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.qaoa_init import QAOA_INIT_BETA, QAOA_INIT_GAMMA
from beat_params import phi_from_c
from explore_OCT29.lib.packing import NU
from explore_OCT29.lib.sat_gen import instances as sat_instances

OUT = S.OUT / "quantum_h_explore" / "smoke_B"
SCALE_BASE = 54_000_000_000  # same base as scale1000 Max-E packs
LAMBDA_GRID = (0.0, 0.15, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0)
PEAK_QUANTUM = 0.5
GAP_MIN = 0.02  # per-qubit energy above E0 that a trivial reference must leave


def log(*a):
    print(time.strftime("%H:%M:%S"), "[smokeB]", *a, flush=True)


# ---------------------------------------------------------------------------
# Hypergraph
# ---------------------------------------------------------------------------

def _span(triple) -> int:
    return len({int(v) // 3 for v in triple})


def cross_triples(n: int) -> list[tuple[int, int, int]]:
    return [t for t in combinations(range(n), 3) if _span(t) >= 2]


def _best_clause_triple(vs, ss):
    """Max cavity-span 3-subset. Tie → lexicographic triple. J = Π signs."""
    best = None
    for drop in range(4):
        raw = [(int(vs[i]), int(ss[i])) for i in range(4) if i != drop]
        raw.sort(key=lambda t: t[0])
        trip = (raw[0][0], raw[1][0], raw[2][0])
        J = int(raw[0][1] * raw[1][1] * raw[2][1])
        sp = _span(trip)
        cand = (sp, trip, J)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
            best = cand
    return best


def _clause_candidates(vs, ss):
    cands = []
    for drop in range(4):
        raw = [(int(vs[i]), int(ss[i])) for i in range(4) if i != drop]
        raw.sort()
        trip = (raw[0][0], raw[1][0], raw[2][0])
        J = int(raw[0][1] * raw[1][1] * raw[2][1]) or 1
        cands.append((_span(trip), trip, J))
    return cands


def reuse_edges(cl, cap: int | None) -> list[tuple[tuple[int, int, int], int]]:
    """One cross-cavity triple per clause, load-balanced across variables.

    Lexicographic tie-break alone piles edges onto low-index qubits (n=6
    collapsed to 10 triples). Among maximum-span triples still unused, pick
    the one with the smallest current degree sum. J is the product of the
    three literal signs from the clause that introduces the triple.
    """
    deg: dict[int, int] = {}
    seen: dict[tuple[int, int, int], int] = {}
    order = []
    for vs, ss in cl:
        cands = [c for c in _clause_candidates(vs, ss) if c[0] >= 2]
        if not cands:
            continue
        mx = max(c[0] for c in cands)
        cands = [c for c in cands if c[0] == mx]
        fresh = [c for c in cands if c[1] not in seen]
        pool = fresh or cands
        _sp, trip, J = min(pool, key=lambda c: (sum(deg.get(v, 0) for v in c[1]), c[1]))
        if trip in seen:
            continue
        seen[trip] = 1 if J >= 0 else -1
        order.append(trip)
        for v in trip:
            deg[v] = deg.get(v, 0) + 1
        if cap is not None and len(order) >= int(cap):
            break
    return [(t, seen[t]) for t in order]


def regen_edges(n: int, n_edges: int, seed: int) -> list[tuple[tuple[int, int, int], int]]:
    pool = cross_triples(n)
    rng = random.Random(int(seed) + 17_000_000_003)
    if n_edges > len(pool):
        n_edges = len(pool)
    picks = rng.sample(pool, n_edges)
    return [(t, rng.choice((-1, 1))) for t in picks]


def jaccard(a_edges, b_edges) -> float:
    A = {e[0] for e in a_edges}
    B = {e[0] for e in b_edges}
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def sign_agree(a_edges, b_edges) -> float:
    A = dict(a_edges)
    B = dict(b_edges)
    common = set(A) & set(B)
    if not common:
        return float("nan")
    return sum(1 for t in common if A[t] == B[t]) / len(common)


def build_instances(n: int, count: int, cap: int | None) -> list[dict]:
    raw = sat_instances(n, count, SCALE_BASE, compute_E=False)
    out = []
    for ins in raw:
        reuse = reuse_edges(ins["cl"], cap)
        regen = regen_edges(n, len(reuse), ins["seed"])
        out.append({
            "seed": int(ins["seed"]),
            "m": int(ins["m"]),
            "reuse": reuse,
            "regen": regen,
            "jaccard": jaccard(reuse, regen),
            "sign_agree": sign_agree(reuse, regen),
            "n_edges": len(reuse),
        })
    return out


# ---------------------------------------------------------------------------
# Diagonal H_c and exact diagonalization
# ---------------------------------------------------------------------------

def z_table(n: int) -> np.ndarray:
    """z[v, x] = ±1, var 0 is MSB."""
    dim = 1 << n
    xs = np.arange(dim, dtype=np.int64)
    bits = [((xs >> (n - 1 - v)) & 1) for v in range(n)]
    return (1 - 2 * np.stack(bits, axis=0)).astype(np.float64)


def hc_diag(n: int, edges, z=None) -> np.ndarray:
    if z is None:
        z = z_table(n)
    acc = np.zeros(z.shape[1], dtype=np.float64)
    for (i, j, k), J in edges:
        acc += -float(J) * z[i] * z[j] * z[k]
    n_e = max(len(edges), 1)
    scale = math.sqrt(n_e / float(n))
    return acc / scale


def build_dense(Hc: np.ndarray, lam: float, n: int) -> np.ndarray:
    dim = Hc.shape[0]
    H = np.zeros((dim, dim), dtype=np.float64)
    idx = np.arange(dim)
    H[idx, idx] = Hc
    for v in range(n):
        bit = n - 1 - v
        H[idx, idx ^ (1 << bit)] -= float(lam)
    return H


def exact_ground(Hc: np.ndarray, lam: float, n: int):
    evals, evecs = np.linalg.eigh(build_dense(Hc, lam, n))
    psi = evecs[:, 0]
    peak = float(np.max(np.abs(psi) ** 2))
    e0 = float(evals[0])
    gap = float(evals[1] - evals[0])
    e_class = float(np.min(Hc))  # bitstrings have ⟨X⟩=0
    e_plus = -float(lam) * n     # ⟨H_c⟩_{|+} = 0 for distinct-index ZZZ
    return {
        "E0": e0,
        "gap": gap,
        "peak": peak,
        "E_class": e_class,
        "E_plus": e_plus,
        "eps_class": (e_class - e0) / n,
        "eps_plus": (e_plus - e0) / n,
    }


def ed_one(n: int, edges, lams, z=None) -> dict:
    Hc = hc_diag(n, edges, z=z)
    rows = {}
    for lam in lams:
        rows[float(lam)] = exact_ground(Hc, float(lam), n)
    return {"Hc": Hc, "by_lam": rows}


def _med(xs) -> float:
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.median(a))


def summarize_lam(per_inst: list[dict], lam: float) -> dict:
    rows = [r["by_lam"][float(lam)] for r in per_inst]
    return {
        "lambda": float(lam),
        "B": len(rows),
        "med_peak": _med(r["peak"] for r in rows),
        "frac_peak_lt_0.5": float(np.mean([r["peak"] < PEAK_QUANTUM for r in rows])),
        "med_gap": _med(r["gap"] for r in rows),
        "med_eps_class": _med(r["eps_class"] for r in rows),
        "med_eps_plus": _med(r["eps_plus"] for r in rows),
        "med_E0": _med(r["E0"] for r in rows),
        "quantum": False,  # filled by caller once both n known
    }


def quantum_ok(summary: dict) -> bool:
    return (
        summary["med_peak"] < PEAK_QUANTUM
        and summary["med_eps_class"] > GAP_MIN
        and summary["med_eps_plus"] > GAP_MIN
        and summary["med_gap"] > 1e-6
    )


def choose_lambda(by_n_graph: dict) -> dict:
    """Pick λ on the reuse/full train split: quantum GS, both sectors open.

    Score = min(ε_class, ε_plus) averaged over n=6 and n=9 (competition).
    """
    # by_n_graph[(n, graph)] = list of per-λ summaries over the TRAIN split
    lams = list(LAMBDA_GRID)
    ranked = []
    for lam in lams:
        if lam <= 0:
            continue
        ok = True
        score_parts = []
        detail = {}
        for n in (6, 9):
            sm = by_n_graph[(n, "reuse")][float(lam)]
            detail[n] = sm
            if not quantum_ok(sm):
                ok = False
            score_parts.append(min(sm["med_eps_class"], sm["med_eps_plus"]))
        ranked.append({
            "lambda": float(lam),
            "pass": ok,
            "score": float(np.mean(score_parts)) if score_parts else -1.0,
            "detail": {str(k): v for k, v in detail.items()},
        })
    passing = [r for r in ranked if r["pass"]]
    if passing:
        best = max(passing, key=lambda r: r["score"])
        reason = "max min(ε_class, ε_plus) inside the quantum window"
    else:
        # fallback: smallest peak among λ with both reference gaps still open
        open_gap = [r for r in ranked if all(
            r["detail"][str(n)]["med_eps_class"] > GAP_MIN and r["detail"][str(n)]["med_eps_plus"] > GAP_MIN
            for n in (6, 9)
        )]
        pool = open_gap or ranked
        best = min(pool, key=lambda r: np.mean([r["detail"][str(n)]["med_peak"] for n in (6, 9)]))
        reason = "no λ passed peak<0.5 and both gaps; fallback to lowest mean peak with open gaps"
    return {"chosen": best, "reason": reason, "grid": ranked}


# ---------------------------------------------------------------------------
# Circuit energies (Z hist + single-qubit X)
# ---------------------------------------------------------------------------

def fock_to_bit(sim, tensor: torch.Tensor) -> torch.Tensor:
    """Map a Fock-grid tensor (real or complex) onto flat bitstring order."""
    n = int(sim.cfg.n)
    dim = 1 << n
    flat = tensor.reshape(-1)
    idx = sim.grid_to_x.reshape(-1).long()
    out = torch.zeros(dim, dtype=tensor.dtype, device=tensor.device)
    out.index_copy_(0, idx, flat)
    return out


def snap_psi(sim, E: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Locked Nov_3rd forward, returning the final Fock state (after last W)."""
    L = sim.L
    gamma = gamma.to(dtype=sim.rdtype, device=sim.lam.device)
    graded = gamma.ndim == 2
    phi = phi.to(dtype=sim.rdtype, device=sim.lam.device)
    psi = sim.vacuum()
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    Cn = None if graded else sim.cost_Cn(E)
    for ell in range(L):
        psi = sim._apply_all(psi, sim.Wmat)
        diag = sim.graded_cost_diag(E, gamma[ell]) if graded else Cn * gamma[ell]
        phase = torch.exp(-1j * diag.to(sim.cdtype))
        psi_new = psi.clone()
        psi_new[sl] = psi[sl] * phase
        psi = psi_new
        psi = sim._apply_all(psi, sim.Wmat)
        th = phi[ell][sim.popcount]
        psi = sim._apply_all(psi, sim._snap_theta(th))
    psi = sim._apply_all(psi, sim.Wmat)
    return psi


def x_sum(psi_x: torch.Tensor, n: int) -> torch.Tensor:
    """Σ_i ⟨X_i⟩ from a bitstring-ordered state. Single-qubit marginals only."""
    dim = psi_x.shape[-1]
    acc = torch.zeros(psi_x.shape[:-1], dtype=psi_x.real.dtype, device=psi_x.device)
    for v in range(n):
        bit = n - 1 - v
        perm = torch.arange(dim, device=psi_x.device) ^ (1 << bit)
        flipped = psi_x.index_select(-1, perm)
        acc = acc + (psi_x.conj() * flipped).sum(dim=-1).real
    return acc


def energy_bits(psi_x: torch.Tensor, Hc: torch.Tensor, lam: float, n: int):
    p = psi_x.real.square() + psi_x.imag.square()
    e_c = (p * Hc).sum(dim=-1)
    e_x = x_sum(psi_x, n)
    e = e_c - float(lam) * e_x
    return e, e_c, e_x


def snap_energy(sim, E, Hc, gamma, phi, lam: float):
    psi = snap_psi(sim, E, gamma, phi)
    amps = fock_to_bit(sim, psi)
    n = int(sim.cfg.n)
    return energy_bits(amps, Hc, lam, n)


def apply_x_mixer(psi: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    c = torch.cos(beta / 2).to(psi.dtype)
    s = (-1j * torch.sin(beta / 2)).to(psi.dtype)
    B = psi.shape[0]
    for q in range(n):
        v = psi.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        psi = torch.stack([c * a0 + s * a1, s * a0 + c * a1], dim=2).reshape(B, -1)
    return psi


def qaoa_energy_batch(betas, gammas, Cn, Hc, lam: float, n: int):
    """Batched Fourier/per-layer QAOA. Cn, Hc: (B, 2^n). Returns ε-ready energies (B,)."""
    B, dim = Cn.shape
    device = Cn.device
    psi = torch.full((B, dim), 1.0 / math.sqrt(dim), dtype=torch.complex64, device=device)
    betas = betas.to(device=device, dtype=torch.float32).reshape(-1)
    gammas = gammas.to(device=device, dtype=torch.float32).reshape(-1)
    for ell in range(int(gammas.numel())):
        psi = psi * torch.exp(-1j * gammas[ell].to(torch.complex64) * Cn.to(torch.complex64))
        psi = apply_x_mixer(psi, betas[ell], n)
    return energy_bits(psi, Hc, lam, n)


def cn_bitstring(sim, E: torch.Tensor) -> torch.Tensor:
    return fock_to_bit(sim, sim.cost_Cn(E))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def paper_angles(L: int, device):
    betas = torch.full((L,), QAOA_INIT_BETA, dtype=torch.float32, device=device)
    gammas = torch.full((L,), QAOA_INIT_GAMMA, dtype=torch.float32, device=device)
    return betas, gammas


def shell_from_beta_gamma(betas, gammas, n: int, p: torch.Tensor | float):
    """power_p angles at size n. p scales c2, c3 only."""
    if not torch.is_tensor(p):
        p = torch.tensor(float(p), dtype=betas.dtype, device=betas.device)
    a = torch.pow(torch.tensor(8.0 / float(n - 1), dtype=p.dtype, device=p.device), p)
    # caller passes c1,c2,c3 separately — this helper is the warm map only
    del a
    gamma = gammas.reshape(-1)
    c1 = -betas.reshape(-1)
    return gamma, c1


def a_of_n(n: int, p: torch.Tensor) -> torch.Tensor:
    # exp/log keeps the graph on p; torch.pow on a 0-dim tensor warns and can drop grad.
    base = torch.as_tensor(8.0 / float(n - 1), dtype=p.dtype, device=p.device)
    return torch.exp(p * torch.log(base))


def phi_power(c1, c2, c3, p, n: int) -> torch.Tensor:
    a = a_of_n(n, p)
    return phi_from_c(c1, c2 * a, c3 * a)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _lbfgs(params, closure_fn, rounds: int, max_iter: int, lr: float, wall_s: float, tag: str):
    opt = torch.optim.LBFGS(
        params, lr=lr, max_iter=max_iter, history_size=10,
        line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9,
    )
    hist = []
    t0 = time.time()
    best = None
    best_vec = None
    stall = 0
    for r in range(rounds):
        if time.time() - t0 > wall_s:
            log(f"{tag} wall cap {wall_s:.0f}s at round {r}")
            break
        try:
            loss = opt.step(closure_fn)
        except Exception as exc:  # line search can fail on a flat patch
            log(f"{tag} LBFGS round {r} failed: {type(exc).__name__}: {exc}")
            break
        val = float(loss.detach())
        hist.append(val)
        if best is None or val < best - 1e-5:
            best = val
            best_vec = [p.detach().clone() for p in params]
            stall = 0
        else:
            stall += 1
        if r == 0 or r + 1 == rounds or (r + 1) % 2 == 0:
            log(f"{tag} round {r+1}/{rounds} loss={val:.6f} best={best:.6f} t={time.time()-t0:.1f}s")
        if stall >= 3 and r >= 3:
            log(f"{tag} early stop at round {r+1}")
            break
    if best_vec is not None:
        with torch.no_grad():
            for p, b in zip(params, best_vec):
                p.copy_(b)
    return {"hist": hist, "best": best, "wall_s": time.time() - t0, "rounds": len(hist)}


def fit_qaoa(bundle, lam: float, rounds: int, max_iter: int, lr: float, wall_s: float, tag: str):
    """Per-size Fourier QAOA. bundle: Cn (B,dim), Hc (B,dim), E0 (B,), n."""
    n = bundle["n"]
    L = bundle["L"]
    device = bundle["Cn"].device
    betas0, gammas0 = paper_angles(L, device)
    betas = torch.nn.Parameter(betas0.clone())
    gammas = torch.nn.Parameter(gammas0.clone())

    def closure():
        opt.zero_grad()
        e, _, _ = qaoa_energy_batch(betas, gammas, bundle["Cn"], bundle["Hc"], lam, n)
        eps = (e - bundle["E0"]) / float(n)
        loss = eps.mean()
        loss.backward()
        return loss

    opt = None  # set inside _lbfgs; closure needs the opt. Rebuild closure each step via nonlocal.
    # _lbfgs creates the optimizer, so closure must close over it. Wire it explicitly.
    holder = {}

    def closure2():
        holder["opt"].zero_grad()
        e, _, _ = qaoa_energy_batch(betas, gammas, bundle["Cn"], bundle["Hc"], lam, n)
        eps = (e - bundle["E0"]) / float(n)
        loss = eps.mean()
        loss.backward()
        return loss

    def run():
        opt_local = torch.optim.LBFGS(
            [betas, gammas], lr=lr, max_iter=max_iter, history_size=10,
            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9,
        )
        holder["opt"] = opt_local
        hist = []
        t0 = time.time()
        best = None
        best_b = best_g = None
        stall = 0
        for r in range(rounds):
            if time.time() - t0 > wall_s:
                log(f"{tag} wall cap at round {r}")
                break
            try:
                loss = opt_local.step(closure2)
            except Exception as exc:
                log(f"{tag} LBFGS fail r{r}: {type(exc).__name__}: {exc}")
                break
            val = float(loss.detach())
            hist.append(val)
            if best is None or val < best - 1e-5:
                best = val
                best_b = betas.detach().clone()
                best_g = gammas.detach().clone()
                stall = 0
            else:
                stall += 1
            if r == 0 or (r + 1) % 2 == 0 or r + 1 == rounds:
                log(f"{tag} round {r+1}/{rounds} train_ε={val:.5f} best={best:.5f} t={time.time()-t0:.1f}s")
            if stall >= 4 and r >= 4:
                log(f"{tag} early stop at round {r+1}")
                break
        if best_b is not None:
            with torch.no_grad():
                betas.copy_(best_b)
                gammas.copy_(best_g)
        return {"hist": hist, "best": best, "wall_s": time.time() - t0}

    info = run()
    return betas.detach(), gammas.detach(), info


def eval_qaoa(betas, gammas, bundle, lam: float) -> np.ndarray:
    n = bundle["n"]
    with torch.no_grad():
        e, _, _ = qaoa_energy_batch(betas, gammas, bundle["Cn"], bundle["Hc"], lam, n)
        eps = (e - bundle["E0"]) / float(n)
    return eps.detach().cpu().numpy().astype(np.float64)


def pack_snap_bundle(sim, items, lam: float, device):
    """items: list of {E tensor, Hc tensor, E0 float}."""
    return {
        "sim": sim,
        "items": items,
        "lam": float(lam),
        "n": int(sim.cfg.n),
        "device": device,
    }


def snap_eps_list(sim, items, gamma, phi, lam: float, n: int):
    eps = []
    for it in items:
        e, _, _ = snap_energy(sim, it["E"], it["Hc"], gamma, phi, lam)
        eps.append((e - it["E0"]) / float(n))
    return torch.stack(eps)


def fit_power_p(bundles: dict[int, dict], lam: float, init_beta, init_gamma,
                 rounds: int, max_iter: int, lr: float, wall_s: float, tag: str,
                 graded: bool = False):
    """Joint (or single-n) power_p. bundles keyed by n."""
    device = next(iter(bundles.values()))["device"]
    L = int(init_gamma.numel()) if init_gamma.ndim == 1 else int(init_gamma.shape[0])
    if graded:
        g0 = init_gamma.detach().to(device).reshape(L, 1).expand(L, 4).contiguous()
        gamma = torch.nn.Parameter(g0.clone())
    else:
        gamma = torch.nn.Parameter(init_gamma.detach().to(device).reshape(L).clone())
    c1 = torch.nn.Parameter((-init_beta.detach().to(device).reshape(L)).clone())
    c2 = torch.nn.Parameter(torch.full((L,), 1e-4, device=device))
    c3 = torch.nn.Parameter(torch.full((L,), 1e-4, device=device))
    p_raw = torch.nn.Parameter(torch.zeros((), device=device))
    params = [gamma, c1, c2, c3, p_raw]

    def angles(n):
        p = 2.0 * torch.tanh(p_raw)
        return gamma, phi_power(c1, c2, c3, p, n)

    def loss_tensor():
        terms = []
        for n, bun in bundles.items():
            g, phi = angles(n)
            eps = snap_eps_list(bun["sim"], bun["items"], g, phi, lam, n)
            terms.append(eps.mean())
        return torch.stack(terms).mean()

    holder = {}

    def closure():
        holder["opt"].zero_grad()
        loss = loss_tensor()
        loss.backward()
        return loss

    opt = torch.optim.LBFGS(
        params, lr=lr, max_iter=max_iter, history_size=10,
        line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9,
    )
    holder["opt"] = opt
    with torch.no_grad():
        init_loss = float(loss_tensor().detach())
    hist = [init_loss]
    t0 = time.time()
    best = init_loss
    best_vec = [p.detach().clone() for p in params]
    stall = 0
    log(f"{tag} init train_ε={init_loss:.5f}")
    for r in range(rounds):
        if time.time() - t0 > wall_s:
            log(f"{tag} wall cap at round {r}")
            break
        try:
            loss = opt.step(closure)
        except Exception as exc:
            log(f"{tag} LBFGS fail r{r}: {type(exc).__name__}: {exc}")
            break
        val = float(loss.detach())
        hist.append(val)
        if best is None or val < best - 1e-5:
            best = val
            best_vec = [p.detach().clone() for p in params]
            stall = 0
        else:
            stall += 1
        if r == 0 or (r + 1) % 2 == 0 or r + 1 == rounds:
            log(f"{tag} round {r+1}/{rounds} train_ε={val:.5f} best={best:.5f} t={time.time()-t0:.1f}s")
        if stall >= 4 and r >= 4:
            break
    if best_vec is not None:
        with torch.no_grad():
            for p, b in zip(params, best_vec):
                p.copy_(b)
    p = 2.0 * torch.tanh(p_raw.detach())
    info = {"hist": hist, "best": best, "wall_s": time.time() - t0, "p": float(p)}
    return {
        "gamma": gamma.detach(),
        "c1": c1.detach(),
        "c2": c2.detach(),
        "c3": c3.detach(),
        "p": p.detach(),
        "graded": graded,
        "info": info,
    }


def eval_power(snap, bundles_eval: dict[int, dict], lam: float) -> dict[int, np.ndarray]:
    out = {}
    with torch.no_grad():
        for n, bun in bundles_eval.items():
            phi = phi_power(snap["c1"], snap["c2"], snap["c3"], snap["p"], n)
            eps = snap_eps_list(bun["sim"], bun["items"], snap["gamma"], phi, lam, n)
            out[n] = eps.detach().cpu().numpy().astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------

def prepare_split(n: int, insts: list[dict], graph: str, lam: float, L: int, device, z=None):
    if z is None:
        z = z_table(n)
    sim = make_sim(n, L, device)
    items = []
    Hcs = []
    Cns = []
    E0s = []
    for ins in insts:
        Hc_np = hc_diag(n, ins[graph], z=z)
        gs = exact_ground(Hc_np, lam, n)
        Hc = torch.as_tensor(Hc_np, dtype=torch.float32, device=device)
        E = Hc.clone()  # circuit diagonal is physical H_c; cost_Cn normalizes
        Cn = cn_bitstring(sim, E).detach()
        items.append({
            "E": E,
            "Hc": Hc,
            "E0": torch.tensor(gs["E0"], dtype=torch.float32, device=device),
            "seed": ins["seed"],
        })
        Hcs.append(Hc)
        Cns.append(Cn)
        E0s.append(gs["E0"])
    qaoa_bun = {
        "n": n,
        "L": L,
        "Cn": torch.stack(Cns, dim=0),
        "Hc": torch.stack(Hcs, dim=0),
        "E0": torch.tensor(E0s, dtype=torch.float32, device=device),
    }
    snap_bun = pack_snap_bundle(sim, items, lam, device)
    return qaoa_bun, snap_bun


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def self_check(device) -> None:
    log("self-check")
    n, L, lam = 6, 2, 1.0
    inst = build_instances(n, 1, cap=None)[0]
    z = z_table(n)
    Hc = hc_diag(n, inst["reuse"], z=z)
    assert abs(float(Hc.mean())) < 1e-8, Hc.mean()
    # |0⟩ and |+⟩
    e0_bit = float(Hc[0])  # all-zero bitstring, ⟨X⟩=0
    amps0 = torch.zeros(1 << n, dtype=torch.complex64, device=device)
    amps0[0] = 1
    Hc_t = torch.as_tensor(Hc, dtype=torch.float32, device=device)
    e, ec, ex = energy_bits(amps0, Hc_t, lam, n)
    if abs(float(e) - e0_bit) > 1e-5 or abs(float(ex)) > 1e-5:
        raise AssertionError(f"|0⟩ energy {float(e)} vs Hc[0]={e0_bit} ex={float(ex)}")
    amps_p = torch.full((1 << n,), 1.0 / math.sqrt(1 << n), dtype=torch.complex64, device=device)
    e, ec, ex = energy_bits(amps_p, Hc_t, lam, n)
    if abs(float(ec)) > 1e-4 or abs(float(ex) - n) > 1e-3 or abs(float(e) + lam * n) > 1e-3:
        raise AssertionError(f"|+⟩ e={float(e)} ec={float(ec)} ex={float(ex)}")
    # ED anchors
    g0 = exact_ground(Hc, 0.0, n)
    if g0["peak"] < 0.99:
        raise AssertionError(f"λ=0 peak {g0['peak']}")
    if abs(g0["E0"] - g0["E_class"]) > 1e-6:
        raise AssertionError("λ=0 E0 != min Hc")
    gL = exact_ground(Hc, 30.0, n)
    if gL["peak"] > 0.05:
        raise AssertionError(f"large-λ peak {gL['peak']}")
    if abs(gL["E0"] + 30.0 * n) > 0.5:
        raise AssertionError(f"large-λ E0 {gL['E0']} vs {-30*n}")
    # readout of the ED ground state must reproduce E0 (catches an X-factor bug)
    ev, evec = np.linalg.eigh(build_dense(Hc, lam, n))
    psi_gs = torch.as_tensor(evec[:, 0], dtype=torch.complex64, device=device)
    e_gs, _, _ = energy_bits(psi_gs, Hc_t, lam, n)
    if abs(float(e_gs) - float(ev[0])) > 2e-3:
        raise AssertionError(f"readout/ED mismatch {float(e_gs)} vs {float(ev[0])}")
    # diagonal pulse on |+⟩ cannot beat |+⟩
    dim = 1 << n
    base = amps_p.clone()
    for gamma in (-0.4, -0.1, 0.2, 0.8):
        phase = torch.exp(-1j * torch.tensor(gamma, dtype=torch.float32) * Hc_t).to(torch.complex64)
        e, _, _ = energy_bits(base * phase, Hc_t, lam, n)
        if float(e) < -lam * n - 1e-4:
            raise AssertionError(f"diagonal pulse beat |+>: γ={gamma} e={float(e)}")
    # warm SNAP == QAOA probabilities and energy
    sim = make_sim(n, L, device)
    E = Hc_t.clone()
    betas, gammas = paper_angles(L, device)
    # perturb so the check is not the trivial γ~0 point
    with torch.no_grad():
        betas.add_(torch.tensor([0.15, -0.22], device=device))
        gammas.add_(torch.tensor([-0.18, 0.07], device=device))
    phi = phi_from_c(-betas, torch.zeros(L, device=device), torch.zeros(L, device=device))
    out = sim.forward_nov3rd(E, gammas, phi)
    psi = snap_psi(sim, E, gammas, phi)
    probs = (psi.real.square() + psi.imag.square())
    if float((probs - out["probs"]).abs().max()) > 1e-5:
        raise AssertionError(f"SNAP psi drifted from forward_nov3rd {(probs-out['probs']).abs().max()}")
    Cn = cn_bitstring(sim, E).reshape(1, -1)
    Hc_b = Hc_t.reshape(1, -1)
    e_q, _, _ = qaoa_energy_batch(betas, gammas, Cn, Hc_b, lam, n)
    e_s, _, _ = snap_energy(sim, E, Hc_t, gammas, phi, lam)
    if abs(float(e_q) - float(e_s)) > 2e-3:
        raise AssertionError(f"QAOA/SNAP energy mismatch q={float(e_q)} s={float(e_s)}")
    # X-only: a product |+⟩ has ex==n; a computational basis state has ex==0. Already checked.
    # Intra-cavity triple must not be selected when a cross triple exists.
    sp, trip, _J = _best_clause_triple([0, 1, 2, 3], [1, 1, 1, -1])
    if _span(trip) < 2:
        raise AssertionError(f"clause triple stayed intra {trip}")
    log(f"self-check OK  |T|={inst['n_edges']} jaccard={inst['jaccard']:.3f} "
        f"ΔE(QAOA,SNAP)={abs(float(e_q)-float(e_s)):.2e}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _ratio(a, b) -> float:
    if b is None or not np.isfinite(b) or abs(b) < 1e-8:
        return float("nan")
    return float(a) / float(b)


def _cell_verdict(R: float) -> str:
    if not np.isfinite(R):
        return "NA"
    if R <= 0.98:
        return "BEAT"
    if R <= 1.02:
        return "TIE"
    return "UNDER"


def family_verdict(cells: list[dict], quantum_pass: bool, grape_ok: bool) -> str:
    """cells are the primary-graph (reuse, full) holdout ratios at each (L, n)."""
    if not grape_ok:
        return "KILL"
    if not quantum_pass:
        return "KILL"
    by_L = {}
    for c in cells:
        by_L.setdefault(c["L"], {})[c["n"]] = c["verdict"]
    both_L_beat = True
    any_beat_both_n = False
    any_under_both = True
    saw = False
    for L, mp in by_L.items():
        if 6 not in mp or 9 not in mp:
            continue
        saw = True
        beat = mp[6] == "BEAT" and mp[9] == "BEAT"
        under = mp[6] == "UNDER" and mp[9] == "UNDER"
        both_L_beat = both_L_beat and beat
        any_beat_both_n = any_beat_both_n or beat
        any_under_both = any_under_both and under
    if not saw:
        return "needs_edit"
    if both_L_beat:
        return "PASS"
    if any_beat_both_n:
        return "needs_edit"
    if any_under_both and not any_beat_both_n:
        # clean loss at every depth on both sizes
        return "KILL"
    return "needs_edit"


def write_results(payload: dict, path: Path) -> None:
    v = payload["verdict"]
    lines = []
    lines.append("# RESULTS — quantum H explore, arm B (sparse p=3 TFIM)")
    lines.append("")
    lines.append(f"_Finished {payload['finished']}. Verdict: **{v}**._")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("Locked gates: BinH `U_c^X = W exp(-i γ H_c) W` with `W = H^{⊗3}`, then bare Hamming-shell SNAP. `power_p` shell `φ_k = c1 k + a_n c2 k² + a_n c3 k³`, `a_n = (8/(n-1))^p`. Fourier QAOA is ideal-qubit QAOA at the same L: per-layer `(β, γ)` from paper init `(0.01, −0.01)`, cost pulse on the **same** normalized `Cn(H_c)`, mixer `exp(-i β X/2)`.")
    lines.append("")
    lines.append("Hamiltonian (stoquastic sign, `| + ⟩^{⊗n}` lowers the field):")
    lines.append("")
    lines.append(r"\[ H = H_c - \lambda \sum_i X_i, \quad H_c = -s^{-1}\sum_{(i,j,k)\in T} J_{ijk} Z_i Z_j Z_k, \quad s=\sqrt{|T|/n},\; J=\pm 1. \]")
    lines.append("")
    lines.append("Readout is Z-histogram of `H_c` plus single-qubit `⟨X_i⟩`. No YY or XX correlators. `ε = (⟨H⟩ − E_0) / n`. BEAT means holdout median `ε` of power_p ≤ `0.98 ×` the same-split Fourier QAOA median, on **both** n=6 and n=9.")
    lines.append("")
    lines.append(f"Instances: satisfiable Max-E 4-SAT supports from `sat_gen` base `{SCALE_BASE}` (scale1000 protocol), first `B_tr={payload['B_tr']}` train / next `B_ho={payload['B_ho']}` holdout. λ chosen on the **train** ED split only.")
    lines.append("")
    lines.append(f"Chosen **λ = {payload['lambda']}** ({payload['lambda_reason']}).")
    lines.append("")
    lines.append("The file `opus_reply_20261104_2150_hamiltonian_families.md` was not in the repository. Family locks are from `PLAN.md` / `SHORTLIST.md` / `SMOKE.md` (same 2026-11-04 ~21:50 note). Rank2 intra-cavity XX/XXX is not trained: XX correlators fail the Z+X readout lock, so it is not a fallback path.")
    lines.append("")
    lines.append("## Why GRAPE of e^{-i H_c} is the right pulse and not the answer")
    lines.append("")
    lines.append("`H_c` is diagonal in the computational basis, so `exp(-i γ H_c)` is the Fock diagonal inside `U_c^X` — the pulse GRAPE synthesizes. `⟨H_c⟩` on `| + ⟩` is 0 (each ZZZ averages to 0). Any diagonal unitary leaves the Z histogram of `| + ⟩` uniform, and `|⟨X_i⟩| ≤ 1`, so")
    lines.append("")
    lines.append(r"\[ \langle + | e^{i\gamma H_c} H e^{-i\gamma H_c} | + \rangle \ge -\lambda n = \langle H\rangle_{|+\rangle}. \]")
    lines.append("")
    lines.append("A diagonal pulse alone cannot undercut `| + ⟩`. The GS sits below both `| + ⟩` and the best bitstring whenever the table shows `ε_plus > 0` and `ε_class > 0`. Hamming-shell SNAP (or the QAOA X mixer) is what moves Z-population. That is the GRAPE justification: the cost layer is exactly `e^{-i H_c}`, and it does not solve `H`.")
    lines.append("")
    lines.append("## ED λ grid (train split, reuse supports)")
    lines.append("")
    lines.append("| n | λ | med peak | frac peak<0.5 | med ε_class | med ε_plus | med gap | quantum |")
    lines.append("|---|---|----------|---------------|-------------|------------|---------|---------|")
    for n in (6, 9):
        for row in payload["ed_train_reuse"][str(n)]:
            q = "yes" if row["quantum_n"] else ""
            lines.append(
                f"| {n} | {row['lambda']:.2f} | {row['med_peak']:.3f} | {row['frac_peak_lt_0.5']:.2f} | "
                f"{row['med_eps_class']:.3f} | {row['med_eps_plus']:.3f} | {row['med_gap']:.4f} | {q} |"
            )
    lines.append("")
    lines.append(f"Quantum window (both n): `{payload['quantum_window']}`.")
    lines.append("")
    gstat = payload["grape"]
    lines.append(
        f"At λ={payload['lambda']}: med `(E_class−E0)/n` = {gstat['eps_class']:.4f}, "
        f"med `(E_+−E0)/n` = {gstat['eps_plus']:.4f}, med bitstring peak = {gstat['peak']:.3f}. "
        f"GRAPE-alone gap is open: **{payload['grape_ok']}**."
    )
    lines.append("")
    lines.append("## Supports: reuse vs regenerate")
    lines.append("")
    lines.append("| n | density | med \\|T\\| | med Jaccard(reuse, regen) | med sign agreement on ∩ |")
    lines.append("|---|---------|-----------|-----------------------------|--------------------------|")
    for row in payload["support_stats"]:
        lines.append(
            f"| {row['n']} | {row['density']} | {row['med_edges']:.1f} | {row['med_jaccard']:.3f} | {row['med_sign']:.3f} |"
        )
    lines.append("")
    lines.append(payload["support_note"])
    lines.append("")
    lines.append("## Holdout smoke")
    lines.append("")
    lines.append("Medians of `ε` on the holdout. R = Med ε(power_p) / Med ε(Fourier QAOA). QAOA is trained **per n**. power_p is trained **jointly** on {6, 9} unless the row says `transfer` (n=6 only, deployed at n=9 with `a_n`).")
    lines.append("")
    lines.append("| graph | density | L | n | train | Med ε SNAP | Med ε QAOA | R | verdict |")
    lines.append("|-------|---------|---|---|-------|------------|------------|---|---------|")
    for c in payload["cells"]:
        lines.append(
            f"| {c['graph']} | {c['density']} | {c['L']} | {c['n']} | {c['train']} | "
            f"{c['med_snap']:.4f} | {c['med_qaoa']:.4f} | {c['R']:.4f} | {c['verdict']} |"
        )
    lines.append("")
    if payload.get("alt_lambda_cells"):
        lines.append(f"### Alternate λ = {payload['alt_lambda']} (L=2 only, because λ* was not a dual-n BEAT)")
        lines.append("")
        lines.append("| graph | L | n | Med ε SNAP | Med ε QAOA | R | verdict |")
        lines.append("|-------|---|---|------------|------------|---|---------|")
        for c in payload["alt_lambda_cells"]:
            lines.append(
                f"| {c['graph']} | {c['L']} | {c['n']} | {c['med_snap']:.4f} | {c['med_qaoa']:.4f} | "
                f"{c['R']:.4f} | {c['verdict']} |"
            )
        lines.append("")
    if payload.get("walsh_cells"):
        lines.append("### Walsh-degree γ (optional, reuse, L=2)")
        lines.append("")
        lines.append("Graded `γ_{ℓ,w}` for Walsh degrees 1..4 of the same `H_c` diagonal, shell still power_p. Scalar `γ` is the GRAPE-exact pulse; Walsh can reweight degrees and is not required for the gate.")
        lines.append("")
        lines.append("| n | Med ε Walsh | Med ε QAOA | R |")
        lines.append("|---|-------------|------------|---|")
        for c in payload["walsh_cells"]:
            lines.append(f"| {c['n']} | {c['med_snap']:.4f} | {c['med_qaoa']:.4f} | {c['R']:.4f} |")
        lines.append("")
    lines.append("## Does regenerating T hurt transfer?")
    lines.append("")
    lines.append(payload["transfer_note"])
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(payload["verdict_paragraph"])
    lines.append("")
    lines.append("## Repro")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python Nov_3rd/scripts/smoke_quantum_h_B.py "
        "--b-tr 16 --b-ho 32 --rounds-l2 18 --rounds-l16 14 --max-iter 6 --wall-fit 240 --threads 4"
    )
    lines.append("```")
    lines.append("")
    lines.append(f"Wall {payload['wall_s']:.1f}s. Device `{payload['device']}`. Script `{payload['script']}`.")
    lines.append("Train69 ladder, phase2 n21, `settings.py`, and `qaoa_angles/` were not written.")
    lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def ed_split(insts, n, graph: str, z):
    rows = []
    for ins in insts:
        rows.append(ed_one(n, ins[graph], LAMBDA_GRID, z=z))
    summary = {}
    for lam in LAMBDA_GRID:
        sm = summarize_lam(rows, lam)
        sm["quantum_n"] = bool(quantum_ok(sm))
        summary[float(lam)] = sm
    return rows, summary


def support_stat(insts, n, density: str) -> dict:
    return {
        "n": n,
        "density": density,
        "med_edges": _med(ins["n_edges"] for ins in insts),
        "med_jaccard": _med(ins["jaccard"] for ins in insts),
        "med_sign": _med(ins["sign_agree"] for ins in insts),
    }


def run_pair(insts_by_n, graph: str, density: str, lam: float, L: int, device,
             b_tr: int, rounds: int, max_iter: int, lr: float, wall_s: float,
             train_mode: str, z_cache, qaoa_cache: dict):
    """Train QAOA per n (cached) and power_p, eval holdout.

    train_mode: 'joint' uses n=6 and n=9; 'transfer' trains n=6 and evals both;
    'per_n' fits a separate shell at each n (capacity, not the train{6,9} gate).
    """
    snap_train = {}
    snap_hold = {}
    qaoa_hold_eps = {}
    qaoa_info = {}
    for n, insts in insts_by_n.items():
        z = z_cache[n]
        tr, ho = insts[:b_tr], insts[b_tr:]
        key = (n, graph, density, L, lam)
        if key not in qaoa_cache:
            q_tr, _s_tr = prepare_split(n, tr, graph, lam, L, device, z=z)
            betas, gammas, info = fit_qaoa(
                q_tr, lam, rounds=rounds, max_iter=max_iter, lr=lr, wall_s=wall_s,
                tag=f"QAOA {graph}/{density} L{L} n{n}",
            )
            q_ho, s_ho = prepare_split(n, ho, graph, lam, L, device, z=z)
            # rebuild train snap bundle from the same sim family
            _q, s_tr = prepare_split(n, tr, graph, lam, L, device, z=z)
            eps_ho = eval_qaoa(betas, gammas, q_ho, lam)
            qaoa_cache[key] = {
                "betas": betas, "gammas": gammas, "info": info,
                "eps_ho": eps_ho, "s_tr": s_tr, "s_ho": s_ho,
            }
        got = qaoa_cache[key]
        qaoa_hold_eps[n] = got["eps_ho"]
        qaoa_info[n] = got["info"]
        snap_train[n] = got["s_tr"]
        snap_hold[n] = got["s_ho"]
    cells = []

    def _cell(n, eps_snap_n, train_best, p_val):
        ms = float(np.median(eps_snap_n))
        mq = float(np.median(qaoa_hold_eps[n]))
        R = _ratio(ms, mq)
        return {
            "graph": graph,
            "density": density,
            "L": L,
            "n": n,
            "train": train_mode,
            "med_snap": ms,
            "med_qaoa": mq,
            "mean_snap": float(np.mean(eps_snap_n)),
            "mean_qaoa": float(np.mean(qaoa_hold_eps[n])),
            "R": R,
            "verdict": _cell_verdict(R),
            "train_best": train_best,
            "p": p_val,
            "qaoa_train_best": qaoa_info[n]["best"],
        }

    if train_mode == "per_n":
        for n in (6, 9):
            warm_n = qaoa_cache[(n, graph, density, L, lam)]
            snap = fit_power_p(
                {n: snap_train[n]}, lam, warm_n["betas"], warm_n["gammas"],
                rounds=rounds, max_iter=max_iter, lr=lr, wall_s=wall_s,
                tag=f"power_p per_n {graph}/{density} L{L} n{n}",
            )
            eps_snap = eval_power(snap, {n: snap_hold[n]}, lam)
            cells.append(_cell(n, eps_snap[n], snap["info"]["best"], snap["info"]["p"]))
        return cells, None

    # joint / transfer warm from the n=6 QAOA solution (one shared angle set).
    warm = qaoa_cache[(6, graph, density, L, lam)]
    use = {6: snap_train[6]} if train_mode == "transfer" else snap_train
    snap = fit_power_p(
        use, lam, warm["betas"], warm["gammas"],
        rounds=rounds, max_iter=max_iter, lr=lr, wall_s=wall_s,
        tag=f"power_p {train_mode} {graph}/{density} L{L}",
    )
    eps_snap = eval_power(snap, snap_hold, lam)
    for n in sorted(eps_snap):
        cells.append(_cell(n, eps_snap[n], snap["info"]["best"], snap["info"]["p"]))
    return cells, snap


def main():
    ap = argparse.ArgumentParser(description="Arm B sparse p=3 TFIM smoke")
    ap.add_argument("--b-tr", type=int, default=8)
    ap.add_argument("--b-ho", type=int, default=16)
    ap.add_argument("--rounds-l2", type=int, default=8)
    ap.add_argument("--rounds-l16", type=int, default=6)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--wall-fit", type=float, default=240.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pilot", action="store_true", help="tiny L=2 reuse-only run")
    ap.add_argument("--self-check-only", action="store_true")
    ap.add_argument("--skip-walsh", action="store_true")
    ap.add_argument("--skip-sparse", action="store_true")
    ap.add_argument("--skip-l16", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    t_all = time.time()
    self_check(device)
    if args.self_check_only:
        return 0

    if args.pilot:
        args.b_tr, args.b_ho = 2, 2
        args.rounds_l2, args.rounds_l16 = 2, 1
        args.max_iter = 2
        args.wall_fit = 60.0
        args.skip_walsh = True
        args.skip_sparse = True
        args.skip_l16 = True

    B = args.b_tr + args.b_ho
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"building instances B={B} (tr {args.b_tr} / ho {args.b_ho})")

    bank = {}
    for n in (6, 9):
        for density, cap in (("full", None), ("sparse", 2 * n)):
            insts = build_instances(n, B, cap=cap)
            bank[(n, density)] = insts
            log(f"n={n} {density}: med|T|={_med(i['n_edges'] for i in insts):.1f} "
                f"J={_med(i['jaccard'] for i in insts):.3f}")

    z_cache = {n: z_table(n) for n in (6, 9)}

    # Full λ grid on the train split of clause-reuse T (the chooser).
    # Other graphs are diagonalized at the chosen λ inside the trainers.
    log("ED λ grid (train, reuse/full)")
    ed_public = {"train": {}, "hold": {}}
    summaries_train = {}
    for n in (6, 9):
        insts = bank[(n, "full")]
        tr, ho = insts[: args.b_tr], insts[args.b_tr :]
        _rows_tr, sm_tr = ed_split(tr, n, "reuse", z_cache[n])
        _rows_ho, sm_ho = ed_split(ho, n, "reuse", z_cache[n])
        ed_public["train"][f"n{n}_full_reuse"] = list(sm_tr.values())
        ed_public["hold"][f"n{n}_full_reuse"] = list(sm_ho.values())
        summaries_train[(n, "full", "reuse")] = sm_tr
        log(f"ED n={n} done")

    # choose λ from reuse/full train
    choose_in = {}
    for n in (6, 9):
        choose_in[(n, "reuse")] = summaries_train[(n, "full", "reuse")]
    choice = choose_lambda(choose_in)
    lam = float(choice["chosen"]["lambda"])
    log(f"chosen λ={lam} ({choice['reason']}) pass={choice['chosen']['pass']}")

    def grape_stats(split_summaries):
        # average the two n's medians
        ec, ep, pk = [], [], []
        for n in (6, 9):
            sm = split_summaries[n][lam]
            ec.append(sm["med_eps_class"])
            ep.append(sm["med_eps_plus"])
            pk.append(sm["med_peak"])
        return {"eps_class": float(np.mean(ec)), "eps_plus": float(np.mean(ep)), "peak": float(np.mean(pk))}

    gstat = grape_stats({n: summaries_train[(n, "full", "reuse")] for n in (6, 9)})
    grape_ok = bool(gstat["eps_class"] > GAP_MIN and gstat["eps_plus"] > GAP_MIN)
    quantum_pass = bool(choice["chosen"]["pass"])
    window = [r["lambda"] for r in choice["grid"] if r["pass"]]

    # mark per-n quantum flag already in summaries
    support_stats = []
    for n in (6, 9):
        for density in ("full", "sparse"):
            support_stats.append(support_stat(bank[(n, density)], n, density))

    full_j_by_n = {n: support_stat(bank[(n, "full")], n, "full")["med_jaccard"] for n in (6, 9)}
    sparse_j_by_n = {n: support_stat(bank[(n, "sparse")], n, "sparse")["med_jaccard"] for n in (6, 9)}
    support_note = (
        f"Full-hypergraph median Jaccard(reuse, regen) is n=6 {full_j_by_n[6]:.2f}, "
        f"n=9 {full_j_by_n[9]:.2f}. "
        f"Sparse cap |T|=2n Jaccard is n=6 {sparse_j_by_n[6]:.2f}, n=9 {sparse_j_by_n[9]:.2f}. "
        "n=6 has only 18 cross-cavity triples and m=60 clauses, so the uncapped reuse set is the "
        "complete cross-cavity hypergraph and regenerating T only rerolls signs. "
        "n=9 has 81 cross-cavity triples and the load-balanced clause extraction does not fill all of them, "
        "so regen changes the support. The |T|=2n rows are the matched-cardinality comparison "
        "where supports differ at both sizes. Intra-cavity triples are not used."
    )

    Ls = [2] if args.skip_l16 else [2, 16]
    densities = ["full"]
    if not args.skip_sparse:
        densities.append("sparse")
    graphs = ["reuse", "regen"]

    cells = []
    qaoa_cache = {}
    for density in densities:
        insts_by_n = {n: bank[(n, density)] for n in (6, 9)}
        for L in Ls:
            rounds = args.rounds_l2 if L == 2 else args.rounds_l16
            # sparse control: L=2 only, to keep the saturation question answered without a second L=16
            if density == "sparse" and L != 2:
                continue
            for graph in graphs:
                pair_cells, _snap = run_pair(
                    insts_by_n, graph, density, lam, L, device,
                    args.b_tr, rounds, args.max_iter, args.lr, args.wall_fit,
                    "joint", z_cache, qaoa_cache,
                )
                cells.extend(pair_cells)
                # transfer: n6 → n9 angles, both sizes evaluated
                tr_cells, _snap_t = run_pair(
                    insts_by_n, graph, density, lam, L, device,
                    args.b_tr, max(3, rounds // 2), args.max_iter, args.lr, args.wall_fit,
                    "transfer", z_cache, qaoa_cache,
                )
                cells.extend(tr_cells)
            if density == "full":
                pn_cells, _pn = run_pair(
                    insts_by_n, "reuse", density, lam, L, device,
                    args.b_tr, rounds, args.max_iter, args.lr, args.wall_fit,
                    "per_n", z_cache, qaoa_cache,
                )
                cells.extend(pn_cells)

    # alternate λ at L=2 reuse+regen full if primary joint is not dual-n BEAT
    primary = [c for c in cells if c["graph"] == "reuse" and c["density"] == "full" and c["train"] == "joint"]
    beat_both = {}
    for c in primary:
        beat_both.setdefault(c["L"], set())
        if c["verdict"] == "BEAT":
            beat_both[c["L"]].add(c["n"])
    dual_beat = any(beat_both.get(L, set()) >= {6, 9} for L in Ls)
    alt_cells = []
    alt_lam = None
    if not dual_beat:
        alts = [r["lambda"] for r in choice["grid"] if r["pass"] and abs(r["lambda"] - lam) > 1e-9]
        if not alts:
            alts = [r["lambda"] for r in choice["grid"] if r["lambda"] not in (0.0, lam)]
        if alts:
            # neighbor with the next-best competition score
            alt_lam = max(
                (r for r in choice["grid"] if r["lambda"] in alts),
                key=lambda r: r["score"],
            )["lambda"]
            log(f"primary not dual-n BEAT; alternate λ={alt_lam} L=2")
            for graph in graphs:
                extra, _s = run_pair(
                    {n: bank[(n, "full")] for n in (6, 9)}, graph, "full", float(alt_lam), 2, device,
                    args.b_tr, args.rounds_l2, args.max_iter, args.lr, args.wall_fit,
                    "joint", z_cache, qaoa_cache,
                )
                for c in extra:
                    c["lambda"] = float(alt_lam)
                alt_cells.extend(extra)

    walsh_cells = []
    if not args.skip_walsh and not args.pilot:
        log("optional Walsh L=2 reuse/full")
        # reuse cached QAOA at λ, L=2, reuse, full
        key6 = (6, "reuse", "full", 2, lam)
        if key6 in qaoa_cache:
            warm = qaoa_cache[key6]
            bundles = {n: qaoa_cache[(n, "reuse", "full", 2, lam)]["s_tr"] for n in (6, 9)}
            holds = {n: qaoa_cache[(n, "reuse", "full", 2, lam)]["s_ho"] for n in (6, 9)}
            snap_w = fit_power_p(
                bundles, lam, warm["betas"], warm["gammas"],
                rounds=max(3, args.rounds_l2 // 2), max_iter=args.max_iter, lr=args.lr,
                wall_s=args.wall_fit, tag="walsh L2 reuse", graded=True,
            )
            eps_w = eval_power(snap_w, holds, lam)
            for n in (6, 9):
                # QAOA medians from a joint cell if present
                qcells = [c for c in cells if c["graph"] == "reuse" and c["density"] == "full"
                          and c["L"] == 2 and c["n"] == n and c["train"] == "joint"]
                mq = qcells[0]["med_qaoa"] if qcells else float("nan")
                ms = float(np.median(eps_w[n]))
                walsh_cells.append({
                    "n": n, "med_snap": ms, "med_qaoa": mq, "R": _ratio(ms, mq),
                    "degree_rms": [float(x) for x in snap_w["gamma"].pow(2).mean(dim=0).sqrt().cpu()],
                })

    verdict = family_verdict(primary, quantum_pass, grape_ok)
    hold_quantum = True
    for n in (6, 9):
        rows = ed_public["hold"][f"n{n}_full_reuse"]
        row = next(r for r in rows if abs(float(r["lambda"]) - lam) < 1e-9)
        if not row["quantum_n"]:
            hold_quantum = False
            log(f"holdout n={n} not quantum at λ={lam}: peak={row['med_peak']:.3f}")
    if verdict == "PASS" and not hold_quantum:
        verdict = "needs_edit"
        log("downgrade PASS → needs_edit: holdout split missed the quantum gate")
    # if alternate λ saves a dual-n BEAT, do not upgrade KILL/needs_edit to PASS:
    # the predeclared λ* is the gate. Mention the alternate in the paragraph.
    alt_primary = [c for c in alt_cells if c["graph"] == "reuse"]
    alt_beat = False
    if alt_primary:
        gotn = {c["n"] for c in alt_primary if c["verdict"] == "BEAT"}
        alt_beat = gotn >= {6, 9}

    # transfer note
    def _find(graph, density, L, n, train):
        for c in cells:
            if (c["graph"], c["density"], c["L"], c["n"], c["train"]) == (graph, density, L, n, train):
                return c
        return None

    lines_tn = []
    for density in densities:
        for L in Ls:
            if density == "sparse" and L != 2:
                continue
            ru = _find("reuse", density, L, 9, "transfer")
            rg = _find("regen", density, L, 9, "transfer")
            ru_j = _find("reuse", density, L, 9, "joint")
            rg_j = _find("regen", density, L, 9, "joint")
            if not (ru and rg and ru_j and rg_j):
                continue
            lines_tn.append(
                f"- L={L} {density}: n=9 transfer R reuse={ru['R']:.3f} (ε={ru['med_snap']:.4f}) vs "
                f"regen={rg['R']:.3f} (ε={rg['med_snap']:.4f}). "
                f"Joint R reuse={ru_j['R']:.3f}, regen={rg_j['R']:.3f}. "
                f"Transfer/joint ε ratio reuse={_ratio(ru['med_snap'], ru_j['med_snap']):.3f}, "
                f"regen={_ratio(rg['med_snap'], rg_j['med_snap']):.3f}."
            )
    if not lines_tn:
        transfer_note = "Transfer rows were not produced."
    else:
        # headline: does regen's transfer/joint ratio exceed reuse's by >5%?
        hurt_bits = []
        for density in densities:
            for L in Ls:
                if density == "sparse" and L != 2:
                    continue
                ru = _find("reuse", density, L, 9, "transfer")
                rg = _find("regen", density, L, 9, "transfer")
                ru_j = _find("reuse", density, L, 9, "joint")
                rg_j = _find("regen", density, L, 9, "joint")
                if not (ru and rg and ru_j and rg_j):
                    continue
                ratio_u = _ratio(ru["med_snap"], ru_j["med_snap"])
                ratio_g = _ratio(rg["med_snap"], rg_j["med_snap"])
                if np.isfinite(ratio_u) and np.isfinite(ratio_g) and ratio_g > ratio_u * 1.05:
                    hurt_bits.append(f"L={L}/{density} (regen transfer/joint {ratio_g:.2f} > reuse {ratio_u:.2f})")
        if hurt_bits:
            headline = (
                "Regenerating T hurts n=6→n=9 transfer relative to reusing Max-E supports at: "
                + "; ".join(hurt_bits) + ". "
                "The shell's a_n map was built for the clause geometry; a fresh hypergraph does not ride it as cleanly."
            )
        else:
            headline = (
                "Regenerating T does not measurably hurt n=6→n=9 transfer versus reusing Max-E supports "
                "(regen transfer/joint ε is within 5% of reuse, or better). "
                "At these sizes the cross-cavity ensemble is small, so support identity is a weak lever "
                "next to the sign pattern and λ."
            )
        transfer_note = headline + "\n\n" + "\n".join(lines_tn)

    # paragraph
    prim_txt = ", ".join(
        f"L={c['L']} n={c['n']} R={c['R']:.3f} {c['verdict']}" for c in primary
    ) or "no primary cells"
    if verdict == "PASS":
        para = (
            f"**PASS.** On clause-reuse T at λ={lam}, joint power_p beats Fourier QAOA on both n "
            f"at every smoked L ({prim_txt}). The GS is quantum (peak<0.5) and neither the best "
            "bitstring nor a diagonal pulse on |+⟩ reaches E0, so the Hamming-shell mixer is doing "
            "work the GRAPE cost pulse cannot. Rank2 intra-cavity XX/XXX stays off the main path: "
            "it needs XX correlators and fails the Z+X readout lock."
        )
    elif verdict == "KILL":
        why = []
        if not quantum_pass:
            why.append("no λ put both n=6 and n=9 into a quantum GS (peak<0.5) while keeping H_c and the field competitive")
        if not grape_ok:
            why.append("the GS is too close to a classical bitstring or to |+⟩, so e^{-i H_c} is not a justified nontrivial pulse")
        unders = [c for c in primary if c["verdict"] == "UNDER"]
        if unders and quantum_pass and grape_ok:
            why.append(
                "power_p stayed above 1.02× Fourier QAOA on the reuse ensemble at the quantum λ "
                f"({prim_txt})"
            )
        if alt_beat:
            why.append(
                f"an alternate λ={alt_lam} did BEAT at L=2 on both n, so this is a λ-window miss rather than a dead gate; "
                "the predeclared chooser still failed the dual-n gate at λ*"
            )
        para = "**KILL.** " + " ".join(why) + " Do not promote arm B onto the ladder."
        if not why:
            para = f"**KILL.** Primary reuse cells did not clear the gate ({prim_txt})."
    else:
        extra = ""
        if alt_beat:
            extra = f" Alternate λ={alt_lam} beats at L=2 on both n; retune the λ chooser before a longer train."
        if not hold_quantum:
            extra += " Holdout split missed the quantum gate at λ*."
        per_n = [c for c in cells if c["train"] == "per_n" and c["density"] == "full" and c["graph"] == "reuse"]
        if per_n:
            extra += " Per-n shell (not the gate): " + ", ".join(
                f"L={c['L']} n={c['n']} R={c['R']:.3f} {c['verdict']}" for c in per_n
            ) + "."
        para = (
            f"**needs_edit.** Quantumness/GRAPE checks "
            f"{'passed' if quantum_pass and grape_ok else 'did not fully pass'}. "
            f"Primary reuse joint cells: {prim_txt}. "
            "The motif is not ready to freeze: a depth, a support cap, or the λ window still misses the dual-n BEAT rule."
            + extra
            + " Rank2 is not opened."
        )

    # json-friendly ed rows for the markdown (reuse full train)
    ed_md = {}
    for n in (6, 9):
        ed_md[str(n)] = ed_public["train"][f"n{n}_full_reuse"]

    payload = {
        "verdict": verdict,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lambda": lam,
        "lambda_reason": choice["reason"],
        "quantum_window": window,
        "quantum_pass": quantum_pass,
        "grape_ok": grape_ok,
        "grape": gstat,
        "B_tr": args.b_tr,
        "B_ho": args.b_ho,
        "ed_train_reuse": ed_md,
        "support_stats": support_stats,
        "support_note": support_note,
        "cells": cells,
        "alt_lambda": alt_lam,
        "alt_lambda_cells": alt_cells,
        "walsh_cells": walsh_cells,
        "transfer_note": transfer_note,
        "verdict_paragraph": para,
        "wall_s": time.time() - t_all,
        "device": str(device),
        "script": "Nov_3rd/scripts/smoke_quantum_h_B.py",
        "pilot": bool(args.pilot),
    }
    # drop non-serializable if any
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    (OUT / "smoke_results.json").write_text(json.dumps(_clean(payload), indent=2) + "\n")
    # full ED dump without markdown-only fields
    (OUT / "ed_grid.json").write_text(json.dumps(_clean({
        "lambda_choice": {"lambda": lam, "reason": choice["reason"], "window": window, "grid": choice["grid"]},
        "ed": ed_public,
        "support_stats": support_stats,
    }), indent=2) + "\n")
    write_results(payload, OUT / "RESULTS.md")
    log(f"VERDICT {verdict}  wrote {OUT/'RESULTS.md'}  wall {time.time()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
