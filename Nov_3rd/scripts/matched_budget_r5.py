#!/usr/bin/env python3
"""Matched-budget bakeoff of the three μ=−0.25 champs (Round 5).

One L-BFGS protocol, taken from PR #19 and PR #23: rounds 18/48 (L=2/L=16),
max_iter 8, lr 0.35, seeds 0–2, B_train=128, B_hold=128, freeze n=12 (B=64)
and n=15 (B=32). Early stop is off. Every head runs the full round count.
The checkpoint is the best {6,9} holdout score inside that budget. n=12 and
n=15 are not in the loss and not in the checkpoint. n=21 is not loaded.

Arms, each against Fourier QAOA on the same H:
  A  gauss J ~ N(0,1) + μ=−0.25, power_p     (PR #21's champ, this budget)
  B  ±1 ZZZ + μ=−0.25, κ=0, power_p           (PR #18 / bakeoff B)
  C  unsat λ=0.75 + μ=−0.25, Walsh            (PR #23's champ, early stop off)

Gate, unchanged: BinH U_c^X then Hamming-shell SNAP. ⟨H_c⟩ is the Z histogram.
⟨X⟩ and intra-cavity ⟨XX⟩ are the X histogram (final W omitted). No YY term.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.sat_gen import clause_count
from Nov_3rd import settings as S
from Nov_3rd.quantum_h_rank2 import (
    Rank2Sim,
    best_product_rank2,
    dense_rank2,
    local_x_eigs,
    self_check as rank2_self_check,
    sparse_ground_rank2,
)
from Nov_3rd.quantum_h_tf import (
    FockSim,
    self_check as fock_self_check,
    zzz_diag,
)
from Nov_3rd.quantum_h_transfer import (
    Circuit,
    center_cn,
    decode,
    energy_from_amps,
    init_fourier,
    load_split,
    n_params,
    power_theta_from_fourier,
    sparse_ground_state,
    summarize_eps,
    walsh_mass_fractions,
    walsh_pieces,
    walsh_theta_from_fourier,
)

TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
J_RNG_SALT = 91  # PR #20 gaussian couplings

# Locked protocol. Do not edit these to chase a ranking.
ROUNDS_L2 = 18
ROUNDS_L16 = 48
MAX_ITER = 8
LR = 0.35
SEEDS = (0, 1, 2)
B_TRAIN = 128
B_HOLD = 128
B_N12 = 64
B_N15 = 32
LS = (2, 16)
BEAT_R = 0.98
TIE_R = 1.02
LAM_A = 0.5
LAM_B = 0.5
LAM_C = 0.75
MU = -0.25
KAPPA = 0.0

# Published first train rows. Used only to prove the diagonal. μ=0 numbers
# are not a checkpoint and are not the trained Hamiltonian for A or C.
REF_SEEDS_N6 = (51060000016, 51060000024, 51060000028, 51060000030)
REF_E0_GAUSS_MU0 = (-4.94513429, -4.76313577, -5.18908633, -5.99268963)
REF_E0_RANK2 = (-5.73314208, -5.73314208, -6.64298697, -5.68868697)
REF_E0_UNSAT_MU0 = (-1.60434386, -1.66938673, -2.11349)

# Prior claims. They are not this budget and they do not enter the ranking.
# A: PR #21, 24 L=16 rounds, B_hold=256, residual checkpoint, n=15 B=16.
# B: PR #22 bakeoff, this protocol.
# C: PR #23, 18/48 rounds with an 8-round early stop.
CLAIMED_N12_R = {"gauss": 0.642, "rank2": 0.701, "unsat": 0.330}

FAMILIES = (
    {
        "id": "gauss",
        "champ": "A gauss J μ=-0.25",
        "head": "power_p",
        "lam": LAM_A,
        "mu": MU,
        "kind": "gauss",
    },
    {
        "id": "rank2",
        "champ": "B ±1 ZZZ μ=-0.25",
        "head": "power_p",
        "lam": LAM_B,
        "mu": MU,
        "kind": "rank2",
    },
    {
        "id": "unsat",
        "champ": "C unsat λ=0.75 μ=-0.25",
        "head": "walsh",
        "lam": LAM_C,
        "mu": MU,
        "kind": "unsat",
    },
)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[r5]", *a, flush=True)


def mark_of(R: float) -> str:
    if not math.isfinite(R):
        return "UNDEF"
    if R <= BEAT_R:
        return "BEAT"
    if R <= TIE_R:
        return "TIE"
    return "UNDER"


def quantum_ok(peak: float, gap: float) -> bool:
    return float(peak) < 0.5 and float(gap) >= 0.02


def apply_gaussian_J(clauses, seed: int, n: int):
    """Independent N(0,1) couplings on a fixed cross-cavity support (PR #20)."""
    if not clauses:
        return []
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(n) + J_RNG_SALT)
    out = []
    for a, b, c, _J in clauses:
        out.append((int(a), int(b), int(c), float(rng.normal())))
    return out


def with_gaussian(inst):
    clauses = apply_gaussian_J(inst.clauses, inst.seed, inst.n)
    return replace(inst, clauses=clauses, diag=zzz_diag(inst.n, clauses))


def cavity_x_strings(pX: torch.Tensor, n_cav: int, sx8: torch.Tensor, xx8: torch.Tensor):
    """Per-cavity X histogram → Σ⟨X⟩, Σ_cav Σ_{i<j}⟨XX⟩."""
    B = int(pX.shape[0])
    C = int(n_cav)
    if int(pX.shape[-1]) != 8 ** C:
        raise ValueError(f"pX {tuple(pX.shape)} != 8^{C}")
    marg = pX.reshape(B, *([8] * C))
    e_sx = torch.zeros(B, dtype=pX.dtype, device=pX.device)
    e_xx = torch.zeros(B, dtype=pX.dtype, device=pX.device)
    for c in range(C):
        axes = tuple(i for i in range(1, C + 1) if i != c + 1)
        m = marg.sum(dim=axes) if axes else marg
        e_sx = e_sx + m @ sx8
        e_xx = e_xx + m @ xx8
    return e_sx, e_xx


class ZXScore:
    """⟨H⟩ from the Z histogram and the X histogram. The gate is ``circ``.

    H = score − λ Σ X + μ Σ_cav Σ_{i<j} XX. ``score`` is the Z diagonal
    (raw unsat). The pulse inside U_c^X stays the centered C_n.
    """

    def __init__(self, circ: Circuit, mu: float):
        self.circ = circ
        self.mu = float(mu)
        self.n = int(circ.n)
        self.e0 = circ.e0
        self.sim = circ.sim
        sx, xx, _xxx = local_x_eigs()
        self.sx8 = torch.as_tensor(sx, dtype=circ.sim.rdtype, device=circ.sim.device)
        self.xx8 = torch.as_tensor(xx, dtype=circ.sim.rdtype, device=circ.sim.device)

    def energy(self, gamma, phi, graded, lam):
        pZ, pX = self.circ.bit_histograms(gamma, phi, graded)
        e_c = (pZ * self.circ.score).sum(dim=-1)
        e_sx, e_xx = cavity_x_strings(pX, self.circ.sim.C, self.sx8, self.xx8)
        return e_c - float(lam) * e_sx + self.mu * e_xx


def product_energy_unsat(theta: np.ndarray, diag: np.ndarray, n: int, lam: float, mu: float) -> float:
    """Real XZ product state. ⟨Z⟩=cos θ, ⟨X⟩=sin θ, ⟨Y⟩=0."""
    n = int(n)
    theta = np.asarray(theta, dtype=np.float64).reshape(n)
    diag = np.asarray(diag, dtype=np.float64).reshape(1 << n)
    c2 = np.cos(0.5 * theta) ** 2
    s2 = np.sin(0.5 * theta) ** 2
    p = np.ones(1, dtype=np.float64)
    for i in range(n):
        p = np.kron(p, np.array([c2[i], s2[i]], dtype=np.float64))
    e = float(p @ diag)
    sx = np.sin(theta)
    e -= float(lam) * float(sx.sum())
    for c0 in range(n // 3):
        i, j, k = 3 * c0, 3 * c0 + 1, 3 * c0 + 2
        e += float(mu) * float(sx[i] * sx[j] + sx[i] * sx[k] + sx[j] * sx[k])
    return float(e)


def best_product_unsat(diag: np.ndarray, n: int, lam: float, mu: float, seed: int = 0) -> float:
    from scipy.optimize import minimize

    n = int(n)
    diag = np.asarray(diag, dtype=np.float64)
    x_best = int(np.argmin(diag))
    bits = np.array([(x_best >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.float64)
    starts = [
        np.zeros(n),
        np.full(n, np.pi),
        np.full(n, 0.5 * np.pi),
        np.full(n, -0.5 * np.pi),
        bits * np.pi,
        (1.0 - bits) * np.pi,
    ]
    mix = int(seed) + 17 * int(n) + int(round(1000 * (float(mu) + 10.0)))
    rng = np.random.default_rng(mix)
    for _ in range(3):
        starts.append(rng.uniform(-np.pi, np.pi, size=n))
    scored = [(product_energy_unsat(th, diag, n, lam, mu), th) for th in starts]
    scored.sort(key=lambda row: row[0])
    best = float(scored[0][0])
    for _e, th in scored[:3]:
        res = minimize(lambda t: product_energy_unsat(t, diag, n, lam, mu), th, method="L-BFGS-B")
        if np.isfinite(res.fun):
            best = min(best, float(res.fun))
    return float(best)


class Rank2Circuit:
    """Same call shape as ``Circuit.energy(gamma, phi, graded, lam)``."""

    def __init__(self, sim: Rank2Sim, Hc: np.ndarray, e0: np.ndarray, mu: float, kappa: float):
        self.sim = sim
        self.n = int(sim.n)
        self.Hc = torch.as_tensor(np.asarray(Hc, dtype=np.float64))
        self.e0 = torch.as_tensor(np.asarray(e0, dtype=np.float64))
        self.mu = float(mu)
        self.kappa = float(kappa)

    def energy(self, gamma, phi, graded, lam):
        if graded:
            raise RuntimeError("rank2 / gauss bakeoff trains power_p, not graded γ")
        return self.sim.energy(self.Hc, gamma, phi, float(lam), self.mu, self.kappa)


class Split:
    def __init__(self, n, role, circuit, seeds, meta):
        self.n = int(n)
        self.role = role
        self.circuit = circuit
        self.seeds = list(seeds)
        self.meta = meta


def _loss(method, theta, batches, lam, L):
    terms = []
    for sp in batches:
        gamma, phi, graded = decode(method, theta, L, sp.n)
        energy = sp.circuit.energy(gamma, phi, graded, lam)
        terms.append(((energy - sp.circuit.e0) / float(sp.n)).mean())
    return torch.stack(terms).mean()


def _hold_view(method, theta, batches, lam, L):
    view = {}
    with torch.no_grad():
        for sp in batches:
            gamma, phi, graded = decode(method, theta, L, sp.n)
            energy = sp.circuit.energy(gamma, phi, graded, lam)
            view[sp.n] = summarize_eps(energy, sp.circuit.e0, sp.n, lam)
    return view


def _select_score(view, ref_med):
    if ref_med is None:
        return 0.5 * (view[6]["med_eps"] + view[9]["med_eps"])
    return max(view[n]["med_eps"] / max(float(ref_med[n]), 1e-8) for n in (6, 9))


def fit_matched(method, L, seed, train, hold, lam, rounds, max_iter, lr, theta0, ref_med, family):
    """Full ``rounds`` L-BFGS steps. No early stop.

    The returned θ is the best {6,9} holdout checkpoint inside that budget.
    A failed L-BFGS step restores θ and still consumes the round.
    """
    dtype = torch.float64
    if theta0 is None:
        if method != "fourier_qaoa":
            raise RuntimeError(f"{method} needs a warm start")
        theta0 = init_fourier(L, seed, dtype, "cpu")
    theta = torch.nn.Parameter(theta0.detach().to(dtype=dtype).clone())
    t0 = time.time()
    hist = []
    failures = 0

    def consider(tag, loss_f=None):
        view = _hold_view(method, theta, hold, lam, L)
        score = _select_score(view, ref_med)
        row = {"tag": tag, "score": float(score), "med6": view[6]["med_eps"], "med9": view[9]["med_eps"]}
        if loss_f is not None:
            row["train_loss"] = float(loss_f)
        hist.append(row)
        return score, view

    score, view = consider("init")
    best = {"score": score, "theta": theta.detach().clone(), "view": view, "tag": "init"}
    log(
        f"  init {family} {method} L={L} s={seed} "
        f"ε6={view[6]['med_eps']:.4f} ε9={view[9]['med_eps']:.4f} score={score:.4f}"
    )
    for r in range(int(rounds)):
        snapshot = theta.detach().clone()
        opt = torch.optim.LBFGS(
            [theta],
            lr=float(lr),
            max_iter=int(max_iter),
            history_size=16,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = _loss(method, theta, train, lam, L)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 1e3)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach())
            if not math.isfinite(loss_f):
                raise RuntimeError("non-finite loss")
        except Exception as exc:
            failures += 1
            with torch.no_grad():
                theta.copy_(snapshot)
            log(f"  LBFGS fail {family} {method} L={L} s={seed} r={r}: {exc}")
            hist.append({"tag": f"r{r}", "fail": True, "score": float("nan"), "med6": float("nan"), "med9": float("nan")})
            continue
        score, view = consider(f"r{r}", loss_f)
        if score < best["score"] - 1e-6:
            best = {"score": score, "theta": theta.detach().clone(), "view": view, "tag": f"r{r}"}
        if r % 6 == 0 or r + 1 == int(rounds):
            log(
                f"  {family} {method} L={L} s={seed} r={r} "
                f"ε6={view[6]['med_eps']:.4f} ε9={view[9]['med_eps']:.4f} "
                f"score={score:.4f} best={best['score']:.4f}@{best['tag']}"
            )
    with torch.no_grad():
        theta.copy_(best["theta"])
    return {
        "family": family,
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "lam": float(lam),
        "n_params": n_params(method, L),
        "best_tag": best["tag"],
        "best_score": float(best["score"]),
        "final_tag": hist[-1]["tag"],
        "final_score": hist[-1]["score"],
        "hold_during_select": {str(k): v for k, v in best["view"].items()},
        "history_len": len(hist),
        "theta": theta.detach().cpu().numpy(),
        "wall_s": time.time() - t0,
        "rounds_budget": int(rounds),
        "rounds_ran": int(rounds),
        "early_stopped": False,
        "lbfgs_failures": int(failures),
        "max_iter": int(max_iter),
        "lr": float(lr),
    }


def evaluate(method, theta, splits, lam, L):
    th = torch.as_tensor(np.asarray(theta, dtype=np.float64))
    out = {}
    with torch.no_grad():
        for sp in splits:
            gamma, phi, graded = decode(method, th, L, sp.n)
            energy = sp.circuit.energy(gamma, phi, graded, lam)
            row = summarize_eps(energy, sp.circuit.e0, sp.n, lam)
            row["role"] = sp.role
            out[int(sp.n)] = row
    return out


def _product_gap(kind, inst, diag_row, e0, lam, mu):
    if kind == "unsat":
        eprod = best_product_unsat(diag_row, inst.n, lam, mu, seed=inst.seed)
    elif kind in ("gauss", "rank2"):
        eprod = best_product_rank2(inst.clauses, inst.n, lam, mu, KAPPA, seed=inst.seed)
    else:
        raise ValueError(kind)
    return float((eprod - float(e0)) / max(abs(float(e0)), 1e-8))


def _cache_path(out: Path, fam, n, role, count) -> Path:
    return out / "cache" / f"{fam}_n{n}_{role}_B{count}.npz"


def _load_cache(path: Path, seeds: list[int], lam: float, mu: float):
    if not path.exists():
        return None
    z = np.load(path)
    if list(np.asarray(z["seeds"]).tolist()) != list(seeds):
        return None
    if "lam" not in z.files or "mu" not in z.files:
        return None
    if abs(float(z["lam"]) - float(lam)) > 1e-12 or abs(float(z["mu"]) - float(mu)) > 1e-12:
        return None
    return z


def build_family_split(fam, insts, raw, n, role, sim_fock, sim_rank, out: Path):
    kind = fam["kind"]
    lam = float(fam["lam"])
    mu = float(fam["mu"])
    seeds = [int(ins.seed) for ins in insts]
    cache = _load_cache(_cache_path(out, kind, n, role, len(insts)), seeds, lam, mu)
    if kind == "gauss":
        insts = [with_gaussian(ins) for ins in insts]
        pulse = np.stack([ins.diag for ins in insts], axis=0)
        score = pulse
    elif kind == "rank2":
        pulse = np.stack([ins.diag for ins in insts], axis=0)
        score = pulse
    elif kind == "unsat":
        m = int(clause_count(n))
        pulse = center_cn(raw, n, m)
        score = np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError(kind)
    if cache is None:
        t0 = time.time()
        e0 = np.empty(len(insts), dtype=np.float64)
        peak = np.empty_like(e0)
        entropy = np.empty_like(e0)
        residual = np.empty_like(e0)
        gap = np.empty_like(e0)
        for i, ins in enumerate(insts):
            row = np.asarray(score[i], dtype=np.float64)
            g = sparse_ground_rank2(row, n, lam, mu, KAPPA)
            e0[i] = g["E0"]
            peak[i] = g["peak"]
            entropy[i] = g["entropy"]
            residual[i] = g["residual"]
            gap[i] = _product_gap(kind, ins, row, g["E0"], lam, mu)
            if (i + 1) % 16 == 0 or i + 1 == len(insts):
                log(
                    f"  ED {kind} n={n} {role} {i+1}/{len(insts)} "
                    f"peak={float(np.median(peak[:i+1])):.3f} gap={float(np.median(gap[:i+1])):.3f} "
                    f"({time.time()-t0:.0f}s)"
                )
        path = _cache_path(out, kind, n, role, len(insts))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seeds=np.asarray(seeds, dtype=np.int64),
            lam=np.float64(lam),
            mu=np.float64(mu),
            E0=e0,
            peak=peak,
            entropy=entropy,
            residual=residual,
            prod_gap=gap,
        )
    else:
        log(f"  cache {kind} n={n} {role} B={len(insts)}")
        e0 = np.asarray(cache["E0"], dtype=np.float64)
        peak = np.asarray(cache["peak"], dtype=np.float64)
        entropy = np.asarray(cache["entropy"], dtype=np.float64)
        residual = np.asarray(cache["residual"], dtype=np.float64)
        gap = np.asarray(cache["prod_gap"], dtype=np.float64)
    if kind == "unsat":
        pieces = walsh_pieces(torch.as_tensor(pulse, dtype=torch.float64))
        base = Circuit(
            sim_fock,
            torch.as_tensor(pulse, dtype=torch.float64),
            torch.as_tensor(score, dtype=torch.float64),
            torch.as_tensor(e0, dtype=torch.float64),
            pieces=pieces,
        )
        circuit = ZXScore(base, mu)
    else:
        circuit = Rank2Circuit(sim_rank, pulse, e0, mu, KAPPA)
    meta = {
        "n": int(n),
        "role": role,
        "B": len(insts),
        "lam": lam,
        "mu": mu,
        "peak_med": float(np.median(peak)),
        "gap_med": float(np.median(gap)),
        "entropy_med": float(np.median(entropy)),
        "residual_max": float(np.max(residual)),
        "quantum": bool(quantum_ok(float(np.median(peak)), float(np.median(gap)))),
        "seeds": seeds,
    }
    return Split(n, role, circuit, seeds, meta), insts, pulse, score


def prepare(out: Path):
    specs = [
        (6, "train", B_TRAIN),
        (6, "hold", B_HOLD),
        (9, "train", B_TRAIN),
        (9, "hold", B_HOLD),
        (12, "transfer", B_N12),
        (15, "transfer", B_N15),
    ]
    if any(n == 21 for n, _r, _c in specs):
        raise RuntimeError("n=21 is out of this bakeoff")
    rows = {}
    for n, role, count in specs:
        kind = "train" if role == "train" else "scale"
        pack = TRAIN_PACKS / f"train_n{n}.npz" if role == "train" else None
        log(f"load n={n} {role} B={count}")
        insts, raw = load_split(n, count, kind, pack)
        rows[(n, role)] = (insts, np.asarray(raw, dtype=np.float64))
    for n in (6, 9):
        tr = {int(ins.seed) for ins in rows[(n, "train")][0]}
        ho = {int(ins.seed) for ins in rows[(n, "hold")][0]}
        if tr & ho:
            raise RuntimeError(f"n={n} train/hold seed overlap")
    sims = {n: FockSim(n, dtype=torch.float64) for n in (6, 9, 12, 15)}
    rank_sims = {n: Rank2Sim(n, dtype=torch.float64) for n in (6, 9, 12, 15)}
    prepared = {fam["id"]: {} for fam in FAMILIES}
    for n, role, _count in specs:
        insts, raw = rows[(n, role)]
        for fam in FAMILIES:
            log(f"build {fam['id']} n={n} {role}")
            sp, insts_h, pulse, score = build_family_split(
                fam, insts, raw, n, role, sims[n], rank_sims[n], out,
            )
            prepared[fam["id"]][(n, role)] = {
                "split": sp,
                "insts": insts_h,
                "pulse": pulse,
                "score": score,
            }
        if (n, role) == (6, "train"):
            check_references(prepared)
    return prepared


def _e0_mu0(diag_row, n, lam) -> float:
    got = sparse_ground_state(np.asarray(diag_row, dtype=np.float64), n, lam)
    if got["residual"] > 1e-6:
        raise RuntimeError(f"μ=0 Lanczos residual {got['residual']}")
    return float(got["E0"])


def check_references(prepared) -> dict:
    """Diagonal identity versus the published first n=6 train rows.

    Gauss and unsat published E0 are the μ=0 Hamiltonians. This run trains
    both at μ=−0.25, so those checks prove the Z diagonal, then a second
    line proves the XX term actually moves E0. Rank2's published E0 is the
    μ=−0.25 Hamiltonian this arm trains.
    """
    report = {"ok": True, "rows": []}
    gauss = prepared["gauss"][(6, "train")]
    rank = prepared["rank2"][(6, "train")]
    unsat = prepared["unsat"][(6, "train")]
    g_seeds = gauss["split"].seeds[:4]
    if tuple(g_seeds) != REF_SEEDS_N6:
        raise RuntimeError(f"n=6 train seeds {g_seeds} != published {REF_SEEDS_N6}")
    if tuple(rank["split"].seeds[:4]) != REF_SEEDS_N6:
        raise RuntimeError("rank2 n=6 train seeds drifted")
    if tuple(unsat["split"].seeds[:4]) != REF_SEEDS_N6:
        raise RuntimeError("unsat n=6 train seeds drifted")

    gauss_mu0 = np.array([_e0_mu0(gauss["score"][i], 6, LAM_A) for i in range(4)])
    unsat_mu0 = np.array([_e0_mu0(unsat["score"][i], 6, LAM_C) for i in range(len(REF_E0_UNSAT_MU0))])
    pairs = (
        ("gauss μ=0 diagonal", gauss_mu0, REF_E0_GAUSS_MU0),
        ("rank2 μ=-0.25", np.asarray(rank["split"].circuit.e0[:4], dtype=np.float64), REF_E0_RANK2),
        ("unsat μ=0 diagonal", unsat_mu0, REF_E0_UNSAT_MU0),
    )
    for name, got, ref in pairs:
        err = float(np.max(np.abs(got - np.asarray(ref, dtype=np.float64))))
        ok = err < 1e-5
        report["rows"].append({"family": name, "max_abs": err, "ok": ok})
        log(f"  ref E0 {name} max|Δ|={err:.2e} {'ok' if ok else 'MISMATCH'}")
        if not ok:
            report["ok"] = False
    # XX is on: trained E0 must move relative to the μ=0 diagonal.
    gauss_shift = float(np.max(np.abs(np.asarray(gauss["split"].circuit.e0[:4]) - gauss_mu0)))
    unsat_shift = float(np.max(np.abs(np.asarray(unsat["split"].circuit.e0[: len(unsat_mu0)]) - unsat_mu0)))
    report["gauss_xx_shift"] = gauss_shift
    report["unsat_xx_shift"] = unsat_shift
    log(f"  XX shift |ΔE0| gauss={gauss_shift:.3f} unsat={unsat_shift:.3f}")
    if gauss_shift < 0.05 or unsat_shift < 0.05:
        report["ok"] = False
        report["rows"].append({"family": "xx shift", "ok": False})
    if not report["ok"]:
        raise RuntimeError(f"published E0 mismatch {report['rows']}")

    Hc = torch.as_tensor(rank["pulse"][:1], dtype=torch.float64)
    g = torch.tensor([0.13, -0.07], dtype=torch.float64)
    phi = torch.tensor([[0.0, -0.2, 0.04, 0.01], [0.0, 0.11, -0.02, 0.03]], dtype=torch.float64)
    e_rank = float(rank["split"].circuit.sim.energy(Hc, g, phi, LAM_B, 0.0, 0.0)[0])
    e_fock = float(FockSim(6, dtype=torch.float64).energy(Hc, g, phi, LAM_B)[0])
    report["mu0_abs"] = abs(e_rank - e_fock)
    log(f"  μ=0 rank2 vs Fock |ΔE|={report['mu0_abs']:.2e}")
    if report["mu0_abs"] > 1e-8:
        raise RuntimeError("rank2 μ=0 energy drifted from FockSim")

    # Unsat Z+X histogram at μ=0 matches the Pauli score on the same gate.
    base = unsat["split"].circuit.circ
    e_zx = float(ZXScore(base, 0.0).energy(g, phi, False, LAM_C)[0])
    amps = base.final_amps(g, phi, False)
    e_pauli = float(energy_from_amps(amps, base.score, LAM_C, base.n)[0])
    report["unsat_mu0_hist_abs"] = abs(e_zx - e_pauli)
    log(f"  unsat μ=0 Z+X vs Pauli |ΔE|={report['unsat_mu0_hist_abs']:.2e}")
    if report["unsat_mu0_hist_abs"] > 1e-8:
        raise RuntimeError("unsat histogram score drifted from the Pauli field")

    # μ=−0.25 histogram matches ⟨ψ|H|ψ⟩ on the first unsat row.
    zx = unsat["split"].circuit
    e_hist = float(zx.energy(g, phi, False, LAM_C)[0])
    psi = base.final_amps(g, phi, False)[0].detach().cpu().numpy()
    H = dense_rank2(np.asarray(unsat["score"][0], dtype=np.float64), 6, LAM_C, MU, KAPPA)
    e_psi = float(np.real(np.vdot(psi, H @ psi)))
    report["unsat_statevector_abs"] = abs(e_hist - e_psi)
    log(f"  unsat μ=-0.25 hist vs ⟨ψ|H|ψ⟩ |ΔE|={report['unsat_statevector_abs']:.2e}")
    if report["unsat_statevector_abs"] > 1e-7:
        raise RuntimeError("unsat XX histogram drifted from the statevector")
    return report


def unsat_histogram_check() -> dict:
    """Tiny synthetic row: μ=0 matches Pauli, μ=−0.25 matches ⟨ψ|H|ψ⟩."""
    rng = np.random.default_rng(0)
    n = 6
    score = rng.integers(0, 6, size=(1, 1 << n)).astype(np.float64)
    pulse = center_cn(score, n, clause_count(n))
    sim = FockSim(n, dtype=torch.float64)
    pieces = walsh_pieces(torch.as_tensor(pulse, dtype=torch.float64))
    circ = Circuit(
        sim,
        torch.as_tensor(pulse, dtype=torch.float64),
        torch.as_tensor(score, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        pieces=pieces,
    )
    g = torch.tensor([0.2, -0.1], dtype=torch.float64)
    phi = torch.tensor([[0.0, -0.15, 0.02, 0.0], [0.0, 0.08, -0.01, 0.02]], dtype=torch.float64)
    e_mu0 = float(ZXScore(circ, 0.0).energy(g, phi, False, LAM_C)[0])
    e_pauli = float(energy_from_amps(circ.final_amps(g, phi, False), circ.score, LAM_C, n)[0])
    e_mu = float(ZXScore(circ, MU).energy(g, phi, False, LAM_C)[0])
    psi = circ.final_amps(g, phi, False)[0].detach().cpu().numpy()
    H = dense_rank2(score[0], n, LAM_C, MU, KAPPA)
    e_psi = float(np.real(np.vdot(psi, H @ psi)))
    out = {"mu0_abs": abs(e_mu0 - e_pauli), "statevector_abs": abs(e_mu - e_psi)}
    if out["mu0_abs"] > 1e-8 or out["statevector_abs"] > 1e-7:
        raise RuntimeError(f"unsat histogram check {out}")
    return out


def _run_path(out: Path, family, L, method, seed) -> Path:
    return out / "runs" / f"{family}_L{L}_{method}_seed{seed}.json"


def _save_run(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = dict(rec)
    blob["theta"] = np.asarray(rec["theta"], dtype=np.float64).tolist()
    if "eval" in blob:
        blob["eval"] = {str(k): v for k, v in blob["eval"].items()}
    blob["protocol"] = "r5-pr19-early-stop-off"
    path.write_text(json.dumps(blob))


def _load_run(path: Path):
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("protocol") != "r5-pr19-early-stop-off":
        return None
    if int(rec.get("rounds_ran", -1)) != int(rec.get("rounds_budget", -2)):
        return None
    if rec.get("early_stopped", True):
        return None
    if int(rec.get("rounds_budget", -1)) != (ROUNDS_L2 if int(rec["L"]) <= 2 else ROUNDS_L16):
        return None
    if abs(float(rec.get("lr", 0)) - LR) > 1e-12:
        return None
    if abs(float(rec.get("lam", -1)) - next(f["lam"] for f in FAMILIES if f["id"] == rec["family"])) > 1e-12:
        return None
    rec["theta"] = np.asarray(rec["theta"], dtype=np.float64)
    return rec


def train_all(prepared, out: Path):
    runs = []
    for fam in FAMILIES:
        fid = fam["id"]
        lam = fam["lam"]
        head = fam["head"]
        train = [prepared[fid][(n, "train")]["split"] for n in (6, 9)]
        hold = [prepared[fid][(n, "hold")]["split"] for n in (6, 9)]
        if any(sp.n >= 12 for sp in train + hold):
            raise RuntimeError("train/hold leaked a transfer n")
        report = [prepared[fid][(n, role)]["split"] for n, role in (
            (6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer"),
        )]
        for L in LS:
            rounds = ROUNDS_L2 if L <= 2 else ROUNDS_L16
            fourier = {}
            for seed in SEEDS:
                path = _run_path(out, fid, L, "fourier_qaoa", seed)
                rec = _load_run(path)
                if rec is None:
                    log(f"train {fid} fourier_qaoa L={L} seed={seed} rounds={rounds}")
                    rec = fit_matched(
                        "fourier_qaoa", L, seed, train, hold, lam,
                        rounds, MAX_ITER, LR, None, None, fid,
                    )
                    rec["eval"] = evaluate("fourier_qaoa", rec["theta"], report, lam, L)
                    _save_run(path, rec)
                else:
                    log(f"resume {fid} fourier_qaoa L={L} seed={seed} tag={rec['best_tag']}")
                    if "eval" not in rec:
                        rec["eval"] = evaluate("fourier_qaoa", rec["theta"], report, lam, L)
                        _save_run(path, rec)
                fourier[seed] = rec
                runs.append(rec)
            for seed in SEEDS:
                path = _run_path(out, fid, L, head, seed)
                rec = _load_run(path)

                def _med(rec_, n, _fourier=fourier):
                    ev = rec_["eval"]
                    if n in ev:
                        return ev[n]["med_eps"]
                    return ev[str(n)]["med_eps"]

                ref = {n: _med(fourier[seed], n) for n in (6, 9)}
                theta_f = torch.as_tensor(np.asarray(fourier[seed]["theta"], dtype=np.float64))
                if head == "power_p":
                    theta0 = power_theta_from_fourier(theta_f, L)
                elif head == "walsh":
                    theta0 = walsh_theta_from_fourier(theta_f, L)
                else:
                    raise RuntimeError(head)
                if rec is None:
                    view0 = _hold_view(head, theta0, hold, lam, L)
                    for n in (6, 9):
                        if abs(view0[n]["med_eps"] - ref[n]) > 1e-5:
                            raise RuntimeError(
                                f"{fid} {head} warm n={n} {view0[n]['med_eps']} vs Fourier {ref[n]}"
                            )
                    log(f"train {fid} {head} L={L} seed={seed} rounds={rounds}")
                    rec = fit_matched(
                        head, L, seed, train, hold, lam,
                        rounds, MAX_ITER, LR, theta0, ref, fid,
                    )
                    rec["eval"] = evaluate(head, rec["theta"], report, lam, L)
                    _save_run(path, rec)
                else:
                    log(f"resume {fid} {head} L={L} seed={seed} tag={rec['best_tag']}")
                    if "eval" not in rec:
                        rec["eval"] = evaluate(head, rec["theta"], report, lam, L)
                        _save_run(path, rec)
                runs.append(rec)
    return runs


def _eval_get(rec, n, key):
    ev = rec["eval"]
    row = ev[n] if n in ev else ev[str(n)]
    return row[key]


def aggregate(runs):
    cells = []
    by = {}
    for rec in runs:
        by.setdefault((rec["family"], rec["method"], int(rec["L"])), []).append(rec)
    for fam in FAMILIES:
        for L in LS:
            q_rows = sorted(by[(fam["id"], "fourier_qaoa", L)], key=lambda r: int(r["seed"]))
            h_rows = sorted(by[(fam["id"], fam["head"], L)], key=lambda r: int(r["seed"]))
            for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
                q = np.array([_eval_get(r, n, "med_eps") for r in q_rows], dtype=np.float64)
                h = np.array([_eval_get(r, n, "med_eps") for r in h_rows], dtype=np.float64)
                qmed = float(np.median(q))
                hmed = float(np.median(h))
                R = float(hmed / qmed) if abs(qmed) > 1e-12 else float("nan")
                seed_R = [float(a / b) if abs(b) > 1e-12 else float("nan") for a, b in zip(h, q)]
                cells.append({
                    "id": fam["id"],
                    "champ": fam["champ"],
                    "head": fam["head"],
                    "L": int(L),
                    "n": int(n),
                    "role": role,
                    "med_eps": hmed,
                    "seed_std": float(np.std(h, ddof=1)),
                    "qaoa_med": qmed,
                    "qaoa_std": float(np.std(q, ddof=1)),
                    "R": R,
                    "mark": mark_of(R),
                    "per_seed_eps": h.tolist(),
                    "per_seed_qaoa": q.tolist(),
                    "per_seed_R": seed_R,
                    "seeds": [int(r["seed"]) for r in h_rows],
                    "n_params_head": int(h_rows[0]["n_params"]),
                    "n_params_qaoa": int(q_rows[0]["n_params"]),
                    "best_tags": [r["best_tag"] for r in h_rows],
                    "failures": int(sum(int(r.get("lbfgs_failures", 0)) for r in list(q_rows) + list(h_rows))),
                    "rounds_ran": [int(r["rounds_ran"]) for r in list(q_rows) + list(h_rows)],
                    "claimed_n12_R": CLAIMED_N12_R[fam["id"]],
                })
    return cells


def rank_transfer(cells):
    """Lower L=16 R at n=12, then at n=15. Stable on family id."""
    picked = []
    for fam in FAMILIES:
        r12 = next(c for c in cells if c["id"] == fam["id"] and c["L"] == 16 and c["n"] == 12)
        r15 = next(c for c in cells if c["id"] == fam["id"] and c["L"] == 16 and c["n"] == 15)
        picked.append({
            "id": fam["id"],
            "champ": fam["champ"],
            "head": fam["head"],
            "R12": r12["R"],
            "R15": r15["R"],
            "mark12": r12["mark"],
            "mark15": r15["mark"],
            "eps12": r12["med_eps"],
            "eps15": r15["med_eps"],
            "claimed_n12_R": CLAIMED_N12_R[fam["id"]],
        })
    ordered = sorted(picked, key=lambda r: (r["R12"], r["R15"], r["id"]))
    for i, row in enumerate(ordered, start=1):
        row["rank"] = i
    return ordered


def protocol_ok(runs) -> tuple[bool, str]:
    expect = len(FAMILIES) * len(LS) * len(SEEDS) * 2
    if len(runs) != expect:
        return False, f"expected {expect} fits, got {len(runs)}"
    for rec in runs:
        budget = ROUNDS_L2 if int(rec["L"]) <= 2 else ROUNDS_L16
        if int(rec["rounds_ran"]) != budget or rec.get("early_stopped", True):
            return False, f"{rec['family']} {rec['method']} L={rec['L']} s={rec['seed']} stopped early"
        if int(rec.get("lbfgs_failures", 0)) > 0:
            return False, f"{rec['family']} {rec['method']} L={rec['L']} s={rec['seed']} had LBFGS failures"
        if abs(float(rec.get("lr", 0.0)) - LR) > 1e-12:
            return False, f"{rec['family']} {rec['method']} lr drifted"
    return True, "full budget, no LBFGS failures, early stop off"


def quantum_table(prepared):
    rows = []
    for fam in FAMILIES:
        for n, role in ((6, "train"), (6, "hold"), (9, "train"), (9, "hold"), (12, "transfer"), (15, "transfer")):
            meta = prepared[fam["id"]][(n, role)]["split"].meta
            rows.append({
                "id": fam["id"],
                "n": n,
                "role": role,
                "peak": meta["peak_med"],
                "gap": meta["gap_med"],
                "entropy": meta["entropy_med"],
                "quantum": meta["quantum"],
                "residual_max": meta["residual_max"],
            })
    return rows


def walsh_note(prepared) -> dict:
    out = {}
    for fid, key in (("gauss", "pulse"), ("unsat", "pulse")):
        diag = prepared[fid][(6, "train")][key][0]
        out[fid] = walsh_mass_fractions(diag)
    return out


def render(cells, ranking, quantum, checks, walsh, runs, wall_s) -> str:
    ok, why = protocol_ok(runs)
    verdict = "PASS" if ok else "needs_edit"
    lines = []
    lines.append("# RESULTS — Round 5 matched bakeoff of the μ=−0.25 champs")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    if ok:
        winner = ranking[0]
        lines.append(
            "Transfer ranking at L=16, lower frozen R first at n=12 then n=15: "
            + ", ".join(
                f"{r['rank']}. {r['champ']} ({r['head']}) R12={r['R12']:.3f} ({r['mark12']}) R15={r['R15']:.3f} ({r['mark15']})"
                for r in ranking
            )
            + f". Winner: {winner['champ']} via {winner['head']}."
        )
    else:
        lines.append(f"Protocol broke: {why}. No ranking is declared.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "Shared budget is PR #19 / PR #23, not the budget that produced the prior claims. "
        f"Train B={B_TRAIN} per n, holdout B={B_HOLD} per n, seeds {list(SEEDS)}. "
        f"L-BFGS rounds {ROUNDS_L2}/{ROUNDS_L16} (L=2/L=16), max_iter {MAX_ITER}, lr={LR}. "
        "Early stop is off. Every head ran the full round count; a failed step restores θ and still counts. "
        "The checkpoint inside that budget minimizes the {6,9} holdout score "
        "(Fourier: mean of the two median residuals; champ: worst of the two ratios versus that seed's Fourier). "
        "n=12 and n=15 are scored after the freeze. Loss is the mean of the per-n mean residuals on n=6 and n=9. "
        "Champ heads warm-start from that seed's Fourier checkpoint "
        "(φ_k=−β k, c2=c3=0, p=0; Walsh copies γ onto degrees 1–4)."
    )
    lines.append("")
    lines.append(
        "PR #21's gauss claim (n=12 R=0.642) used 24 L=16 rounds, B_hold=256, a residual checkpoint, and n=15 B=16. "
        "PR #23's unsat claim (n=12 R=0.330) used this round count with an 8-round early stop. "
        "PR #22's ±1 claim (n=12 R=0.701) is this protocol. Those three numbers are prior claims. "
        "This table retrains all three and does not reuse their angles."
    )
    lines.append("")
    lines.append(
        "Every arm has intra-cavity XX at μ=−0.25 and κ=0. "
        "A: H = Σ J_abc ZZZ − 0.5 Σ X + μ Σ_cav Σ_{i<j} XX, J~N(0,1) on the seal-A support (salt 91). "
        "B: the same with J=±1. "
        "C: H = unsat − 0.75 Σ X + μ Σ_cav Σ_{i<j} XX, and U_c^X pulses centered C_n = (n/m) unsat − max. "
        "Gate sequence is BinH U_c^X (W = H^{⊗3}) then the Hamming-shell SNAP. "
        "⟨H_c⟩ is the Z histogram. ⟨X⟩ and ⟨XX⟩ are the per-cavity X histogram. needs YY: no."
    )
    lines.append("")
    lines.append(
        "Gate, unchanged: BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "R = median_seed(Med ε_head) / median_seed(Med ε_Fourier) on the same H. "
        "Quantum pass: median bitstring peak < 0.5 and median product gap (E_prod−E_0)/|E_0| ≥ 2%. "
        "Product states are XZ-plane. Ranking uses frozen L=16 R and does not drop a champ for the quantum note."
    )
    lines.append("")
    lines.append(
        f"Reference checks: {checks['rows']}. "
        f"XX |ΔE0| versus the μ=0 diagonal: gauss {checks['gauss_xx_shift']:.3f}, unsat {checks['unsat_xx_shift']:.3f}. "
        f"μ=0 rank2 versus Fock |ΔE|={checks['mu0_abs']:.2e}. "
        f"Unsat μ=0 histogram versus Pauli |ΔE|={checks['unsat_mu0_hist_abs']:.2e}. "
        f"Unsat μ=−0.25 histogram versus ⟨ψ|H|ψ⟩ |ΔE|={checks['unsat_statevector_abs']:.2e}."
    )
    lines.append("")
    lines.append("## Ranking table")
    lines.append("")
    lines.append(
        "n=6 and n=9 are the selection holdout, so those medians are slightly optimistic. "
        "n=12 and n=15 are not. Lower ε is better. Rank is filled only on the L=16 n=12 row."
    )
    lines.append("")
    lines.append("| rank | champ | head | L | n | role | Med ε | seed std | Fourier Med ε | R | mark | med peak | med prod gap | quantum |")
    lines.append("|-----:|:------|:-----|--:|--:|:-----|------:|---------:|--------------:|--:|:-----|---------:|-------------:|:--------|")
    rank_of = {r["id"]: r["rank"] for r in ranking} if ok else {f["id"]: "" for f in FAMILIES}
    for c in cells:
        role_q = "hold" if c["n"] in (6, 9) else "transfer"
        q = next(row for row in quantum if row["id"] == c["id"] and row["n"] == c["n"] and row["role"] == role_q)
        rk = rank_of[c["id"]] if c["L"] == 16 and c["n"] == 12 else ""
        lines.append(
            f"| {rk} | {c['champ']} | {c['head']} | {c['L']} | {c['n']} | {c['role']} | "
            f"{c['med_eps']:.4f} | {c['seed_std']:.4f} | {c['qaoa_med']:.4f} | {c['R']:.3f} | {c['mark']} | "
            f"{q['peak']:.3f} | {q['gap']:.3f} | {'yes' if q['quantum'] else 'no'} |"
        )
    lines.append("")
    lines.append("Per-seed median ε (head) and per-seed R, L=16 transfer:")
    lines.append("")
    for c in cells:
        if c["L"] != 16 or c["n"] not in (12, 15):
            continue
        eps = ", ".join(f"{v:.4f}" for v in c["per_seed_eps"])
        rs = ", ".join(f"{v:.3f}" for v in c["per_seed_R"])
        lines.append(f"- {c['champ']} n={c['n']} seeds {c['seeds']}: ε {eps}; R {rs}")
    lines.append("")
    lines.append("Prior n=12 claims, not used for the rank: " + ", ".join(
        f"{fid} {CLAIMED_N12_R[fid]:.3f}" for fid in ("gauss", "unsat", "rank2")
    ) + ".")
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append("Split medians (peak, product gap). The ranking table already carries the scored split.")
    lines.append("")
    for q in quantum:
        lines.append(
            f"- {q['id']} {q['role']} n={q['n']}: peak {q['peak']:.3f}, gap {q['gap']:.3f}, "
            f"S {q['entropy']:.3f}, {'pass' if q['quantum'] else 'FAIL'}"
        )
    lines.append("")
    g = walsh.get("gauss", {})
    u = walsh.get("unsat", {})
    lines.append(
        f"Walsh mass on the first n=6 train pulse: gauss degree-3 fraction {g.get('frac_deg_3', float('nan')):.3f} "
        f"(degrees 1–4 share {g.get('frac_deg_1_to_4', float('nan')):.3f}), so the champ head is power_p. "
        f"Unsat degrees 1–4 share {u.get('frac_deg_1_to_4', float('nan')):.3f} "
        f"(degree 3 is {u.get('frac_deg_3', float('nan')):.3f}), so the champ head is Walsh."
    )
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "PASS when every Fourier and champ fit ran the full round budget with early stop off and no LBFGS failure, "
        "the published diagonal checks matched, the XX term moved E0, the Z+X histogram matched ⟨ψ|H|ψ⟩, "
        "and the L=16 transfer ranking is filled for n=12 then n=15. "
        "needs_edit when that protocol broke. The quantum note is reported and does not change the rank."
    )
    lines.append("")
    lines.append(f"Protocol status: {why}.")
    lines.append("")
    fails = sum(int(r.get("lbfgs_failures", 0)) for r in runs)
    lines.append(f"LBFGS failures across fits: {fails}. Wall clock {wall_s:.1f}s. n=21 was not loaded.")
    lines.append("")
    lines.append("Branch `cursor/r5-champ-bakeoff-bd3c`.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(out: Path, cells, ranking, quantum, checks, walsh, runs, wall_s, circuit_check):
    ok, why = protocol_ok(runs)
    payload = {
        "verdict": "PASS" if ok else "needs_edit",
        "protocol_ok": ok,
        "protocol_note": why,
        "ranking": ranking,
        "cells": cells,
        "quantum": quantum,
        "checks": checks,
        "claimed_n12_R": CLAIMED_N12_R,
        "walsh": {
            k: {"frac_deg_3": v.get("frac_deg_3"), "frac_deg_1_to_4": v.get("frac_deg_1_to_4")}
            for k, v in walsh.items()
        },
        "circuit_check": circuit_check,
        "rounds_L2": ROUNDS_L2,
        "rounds_L16": ROUNDS_L16,
        "early_stop": False,
        "seeds": list(SEEDS),
        "B_train": B_TRAIN,
        "B_hold": B_HOLD,
        "B_n12": B_N12,
        "B_n15": B_N15,
        "mu": MU,
        "wall_s": wall_s,
        "n21_loaded": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    text = render(cells, ranking, quantum, checks, walsh, runs, wall_s)
    (out / "RESULTS.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=S.OUT / "quantum_h_explore" / "matched_budget_r5")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    log("self-check BinH Uc^X")
    circuit_check = fock_self_check("cpu")
    log("self-check rank2 Z+X histogram")
    rank_check = rank2_self_check()
    log("self-check unsat Z+X histogram")
    unsat_check = unsat_histogram_check()
    log(
        "self-check ok",
        {
            "pZ": circuit_check.get("circuit_max_abs_pZ"),
            "pX": circuit_check.get("circuit_max_abs_pX"),
            "rank2": rank_check.get("statevector_max_abs"),
            "unsat": unsat_check,
        },
    )
    prepared = prepare(args.out)
    checks = check_references(prepared)
    if args.check_only:
        log(f"check-only done in {time.time()-t0:.1f}s")
        return
    runs = train_all(prepared, args.out)
    cells = aggregate(runs)
    ranking = rank_transfer(cells)
    quantum = quantum_table(prepared)
    walsh = walsh_note(prepared)
    wall = time.time() - t0
    text = write_outputs(args.out, cells, ranking, quantum, checks, walsh, runs, wall, {
        "circuit_max_abs_pZ": circuit_check.get("circuit_max_abs_pZ"),
        "circuit_max_abs_pX": circuit_check.get("circuit_max_abs_pX"),
        "rank2_statevector_max_abs": rank_check.get("statevector_max_abs"),
        "unsat_histogram": unsat_check,
        "needs_YY": False,
    })
    log(text.splitlines()[2] if len(text.splitlines()) > 2 else "done")
    ok, why = protocol_ok(runs)
    log(("PASS " if ok else "needs_edit ") + why)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
