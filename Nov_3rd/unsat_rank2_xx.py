"""Unsat λ=0.75 plus the Rank2 intra-cavity XX term.

    H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j

λ = 0.75, μ = −0.25, κ = 0. The gate is the locked Nov_3rd sequence:
U_c^X still pulses C_n from the unsat counts, then the Hamming-shell SNAP.
⟨unsat⟩ comes from the Z histogram. ⟨X⟩ and ⟨XX⟩ come from the X histogram
(final W omitted). No Y correlator enters H.

The pass gate is the frozen n=12 L=16 Walsh ratio against same-depth Fourier
QAOA on this H, compared with the published μ=0 unsat Walsh seal (PR #19,
R = 0.940). power_p is the scalar-γ control and does not gate the call.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from Nov_3rd.quantum_h_rank2 import (
    CLEAR_DROP,
    dense_rank2,
    local_x_eigs,
    sparse_ground_rank2,
)
from Nov_3rd.quantum_h_transfer import Circuit
from Nov_3rd.seal_unsat import BEAT_R, FLIP_R, REVERSE_R, TIE_R, mark_of

LAM = 0.75
MU = -0.25
KAPPA = 0.0

# PR #19 seal, B_train=128, ratio of median-over-seeds of median ε.
# μ = 0 on the same unsat diagonal. Not retrained here.
PUBLISHED_WALSH_R = {
    (2, 6): 0.670,
    (2, 9): 0.705,
    (2, 12): 0.761,
    (2, 15): 0.733,
    (16, 6): 0.925,
    (16, 9): 0.895,
    (16, 12): 0.940,
    (16, 15): 0.914,
}
PUBLISHED_POWER_R = {
    (2, 6): 0.915,
    (2, 9): 0.961,
    (2, 12): 1.000,
    (2, 15): 0.994,
    (16, 6): 0.978,
    (16, 9): 0.951,
    (16, 12): 0.990,
    (16, 15): 0.992,
}
PUBLISHED_N12_L16 = float(PUBLISHED_WALSH_R[(16, 12)])
# PR #18 used this drop before calling a μ move clear.
MIN_DROP = float(CLEAR_DROP)
# Same product-state bar as the Rank2 μ transfer.
PROD_GAP_MIN = 0.02
# Round 7 seal: this fraction of n=12 and of n=15 instances must sit at or above the bar.
GAP_COVERAGE = 0.95


def cavity_x_strings(pX: torch.Tensor, n_cav: int, sx8: torch.Tensor, xx8: torch.Tensor):
    """Per-cavity X histogram → Σ⟨X⟩, Σ_cav Σ_{i<j}⟨XX⟩.

    ``pX`` is bitstring order, var 0 = MSB, cavity 0 in the high bits.
    """
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
    """⟨H⟩ from the Z histogram and the X histogram. The gate is ``circ``."""

    def __init__(self, circ: Circuit, mu: float):
        self.circ = circ
        self.mu = float(mu)
        self.n = int(circ.n)
        self.e0 = circ.e0
        self.sim = circ.sim
        sx, xx, _xxx = local_x_eigs()
        self.sx8 = torch.as_tensor(sx, dtype=circ.sim.rdtype, device=circ.sim.device)
        self.xx8 = torch.as_tensor(xx, dtype=circ.sim.rdtype, device=circ.sim.device)

    def energy(self, gamma: torch.Tensor, phi: torch.Tensor, graded: bool, lam: float) -> torch.Tensor:
        pZ, pX = self.circ.bit_histograms(gamma, phi, graded)
        e_c = (pZ * self.circ.score).sum(dim=-1)
        e_sx, e_xx = cavity_x_strings(pX, self.circ.sim.C, self.sx8, self.xx8)
        return e_c - float(lam) * e_sx + self.mu * e_xx


def plus_energy(diag_row: np.ndarray, n: int, lam: float, mu: float) -> float:
    """⟨+|H|+⟩. |+⟩ has ⟨X_i⟩=1 and ⟨X_i X_j⟩=1 on every intra-cavity pair.

    There are n pairs (3 per cavity, n/3 cavities), so the XX shift is μ n.
    """
    n = int(n)
    if n % 3 != 0:
        raise ValueError(n)
    return float(np.mean(diag_row) - float(lam) * n + float(mu) * n)


def product_energy_unsat(theta: np.ndarray, diag: np.ndarray, n: int, lam: float, mu: float) -> float:
    """Real XZ product state. ⟨Z⟩=cos θ, ⟨X⟩=sin θ, ⟨Y⟩=0.

    Amplitudes are cos(θ/2)|0⟩ + sin(θ/2)|1⟩, so the Z distribution is a
    product of cos²(θ/2) and sin²(θ/2). H has no Y, and at μ≤0, λ≥0 it is
    stoquastic, so this family contains the relevant product states.
    """
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
    """Lowest energy among a fixed set of XZ product starts, then L-BFGS on the best."""
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


def _ground_job(payload):
    diag, n, lam, mu, seed, want_product = payload
    row = sparse_ground_rank2(np.asarray(diag, dtype=np.float64), int(n), float(lam), float(mu), KAPPA)
    if row["residual"] > 1e-5:
        raise RuntimeError(f"n={n} Lanczos residual {row['residual']:.2e}")
    e_prod = float("nan")
    if want_product:
        e_prod = best_product_unsat(diag, int(n), float(lam), float(mu), seed=int(seed))
    return (
        float(row["E0"]),
        float(row["peak"]),
        float(row["entropy"]),
        float(row["residual"]),
        float(e_prod),
    )


def batch_ground(diags: np.ndarray, n: int, lam: float, mu: float, workers: int, seeds=None) -> dict:
    """Lanczos E0 of diag(unsat) − λ ΣX + μ ΣXX, plus an XZ product upper bound."""
    diags = np.asarray(diags, dtype=np.float64)
    if seeds is None:
        seeds = np.arange(diags.shape[0], dtype=np.int64)
    payloads = [
        (diags[i], int(n), float(lam), float(mu), int(seeds[i]), True)
        for i in range(diags.shape[0])
    ]
    e0 = np.empty(len(payloads), dtype=np.float64)
    peak = np.empty_like(e0)
    entropy = np.empty_like(e0)
    residual = np.empty_like(e0)
    e_prod = np.empty_like(e0)
    if workers <= 1 or len(payloads) < 4:
        rows = [_ground_job(p) for p in payloads]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            rows = list(pool.map(_ground_job, payloads))
    for i, (ev, pk, ent, res, ep) in enumerate(rows):
        e0[i] = ev
        peak[i] = pk
        entropy[i] = ent
        residual[i] = res
        e_prod[i] = ep
    denom = np.maximum(np.abs(e0), 1e-8)
    prod_gap = (e_prod - e0) / denom
    return {
        "E0": e0,
        "peak": peak,
        "entropy": entropy,
        "residual": residual,
        "E_prod": e_prod,
        "prod_gap": prod_gap,
    }


def dense_gap(diag: np.ndarray, n: int, lam: float, mu: float) -> float:
    """|E0_Lanczos − E0_dense| for one unsat diagonal."""
    got = sparse_ground_rank2(np.asarray(diag, dtype=np.float64), int(n), float(lam), float(mu), KAPPA)
    if got["residual"] > 1e-6:
        raise RuntimeError(f"Lanczos residual {got['residual']}")
    ev = float(np.linalg.eigvalsh(dense_rank2(np.asarray(diag, dtype=np.float64), int(n), float(lam), float(mu), KAPPA))[0])
    return abs(float(got["E0"]) - ev)


def statevector_energy(psi: np.ndarray, diag: np.ndarray, n: int, lam: float, mu: float) -> float:
    """⟨ψ|H|ψ⟩ in the Z basis. psi is bitstring order."""
    H = dense_rank2(np.asarray(diag, dtype=np.float64), int(n), float(lam), float(mu), KAPPA)
    v = np.asarray(psi, dtype=np.complex128).reshape(-1)
    return float(np.real(np.vdot(v, H @ v)))


def clearly_below(R: float, published: float, seed_Rs, min_drop: float = MIN_DROP) -> dict:
    """Whether the ratio-of-medians and the median per-seed ratio both drop by ``min_drop``.

    The headline number is the ratio of medians, which is what PR #19 published
    as 0.940. One seed sitting on the Fourier point does not count as a clear
    move unless the median seed drops by the same bar.
    """
    seed_Rs = [float(x) for x in seed_Rs]
    if len(seed_Rs) < 1 or not all(math.isfinite(r) for r in seed_Rs):
        raise ValueError("per-seed ratios")
    if not math.isfinite(float(R)) or not math.isfinite(float(published)):
        raise ValueError("R")
    seed_med = float(np.median(seed_Rs))
    drop = float(published) - float(R)
    seed_drop = float(published) - seed_med
    clear = bool(drop + 1e-12 >= float(min_drop) and seed_drop + 1e-12 >= float(min_drop))
    return {
        "drop": float(drop),
        "seed_median": seed_med,
        "seed_drop": float(seed_drop),
        "clear": clear,
        "min_drop": float(min_drop),
        "published": float(published),
        "R": float(R),
    }


def decide_unsat_xx(
    cells: dict,
    quantum: dict,
    published_R: float = PUBLISHED_N12_L16,
    min_drop: float = MIN_DROP,
) -> dict:
    """PASS / needs_edit / KILL for μ=−0.25 on the unsat Walsh champ.

    PASS: frozen n=12 L=16 Walsh R is a BEAT against Fourier on this H and
    clearly below the published μ=0 ratio 0.940 (drop ≥ 0.02 on both the
    ratio of medians and the median per-seed ratio); at least two of three
    seeds beat; the n=12 ground state is not a bitstring and sits at least
    2% below the best XZ product state; n=15 does not flip past 1.10.

    A drop under 0.02, with no kill, is a null: needs_edit.

    KILL: a gated ground state is a bitstring; n=12 loses to Fourier and is
    not below the μ=0 ratio; or the L=16 Walsh advantage reverses from n=6
    to n=12.
    """
    for n in (6, 9, 12, 15):
        if n not in quantum:
            return {
                "verdict": "needs_edit",
                "reason": f"quantumness missing at n={n}",
                "null": False,
            }
    for n in (6, 9, 12):
        peak = float(quantum[n]["peak_med"])
        if peak >= 0.5:
            return {
                "verdict": "KILL",
                "reason": f"n={n} ground state is a bitstring (peak med {peak:.3f} ≥ 0.5).",
                "null": False,
            }

    cell = cells[("walsh", 16, 12)]
    R = float(cell["R"])
    seed_Rs = [float(x) for x in cell["seed_Rs"]]
    if len(seed_Rs) < 3:
        return {
            "verdict": "needs_edit",
            "reason": f"n=12 L=16 Walsh has {len(seed_Rs)} seeds; the comparison needs at least 3.",
            "null": False,
        }
    clear = clearly_below(R, published_R, seed_Rs, min_drop)
    r6 = float(cells[("walsh", 16, 6)]["R"])
    r9 = float(cells[("walsh", 16, 9)]["R"])
    r15 = float(cells[("walsh", 16, 15)]["R"])
    mark6 = cells[("walsh", 16, 6)]["mark"]
    mark9 = cells[("walsh", 16, 9)]["mark"]
    info = {
        "clear": clear,
        "R12": R,
        "R6": r6,
        "R9": r9,
        "R15": r15,
        "published_R": float(published_R),
        "null": False,
    }
    if mark6 == "BEAT" and mark9 == "UNDER" and R >= REVERSE_R:
        info["verdict"] = "KILL"
        info["reason"] = (
            f"L=16 Walsh advantage reverses: n=6 R={r6:.3f}, n=9 R={r9:.3f}, "
            f"n=12 R={R:.3f} ≥ {REVERSE_R}."
        )
        return info
    if R > TIE_R and R + 1e-12 >= float(published_R):
        info["verdict"] = "KILL"
        info["reason"] = (
            f"frozen n=12 Walsh R={R:.3f} does not beat Fourier QAOA and is not below "
            f"the μ=0 unsat Walsh ratio {float(published_R):.3f}."
        )
        return info

    n_seed_beat = sum(r <= BEAT_R for r in seed_Rs)
    beat = R <= BEAT_R
    peak12 = float(quantum[12]["peak_med"])
    gap12 = float(quantum[12]["prod_gap_med"])
    quantum_ok = peak12 < 0.5 and gap12 + 1e-12 >= PROD_GAP_MIN
    n15_ok = r15 < FLIP_R and float(quantum[15]["peak_med"]) < 0.5
    info["n_seed_beat"] = int(n_seed_beat)
    info["quantum_ok"] = bool(quantum_ok)
    info["n15_ok"] = bool(n15_ok)
    if clear["clear"] and beat and n_seed_beat >= 2 and quantum_ok and n15_ok:
        info["verdict"] = "PASS"
        info["reason"] = (
            f"frozen n=12 L=16 Walsh R={R:.3f} is a BEAT and clearly below the μ=0 "
            f"unsat Walsh ratio {float(published_R):.3f} "
            f"(drop {clear['drop']:.3f}, seed-median R={clear['seed_median']:.3f}). "
            f"{n_seed_beat}/{len(seed_Rs)} seeds beat. "
            f"n=15 R={r15:.3f}."
        )
        return info
    if not clear["clear"]:
        info["verdict"] = "needs_edit"
        info["null"] = True
        info["reason"] = (
            f"null: frozen n=12 Walsh R={R:.3f} is not clearly below the μ=0 unsat Walsh "
            f"ratio {float(published_R):.3f} (drop {clear['drop']:.3f}, "
            f"seed-median drop {clear['seed_drop']:.3f}, bar {float(min_drop):.2f})."
        )
        return info
    missed = []
    if not beat:
        missed.append(f"n=12 R={R:.3f} is above {BEAT_R}")
    if n_seed_beat < 2:
        missed.append(
            f"only {n_seed_beat}/{len(seed_Rs)} seeds beat at n=12 L=16 "
            f"(per-seed R={', '.join(f'{r:.3f}' for r in seed_Rs)})"
        )
    if not quantum_ok:
        missed.append(
            f"n=12 quantumness fails (peak={peak12:.3f}, product gap={gap12:.3f})"
        )
    if not n15_ok:
        missed.append(
            f"n=15 does not hold (R={r15:.3f}, peak={float(quantum[15]['peak_med']):.3f})"
        )
    info["verdict"] = "needs_edit"
    info["reason"] = "; ".join(missed) if missed else "bars not all met"
    return info


def paired_instance_counts(eps_method, eps_base, tol: float = 0.0) -> dict:
    """Instance-paired comparison of two ε vectors on one frozen seed.

    A win is method ε strictly below the Fourier ε by more than ``tol``.
    """
    a = np.asarray(eps_method, dtype=np.float64).reshape(-1)
    b = np.asarray(eps_base, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 1:
        raise ValueError(f"paired shapes {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite ε")
    delta = a - b
    win = int(np.sum(delta < -float(tol)))
    tie = int(np.sum(np.abs(delta) <= float(tol)))
    loss = int(np.sum(delta > float(tol)))
    if win + tie + loss != int(a.size):
        raise RuntimeError("paired partition")
    return {
        "win": win,
        "tie": tie,
        "loss": loss,
        "n": int(a.size),
        "win_frac": float(win / int(a.size)),
    }


def paired_seed_table(eps_method, eps_base, tol: float = 0.0) -> dict:
    """Per-seed paired counts. A majority seed has strictly more wins than half the batch."""
    if len(eps_method) != len(eps_base) or len(eps_method) < 1:
        raise ValueError("paired seeds")
    rows = [paired_instance_counts(a, b, tol=tol) for a, b in zip(eps_method, eps_base)]
    return {
        "per_seed": rows,
        "n_majority": int(sum(1 for r in rows if r["win"] * 2 > r["n"])),
        "n_seeds": len(rows),
        "win": int(sum(r["win"] for r in rows)),
        "tie": int(sum(r["tie"] for r in rows)),
        "loss": int(sum(r["loss"] for r in rows)),
        "n": int(sum(r["n"] for r in rows)),
    }


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials. Descriptive on pooled pairs."""
    k = int(k)
    n = int(n)
    if n < 1 or k < 0 or k > n:
        raise ValueError(f"wilson k={k} n={n}")
    p = k / n
    z2 = float(z) ** 2
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = float(z) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return float(center - half), float(center + half)


def _median_ratio(eps_method, eps_base) -> float:
    med_m = float(np.median([float(np.median(a)) for a in eps_method]))
    med_b = float(np.median([float(np.median(b)) for b in eps_base]))
    if abs(med_b) < 1e-15:
        return float("nan")
    return float(med_m / med_b)


def bootstrap_ratio(eps_method, eps_base, n_boot: int = 4000, seed: int = 5136, mode: str = "instance") -> dict:
    """95% interval for the ratio of medians.

    The point is median-over-seeds of median-over-instances, the same R as the
    transfer table. ``mode='instance'`` resamples instances inside each seed
    and keeps every seed. ``mode='seed'`` resamples seeds and keeps their
    instances. ``mode='both'`` resamples both.
    """
    if mode not in ("instance", "seed", "both"):
        raise ValueError(mode)
    methods = [np.asarray(a, dtype=np.float64).reshape(-1) for a in eps_method]
    bases = [np.asarray(b, dtype=np.float64).reshape(-1) for b in eps_base]
    if len(methods) < 2 or len(methods) != len(bases):
        raise ValueError("bootstrap seeds")
    width = methods[0].size
    for a, b in zip(methods, bases):
        if a.shape != b.shape or int(a.size) != int(width) or int(a.size) < 2:
            raise ValueError("bootstrap instance axis")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("non-finite ε")
    point = _median_ratio(methods, bases)
    if not math.isfinite(point):
        raise RuntimeError("ratio of medians is not finite")
    rng = np.random.default_rng(int(seed))
    n_seed = len(methods)
    draws = np.empty(int(n_boot), dtype=np.float64)
    for t in range(int(n_boot)):
        if mode == "instance":
            pick = range(n_seed)
        else:
            pick = rng.integers(0, n_seed, size=n_seed)
        ms = []
        bs = []
        for s in pick:
            if mode == "seed":
                ms.append(methods[int(s)])
                bs.append(bases[int(s)])
            else:
                idx = rng.integers(0, width, size=width)
                ms.append(methods[int(s)][idx])
                bs.append(bases[int(s)][idx])
        draws[t] = _median_ratio(ms, bs)
    ok = draws[np.isfinite(draws)]
    if int(ok.size) < int(n_boot) * 0.95:
        raise RuntimeError("bootstrap ratios were non-finite")
    lo, hi = np.quantile(ok, [0.025, 0.975])
    return {
        "R": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(ok.size),
        "n_seeds": int(n_seed),
        "n_inst": int(width),
        "seed": int(seed),
        "mode": mode,
    }


def _n18_sentence(n18) -> str:
    if not n18:
        return ""
    status = str(n18.get("status", ""))
    if status == "probe" and n18.get("R") is not None:
        peak = n18.get("peak_med")
        gap = n18.get("prod_gap_med")
        return (
            f"n=18 probe L=16 Walsh R={float(n18['R']):.3f} "
            f"Med ε={float(n18['med']):.4f} on B={int(n18['B'])} "
            f"(peak {float(peak):.3f}, product gap {float(gap):.3f}); exploratory, not gated."
        )
    detail = n18.get("detail") or "Lanczos or the product bound did not finish in the probe budget."
    return f"n=18 probe was not scored ({detail})."


def _attach_n18(info: dict, n18) -> dict:
    info["n18"] = n18
    sentence = _n18_sentence(n18)
    if sentence:
        info["reason"] = info["reason"].rstrip() + " " + sentence
    return info


def product_gap_tail(prod_gap: np.ndarray, peak: np.ndarray | None = None) -> dict:
    """Instance counts under the product-gap bars 0.02 and 0.05.

    An instance is under a bar when ``(E_prod − E_0) / |E_0|`` is strictly below it.
    ``frac_gap_ge_0p02`` is the complement of the 0.02 count, which is the Round 7 coverage.
    """
    gap = np.asarray(prod_gap, dtype=np.float64).reshape(-1)
    n = int(gap.size)
    if n < 1:
        raise ValueError("empty product-gap sample")
    n02 = int(np.sum(gap < PROD_GAP_MIN))
    n05 = int(np.sum(gap < 0.05))
    out = {
        "gap_min": float(np.min(gap)),
        "gap_med": float(np.median(gap)),
        "n_inst": n,
        "n_gap_lt_0p02": n02,
        "n_gap_lt_0p05": n05,
        "frac_gap_lt_0p02": float(n02 / n),
        "frac_gap_lt_0p05": float(n05 / n),
        "n_gap_ge_0p02": int(n - n02),
        "frac_gap_ge_0p02": float((n - n02) / n),
    }
    if peak is not None:
        out["peak_max"] = float(np.max(np.asarray(peak, dtype=np.float64)))
    return out


def _coverage_misses(quantum: dict, gap_coverage: float | None) -> list:
    """Round 7 gate. Empty when ``gap_coverage`` is unset, so earlier seals stay put."""
    if gap_coverage is None:
        return []
    misses = []
    need = float(gap_coverage)
    for n in (12, 15):
        row = quantum[n]
        if "frac_gap_ge_0p02" not in row or "n_inst" not in row:
            misses.append(f"quantumness fail: n={n} instance product-gap counts are missing")
            continue
        frac = float(row["frac_gap_ge_0p02"])
        n_inst = int(row["n_inst"])
        if frac + 1e-12 < need:
            misses.append(
                f"quantumness fail: n={n} product gap ≥ {PROD_GAP_MIN:.2f} on {frac:.1%} of instances "
                f"({int(row['n_gap_lt_0p02'])}/{n_inst} below {PROD_GAP_MIN:.2f}, "
                f"frac < 0.02 = {float(row['frac_gap_lt_0p02']):.3f}, "
                f"frac < 0.05 = {float(row['frac_gap_lt_0p05']):.3f}); "
                f"need ≥ {need:.0%}"
            )
    return misses


def _coverage_sentence(quantum: dict, gap_coverage: float | None) -> str:
    if gap_coverage is None:
        return ""
    bits = []
    for n in (12, 15):
        row = quantum[n]
        bits.append(
            f"n={n} frac gap<0.02 = {float(row['frac_gap_lt_0p02']):.3f} "
            f"({int(row['n_gap_lt_0p02'])}/{int(row['n_inst'])}), "
            f"frac gap<0.05 = {float(row['frac_gap_lt_0p05']):.3f}"
        )
    return (
        " Instance product-gap tails: "
        + "; ".join(bits)
        + f". At least {float(gap_coverage):.0%} of each sealed size has gap ≥ {PROD_GAP_MIN:.2f}."
    )


def decide_deepen(
    cells: dict,
    quantum: dict,
    paired: dict,
    ci: dict,
    n18=None,
    published_R: float = PUBLISHED_N12_L16,
    min_drop: float = MIN_DROP,
    gap_coverage: float | None = None,
) -> dict:
    """Round-5 seal of the μ=−0.25 Walsh freeze.

    PASS when the frozen L=16 Walsh ratios at n=12 and n=15 both stay BEAT,
    the instance-bootstrap 95% interval lies inside BEAT, every seed beats
    its own Fourier, every seed wins a strict instance majority, the n=12
    drop versus the μ=0 unsat seal stays at least 0.02, and the ground states
    stay quantum. n=18 is reported and does not move the call.

    ``gap_coverage`` is the Round 7 instance gate. When set, n=12 and n=15
    each need that fraction of instances at product gap ≥ 0.02. A shortfall
    is needs_edit and the reason starts with ``quantumness fail``. Leaving it
    unset keeps the median-only quantumness check used by the earlier seals.

    KILL matches the three-seed rule: a bitstring ground state at n=6, 9, or
    12; n=12 loses to Fourier and is not below the μ=0 ratio; or the L=16
    Walsh advantage reverses from n=6 to n=12.
    """
    return _attach_n18(
        _decide_deepen_body(cells, quantum, paired, ci, published_R, min_drop, gap_coverage),
        n18,
    )


def _decide_deepen_body(cells, quantum, paired, ci, published_R, min_drop, gap_coverage=None) -> dict:
    for n in (6, 9, 12, 15):
        if n not in quantum:
            return {
                "verdict": "needs_edit",
                "null": False,
                "reason": f"quantumness missing at n={n}",
            }
    try:
        cell12 = cells[("walsh", 16, 12)]
        cell15 = cells[("walsh", 16, 15)]
        r6 = float(cells[("walsh", 16, 6)]["R"])
        r9 = float(cells[("walsh", 16, 9)]["R"])
        mark6 = cells[("walsh", 16, 6)]["mark"]
        mark9 = cells[("walsh", 16, 9)]["mark"]
    except KeyError as exc:
        return {"verdict": "needs_edit", "null": False, "reason": f"missing cell {exc}"}
    seed12 = [float(x) for x in cell12["seed_Rs"]]
    seed15 = [float(x) for x in cell15["seed_Rs"]]
    n_seeds = len(seed12)
    if n_seeds < 5 or len(seed15) != n_seeds:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": f"deepen needs at least 5 matched seeds; got {n_seeds} and {len(seed15)}.",
        }
    for n in (6, 9, 12):
        peak = float(quantum[n]["peak_med"])
        if peak >= 0.5:
            return {
                "verdict": "KILL",
                "null": False,
                "reason": f"n={n} ground state is a bitstring (peak med {peak:.3f} ≥ 0.5).",
            }
    R = float(cell12["R"])
    r15 = float(cell15["R"])
    if mark6 == "BEAT" and mark9 == "UNDER" and R >= REVERSE_R:
        return {
            "verdict": "KILL",
            "null": False,
            "R12": R,
            "R15": r15,
            "reason": (
                f"L=16 Walsh advantage reverses: n=6 R={r6:.3f}, n=9 R={r9:.3f}, "
                f"n=12 R={R:.3f} ≥ {REVERSE_R}."
            ),
        }
    if R > TIE_R and R + 1e-12 >= float(published_R):
        return {
            "verdict": "KILL",
            "null": False,
            "R12": R,
            "R15": r15,
            "reason": (
                f"frozen n=12 Walsh R={R:.3f} does not beat Fourier QAOA and is not below "
                f"the μ=0 unsat Walsh ratio {float(published_R):.3f}."
            ),
        }
    clear = clearly_below(R, published_R, seed12, min_drop)
    info = {
        "clear": clear,
        "R12": R,
        "R15": r15,
        "R6": r6,
        "R9": r9,
        "published_R": float(published_R),
        "n_seeds": int(n_seeds),
        "null": False,
    }
    if not clear["clear"]:
        info["verdict"] = "needs_edit"
        info["null"] = True
        info["reason"] = (
            f"null: frozen n=12 Walsh R={R:.3f} is not clearly below the μ=0 unsat Walsh "
            f"ratio {float(published_R):.3f} (drop {clear['drop']:.3f}, "
            f"seed-median drop {clear['seed_drop']:.3f}, bar {float(min_drop):.2f})."
        )
        return info

    def _pull(store, n):
        key = ("walsh", 16, n)
        if key not in store:
            return None
        return store[key]

    missed = []
    ci12 = _pull(ci, 12)
    ci15 = _pull(ci, 15)
    p12 = _pull(paired, 12)
    p15 = _pull(paired, 15)
    if ci12 is None or ci15 is None:
        missed.append("instance bootstrap CI missing at n=12 or n=15")
    else:
        for label, block, point in (("n=12", ci12, R), ("n=15", ci15, r15)):
            if abs(float(block["R"]) - float(point)) > 1e-6:
                missed.append(f"{label} CI point {float(block['R']):.4f} drifted from R {float(point):.4f}")
            if float(block["hi"]) > BEAT_R + 1e-12:
                missed.append(
                    f"{label} instance 95% CI upper {float(block['hi']):.3f} is above {BEAT_R}"
                )
        info["ci12"] = [float(ci12["lo"]), float(ci12["hi"])]
        info["ci15"] = [float(ci15["lo"]), float(ci15["hi"])]
    n_beat12 = sum(r <= BEAT_R for r in seed12)
    n_beat15 = sum(r <= BEAT_R for r in seed15)
    info["n_seed_beat_12"] = int(n_beat12)
    info["n_seed_beat_15"] = int(n_beat15)
    if R > BEAT_R:
        missed.append(f"n=12 R={R:.3f} is above {BEAT_R}")
    if r15 > BEAT_R:
        missed.append(f"n=15 R={r15:.3f} is above {BEAT_R}")
    if n_beat12 < n_seeds:
        missed.append(f"n=12 seeds beating Fourier {n_beat12}/{n_seeds}")
    if n_beat15 < n_seeds:
        missed.append(f"n=15 seeds beating Fourier {n_beat15}/{n_seeds}")
    if clear["seed_median"] > BEAT_R:
        missed.append(f"n=12 median per-seed R={clear['seed_median']:.3f} is above {BEAT_R}")
    if float(np.median(seed15)) > BEAT_R:
        missed.append(f"n=15 median per-seed R={float(np.median(seed15)):.3f} is above {BEAT_R}")
    for label, block in (("n=12", p12), ("n=15", p15)):
        if block is None:
            missed.append(f"paired counts missing at {label}")
            continue
        if int(block["n_seeds"]) != n_seeds:
            missed.append(f"{label} paired seed count {block['n_seeds']} != {n_seeds}")
        elif int(block["n_majority"]) < n_seeds:
            missed.append(
                f"{label} instance majority on {int(block['n_majority'])}/{n_seeds} seeds "
                f"(pooled wins {int(block['win'])}/{int(block['n'])})"
            )
    peak12 = float(quantum[12]["peak_med"])
    gap12 = float(quantum[12]["prod_gap_med"])
    peak15 = float(quantum[15]["peak_med"])
    gap15 = float(quantum[15]["prod_gap_med"])
    if not (peak12 < 0.5 and gap12 + 1e-12 >= PROD_GAP_MIN):
        missed.append(f"n=12 quantumness fails (peak={peak12:.3f}, product gap={gap12:.3f})")
    if not (peak15 < 0.5 and gap15 + 1e-12 >= PROD_GAP_MIN):
        missed.append(f"n=15 quantumness fails (peak={peak15:.3f}, product gap={gap15:.3f})")
    coverage = _coverage_misses(quantum, gap_coverage)
    if coverage:
        missed = coverage + missed
    med_ok = (
        peak12 < 0.5 and gap12 + 1e-12 >= PROD_GAP_MIN and peak15 < 0.5 and gap15 + 1e-12 >= PROD_GAP_MIN
    )
    info["quantum_ok"] = bool(med_ok and not coverage)
    if gap_coverage is not None:
        info["gap_coverage"] = float(gap_coverage)
    if not missed:
        info["verdict"] = "PASS"
        info["reason"] = (
            f"frozen n=12 L=16 Walsh R={R:.3f} "
            f"(instance 95% CI {info['ci12'][0]:.3f}–{info['ci12'][1]:.3f}) and "
            f"n=15 R={r15:.3f} (CI {info['ci15'][0]:.3f}–{info['ci15'][1]:.3f}) both stay BEAT. "
            f"{n_beat12}/{n_seeds} and {n_beat15}/{n_seeds} seeds beat. "
            f"Paired majorities {int(p12['n_majority'])}/{n_seeds} and {int(p15['n_majority'])}/{n_seeds}. "
            f"Drop vs μ=0 unsat Walsh {clear['drop']:.3f}."
            + _coverage_sentence(quantum, gap_coverage)
        )
        return info
    info["verdict"] = "needs_edit"
    info["reason"] = "; ".join(missed)
    return info
