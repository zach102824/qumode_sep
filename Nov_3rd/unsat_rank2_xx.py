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


def cache_filename(n: int, role: str, count: int, lam: float, mu: float) -> str:
    """Pack cache stem. λ=0.75, μ=−0.50 is the PR #26 name ``l0.75_m-0.50``."""
    mtag = f"m{float(mu):+.2f}".replace("+", "")
    return f"unsat_xx_n{int(n)}_{role}_B{int(count)}_l{float(lam):.2f}_{mtag}.npz"


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
