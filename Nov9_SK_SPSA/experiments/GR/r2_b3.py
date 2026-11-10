"""R2 B3: matched Nov3rd vs QAOA at the per-instance entropy field B*.

Recipe id N3X_matched_v1. Caps 128 and 1024 are separate runs. Every energy
evaluation that the run performs is inside the cap, including the four initial
states. Monitoring queries are zero. Fidelity is computed after energy
selection from the state of the charged winning evaluation.

This is not the local cumulative 2048/3072 continuation, and it does not
rerun true displacement.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh

from common import RESULT_DIR
from hamiltonian import apply_X_sum, make_J, pair_K, pair_diag, upper, hat, rms, z_table
from encoding import feature_M
from model import apply_H_site, apply_rx_site, plus_state, pop_phase_logical
from spsa import _grad, _rademacher

RECIPE = "N3X_matched_v1"
OUT = RESULT_DIR / "r2_b3"
SEEDS = list(range(10000, 10008))
NS = (6, 9, 12)
CAPS = (128, 1024)
LS = (1, 2, 3)
METHODS = ("Nov3rd", "QAOA")
METHOD_ID = {"QAOA": 0, "Nov3rd": 1}
# Published LE diagnostic, same cut and grid. Used only as a reproduction check.
LE_BSTAR = {
    (6, 10000): 0.2961253608937295,
    (6, 10001): 0.27359009760316433,
    (6, 10002): 0.13616946164261357,
    (6, 10003): 0.16470075655115857,
    (9, 10000): 0.5672817157081989,
    (9, 10001): 0.38776277976044254,
    (9, 10002): 0.5914517606103161,
    (9, 10003): 0.5681240500038138,
}


def packing(n: int) -> list:
    m = n // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def entropy(gs: np.ndarray, n: int) -> float:
    """LE definition: natural log, singular values squared, cutoff 1e-300."""
    n_a = n // 2
    psi = np.asarray(gs, dtype=np.float64).reshape(1 << (n - n_a), 1 << n_a)
    sv = np.linalg.svd(psi, compute_uv=False) ** 2
    sv = sv[sv > 1e-300]
    return float(-(sv * np.log(sv)).sum())


def _mv(hc: np.ndarray, b: float, n: int):
    def mv(v):
        v = np.asarray(v, dtype=np.float64)
        return hc * v - float(b) * apply_X_sum(v, n)

    return mv


def ground_state(hc: np.ndarray, b: float, n: int) -> np.ndarray:
    dim = 1 << n
    op = LinearOperator((dim, dim), matvec=_mv(hc, b, n), dtype=np.float64)
    _w, v = eigsh(op, k=1, which="SA", tol=1e-12)
    g = np.asarray(v[:, 0], dtype=np.float64)
    if float(g.sum()) < 0.0:
        g = -g
    return g / np.linalg.norm(g)


def bstar_of(J: np.ndarray, n: int) -> tuple[float, float, int]:
    grid = np.linspace(0.05, 3.0, 60)
    hc = pair_diag(J, z_table(n))
    values = [entropy(ground_state(hc, float(b), n), n) for b in grid]
    k = int(np.argmax(values))
    if not (0 < k < len(grid) - 1):
        raise RuntimeError(f"entropy peak on a grid edge at N={n}")
    bracket = (float(grid[k - 1]), float(grid[k + 1]))

    def neg(b):
        return -entropy(ground_state(hc, float(b), n), n)

    refined = minimize_scalar(neg, bounds=bracket, method="bounded", options={"xatol": 1e-6})
    return float(refined.x), float(-refined.fun), k


def spectrum(hc: np.ndarray, b: float, n: int) -> dict:
    dim = 1 << n
    mv = _mv(hc, b, n)
    k = min(12, dim - 1)
    op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    w, v = eigsh(op, k=k, which="SA", tol=1e-12)
    order = np.argsort(w)
    w = w[order]
    v = v[:, order]
    idx = np.arange(dim)
    mask = dim - 1
    parity = [float(v[:, i] @ v[idx ^ mask, i]) for i in range(k)]
    g = np.asarray(v[:, 0], dtype=np.float64)
    if float(g.sum()) < 0.0:
        g = -g
    g = g / np.linalg.norm(g)
    e0 = float(w[0])
    odd = [float(w[i]) for i in range(k) if parity[i] < -0.5]
    even_ex = [float(w[i]) for i in range(k) if parity[i] > 0.5 and float(w[i]) > e0 + 1e-8]
    if not odd or not even_ex:
        raise RuntimeError(f"gap sector missing at N={n} b={b}")
    return {
        "E0": e0,
        "E1": float(w[1]),
        "gap": float(w[1] - e0),
        "E0_odd": float(min(odd)),
        "delta_odd": float(min(odd) - e0),
        "delta_even": float(min(even_ex) - e0),
        "E1_is_odd": bool(parity[1] < -0.5),
        "gs": g,
        "residual": float(np.linalg.norm(mv(g) - e0 * g)),
        "gs_min": float(np.min(g)),
        "gs_parity": float(g @ g[idx ^ mask]),
        "F_plus": float(np.sum(g) ** 2 / dim),
        "S_A": entropy(g, n),
    }


def prepare_instances() -> list[dict]:
    rows = []
    for n in NS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            J = make_J(seed, n)
            b, s_grid, k = bstar_of(J, n)
            hc = pair_diag(J, z_table(n))
            spec = spectrum(hc, b, n)
            published = LE_BSTAR.get((n, seed))
            agree = None if published is None else abs(b - published)
            if agree is not None and agree > 1e-5:
                raise RuntimeError(f"B* mismatch N={n} seed={seed}: {b} vs LE {published}")
            if spec["gs_min"] <= -1e-8 or spec["gs_parity"] < 1.0 - 1e-6 or spec["residual"] > 1e-8:
                raise RuntimeError(f"GS contract failed N={n} seed={seed}")
            if spec["gap"] <= 0.0 or spec["delta_odd"] <= 0.0 or spec["delta_even"] <= 0.0:
                raise RuntimeError(f"gap contract failed N={n} seed={seed}")
            rows.append(
                {
                    "seed": seed,
                    "N": n,
                    "B_star": b,
                    "S_A": spec["S_A"],
                    "S_A_nats": spec["S_A"],
                    "entropy_units": "nats_natural_log_LE_svd_cutoff_1e-300",
                    "cut": list(range(n // 2)),
                    "grid_argmax_index": k,
                    "interior": True,
                    "spsa_b_key": int(round(10.0 * b)),
                    "E0": spec["E0"],
                    "E1": spec["E1"],
                    "gap": spec["gap"],
                    "delta_odd": spec["delta_odd"],
                    "delta_even": spec["delta_even"],
                    "E1_is_odd": spec["E1_is_odd"],
                    "F_plus": spec["F_plus"],
                    "E_plus": -float(b) * n,
                    "residual": spec["residual"],
                    "gs_min": spec["gs_min"],
                    "gs_parity": spec["gs_parity"],
                    "le_abs_diff": agree,
                    "seconds": time.perf_counter() - t0,
                    "gs": spec["gs"],
                    "hc": hc,
                    "J": J,
                }
            )
            print(f"B* N={n} seed={seed} b={b:.6f} S={spec['S_A']:.4f} gap={spec['gap']:.4f} F+={spec['F_plus']:.4f}", flush=True)
    return rows


def features(J: np.ndarray, n: int, hc: np.ndarray):
    z = z_table(n)
    rms_j = rms(upper(J))
    k_hat, _sk = hat(pair_K(J), rms_j)
    groups = packing(n)
    _raw, _mh, _sm, d_m = feature_M(J, groups, rms_j, z)
    d_k = pair_diag(k_hat, z)
    return groups, d_k, d_m, hc


def nov3_psi(x: np.ndarray, L: int, hc, d_k, d_m, groups, n: int) -> np.ndarray:
    gamma, th_k, th_m = float(x[0]), float(x[1]), float(x[2])
    shells = np.asarray(x[3:], dtype=np.float64).reshape(L, 3)
    gdiag = hc + th_k * d_k + th_m * d_m
    psi = plus_state(n)
    for ell in range(L):
        psi = psi * np.exp(-1j * gamma * gdiag)
        for cav in groups:
            for site in cav:
                psi = apply_H_site(psi, int(site))
            psi = psi * np.exp(-1j * pop_phase_logical(n, cav, float(shells[ell, 0]), float(shells[ell, 1]), float(shells[ell, 2])))
            for site in cav:
                psi = apply_H_site(psi, int(site))
    return psi


def qaoa_psi(x: np.ndarray, L: int, hc, n: int) -> np.ndarray:
    psi = plus_state(n)
    for ell in range(L):
        psi = psi * np.exp(-1j * float(x[2 * ell]) * hc)
        beta = float(x[2 * ell + 1])
        for q in range(n):
            psi = apply_rx_site(psi, q, beta)
    return psi


def energy_of(psi: np.ndarray, hc, b: float, n: int) -> float:
    hpsi = hc * psi - float(b) * apply_X_sum(psi, n)
    return float(np.vdot(psi, hpsi).real)


def n_controls(method: str, L: int) -> int:
    if method == "Nov3rd":
        return 3 + 3 * L
    if method == "QAOA":
        return 2 * L
    raise ValueError(method)


def sigmas_of(method: str, L: int) -> np.ndarray:
    if method == "Nov3rd":
        sig = np.empty(3 + 3 * L, dtype=np.float64)
        sig[0] = 0.10
        sig[1] = 0.25
        sig[2] = 0.25
        sig[3:] = 0.15
        return sig
    return np.full(2 * L, 0.10, dtype=np.float64)


def x0_of(method: str, L: int, kind: str, rng) -> np.ndarray:
    if method == "Nov3rd":
        x = np.zeros(3 + 3 * L, dtype=np.float64)
        if kind == "E":
            x[0] = 0.25
            return x
        x[0] = float(rng.uniform(-0.5, 0.5))
        x[1] = float(rng.uniform(-0.5, 0.5))
        x[2] = float(rng.uniform(-0.5, 0.5))
        for t in range(3 * L):
            x[3 + t] = float(rng.uniform(-0.3, 0.3))
        return x
    x = np.zeros(2 * L, dtype=np.float64)
    if kind == "E":
        x[:] = 0.25
        return x
    for ell in range(L):
        x[2 * ell] = float(rng.uniform(-0.5, 0.5))
        x[2 * ell + 1] = float(rng.uniform(-0.5, 0.5))
    return x


def iterations_for(cap: int) -> int:
    per = cap // 4
    if per < 12 or (per - 12) % 2 != 0:
        raise ValueError(f"cap {cap} does not fit 1 init + 8 calibration + 3 selection + even iterations")
    return (per - 12) // 2


class Counter:
    def __init__(self):
        self.init = 0
        self.calibration = 0
        self.iteration = 0
        self.candidate = 0
        self.monitoring = 0

    def add(self, kind: str) -> None:
        setattr(self, kind, getattr(self, kind) + 1)

    def total(self) -> int:
        return self.init + self.calibration + self.iteration + self.candidate + self.monitoring

    def as_dict(self) -> dict:
        return {
            "init": self.init,
            "calibration": self.calibration,
            "iteration": self.iteration,
            "candidate": self.candidate,
            "monitoring": self.monitoring,
            "total": self.total(),
        }


def run_matched(propagate, x0s, sigmas, cap: int, rngs) -> dict:
    """propagate(x) -> (energy, state). Selector reads energy only."""
    n_iter = iterations_for(cap)
    counter = Counter()
    c = 0.1
    A = 25.0
    best = None
    a_gains = []

    def call(x, kind: str):
        counter.add(kind)
        energy, psi = propagate(np.asarray(x, dtype=np.float64))
        if not np.isfinite(energy):
            raise RuntimeError("non-finite energy")
        return float(energy), psi

    for start_id, (x0, rng) in enumerate(zip(x0s, rngs)):
        x = np.array(x0, dtype=np.float64, copy=True)
        e_init, psi_init = call(x, "init")
        grads = []
        for _ in range(4):
            delta = _rademacher(rng, x.size)
            step = c * sigmas * delta
            ep, _ = call(x + step, "calibration")
            em, _ = call(x - step, "calibration")
            grads.append(_grad(ep, em, c, sigmas, delta))
        ratios = [abs(g[j]) / sigmas[j] for g in grads for j in range(x.size) if sigmas[j] > 0.0]
        med = float(np.median(ratios)) if ratios else 0.0
        a0 = 0.0 if med < 1e-15 else 0.5 / med
        a_gain = a0 * (1.0 + A) ** 0.602
        history = []
        for k in range(n_iter):
            ck = c / (k + 1) ** 0.101
            ak = a_gain / (k + 1.0 + A) ** 0.602
            delta = _rademacher(rng, x.size)
            step = ck * sigmas * delta
            ep, _ = call(x + step, "iteration")
            em, _ = call(x - step, "iteration")
            g = _grad(ep, em, ck, sigmas, delta)
            update = ak * g
            limit = 2.0 * sigmas
            update = np.clip(update, -limit, limit)
            x = x - update
            history.append(x.copy())
        last = history[-1]
        poly = np.mean(np.stack(history[-min(64, len(history)) :], axis=0), axis=0)
        mid = history[len(history) // 2]
        e_last, psi_last = call(last, "candidate")
        e_poly, psi_poly = call(poly, "candidate")
        e_mid, psi_mid = call(mid, "candidate")
        options = (
            ("init", e_init, x0, psi_init),
            ("last", e_last, last, psi_last),
            ("polyak64", e_poly, poly, psi_poly),
            ("mid_iterate", e_mid, mid, psi_mid),
        )
        choice = min(options, key=lambda item: (item[1], item[0] != "init"))
        if best is None or choice[1] < best["E"] or (choice[1] == best["E"] and start_id < best["start_id"]):
            best = {
                "E": float(choice[1]),
                "x": np.asarray(choice[2], dtype=np.float64).copy(),
                "psi": choice[3],
                "which": choice[0],
                "start_id": start_id,
            }
        a_gains.append(float(a_gain))
    if counter.total() != cap or counter.monitoring != 0:
        raise RuntimeError(f"ledger {counter.as_dict()} != cap {cap}")
    best["queries"] = counter.as_dict()
    best["n_iter"] = n_iter
    best["a_gains"] = a_gains
    return best


def _rng(seed, n, b, L, method_id, start_id, stream):
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(n), int(round(10.0 * float(b))), int(L), int(method_id), int(start_id), int(stream)])
    )


def _propagate_factory(method: str, L: int, packed):
    groups, d_k, d_m, hc, b, n = packed

    def propagate(x):
        if method == "Nov3rd":
            psi = nov3_psi(x, L, hc, d_k, d_m, groups, n)
        else:
            psi = qaoa_psi(x, L, hc, n)
        return energy_of(psi, hc, b, n), psi

    return propagate


def run_job(job: dict) -> dict:
    t0 = time.perf_counter()
    n = int(job["N"])
    seed = int(job["seed"])
    b = float(job["B_star"])
    L = int(job["L"])
    method = job["method"]
    cap = int(job["cap"])
    J = make_J(seed, n)
    hc = pair_diag(J, z_table(n))
    groups, d_k, d_m, hc = features(J, n, hc)
    propagate = _propagate_factory(method, L, (groups, d_k, d_m, hc, b, n))
    sig = sigmas_of(method, L)
    x0s = []
    rngs = []
    for s in range(4):
        rngs.append(_rng(seed, n, b, L, METHOD_ID[method], s, 0))
        init_rng = None if s == 0 else _rng(seed, n, b, L, METHOD_ID[method], s, 2)
        x0s.append(x0_of(method, L, "E" if s == 0 else "R", init_rng))
    out = run_matched(propagate, x0s, sig, cap, rngs)
    return {
        "x": out["x"],
        "E": out["E"],
        "psi": out["psi"],
        "which": out["which"],
        "start_id": out["start_id"],
        "a_gains": out["a_gains"],
        "queries": out["queries"],
        "n_iter": out["n_iter"],
        "seconds": time.perf_counter() - t0,
        "n_controls": n_controls(method, L),
    }


def fidelity(gs: np.ndarray, psi: np.ndarray) -> float:
    return float(np.abs(np.vdot(gs, psi)) ** 2)


def curvature_l1(rows: list[dict]) -> list[dict]:
    """2x2 energy Hessian at the zero vector. Not a selector and not inside the cap."""
    out = []
    calls = 0
    h = 1e-4
    for row in rows:
        for method, i, j in (("Nov3rd", 0, 3), ("QAOA", 0, 1)):
            L = 1
            groups, d_k, d_m, hc = features(row["J"], row["N"], row["hc"])
            propagate = _propagate_factory(method, L, (groups, d_k, d_m, hc, row["B_star"], row["N"]))
            dim = n_controls(method, L)
            cache = {}

            def eat(vec, _prop=propagate):
                nonlocal calls
                key = tuple(np.round(vec, decimals=12))
                if key not in cache:
                    calls += 1
                    cache[key] = _prop(vec)[0]
                return cache[key]

            def point(di, dj):
                x = np.zeros(dim)
                x[i] += di * h
                x[j] += dj * h
                return eat(x)

            hii = (point(1, 0) - 2 * point(0, 0) + point(-1, 0)) / h**2
            hjj = (point(0, 1) - 2 * point(0, 0) + point(0, -1)) / h**2
            hij = (point(1, 1) - point(1, -1) - point(-1, 1) + point(-1, -1)) / (4 * h**2)
            vals = np.linalg.eigvalsh(np.array([[hii, hij], [hij, hjj]], dtype=np.float64))
            out.append(
                {
                    "N": row["N"],
                    "seed": row["seed"],
                    "method": method,
                    "coordinates": ["gamma", "s1" if method == "Nov3rd" else "beta"],
                    "hessian": [hii, hij, hjj],
                    "eig": [float(vals[0]), float(vals[1])],
                    "selector": False,
                }
            )
    return out, calls


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_summary(records: list[dict], curvature_calls: int) -> None:
    def cell(n, L, cap, method):
        hit = [r for r in records if r["N"] == n and r["L"] == L and r["cap"] == cap and r["method"] == method]
        hit.sort(key=lambda r: r["seed"])
        return hit

    lines = []
    lines.append("# R2 B3 matched Nov3rd vs QAOA at B*")
    lines.append("")
    lines.append("Entropy case only. Recipe `N3X_matched_v1`: 4 starts, caps 128 and 1024, every initial, calibration, and selection energy call inside the cap, monitoring 0. Fidelity is post hoc on the selected state.")
    lines.append("This is not the local cumulative continuation (2048/3072 queries) and it is not the earlier b = 1 / 1.5 reference.")
    lines.append("")
    lines.append("## External local feedback, different recipe")
    lines.append("")
    lines.append("Four dev instances per N, L = 3, cumulative charged queries, energy selection. Not merged with the table below. No scaling claim from these four.")
    lines.append("")
    lines.append("| N | Nov3 F | QAOA F | cumulative queries |")
    lines.append("|---:|---:|---:|---:|")
    lines.append("| 9 | 0.8326 | 0.7823 | 2048 |")
    lines.append("| 12 | 0.7753 | 0.7700 | 3072 |")
    lines.append("| 15 | 0.7033 | 0.7468 | 2048 |")
    lines.append("")
    lines.append("Local Nov3 at N = 9: all 4 instances F ≥ 0.7 and 3 of 4 F ≥ 0.8. Local’s next step is a common 3072/4096 budget and fresh N = 15 dev seeds. That extension is not this run.")
    lines.append("")
    lines.append("## This confirmation")
    lines.append("")
    lines.append("Seeds 10000–10007. Same J and same B* for both methods. Nov3rd has one shared γ, two feature angles, and 3 shell phases per layer (6 controls at L = 1). The shell chart is the polynomial `s1 p + s2 C(p,2) + s3 [p=3]`, not the Walsh `t1,t2,t3` chart. QAOA has 2L independent angles, `exp(-i γ H_ZZ) exp(-i β Σ X)`.")
    lines.append("Per start at cap 1024: 1 init + 8 calibration + 122×2 iteration + 3 selection = 256, times 4 = 1024. At cap 128 the iteration count is 10. The selectable points are the init, the last iterate, the mean of the last min(64, T) iterates, and the midpoint iterate. Lowest energy wins. Gains use c = 0.1 and a median initial step of 0.5σ.")
    lines.append("")
    lines.append("| N | L | cap | Nov3 med F | QAOA med F | med ΔF | F wins | Nov3 med E | QAOA med E | med ΔE | Nov3 η | Nov3 η_E | F+ | frac Nov3 ≥0.7 | ≥0.8 | frac QAOA ≥0.7 | ≥0.8 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in NS:
        for L in LS:
            for cap in CAPS:
                a = cell(n, L, cap, "Nov3rd")
                q = cell(n, L, cap, "QAOA")
                qm = {r["seed"]: r for r in q}
                dF = [r["F"] - qm[r["seed"]]["F"] for r in a]
                dE = [r["E"] - qm[r["seed"]]["E"] for r in a]
                wins = sum(1 for v in dF if v > 0.0)
                def med(xs):
                    return float(np.median(xs))
                def frac(rows, thr):
                    return sum(1 for r in rows if r["F"] >= thr) / len(rows)
                lines.append(
                    f"| {n} | {L} | {cap} | {med([r['F'] for r in a]):.4f} | {med([r['F'] for r in q]):.4f} | {med(dF):.4f} | {wins}/8 | {med([r['E'] for r in a]):.4f} | {med([r['E'] for r in q]):.4f} | {med(dE):.4f} | {med([r['eta'] for r in a]):.3f} | {med([r['eta_E'] for r in a]):.3f} | {med([r['F_plus'] for r in a]):.3f} | {frac(a, 0.7):.2f} | {frac(a, 0.8):.2f} | {frac(q, 0.7):.2f} | {frac(q, 0.8):.2f} |"
                )
    lines.append("")
    lines.append("## Tails at cap 1024")
    lines.append("")
    lines.append("| N | L | method | min F | max F | min η_E |")
    lines.append("|---:|---:|---|---:|---:|---:|")
    for n in NS:
        for L in LS:
            for method in METHODS:
                rows = cell(n, L, 1024, method)
                lines.append(
                    f"| {n} | {L} | {method} | {min(r['F'] for r in rows):.4f} | {max(r['F'] for r in rows):.4f} | {min(r['eta_E'] for r in rows):.3f} |"
                )
    lines.append("")
    total = sum(r["queries"]["total"] for r in records)
    lines.append(f"Charged optimization queries: {total}. Every stored run totals its cap. Monitoring: 0. Curvature finite-difference calls, not selectors: {curvature_calls}.")
    lines.append("N = 15 was not optimized. Imaginary-axis displacement is LE’s assignment. The real-α basin audit is FT’s assignment.")
    lines.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def self_test_budget() -> None:
    calls = {"n": 0}

    def propagate(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), np.zeros(2)

    for cap in (128, 1024):
        calls["n"] = 0
        x0s = [np.zeros(2) for _ in range(4)]
        rngs = [np.random.default_rng(i) for i in range(4)]
        out = run_matched(propagate, x0s, np.full(2, 0.1), cap, rngs)
        if calls["n"] != cap or out["queries"]["total"] != cap or out["queries"]["monitoring"] != 0:
            raise RuntimeError(f"budget test failed at {cap}: {calls['n']} {out['queries']}")
        if out["queries"]["init"] != 4:
            raise RuntimeError("init calls were not charged")


def main() -> None:
    if any(seed >= 11000 for seed in SEEDS):
        raise SystemExit("blind seed requested")
    self_test_budget()
    OUT.mkdir(parents=True, exist_ok=True)
    print("budget self-test passed", flush=True)
    instances = prepare_instances()
    public = []
    for row in instances:
        item = {k: v for k, v in row.items() if k not in {"gs", "hc", "J"}}
        public.append(item)
    (OUT / "bstar.json").write_text(json.dumps({"recipe": RECIPE, "cut": "sites 0..floor(N/2)-1", "instances": public}, indent=2) + "\n")
    curv, curv_calls = curvature_l1(instances)
    (OUT / "curvature_l1.json").write_text(json.dumps(curv, indent=2) + "\n")
    by_key = {(r["N"], r["seed"]): r for r in instances}
    jobs = []
    for row in instances:
        for L in LS:
            for cap in CAPS:
                for method in METHODS:
                    jobs.append({"N": row["N"], "seed": row["seed"], "B_star": row["B_star"], "L": L, "cap": cap, "method": method})
    print(f"jobs {len(jobs)}", flush=True)
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    records = []
    done = 0
    with ctx.Pool(4) as pool:
        for job, out in zip(jobs, pool.imap(run_job, jobs, chunksize=1)):
            # imap preserves order, so zip is safe. Still write as we go.
            done += 1
            inst = by_key[(job["N"], job["seed"])]
            F = fidelity(inst["gs"], out["psi"])
            e_check = energy_of(out["psi"], inst["hc"], inst["B_star"], inst["N"])
            if abs(e_check - out["E"]) > 1e-8:
                raise RuntimeError("returned state does not match reported energy")
            denom = 1.0 - inst["F_plus"]
            eta = (F - inst["F_plus"]) / denom if denom > 1e-15 else float("nan")
            span = inst["E_plus"] - inst["E0"]
            eta_e = (inst["E_plus"] - out["E"]) / span if abs(span) > 1e-15 else float("nan")
            rec = {
                "recipe": RECIPE,
                "status": "ok",
                "seed": job["seed"],
                "N": job["N"],
                "L": job["L"],
                "cap": job["cap"],
                "method": job["method"],
                "B_star": inst["B_star"],
                "spsa_b_key": inst["spsa_b_key"],
                "S_A": inst["S_A"],
                "F_plus": inst["F_plus"],
                "E_plus": inst["E_plus"],
                "E0": inst["E0"],
                "gap": inst["gap"],
                "delta_odd": inst["delta_odd"],
                "delta_even": inst["delta_even"],
                "E": out["E"],
                "F": F,
                "F_role": "posthoc_on_charged_selected_state",
                "eta": eta,
                "eta_E": eta_e,
                "queries": out["queries"],
                "n_iter_per_start": out["n_iter"],
                "selected_start": out["start_id"],
                "selected_point": out["which"],
                "a_gains": out["a_gains"],
                "n_controls": out["n_controls"],
                "seconds": out["seconds"],
                "chart": "nov3_polynomial_shell" if job["method"] == "Nov3rd" else "qaoa_independent_2p",
                "hardware_claim": False,
                "entropy_case_only": True,
            }
            if rec["queries"]["total"] != job["cap"]:
                raise RuntimeError("cap mismatch")
            path = OUT / "runs" / f"n{job['N']}_s{job['seed']}_L{job['L']}_{job['method']}_c{job['cap']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_clean(rec), indent=2) + "\n")
            records.append(rec)
            print(f"[{done}/{len(jobs)}] N={job['N']} s={job['seed']} L={job['L']} {job['method']} c={job['cap']} F={F:.4f} E={out['E']:.4f} q={rec['queries']['total']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(_clean({"recipe": RECIPE, "n": len(records), "curvature_calls": curv_calls, "records_omit_psi": True}), indent=2) + "\n")
    write_summary(records, curv_calls)
    # Throughput only. Not an entropy instance and not an optimization.
    t0 = time.perf_counter()
    J = make_J(10000, 15)
    hc = pair_diag(J, z_table(15))
    groups, d_k, d_m, hc = features(J, 15, hc)
    prop = _propagate_factory("Nov3rd", 1, (groups, d_k, d_m, hc, 0.5, 15))
    for _ in range(5):
        prop(np.zeros(6))
    dt = (time.perf_counter() - t0) / 5.0
    (OUT / "n15_throughput.json").write_text(
        json.dumps(
            {
                "label": "throughput_only_not_Bstar_not_optimized",
                "N": 15,
                "b_used": 0.5,
                "seconds_per_nov3_L1_energy": dt,
                "projected_1024_seconds": dt * 1024,
                "optimized": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", OUT / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
