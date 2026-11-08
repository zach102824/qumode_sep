"""Disorder-averaged Binder scan for the finite-N pseudocritical field.

Ground states are used only for this field definition. No circuit is built.
"""
from __future__ import annotations

import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
PREREG_PATH = HERE / "PREREG_SCAN.json"
PREREG_SHA = "0226715302ca7d0e30c4bee8182f05c367a93432c5d06fff99d376b3fb988629"
PREREG = json.loads(PREREG_PATH.read_text())


def check_hash():
    digest = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")


def gamma_grid():
    spec = PREREG["Gamma_milli"]
    milli = np.arange(spec["start"], spec["stop"] + 1, spec["step"], dtype=int)
    return milli / 1000.0


def make_J(N, k):
    rng = np.random.default_rng(np.random.SeedSequence([20261170, int(N), int(k)]))
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = J[j, i] = float(rng.normal())
    return J


def zz_diag(J):
    N = J.shape[0]
    dim = 1 << N
    hc = np.zeros(dim)
    idx = np.arange(dim)
    for i in range(N):
        zi = 1.0 - 2.0 * ((idx >> i) & 1)
        for j in range(i + 1, N):
            zj = 1.0 - 2.0 * ((idx >> j) & 1)
            hc += J[i, j] * zi * zj
    return hc


def x_sum(N):
    dim = 1 << N
    idx = np.arange(dim)
    data = []
    rows = []
    cols = []
    for q in range(N):
        data.append(np.ones(dim))
        rows.append(idx)
        cols.append(idx ^ (1 << q))
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    )


def fwht_unnorm(a):
    b = np.asarray(a, dtype=np.float64).copy()
    n = b.size
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = b[i : i + h].copy()
            y = b[i + h : i + 2 * h].copy()
            b[i : i + h] = x + y
            b[i + h : i + 2 * h] = x - y
        h *= 2
    return b


def moments(psi, N):
    p = np.abs(np.asarray(psi, dtype=np.complex128)) ** 2
    p = np.real(p)
    p /= p.sum()
    c = fwht_unnorm(p)
    ii, jj = np.indices((N, N))
    q2 = float(np.mean(c[(1 << ii) ^ (1 << jj)] ** 2))
    ii, jj, kk, ll = np.indices((N, N, N, N))
    masks = (1 << ii) ^ (1 << jj) ^ (1 << kk) ^ (1 << ll)
    q4 = float(np.mean(c[masks] ** 2))
    q1 = float(np.mean(c[1 << np.arange(N)] ** 2))
    return q1, q2, q4, float(c[0])


def parity(psi):
    v = np.asarray(psi, dtype=np.complex128).ravel()
    flip = np.arange(v.size) ^ (v.size - 1)
    return float(np.vdot(v, v[flip]).real)


def q2_spread(vecs, N, gap):
    if gap >= PREREG["adoption"]["degeneracy"]["ordinary_gap_flag"]:
        return 0.0
    cols = [vecs[:, 0], vecs[:, 1]]
    s = vecs[:, 0] + vecs[:, 1]
    d = vecs[:, 0] - vecs[:, 1]
    s = s / np.linalg.norm(s)
    d = d / np.linalg.norm(d)
    cols.extend([s, d])
    vals = [moments(v, N)[1] for v in cols]
    return float(max(vals) - min(vals))


def lowest(H, k):
    evals, vecs = eigsh(H, k=k, which="SA", tol=1e-9, maxiter=4000)
    order = np.argsort(evals)
    return np.real(evals[order]), np.real(vecs[:, order])


def one_instance(task):
    N, k, gammas, do_gap = task
    J = make_J(N, k)
    hc = zz_diag(J)
    dim = 1 << N
    D = sparse.diags(hc)
    X = x_sum(N)
    q1 = np.zeros(gammas.size)
    q2 = np.zeros(gammas.size)
    q4 = np.zeros(gammas.size)
    gap = np.zeros(gammas.size)
    par = np.zeros(gammas.size)
    spread = np.zeros(gammas.size)
    prob0 = np.zeros(gammas.size)
    for i, gamma in enumerate(gammas):
        B = float(gamma) * np.sqrt(N)
        evals, vecs = lowest(D + B * X, 2)
        m1, m2, m4, c0 = moments(vecs[:, 0], N)
        q1[i], q2[i], q4[i], prob0[i] = m1, m2, m4, c0
        gap[i] = float(evals[1] - evals[0])
        par[i] = parity(vecs[:, 0])
        spread[i] = q2_spread(vecs, N, gap[i])
    # historical author field, separate from the Gamma grid
    evals_h, vecs_h = lowest(D + 0.1 * X, 2)
    _, q2_hist, _, _ = moments(vecs_h[:, 0], N)
    gap_diag = None
    if do_gap:
        gap_diag = coupled_min(D, X, hc, N, gammas)
    return {
        "N": int(N),
        "k": int(k),
        "q1": q1,
        "q2": q2,
        "q4": q4,
        "gap": gap,
        "parity": par,
        "spread": spread,
        "prob0": prob0,
        "q2_B01": float(q2_hist),
        "gap_ord_B01": float(evals_h[1] - evals_h[0]),
        "coupled": gap_diag,
    }


def coupled_min(D, X, hc, N, gammas):
    """Same-parity gap along the grid. Full spectrum at N<=9. Not an adoption input."""
    dim = 1 << N
    best = None
    for gamma in gammas:
        B = float(gamma) * np.sqrt(N)
        if N <= 9:
            evals, vecs = np.linalg.eigh(np.diag(hc) + B * np.eye(dim) * 0 + B * X.toarray())
            # X.toarray() each time is wasteful but N<=9 and k is small
        else:
            evals, vecs = lowest(D + B * X, min(16, dim - 1))
        p0 = round(parity(vecs[:, 0]), 5)
        coupled = None
        for i in range(1, evals.size):
            if round(parity(vecs[:, i]), 5) == p0:
                coupled = float(evals[i] - evals[0])
                break
        if coupled is None:
            continue
        if best is None or coupled < best[0]:
            best = (coupled, float(gamma), float(B), "full" if N <= 9 else "lanczos16")
    if best is None:
        return None
    return {"delta": best[0], "Gamma": best[1], "B": best[2], "spectrum": best[3]}


def crossings(grid, d):
    out = []
    for i in range(grid.size - 1):
        if d[i] == 0.0:
            out.append(float(grid[i]))
        elif d[i] * d[i + 1] < 0.0:
            t = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]))
            out.append(float(grid[i] + t * (grid[i + 1] - grid[i])))
    return out


def in_window(values, window):
    lo, hi = window
    return [v for v in values if lo <= v <= hi]


def binder_from_ratio(ratio):
    return 0.5 * (3.0 - np.mean(ratio, axis=0))


def startup_checks(gammas):
    # FWHT moments against a direct pair sum
    rng = np.random.default_rng(0)
    n = 5
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    q1, q2, q4, c0 = moments(psi, n)
    p = np.abs(psi) ** 2
    p /= p.sum()
    idx = np.arange(1 << n)
    acc = 0.0
    for i in range(n):
        zi = 1.0 - 2.0 * ((idx >> i) & 1)
        for j in range(n):
            zj = 1.0 - 2.0 * ((idx >> j) & 1)
            acc += float(np.dot(p, zi * zj)) ** 2
    direct_q2 = acc / n**2
    if abs(q2 - direct_q2) > 1e-10 or abs(c0 - 1.0) > 1e-10:
        raise SystemExit(f"moment check failed {q2} {direct_q2} {c0}")
    # Author B = Gamma * sqrt(N) matches the variance-1/N SK operator.
    n = 4
    J = make_J(n, 0)
    hc = zz_diag(J)
    gamma = 1.25
    B = gamma * np.sqrt(n)
    x = x_sum(n).toarray()
    ea = np.linalg.eigvalsh(np.diag(hc) + B * x)
    es = np.linalg.eigvalsh(np.diag(hc / np.sqrt(n)) + gamma * x)
    scale_resid = float(np.max(np.abs(ea / np.sqrt(n) - es)))
    if scale_resid > 1e-8:
        raise SystemExit(f"normalization map residual {scale_resid}")
    return {"moment_q2_abs": abs(q2 - direct_q2), "scale_residual": scale_resid, "n_gamma": int(gammas.size)}


def decide(store, gammas):
    rule = PREREG["adoption"]
    window = rule["window"]
    by_n = {}
    for N in PREREG["N"]:
        rows = [r for r in store if r["N"] == N]
        rows.sort(key=lambda r: r["k"])
        ratio = np.array([r["q4"] / np.maximum(r["q2"] ** 2, 1e-30) for r in rows])
        g = binder_from_ratio(ratio)
        q2_mean = np.mean([r["q2"] for r in rows], axis=0)
        by_n[N] = {"g": g, "ratio": ratio, "q2_mean": q2_mean, "rows": rows}
    pairs = [(6, 9), (9, 12)]
    pair_info = {}
    primary = {}
    for a, b in pairs:
        xs = crossings(gammas, by_n[a]["g"] - by_n[b]["g"])
        inside = in_window(xs, window)
        pair_info[f"{a}_{b}"] = {"all": xs, "in_window": inside}
        primary[f"{a}_{b}"] = inside
    reasons = []
    gamma_pseudo = None
    for key, inside in primary.items():
        if len(inside) != 1:
            reasons.append(f"{key} has {len(inside)} in-window crossings")
    if not reasons:
        g69 = primary["6_9"][0]
        g912 = primary["9_12"][0]
        if abs(g69 - g912) > rule["max_abs_pair_difference"]:
            reasons.append(f"pair difference {abs(g69 - g912)} exceeds {rule['max_abs_pair_difference']}")
        else:
            gamma_pseudo = 0.5 * (g69 + g912)
    # bootstrap
    rng = np.random.default_rng(np.random.SeedSequence([rule["bootstrap"]["seed"]]))
    boot = {f"{a}_{b}": [] for a, b in pairs}
    missing = {f"{a}_{b}": 0 for a, b in pairs}
    n_inst = by_n[6]["ratio"].shape[0]
    for _ in range(rule["bootstrap"]["resamples"]):
        take = rng.integers(0, n_inst, size=n_inst)
        gs = {}
        for N in PREREG["N"]:
            gs[N] = binder_from_ratio(by_n[N]["ratio"][take])
        for a, b in pairs:
            key = f"{a}_{b}"
            inside = in_window(crossings(gammas, gs[a] - gs[b]), window)
            if len(inside) != 1:
                missing[key] += 1
            else:
                boot[key].append(inside[0])
    boot_summary = {}
    for key, vals in boot.items():
        arr = np.array(vals, float)
        frac = missing[key] / rule["bootstrap"]["resamples"]
        boot_summary[key] = {
            "n_unique": int(arr.size),
            "missing_fraction": float(frac),
            "p2.5": float(np.percentile(arr, 2.5)) if arr.size else None,
            "p50": float(np.percentile(arr, 50)) if arr.size else None,
            "p97.5": float(np.percentile(arr, 97.5)) if arr.size else None,
        }
        if frac > rule["bootstrap"]["fail_if_primary_missing_fraction_above"]:
            reasons.append(f"{key} bootstrap missing fraction {frac}")
    # degeneracy in the pre-registered window
    lo, hi = rule["degeneracy"]["check_window"]
    mask = (gammas >= lo) & (gammas <= hi)
    flagged = 0
    total = 0
    for row in store:
        flags = row["spread"][mask] > rule["degeneracy"]["Q2_spread_flag"]
        flagged += int(np.sum(flags))
        total += int(flags.size)
    frac_flag = flagged / total if total else 1.0
    if frac_flag > rule["degeneracy"]["withhold_if_flagged_fraction_above"]:
        reasons.append(f"degeneracy-flagged fraction {frac_flag}")
    adopted = gamma_pseudo is not None and not reasons
    b_cri = None
    if adopted:
        b_cri = {str(N): float(gamma_pseudo * np.sqrt(N)) for N in PREREG["N"]}
    return {
        "adopted": bool(adopted),
        "reasons": reasons,
        "Gamma_pseudo": None if gamma_pseudo is None else float(gamma_pseudo),
        "B_cri": b_cri,
        "pairs": pair_info,
        "bootstrap": boot_summary,
        "degeneracy_flagged_fraction": float(frac_flag),
        "g": {str(N): by_n[N]["g"].tolist() for N in PREREG["N"]},
        "q2_mean": {str(N): by_n[N]["q2_mean"].tolist() for N in PREREG["N"]},
    }


def diagnostics(store, gammas):
    """Recorded after the decision inputs are fixed. Not read by decide()."""
    per_n = {}
    for N in PREREG["N"]:
        rows = [r for r in store if r["N"] == N]
        drops = []
        for r in rows:
            dq = np.diff(r["q2"])
            # largest drop along increasing Gamma
            i = int(np.argmin(dq))
            # interpolate the two-point slope location at the midpoint
            drops.append(0.5 * (gammas[i] + gammas[i + 1]))
        arr = np.array(drops)
        q1 = np.concatenate([r["q1"] for r in rows])
        per_n[str(N)] = {
            "q2_drop_Gamma_median": float(np.median(arr)),
            "q2_drop_Gamma_p25": float(np.percentile(arr, 25)),
            "q2_drop_Gamma_p75": float(np.percentile(arr, 75)),
            "q1_max": float(np.max(q1)),
            "q2_at_B_0.1_median": float(np.median([r["q2_B01"] for r in rows])),
            "ordinary_gap_at_B_0.1_median": float(np.median([r["gap_ord_B01"] for r in rows])),
        }
    coupled = []
    for r in store:
        if r["coupled"] is not None:
            coupled.append({"N": r["N"], "k": r["k"], **r["coupled"]})
    return {"per_instance_q2_drop": per_n, "coupled_gap_argmin_k0_to_7": coupled}


def main():
    check_hash()
    gammas = gamma_grid()
    checks = startup_checks(gammas)
    tasks = []
    for N in PREREG["N"]:
        for k in range(PREREG["k_inclusive"][0], PREREG["k_inclusive"][1] + 1):
            tasks.append((N, k, gammas, k <= 7))
    t0 = time.perf_counter()
    with Pool(4) as pool:
        store = pool.map(one_instance, tasks, chunksize=1)
    elapsed = time.perf_counter() - t0
    decision = decide(store, gammas)
    diag = diagnostics(store, gammas)
    # persist compact arrays
    out_dir = HERE / "scan"
    out_dir.mkdir(exist_ok=True)
    payload = {str(N): {} for N in PREREG["N"]}
    for row in store:
        payload[str(row["N"])][str(row["k"])] = {
            "q2": row["q2"],
            "q4": row["q4"],
            "q1": row["q1"],
            "gap": row["gap"],
            "parity": row["parity"],
            "spread": row["spread"],
        }
    np.savez_compressed(
        out_dir / "curves.npz",
        gamma=gammas,
        **{f"q2_{N}": np.stack([payload[str(N)][str(k)]["q2"] for k in range(64)]) for N in PREREG["N"]},
        **{f"q4_{N}": np.stack([payload[str(N)][str(k)]["q4"] for k in range(64)]) for N in PREREG["N"]},
    )
    summary = {
        "prereg_sha256": PREREG_SHA,
        "seconds": elapsed,
        "startup": checks,
        "gamma": gammas.tolist(),
        "decision": {k: v for k, v in decision.items() if k not in ("g", "q2_mean")},
        "g": decision["g"],
        "q2_mean": decision["q2_mean"],
        "diagnostics": diag,
        "thermodynamic_B": {str(N): 1.62 * float(np.sqrt(N)) for N in PREREG["N"]},
        "historical_B": 0.1,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["decision"], indent=2))
    print("seconds", elapsed)


if __name__ == "__main__":
    main()
