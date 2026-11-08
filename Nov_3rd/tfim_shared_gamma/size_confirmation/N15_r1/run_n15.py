"""N=15 fixed-field confirmation. Reads PREREG.json and does not edit it."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h
import reference as ref

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PREREG = json.loads((ROOT / "PREREG.json").read_text())
B = float("-0.3444")
CAND_CAP = 100000
TOTAL_CAP = 800000
MAXFUN = 40


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen():
    if sha_file(ROOT / "target_manifest.json") != PREREG["manifest_sha256"]:
        raise SystemExit("manifest hash")
    if sha_file(ROOT / "packing.json") != PREREG["packing_sha256"]:
        raise SystemExit("packing hash")
    if sha_file(ROOT.parents[1] / "size_confirmation" / "harness.py") != PREREG["harness_sha256"]:
        raise SystemExit("harness hash")
    if sha_file(ROOT / "reference.py") != PREREG["reference_sha256"]:
        raise SystemExit("reference hash")


def clip_box(x, limit):
    return np.clip(np.asarray(x, dtype=np.float64), -limit, limit)


def lbfgs(fun, x0, maxfun, counter, cap):
    trace = []

    def wrapped(x):
        if counter[0] >= cap:
            raise RuntimeError("cap")
        E, g = fun(np.asarray(x, dtype=np.float64))
        counter[0] += 1
        trace.append(float(E))
        return float(E), np.asarray(g, dtype=np.float64)

    try:
        res = minimize(wrapped, np.asarray(x0, dtype=np.float64), method="L-BFGS-B", jac=True,
                       options={"maxiter": 1500, "maxfun": maxfun, "ftol": 1e-14, "gtol": 1e-8})
    except RuntimeError:
        return None, trace
    return np.asarray(res.x, dtype=np.float64), trace


def centers(n):
    return (np.arange(1, n + 1) - 0.5) / n


def resample_shells(shells, n_new):
    src = np.asarray(shells, dtype=np.float64)
    xs = centers(len(src))
    xt = np.clip(centers(n_new), xs[0], xs[-1])
    cubic = len(src) >= 3
    out = np.zeros((n_new, 3))
    for k in range(3):
        y = src[:, k]
        out[:, k] = CubicSpline(xs, y, bc_type="natural")(xt) if cubic else np.interp(xt, xs, y)
    return out


def endpoint_resample(values, p_new):
    src = np.asarray(values, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(src))
    xt = np.linspace(0.0, 1.0, p_new)
    if p_new <= 3 or len(src) < 3:
        return np.interp(xt, xs, src)
    return CubicSpline(xs, src, bc_type="natural")(xt)


def fourier(values):
    y = np.asarray(values, dtype=np.float64)
    L = len(y)
    if L < 2:
        return y.copy()
    cols = [np.sin((k - 0.5) * (np.arange(L) + 0.5) * np.pi / L) for k in (1, 2)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


def pack_obs(psi, E, spec):
    return {
        "E": E,
        "energy_error": E - spec["E0"],
        "F": h.fidelity(spec["gs"], psi),
        "var_H": None,
        "eta": h.eta(E, spec["E0"], spec["Emax"]),
        "norm": float(np.linalg.norm(psi)),
        "plus_sector_weight": h.parity_weight(psi),
        "state_hash": h.state_hash(psi),
    }


def wilson(k, n):
    if n == 0:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, center - half)), float(min(1, center + half))]


def main():
    assert_frozen()
    t0 = time.perf_counter()
    deadline = t0 + 68 * 60
    target = h.Target(15, B)
    spec = ref.ground_state(target)
    from scipy.sparse.linalg import LinearOperator, eigsh
    op = LinearOperator(
        (target.dim, target.dim),
        matvec=lambda v: ref.apply_h(target, np.asarray(v, dtype=np.complex128)).real,
        dtype=np.float64,
    )
    spec["Emax"] = float(eigsh(op, k=1, which="LA", tol=1e-10, maxiter=3000, return_eigenvectors=False)[0])
    print("spectrum", spec["E0"], spec["Emax"], spec["full_gap"], flush=True)
    counter = [0]
    records = []

    def observe(psi, E):
        Hpsi = target.hc * psi + target.B * h.apply_sum_x(psi, target.n)
        var = float(np.vdot(Hpsi, Hpsi).real - E * E)
        row = pack_obs(psi, E, spec)
        row["var_H"] = var
        return row

    print("candidate", flush=True)
    chains = [None] * 48
    for L in range(1, 7):
        for k in range(48):
            if counter[0] >= CAND_CAP or time.perf_counter() > deadline:
                records.append({"arm": "candidate", "L": L, "k": k, "censored": True})
                continue
            if L == 1:
                rng = np.random.default_rng(np.random.SeedSequence([3115, 15, 0, L, k]))
                x0 = rng.normal(0.0, 0.35, size=3 + 3 * L)
            else:
                if chains[k] is None:
                    records.append({"arm": "candidate", "L": L, "k": k, "censored": True})
                    continue
                g, tk, tm = chains[k][:3]
                shells = chains[k][3:].reshape(L - 1, 3)
                x0 = np.concatenate([[g, tk, tm], resample_shells(shells, L).reshape(-1)])
            def fun(x, L=L):
                _, E, g, _ = h.logical_grad(target, x)
                return E, g
            x, trace = lbfgs(fun, x0, MAXFUN, counter, CAND_CAP)
            if x is None:
                records.append({"arm": "candidate", "L": L, "k": k, "censored": True, "nfev": len(trace)})
                continue
            psi, E, _, _ = h.logical_grad(target, x)
            counter[0] += 1
            row = observe(psi, E)
            row.update({"arm": "candidate", "L": L, "k": k, "censored": False, "nfev": len(trace) + 1,
                        "x": x.tolist(), "seed": [3115, 15, 0, L, k]})
            records.append(row)
            chains[k] = x
        done = [r for r in records if r.get("arm") == "candidate" and r.get("L") == L and not r.get("censored")]
        if done:
            best = min(done, key=lambda r: r["E"])
            print(f"  L{L} best {best['energy_error']:.6f} F {best['F']:.4f} n {len(done)} evals {counter[0]}", flush=True)

    cand_evals = counter[0]
    qaoa_floor = min(4 * cand_evals, TOTAL_CAP - cand_evals)
    print("qaoa floor frozen from candidate ledger", qaoa_floor, "candidate", cand_evals, flush=True)
    qcounter = [0]
    schedules = [None] * 16
    for p in range(1, 13):
        if time.perf_counter() > deadline or cand_evals + qcounter[0] >= TOTAL_CAP:
            break
        for k in range(16):
            if schedules[k] is None and p > 1:
                continue
            if p == 1:
                bases = []
                for s in range(3):
                    rng = np.random.default_rng(np.random.SeedSequence([3115, 15, 1, p, k, s]))
                    bases.append(("random", rng.uniform(-0.8, 0.8, size=2)))
            elif p == 2:
                g1, b1 = schedules[k]
                base = np.array([g1, b1, g1 + 0.2, b1 + 0.06888])
                bases = [("p2_mapped", base)]
                for s in range(2):
                    rng = np.random.default_rng(np.random.SeedSequence([3115, 15, 1, p, k, s]))
                    bases.append(("p2_noise", base + rng.normal(0, 0.05, size=4)))
            else:
                g = schedules[k][0::2]
                b = schedules[k][1::2]
                base = np.empty(2 * p)
                base[0::2] = endpoint_resample(g, p)
                base[1::2] = endpoint_resample(b, p)
                bases = [("endpoint", base)]
                for s in range(2):
                    rng = np.random.default_rng(np.random.SeedSequence([3115, 15, 1, p, k, s]))
                    bases.append(("endpoint_noise", base + rng.normal(0, 0.05, size=base.size)))
            best_local = None
            for s, (label, x0) in enumerate(bases):
                if qcounter[0] >= qaoa_floor or time.perf_counter() > deadline:
                    records.append({"arm": "qaoa", "p": p, "k": k, "s": s, "censored": True, "label": label})
                    continue
                def fun(x):
                    _, E, g, _ = h.qaoa_grad(target, x)
                    return E, g
                x, trace = lbfgs(fun, x0, MAXFUN, qcounter, qaoa_floor)
                if x is None:
                    records.append({"arm": "qaoa", "p": p, "k": k, "s": s, "censored": True, "label": label, "nfev": len(trace)})
                    continue
                psi, E, _, _ = h.qaoa_grad(target, x)
                qcounter[0] += 1
                row = observe(psi, E)
                row.update({"arm": "qaoa", "p": p, "k": k, "s": s, "label": label, "censored": False,
                            "nfev": len(trace) + 1, "x": x.tolist()})
                records.append(row)
                if best_local is None or E < best_local[0]:
                    best_local = (E, x)
            if best_local is not None:
                schedules[k] = best_local[1]
        done = [r for r in records if r.get("arm") == "qaoa" and r.get("p") == p and not r.get("censored")]
        if done:
            best = min(done, key=lambda r: r["E"])
            print(f"  p{p} best {best['energy_error']:.6f} F {best['F']:.4f} n {len(done)} qevals {qcounter[0]}", flush=True)

    # Priority extras: random, Fourier, center, cycling p12, p11, p9.
    cycle = [12, 11, 9]
    extra_i = 0
    while qcounter[0] < qaoa_floor and time.perf_counter() < deadline and cand_evals + qcounter[0] < TOTAL_CAP:
        p = cycle[extra_i % 3]
        extra_i += 1
        pool = [r for r in records if r.get("arm") == "qaoa" and r.get("p") == p and not r.get("censored")]
        if not pool:
            break
        parent = min(pool, key=lambda r: r["E"])
        x_parent = np.asarray(parent["x"], dtype=np.float64)
        kind = extra_i % 5
        rng = np.random.default_rng(np.random.SeedSequence([3115, 15, 2, p, extra_i]))
        if kind == 0:
            label = "fourier"
            x0 = np.empty_like(x_parent)
            x0[0::2] = fourier(x_parent[0::2])
            x0[1::2] = fourier(x_parent[1::2])
        elif kind == 1:
            label = "center"
            x0 = np.empty_like(x_parent)
            # center knots on the parent schedule of the same length: identity, plus noise so it is a restart
            x0 = x_parent + rng.normal(0, 0.05, size=x_parent.size)
        else:
            label = "random"
            x0 = np.empty(2 * p)
            x0[0::2] = rng.uniform(-1.2, 1.2, size=p)
            x0[1::2] = rng.uniform(-1.0, 1.0, size=p)
        def fun(x):
            _, E, g, _ = h.qaoa_grad(target, x)
            return E, g
        x, trace = lbfgs(fun, x0, MAXFUN, qcounter, qaoa_floor)
        if x is None:
            records.append({"arm": "qaoa_extra", "p": p, "k": extra_i, "label": label, "censored": True, "nfev": len(trace)})
            continue
        psi, E, _, _ = h.qaoa_grad(target, x)
        qcounter[0] += 1
        row = observe(psi, E)
        row.update({"arm": "qaoa_extra", "p": p, "k": extra_i, "label": label, "censored": False,
                    "nfev": len(trace) + 1, "x": x.tolist()})
        records.append(row)
        if extra_i % 30 == 0:
            print(f"  extra {extra_i} p{p} {label} qevals {qcounter[0]}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "records.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    sparse = ref.sparse_h(target)
    checks = []
    for label, pred, rebuild in (
        ("L6", lambda r: r.get("arm") == "candidate" and r.get("L") == 6 and not r.get("censored"),
         lambda row: h.logical_candidate(target, row["x"][0], row["x"][1], row["x"][2], np.asarray(row["x"][3:]).reshape(6, 3))),
        ("p12", lambda r: r.get("p") == 12 and not r.get("censored"),
         lambda row: h.qaoa_state(target, np.asarray(row["x"][0::2]), np.asarray(row["x"][1::2]))),
    ):
        pool = [r for r in records if pred(r)]
        if not pool:
            continue
        row = min(pool, key=lambda r: r["E"])
        psi = rebuild(row)
        Es = float(np.vdot(psi, sparse @ psi).real)
        checks.append({
            "label": label,
            "logged_E": row["E"],
            "sparse_E": Es,
            "delta": Es - row["E"],
            "hash_match": h.state_hash(psi) == row["state_hash"],
            "residual": float(np.linalg.norm(sparse @ psi - Es * psi)),
        })
    summary = summarize(records, spec, cand_evals, qcounter[0], t0)
    summary["independent_sparse_checks"] = checks
    summary["qaoa_floor"] = qaoa_floor
    summary["qaoa_shortfall"] = max(0, qaoa_floor - qcounter[0])
    summary["qaoa_floor_formula"] = "4 * completed candidate optimizer evaluations, capped by 800000 minus that ledger"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "REPORT.md").write_text(render(summary))
    plot(records)
    print("done", summary["candidate_evals"], summary["qaoa_evals"], flush=True)


def matched(L):
    return int(np.ceil((3 * L + 3) / 2))


def summarize(records, spec, cand_evals, qaoa_evals, t0):
    def finished(arm, **kw):
        rows = [r for r in records if r.get("arm") == arm and not r.get("censored") and all(r.get(a) == b for a, b in kw.items())]
        return rows
    depths = []
    for L in range(1, 7):
        rows = finished("candidate", L=L)
        if not rows:
            depths.append({"L": L, "n": 0})
            continue
        err = np.array([r["energy_error"] for r in rows])
        best = min(rows, key=lambda r: r["E"])
        q_rows = [r for r in records if r.get("p") == L and not r.get("censored") and r.get("arm", "").startswith("qaoa")]
        qbest = min(q_rows, key=lambda r: r["E"]) if q_rows else None
        mp = matched(L)
        mq = [r for r in records if r.get("p") == mp and not r.get("censored") and r.get("arm", "").startswith("qaoa")]
        mbest = min(mq, key=lambda r: r["E"]) if mq else None
        nxt = [r for r in records if r.get("p") == mp + 1 and not r.get("censored") and r.get("arm", "").startswith("qaoa")]
        nbest = min(nxt, key=lambda r: r["E"]) if nxt else None
        both_eq = None if qbest is None else (best["E"] < qbest["E"] - 1e-9 and best["F"] > qbest["F"] + 1e-9)
        both_m = None if mbest is None else (best["E"] < mbest["E"] - 1e-9 and best["F"] > mbest["F"] + 1e-9)
        depths.append({
            "L": L, "P": 3 * L + 3, "n": len(rows),
            "best_E": best["E"], "best_error": best["energy_error"], "best_F": best["F"],
            "best_var": best["var_H"], "best_eta": best["eta"], "best_hash": best["state_hash"], "best_k": best["k"],
            "median_error": float(np.median(err)),
            "q25_error": float(np.percentile(err, 25)),
            "q75_error": float(np.percentile(err, 75)),
            "equal_L_qaoa_error": None if qbest is None else qbest["energy_error"],
            "equal_L_qaoa_F": None if qbest is None else qbest["F"],
            "both_vs_equal_L": both_eq,
            "matched_p": mp,
            "matched_error": None if mbest is None else mbest["energy_error"],
            "matched_F": None if mbest is None else mbest["F"],
            "both_vs_matched": both_m,
            "next_p": mp + 1,
            "next_error": None if nbest is None else nbest["energy_error"],
            "next_F": None if nbest is None else nbest["F"],
        })
    p12 = [r for r in records if r.get("p") == 12 and not r.get("censored")]
    p12_best = min(p12, key=lambda r: r["E"]) if p12 else None
    return {
        "E0": spec["E0"], "Emax": spec["Emax"], "full_gap": spec["full_gap"],
        "candidate_evals": cand_evals, "qaoa_evals": qaoa_evals,
        "qaoa_floor": None, "qaoa_shortfall": None,
        "total_evals": cand_evals + qaoa_evals,
        "wall_s": time.perf_counter() - t0,
        "depths": depths,
        "p12": None if p12_best is None else {"E": p12_best["E"], "F": p12_best["F"], "error": p12_best["energy_error"], "n": len(p12)},
        "feature_rank": 3,
    }


def render(summary):
    lines = [
        "# N=15 fixed-field confirmation",
        "",
        "Target `de5f47b1bd1b4e3cfced8694c892d41f8cab9eb43f41ed285c3fccff6c9b0892`, "
        "packing `dcd102b8bc45bec941dc9f3e474781ba9cf19c12dd893d068e83d50e09d41074`. "
        f"B = -0.3444. E0 = {summary['E0']:.12f}, full gap {summary['full_gap']:.6e}. "
        "The ground state is a Krylov vector, not a dense diagonalization. "
        "The full gap is measured. The first excited state is the opposite-parity partner.",
        "",
        "| L | best error | F | median | q25 | q75 | equal-L QAOA | matched QAOA | both equal | both matched |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in summary["depths"]:
        if d.get("n", 0) == 0:
            lines.append(f"| {d['L']} | | | | | | | | | |")
            continue
        lines.append(
            f"| {d['L']} | {d['best_error']:.6f} | {d['best_F']:.4f} | {d['median_error']:.4f} | "
            f"{d['q25_error']:.4f} | {d['q75_error']:.4f} | {d['equal_L_qaoa_error']} | "
            f"p{d['matched_p']} {d['matched_error']} | {d['both_vs_equal_L']} | {d['both_vs_matched']} |"
        )
    lines += [
        "",
        f"Candidate optimizer evaluations {summary['candidate_evals']} of the ceiling 100000. "
        f"QAOA optimizer evaluations {summary['qaoa_evals']} of the frozen floor {summary['qaoa_floor']}. "
        f"Shortfall {summary['qaoa_shortfall']}.",
        "",
        "A missing QAOA cell is an incomplete comparison, not a win. "
        "Eta is post hoc and is not fidelity. The published 94–96% figures are eta on a different target.",
        "",
        f"Wall {summary['wall_s']:.1f} s.",
    ]
    return "\n".join(lines) + "\n"


def plot(records):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for L, color in zip(range(1, 7), ["C0", "C1", "C2", "C3", "C4", "C5"]):
        err = np.sort([r["energy_error"] for r in records if r.get("arm") == "candidate" and r.get("L") == L and not r.get("censored")])
        if len(err):
            ax.step(err, np.arange(1, len(err) + 1) / len(err), where="post", color=color, label=f"L{L}")
    ax.set_xlabel("energy error")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "candidate_cdf.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
