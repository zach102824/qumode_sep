"""Held-out equal-depth screen. Reads PREREG.json and does not edit it."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as h

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PREREG = json.loads((ROOT / "PREREG.json").read_text())
B = float("-0.3444")
TAU = float(PREREG["tau"])
S = int(PREREG["L1_starts"])
L1_MAX = int(PREREG["L1_maxfun"])
CONT_MAX = int(PREREG["continuation_maxfun"])


def centers(n):
    return (np.arange(n) + 0.5) / n


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


def lbfgs(fun, x0, maxfun, counter):
    trace = []

    def wrapped(x):
        counter[0] += 1
        E, g = fun(x)
        trace.append(float(E))
        return E, g

    res = minimize(wrapped, np.asarray(x0, dtype=np.float64), method="L-BFGS-B", jac=True,
                   options={"maxfun": maxfun, "maxiter": maxfun, "ftol": 1e-12})
    if res.x is None:
        return None, trace
    return np.asarray(res.x, dtype=np.float64), trace


def wilson(k, n):
    if n == 0:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, center - half)), float(min(1, center + half))]


def classify(E, F, bar):
    ewin = E < bar[0] - TAU
    fwin = F > bar[1] + TAU
    if ewin and fwin:
        return "BOTH"
    if ewin and not fwin:
        return "betterE/worseF"
    if (not ewin) and fwin:
        return "worseE/betterF"
    return "worseboth"


def observe(target, spec, x, L):
    shells = np.asarray(x[3:], dtype=np.float64).reshape(L, 3)
    psi, E, grad, var = h.logical_grad(target, x)
    return {
        "E": E,
        "energy_error": E - spec["E0"],
        "F": h.fidelity(spec["gs"], psi),
        "var_H": var,
        "eta": h.eta(E, spec["E0"], spec["Emax"]),
        "grad_norm": float(np.linalg.norm(grad)),
        "state_hash": h.state_hash(psi),
        "plus_sector_weight": h.parity_weight(psi),
        "x": x.tolist(),
    }


def run_size(n, seed_head, bars, deadline, qaoa_mode):
    print(f"size {n}", flush=True)
    target = h.Target(n, B)
    if n <= 12:
        spec = h.ground_state(target)
    else:
        raise SystemExit("this slice does not run above N=12")
    print(f"  E0 {spec['E0']:.12f} gap {spec['full_gap']:.6e}", flush=True)
    counter = [0]
    records = []
    path = OUT / f"N{n}_records.jsonl"
    path.write_text("")

    def flush(row):
        records.append(row)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    for k in range(48):
        if time.perf_counter() > deadline:
            flush({"arm": "procedure", "n": n, "k": k, "censored": True, "reason": "wall"})
            continue
        raw = []
        for s in range(S):
            rng = np.random.default_rng(np.random.SeedSequence([seed_head, n, 0, k, s]))
            x0 = rng.normal(0.0, 0.35, size=6)
            def fun(z):
                _, E, g, _ = h.logical_grad(target, z)
                return E, g
            x, trace = lbfgs(fun, x0, L1_MAX, counter)
            if x is None:
                flush({"arm": "l1_raw", "n": n, "k": k, "s": s, "censored": True, "nfev": len(trace)})
                continue
            psi, E, _, _ = h.logical_grad(target, x)
            counter[0] += 1
            row = observe(target, spec, x, 1)
            row.update({"arm": "l1_raw", "n": n, "k": k, "s": s, "L": 1, "censored": False,
                        "nfev": len(trace) + 1, "seed": [seed_head, n, 0, k, s]})
            flush(row)
            raw.append(row)
        if not raw:
            flush({"arm": "procedure", "n": n, "k": k, "censored": True, "reason": "no L1"})
            continue
        chosen = min(raw, key=lambda r: (r["E"], r["s"]))
        x = np.asarray(chosen["x"], dtype=np.float64)
        chosen_mark = dict(chosen)
        chosen_mark["arm"] = "procedure"
        chosen_mark["selected_s"] = chosen["s"]
        flush(chosen_mark)
        for L in range(2, 7):
            if time.perf_counter() > deadline:
                flush({"arm": "procedure", "n": n, "k": k, "L": L, "censored": True, "reason": "wall"})
                break
            shells = x[3:].reshape(L - 1, 3)
            x0 = np.concatenate([[x[0], x[1], x[2]], resample_shells(shells, L).reshape(-1)])
            def fun(z, L=L):
                _, E, g, _ = h.logical_grad(target, z)
                return E, g
            x_new, trace = lbfgs(fun, x0, CONT_MAX, counter)
            if x_new is None:
                flush({"arm": "procedure", "n": n, "k": k, "L": L, "censored": True, "nfev": len(trace)})
                break
            psi, E, _, _ = h.logical_grad(target, x_new)
            counter[0] += 1
            row = observe(target, spec, x_new, L)
            row.update({"arm": "procedure", "n": n, "k": k, "L": L, "censored": False,
                        "nfev": len(trace) + 1, "selected_s": chosen["s"]})
            flush(row)
            x = x_new
        if (k + 1) % 8 == 0:
            print(f"  chain {k + 1} evals {counter[0]}", flush=True)

    qaoa_evals = 0
    if qaoa_mode == "fresh":
        qaoa_evals = run_qaoa(target, spec, n, counter[0], deadline, records, path)
    summary = summarize(n, records, spec, counter[0], qaoa_evals, bars)
    (OUT / f"N{n}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  done N{n} evals {counter[0]} qaoa {qaoa_evals}", flush=True)
    return summary


def run_qaoa(target, spec, n, floor, deadline, records, path):
    print(f"  qaoa floor {floor}", flush=True)
    counter = [0]
    schedules = [None] * 16

    def flush(row):
        records.append(row)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    for p in range(1, 7):
        if time.perf_counter() > deadline:
            break
        for k in range(16):
            if schedules[k] is None and p > 1:
                continue
            if p == 1:
                bases = []
                for s in range(3):
                    rng = np.random.default_rng(np.random.SeedSequence([4212, n, 1, p, k, s]))
                    bases.append(rng.uniform(-0.8, 0.8, size=2))
            elif p == 2:
                g1, b1 = schedules[k]
                base = np.array([g1, b1, g1 + 0.2, b1 + B * (-0.2)])
                bases = [base]
                for s in range(2):
                    rng = np.random.default_rng(np.random.SeedSequence([4212, n, 1, p, k, s]))
                    bases.append(base + rng.normal(0, 0.05, size=4))
            else:
                g = schedules[k][0::2]
                b = schedules[k][1::2]
                base = np.empty(2 * p)
                base[0::2] = endpoint_resample(g, p)
                base[1::2] = endpoint_resample(b, p)
                bases = [base]
                for s in range(2):
                    rng = np.random.default_rng(np.random.SeedSequence([4212, n, 1, p, k, s]))
                    bases.append(base + rng.normal(0, 0.05, size=base.size))
            best_local = None
            for s, x0 in enumerate(bases):
                if counter[0] >= floor or time.perf_counter() > deadline:
                    flush({"arm": "qaoa", "n": n, "p": p, "k": k, "s": s, "censored": True})
                    continue
                def fun(z):
                    _, E, g, _ = h.qaoa_grad(target, z)
                    return E, g
                x, trace = lbfgs(fun, x0, 80, counter)
                if x is None:
                    flush({"arm": "qaoa", "n": n, "p": p, "k": k, "s": s, "censored": True, "nfev": len(trace)})
                    continue
                psi, E, _, _ = h.qaoa_grad(target, x)
                counter[0] += 1
                row = {
                    "arm": "qaoa", "n": n, "p": p, "k": k, "s": s, "censored": False,
                    "E": E, "F": h.fidelity(spec["gs"], psi), "energy_error": E - spec["E0"],
                    "nfev": len(trace) + 1, "x": x.tolist(), "state_hash": h.state_hash(psi),
                }
                flush(row)
                if best_local is None or E < best_local[0]:
                    best_local = (E, x)
            if best_local is not None:
                schedules[k] = best_local[1]
        done = [r for r in records if r.get("arm") == "qaoa" and r.get("p") == p and not r.get("censored")]
        if done:
            best = min(done, key=lambda r: r["E"])
            print(f"  p{p} {best['energy_error']:.4f} {best['F']:.4f} qevals {counter[0]}", flush=True)
    extra = 0
    while counter[0] < floor and time.perf_counter() < deadline:
        p = (6, 5, 4)[extra % 3]
        extra += 1
        pool = [r for r in records if r.get("arm") == "qaoa" and r.get("p") == p and not r.get("censored")]
        if not pool:
            break
        parent = min(pool, key=lambda r: r["E"])
        rng = np.random.default_rng(np.random.SeedSequence([4212, n, 2, p, extra]))
        x0 = np.asarray(parent["x"]) + rng.normal(0, 0.05, size=2 * p)
        def fun(z):
            _, E, g, _ = h.qaoa_grad(target, z)
            return E, g
        x, trace = lbfgs(fun, x0, 80, counter)
        if x is None:
            continue
        psi, E, _, _ = h.qaoa_grad(target, x)
        counter[0] += 1
        flush({"arm": "qaoa_extra", "n": n, "p": p, "k": extra, "censored": False, "E": E,
               "F": h.fidelity(spec["gs"], psi), "energy_error": E - spec["E0"],
               "nfev": len(trace) + 1, "x": x.tolist(), "state_hash": h.state_hash(psi)})
    return counter[0]


def summarize(n, records, spec, cand_evals, qaoa_evals, stored_bars):
    fresh = {}
    for L in range(1, 7):
        pool = [r for r in records if r.get("p") == L and not r.get("censored") and str(r.get("arm", "")).startswith("qaoa")]
        if pool:
            best = min(pool, key=lambda r: r["E"])
            fresh[L] = [best["E"], best["F"], best.get("arm")]
    depths = []
    for L in range(1, 7):
        rows = [r for r in records if r.get("arm") == "procedure" and r.get("L") == L and not r.get("censored")]
        stored = stored_bars[L - 1]
        bar = stored
        bar_source = "stored"
        if L in fresh and fresh[L][0] < stored[0]:
            bar = fresh[L][:2]
            bar_source = "this_slice"
        counts = {"BOTH": 0, "betterE/worseF": 0, "worseE/betterF": 0, "worseboth": 0}
        errs = []
        for r in rows:
            label = classify(r["E"], r["F"], bar)
            counts[label] += 1
            errs.append(r["energy_error"])
        errs = np.asarray(errs) if errs else np.array([np.nan])
        depths.append({
            "L": L, "n": len(rows), "counts": counts,
            "wilson_BOTH": wilson(counts["BOTH"], len(rows)),
            "median_error": None if not rows else float(np.median(errs)),
            "q25_error": None if not rows else float(np.percentile(errs, 25)),
            "q75_error": None if not rows else float(np.percentile(errs, 75)),
            "best_error": None if not rows else float(np.min(errs)),
            "best_F": None if not rows else float(max(rows, key=lambda r: -r["E"])["F"]),
            "bar_E": bar[0], "bar_F": bar[1], "bar_source": bar_source,
            "bar_error": bar[0] - spec["E0"],
        })
    raw = [r for r in records if r.get("arm") == "l1_raw" and not r.get("censored")]
    raw_both = 0
    if raw:
        bar = stored_bars[0]
        if 1 in fresh and fresh[1][0] < bar[0]:
            bar = fresh[1][:2]
        raw_both = sum(classify(r["E"], r["F"], bar) == "BOTH" for r in raw)
    # replay the best stored L6 procedure
    replay = None
    finals = [r for r in records if r.get("arm") == "procedure" and r.get("L") == 6 and not r.get("censored")]
    if finals:
        best = min(finals, key=lambda r: r["E"])
        x = np.asarray(best["x"])
        psi, E, _, _ = h.logical_grad(h.Target(n, B), x)
        replay = {"logged_E": best["E"], "replay_E": E, "delta": E - best["E"],
                  "hash_match": h.state_hash(psi) == best["state_hash"]}
    return {
        "n": n, "E0": spec["E0"], "full_gap": spec["full_gap"],
        "candidate_evals": cand_evals, "qaoa_evals": qaoa_evals,
        "qaoa_shortfall": max(0, cand_evals - qaoa_evals) if qaoa_evals else None,
        "depths": depths,
        "raw_L1_starts": len(raw), "raw_L1_BOTH": raw_both,
        "raw_L1_wilson": wilson(raw_both, len(raw)),
        "replay_L6": replay,
    }


def render(summaries):
    lines = [
        "# Equal-depth L=1 screen, held out",
        "",
        "Recipe frozen in `PREREG.json` before these chains. Six energy-only L=1 starts, then one continuation. Fidelity was not used to choose a start. No angle was copied from N=6, N=9, or N=12.",
        "",
    ]
    for s in summaries:
        lines.append(f"## N={s['n']}")
        lines.append("")
        lines.append(f"E0 = {s['E0']:.12f}. Candidate optimizer evaluations {s['candidate_evals']}.")
        if s["qaoa_evals"]:
            lines.append(f"Fresh equal-depth QAOA evaluations {s['qaoa_evals']}. Shortfall against the candidate ledger: {s['qaoa_shortfall']}.")
        lines.append("")
        lines.append("| L | procedures | BOTH | Wilson | worseE/betterF | worse both | median error | bar error | bar |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for d in s["depths"]:
            c = d["counts"]
            w = d["wilson_BOTH"]
            wtxt = "—" if w is None else f"[{w[0]:.3f}, {w[1]:.3f}]"
            med = "—" if d["median_error"] is None else f"{d['median_error']:.4f}"
            bar = "—" if d["bar_error"] is None else f"{d['bar_error']:.4f}"
            lines.append(f"| {d['L']} | {d['n']} | {c['BOTH']} | {wtxt} | {c['worseE/betterF']} | {c['worseboth']} | {med} | {bar} | {d['bar_source']} |")
        lines.append("")
        lines.append(f"Raw L=1 starts, before the screen keeps one: {s['raw_L1_BOTH']} / {s['raw_L1_starts']} BOTH. Wilson {s['raw_L1_wilson']}.")
        lines.append(f"Replay of the stored L=6 best: {s['replay_L6']}.")
        lines.append("")
    lines.append("Eta is not fidelity. A chain that misses is not a proof that no angles of this generator can win. N=15 was not continued and N=18 was not started.")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    deadline = t0 + 50 * 60
    plan = [
        (9, 4209, PREREG["qaoa_bars_N9_E_F"], None),
        (6, 4206, PREREG["qaoa_bars_N6_E_F"], None),
        (12, 4212, PREREG["qaoa_bars_N12_historical_round3_E_F"], "fresh"),
    ]
    summaries = []
    for n, seed, bars, mode in plan:
        if time.perf_counter() > deadline:
            print("wall before size", n, flush=True)
            break
        summaries.append(run_size(n, seed, bars, deadline, mode))
        (OUT / "REPORT.md").write_text(render(summaries))
    print("finished", len(summaries), flush=True)


if __name__ == "__main__":
    main()
