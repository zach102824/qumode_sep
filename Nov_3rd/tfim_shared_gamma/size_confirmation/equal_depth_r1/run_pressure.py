"""Pressure-test the stored six-start states. Does not edit the prereg."""

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
SIZE = ROOT.parent
GAMMA = SIZE.parent
OUT = ROOT / "pressure"
PRE = json.loads((ROOT / "PRESSURE_PREREG.json").read_text())
B = float("-0.3444")
TAU = 1e-9


def load_jsonl(path):
    rows = []
    for line in Path(path).open():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def wilson(k, n):
    if not n:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def both(E, F, bar):
    return bool(E < bar[0] - TAU and F > bar[1] + TAU)


def best_by_p(rows):
    best = {}
    for r in rows:
        if r.get("censored") or r.get("p") is None:
            continue
        if not str(r.get("arm", "")).startswith("qaoa"):
            continue
        p = int(r["p"])
        if p not in best or r["E"] < best[p]["E"]:
            best[p] = r
    return best


def bars_from_files():
    out = {}
    for n in (6, 9):
        rows = []
        for rel in PRE["bar_files"][str(n)]:
            rows.extend(load_jsonl(SIZE / rel))
        best = best_by_p(rows)
        out[n] = {p: [best[p]["E"], best[p]["F"]] for p in range(1, 13)}
    rows = []
    for rel in PRE["bar_files"]["12_equal"]:
        rows.extend(load_jsonl(SIZE / rel))
    best = best_by_p(rows)
    out[12] = {p: [best[p]["E"], best[p]["F"]] for p in range(1, 7)}
    for rel in PRE["bar_files"]["12_matched_cells"]:
        cell = json.loads((GAMMA / rel).read_text())
        p = int(cell["L"])
        out[12][p] = [cell["E"], cell["F"]]
    return out


def procedures(n):
    rows = load_jsonl(ROOT / "results" / f"N{n}_records.jsonl")
    return [r for r in rows if r.get("arm") == "procedure" and not r.get("censored")]


def score_equal(bars, n):
    proc = procedures(n)
    rows = []
    for L in range(1, 7):
        bar = bars[n][L]
        group = [r for r in proc if r.get("L") == L]
        wins = []
        margins_e = []
        margins_f = []
        for r in group:
            ok = both(r["E"], r["F"], bar)
            wins.append(ok)
            margins_e.append(bar[0] - r["E"])
            margins_f.append(r["F"] - bar[1])
        rows.append({
            "L": L, "n": len(group), "BOTH": int(sum(wins)), "wilson": wilson(int(sum(wins)), len(group)),
            "median_error": float(np.median([r["energy_error"] for r in group])),
            "worst_error": float(max(r["energy_error"] for r in group)),
            "worst_energy_margin": float(min(margins_e)),
            "worst_F_margin": float(min(margins_f)),
            "bar_error": None,
            "bar_E": bar[0], "bar_F": bar[1],
        })
    return rows


def score_matched(bars, n, spec_E0):
    proc = procedures(n)
    rows = []
    for L in range(1, 7):
        p_match = int(np.ceil((3 * L + 3) / 2))
        group = [r for r in proc if r.get("L") == L]
        entry = {"L": L, "n": len(group)}
        for name, p in (("matched", p_match), ("next", p_match + 1), ("p12", 12)):
            if p not in bars[n]:
                entry[name] = None
                continue
            bar = bars[n][p]
            wins = sum(both(r["E"], r["F"], bar) for r in group)
            margins = [bar[0] - r["E"] for r in group]
            entry[name] = {
                "p": p, "BOTH": int(wins), "wilson": wilson(int(wins), len(group)),
                "bar_error": bar[0] - spec_E0[n], "bar_F": bar[1],
                "worst_energy_margin": float(min(margins)),
            }
        rows.append(entry)
    return rows


def endpoint_resample(values, p_new):
    src = np.asarray(values, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(src))
    xt = np.linspace(0.0, 1.0, p_new)
    if p_new <= 3 or len(src) < 3:
        return np.interp(xt, xs, src)
    return CubicSpline(xs, src, bc_type="natural")(xt)


def fourier(values):
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2:
        return y.copy()
    cols = [np.sin((k - 0.5) * (np.arange(len(y)) + 0.5) * np.pi / len(y)) for k in (1, 2)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return np.column_stack(cols) @ coef


def lbfgs(fun, x0, maxfun, counter, ceiling):
    trace = []

    def wrapped(x):
        if counter[0] >= ceiling:
            raise RuntimeError("ceiling")
        counter[0] += 1
        E, g = fun(x)
        trace.append(float(E))
        return E, g

    try:
        res = minimize(wrapped, np.asarray(x0, dtype=np.float64), method="L-BFGS-B", jac=True,
                       options={"maxfun": maxfun, "maxiter": maxfun, "ftol": 1e-12})
    except RuntimeError:
        return None, trace
    if res.x is None:
        return None, trace
    return np.asarray(res.x, dtype=np.float64), trace


def record_qaoa(target, spec, x, counter, trace):
    if counter[0] >= 10**12:
        return None
    psi, E, _, _ = h.qaoa_grad(target, x)
    counter[0] += 1
    return {"E": float(E), "F": h.fidelity(spec["gs"], psi), "energy_error": float(E - spec["E0"]),
            "nfev": len(trace) + 1, "x": np.asarray(x, dtype=np.float64).tolist(),
            "state_hash": h.state_hash(psi), "censored": False}


def run_qaoa_depths(n, ceiling, depths_extra, seed_head, deadline, full_ladder=True):
    target = h.Target(n, B)
    spec = h.ground_state(target)
    counter = [0]
    records = []
    schedules = [None] * 16
    if not full_ladder:
        extra = 0
        while counter[0] < ceiling and time.perf_counter() < deadline:
            p = depths_extra[0]
            extra += 1
            rng = np.random.default_rng(np.random.SeedSequence(seed_head + [6, extra]))
            kind = extra % 3
            if kind == 2 or not records:
                x0 = np.empty(2 * p)
                x0[0::2] = rng.uniform(-1.2, 1.2, size=p)
                x0[1::2] = rng.uniform(-1.0, 1.0, size=p)
                label = "random"
            else:
                parent = min(records, key=lambda r: r["E"])
                xp = np.asarray(parent["x"])
                if kind == 0:
                    x0 = np.array(xp, copy=True)
                    x0[0::2] = fourier(xp[0::2])
                    x0[1::2] = fourier(xp[1::2])
                    label = "fourier"
                else:
                    x0 = xp + rng.normal(0, 0.05, size=xp.size)
                    label = "noise"
            def fun(z):
                _, E, gv, _ = h.qaoa_grad(target, z)
                return E, gv
            x, trace = lbfgs(fun, x0, 80, counter, ceiling)
            if x is None:
                continue
            row = record_qaoa(target, spec, x, counter, trace)
            row.update({"arm": "qaoa_extra", "p": p, "k": extra, "label": label})
            records.append(row)
            if extra % 40 == 0:
                print(f"  p{p} {label} evals {counter[0]} best {min(r['energy_error'] for r in records):.4f}", flush=True)
        return spec, records, counter[0]
    for k in range(16):
        if counter[0] >= ceiling or time.perf_counter() > deadline:
            break
        best = None
        for s in range(6):
            rng = np.random.default_rng(np.random.SeedSequence(seed_head + [1, k, s]))
            x0 = rng.uniform(-0.8, 0.8, size=2)
            def fun(z):
                _, E, g, _ = h.qaoa_grad(target, z)
                return E, g
            x, trace = lbfgs(fun, x0, 40, counter, ceiling)
            if x is None:
                continue
            row = record_qaoa(target, spec, x, counter, trace)
            row.update({"arm": "qaoa", "p": 1, "k": k, "s": s})
            records.append(row)
            if best is None or row["E"] < best[0]:
                best = (row["E"], x)
        if best is None:
            continue
        schedules[k] = best[1]
        for p in range(2, 7):
            if counter[0] >= ceiling or time.perf_counter() > deadline:
                break
            g = schedules[k][0::2]
            b = schedules[k][1::2]
            if p == 2:
                base = np.array([g[0], b[0], g[0] + 0.2, b[0] + 0.06888])
            else:
                base = np.empty(2 * p)
                base[0::2] = endpoint_resample(g, p)
                base[1::2] = endpoint_resample(b, p)
            rng = np.random.default_rng(np.random.SeedSequence(seed_head + [p, k]))
            variants = [base, base + rng.normal(0, 0.05, size=base.size)]
            four = np.array(base, copy=True)
            four[0::2] = fourier(base[0::2])
            four[1::2] = fourier(base[1::2])
            variants.append(four)
            local = None
            for s, x0 in enumerate(variants):
                if counter[0] >= ceiling:
                    break
                def fun(z):
                    _, E, gv, _ = h.qaoa_grad(target, z)
                    return E, gv
                x, trace = lbfgs(fun, x0, 60, counter, ceiling)
                if x is None:
                    continue
                row = record_qaoa(target, spec, x, counter, trace)
                row.update({"arm": "qaoa", "p": p, "k": k, "s": s})
                records.append(row)
                if local is None or row["E"] < local[0]:
                    local = (row["E"], x)
            if local is not None:
                schedules[k] = local[1]
        print(f"  chain {k} evals {counter[0]}", flush=True)
    extra = 0
    while counter[0] < ceiling and time.perf_counter() < deadline:
        p = depths_extra[extra % len(depths_extra)]
        extra += 1
        pool = [r for r in records if r.get("p") == p and not r.get("censored")]
        if not pool:
            break
        parent = min(pool, key=lambda r: r["E"])
        rng = np.random.default_rng(np.random.SeedSequence(seed_head + [4, p, extra]))
        kind = extra % 3
        xp = np.asarray(parent["x"], dtype=np.float64)
        if kind == 0:
            x0 = np.array(xp, copy=True)
            x0[0::2] = fourier(xp[0::2])
            x0[1::2] = fourier(xp[1::2])
            label = "fourier"
        elif kind == 1:
            x0 = xp + rng.normal(0, 0.05, size=xp.size)
            label = "noise"
        else:
            x0 = np.empty(2 * p)
            x0[0::2] = rng.uniform(-1.2, 1.2, size=p)
            x0[1::2] = rng.uniform(-1.0, 1.0, size=p)
            label = "random"
        def fun(z):
            _, E, gv, _ = h.qaoa_grad(target, z)
            return E, gv
        x, trace = lbfgs(fun, x0, 60, counter, ceiling)
        if x is None:
            continue
        row = record_qaoa(target, spec, x, counter, trace)
        row.update({"arm": "qaoa_extra", "p": p, "k": extra, "label": label})
        records.append(row)
        if extra % 50 == 0:
            print(f"  extra {extra} p{p} evals {counter[0]}", flush=True)
    return spec, records, counter[0]


def centers(n):
    return (np.arange(n) + 0.5) / n


def resample(shells, n_new):
    src = np.asarray(shells, dtype=np.float64)
    xs = centers(len(src))
    xt = np.clip(centers(n_new), xs[0], xs[-1])
    out = np.zeros((n_new, 3))
    cubic = len(src) >= 3
    for j in range(3):
        y = src[:, j]
        out[:, j] = CubicSpline(xs, y, bc_type="natural")(xt) if cubic else np.interp(xt, xs, y)
    return out


def continue_candidate(target, spec, x, counter, ceiling, maxfuns):
    path = []
    for L, mf in maxfuns:
        if counter[0] >= ceiling:
            path.append({"L": L, "censored": True})
            break
        x0 = np.concatenate([[x[0], x[1], x[2]], resample(x[3:].reshape(L - 1, 3), L).ravel()])
        def fun(z):
            _, E, g, _ = h.logical_grad(target, z)
            return E, g
        x_new, trace = lbfgs(fun, x0, mf, counter, ceiling)
        if x_new is None or counter[0] >= ceiling:
            path.append({"L": L, "censored": True, "nfev": len(trace)})
            break
        psi, E, g, _ = h.logical_grad(target, x_new)
        counter[0] += 1
        path.append({"L": L, "censored": False, "E": float(E), "energy_error": float(E - spec["E0"]),
                     "F": h.fidelity(spec["gs"], psi), "grad_norm": float(np.linalg.norm(g)),
                     "x": x_new.tolist(), "state_hash": h.state_hash(psi), "nfev": len(trace) + 1})
        x = x_new
    return path


def candidate_rescue(deadline):
    target = h.Target(9, B)
    spec = h.ground_state(target)
    counter = [0]
    ceiling = int(PRE["candidate_rescue"]["ceiling"])
    # resume the stored rare state
    raw = load_jsonl(SIZE / "N9_r1/results/records.jsonl")
    rare = next(r for r in raw if r.get("arm") == "candidate" and r.get("L") == 6 and r.get("k") == 40 and not r.get("censored"))
    x = np.asarray(rare["x"], dtype=np.float64)
    def fun(z):
        _, E, g, _ = h.logical_grad(target, z)
        return E, g
    x2, trace = lbfgs(fun, x, 600, counter, ceiling)
    if x2 is None:
        x2 = x
    psi, E, g, _ = h.logical_grad(target, x2)
    counter[0] += 1
    resumed = {"start_error": rare["energy_error"], "E": float(E), "energy_error": float(E - spec["E0"]),
               "F": h.fidelity(spec["gs"], psi), "grad_norm": float(np.linalg.norm(g)),
               "nfev": len(trace) + 1, "state_hash": h.state_hash(psi), "x": x2.tolist(),
               "label": "resume stored k=40, not a new procedure"}
    print("  resumed", resumed["energy_error"], resumed["grad_norm"], flush=True)
    # fresh second branches
    fresh = []
    for k in range(6):
        if counter[0] >= ceiling or time.perf_counter() > deadline:
            break
        raws = []
        for s in range(8):
            rng = np.random.default_rng(np.random.SeedSequence([7409, 9, 1, k, s]))
            x0 = rng.normal(0, 0.35, size=6)
            xx, tr = lbfgs(fun, x0, 50, counter, ceiling)
            if xx is None:
                continue
            psi, E, _, _ = h.logical_grad(target, xx)
            counter[0] += 1
            raws.append({"E": float(E), "s": s, "x": xx.tolist(), "energy_error": float(E - spec["E0"])})
        if not raws:
            continue
        best = min(raws, key=lambda r: (r["E"], r["s"]))
        pool = [r for r in raws if r["E"] > best["E"] + 0.3]
        second = min(pool, key=lambda r: (r["E"], r["s"])) if pool else None
        branch = second if second is not None else best
        path = continue_candidate(target, spec, np.asarray(branch["x"]), counter, ceiling,
                                  [(2, 200), (3, 200), (4, 200), (5, 200), (6, 600)])
        done = [r for r in path if r.get("L") == 6 and not r.get("censored")]
        fresh.append({"k": k, "used_second": second is not None,
                      "L1_error": branch["energy_error"],
                      "L6": done[0] if done else {"censored": True}})
        print(f"  fresh {k} evals {counter[0]}", flush=True)
    # local restarts of the common basin
    proc = [r for r in procedures(9) if r.get("L") == 6]
    local = []
    for r in proc:
        if counter[0] >= ceiling or time.perf_counter() > deadline:
            break
        rng = np.random.default_rng(np.random.SeedSequence([7409, 9, 2, r["k"]]))
        x0 = np.asarray(r["x"]) + rng.normal(0, 0.2, size=len(r["x"]))
        xx, tr = lbfgs(fun, x0, 40, counter, ceiling)
        if xx is None:
            continue
        psi, E, g, _ = h.logical_grad(target, xx)
        counter[0] += 1
        local.append({"k": r["k"], "start_error": r["energy_error"], "energy_error": float(E - spec["E0"]),
                      "F": h.fidelity(spec["gs"], psi), "grad_norm": float(np.linalg.norm(g)),
                      "improved": E < r["E"] - TAU})
    return {"evals": counter[0], "resumed": resumed, "fresh": fresh, "local_improved": sum(1 for r in local if r["improved"]),
            "local_n": len(local), "local_best_error": None if not local else min(r["energy_error"] for r in local)}


def replay_pair(n, row, kind):
    target = h.Target(n, B)
    spec = h.ground_state(target)
    x = np.asarray(row["x"])
    if kind == "qaoa":
        psi = h.qaoa_state(target, x[0::2], x[1::2])
        Hpsi = target.hc * psi + target.B * h.apply_sum_x(psi, n)
        E = float(np.vdot(psi, Hpsi).real)
        gnorm = None
    else:
        psi, E, g, _ = h.logical_grad(target, x)
        gnorm = float(np.linalg.norm(g))
    return {"logged_E": row["E"], "replay_E": float(E), "delta": float(E - row["E"]),
            "hash_match": h.state_hash(psi) == row["state_hash"], "grad_norm": gnorm,
            "F": h.fidelity(spec["gs"], psi)}


def fmt_row(d):
    w = d["wilson"]
    wtxt = "—" if not w else f"[{w[0]:.3f}, {w[1]:.3f}]"
    return f"| {d['L']} | {d['BOTH']}/{d['n']} | {wtxt} | {d['median_error']:.4f} | {d['worst_error']:.4f} | {d['worst_energy_margin']:.4f} | {d['worst_F_margin']:.4f} |"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + 45 * 60
    bars = bars_from_files()
    specs = {}
    for n in (6, 9, 12):
        specs[n] = h.ground_state(h.Target(n, B))["E0"]
    before_eq = {n: score_equal(bars, n) for n in (6, 9, 12)}
    before_m = {n: score_matched(bars, n, specs) for n in (6, 9, 12)}
    (OUT / "before.json").write_text(json.dumps({"equal": before_eq, "matched": before_m}, indent=2) + "\n")
    print("n12 qaoa", flush=True)
    spec12, rec12, spent12 = run_qaoa_depths(12, 50489, [6, 5, 4], [7412, 12, 5], deadline)
    (OUT / "N12_qaoa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec12))
    print("n12 spent", spent12, flush=True)
    for r in rec12:
        if r.get("censored"):
            continue
        p = r["p"]
        if p not in bars[12] or r["E"] < bars[12][p][0]:
            bars[12][p] = [r["E"], r["F"]]
    n6_spent = 0
    if time.perf_counter() < deadline - 8 * 60:
        print("n6 qaoa", flush=True)
        spec6, rec6, n6_spent = run_qaoa_depths(6, 12000, [6], [7406, 6, 5], deadline, full_ladder=False)
        (OUT / "N6_qaoa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec6))
        for r in rec6:
            if not r.get("censored") and r["E"] < bars[6][r["p"]][0]:
                bars[6][r["p"]] = [r["E"], r["F"]]
    print("candidate rescue", flush=True)
    rescue = candidate_rescue(deadline)
    (OUT / "rescue.json").write_text(json.dumps(rescue, indent=2) + "\n")
    after_eq = {str(n): score_equal(bars, n) for n in (6, 9, 12)}
    after_m = {str(n): score_matched(bars, n, specs) for n in (6, 9, 12)}
    # replay new N12 p6 best and N9 stored worst/best already known; replay resumed and a stored procedure
    p6 = [r for r in rec12 if r.get("p") == 6 and not r.get("censored")]
    replays = {"N12_p6": replay_pair(12, min(p6, key=lambda r: r["E"]), "qaoa") if p6 else None}
    proc9 = [r for r in procedures(9) if r.get("L") == 6]
    replays["N9_procedure_worst"] = replay_pair(9, max(proc9, key=lambda r: r["E"]), "candidate")
    (OUT / "summary.json").write_text(json.dumps({
        "n12_qaoa_evals": spent12, "n12_ceiling": 50489, "n12_shortfall": max(0, 50489 - spent12),
        "n6_qaoa_evals": n6_spent, "n6_ceiling": 12000 if n6_spent else 0,
        "equal": after_eq, "matched": after_m, "rescue": rescue, "replays": replays,
    }, indent=2) + "\n")
    lines = ["# Six-start pressure test", "",
             "The 48 stored procedures were not retrained. Tau is 1e-9. Equal depth and parameter-matched depths are separate ledgers. The L counts are the same chains, not independent samples.",
             "", f"N=12 new QAOA evaluations {spent12} of 50489. N=6 extra {n6_spent}.", ""]
    lines.append("## Equal depth")
    lines.append("")
    lines.append("| N | L | BOTH | Wilson | median error | worst error | worst energy margin | worst F margin |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n in (6, 9, 12):
        for d in after_eq[str(n)]:
            w = d["wilson"]
            wtxt = "—" if not w else f"[{w[0]:.3f}, {w[1]:.3f}]"
            lines.append(f"| {n} | {d['L']} | {d['BOTH']}/{d['n']} | {wtxt} | {d['median_error']:.4f} | {d['worst_error']:.4f} | {d['worst_energy_margin']:.4f} | {d['worst_F_margin']:.4f} |")
    lines.append("")
    lines.append("## Parameter matched, next depth, and p=12")
    lines.append("")
    lines.append("| N | L | matched BOTH | next BOTH | p12 BOTH | matched bar error |")
    lines.append("|---|---|---|---|---|---|")
    for n in (6, 9, 12):
        for d in after_m[str(n)]:
            def cell(name):
                item = d[name]
                if not item:
                    return "—"
                return f"p{item['p']}: {item['BOTH']}/{d['n']}"
            bar = "—" if not d["matched"] else f"{d['matched']['bar_error']:.4f}"
            lines.append(f"| {n} | {d['L']} | {cell('matched')} | {cell('next')} | {cell('p12')} | {bar} |")
    lines.append("")
    lines.append(f"Shared-gamma rescue, not a new 48-procedure sample: {rescue['evals']} evaluations. Resume of stored k=40 ended at error {rescue['resumed']['energy_error']:.6f}, gradient {rescue['resumed']['grad_norm']:.3e}. Local restarts improved {rescue['local_improved']} of {rescue['local_n']}; best error {rescue['local_best_error']}.")
    lines.append(f"Replays: {replays}.")
    lines.append("")
    lines.append("A failed search is not an expressivity ceiling. N=15 and N=18 were not run.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("done", spent12, n6_spent, flush=True)


if __name__ == "__main__":
    main()
