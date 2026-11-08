"""Fresh paired test of the frozen L1 stopping rule. Does not edit the prereg."""

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
OUT = ROOT / "screen_rule"
PRE = json.loads((ROOT / "SCREEN_PREREG.json").read_text())
B = float("-0.3444")
TAU = float(PRE["tau"])
CAP = int(PRE["per_procedure_ceiling"])
DELTA = 1e-4
PATIENCE = 2


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


def wilson(k, n):
    if not n:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def beats(E, F, bar):
    return bool(E < bar[0] - TAU and F > bar[1] + TAU)


def stop_index(starts):
    best = starts[0]["E"]
    stall = 0
    for i, row in enumerate(starts[1:], start=1):
        if row["E"] < best - DELTA:
            best = row["E"]
            stall = 0
        else:
            stall += 1
        if stall >= PATIENCE:
            return i + 1
    return len(starts)


def continue_from(target, spec, x, counter):
    path = []
    for L in range(2, 7):
        shells = x[3:].reshape(L - 1, 3)
        x0 = np.concatenate([[x[0], x[1], x[2]], resample(shells, L).ravel()])
        def fun(z):
            _, E, g, _ = h.logical_grad(target, z)
            return E, g
        xn, trace = lbfgs(fun, x0, 250, counter, CAP)
        if xn is None or counter[0] >= CAP:
            path.append({"L": L, "censored": True, "nfev": len(trace)})
            break
        psi, E, g, _ = h.logical_grad(target, xn)
        counter[0] += 1
        path.append({"L": L, "censored": False, "E": float(E), "error": float(E - spec["E0"]),
                     "F": h.fidelity(spec["gs"], psi), "grad": float(np.linalg.norm(g)),
                     "x": xn.tolist(), "hash": h.state_hash(psi), "nfev": len(trace) + 1})
        x = xn
    return path


def run_size(n, deadline):
    target = h.Target(n, B)
    spec = h.ground_state(target)
    seed_head = PRE["seeds"][str(n)]
    records = []
    executed = 0
    for k in range(48):
        if time.perf_counter() > deadline:
            records.append({"k": k, "censored": True, "reason": "wall"})
            continue
        starts = []
        l1_counter = [0]
        for s in range(6):
            rng = np.random.default_rng(np.random.SeedSequence(seed_head[:3] + [k, s]))
            x0 = rng.normal(0.0, 0.35, size=6)
            def fun(z):
                _, E, g, _ = h.logical_grad(target, z)
                return E, g
            xn, trace = lbfgs(fun, x0, 80, l1_counter, 10**9)
            if xn is None:
                starts.append({"s": s, "censored": True, "nfev": len(trace)})
                continue
            psi, E, g, _ = h.logical_grad(target, xn)
            l1_counter[0] += 1
            starts.append({"s": s, "censored": False, "E": float(E), "error": float(E - spec["E0"]),
                           "F": h.fidelity(spec["gs"], psi), "x": xn.tolist(),
                           "hash": h.state_hash(psi), "nfev": len(trace) + 1, "grad": float(np.linalg.norm(g))})
        good = [r for r in starts if not r.get("censored")]
        if len(good) < 6:
            records.append({"k": k, "censored": True, "reason": "l1", "starts": starts})
            executed += l1_counter[0]
            continue
        cut = stop_index(good)
        rule_pool = good[:cut]
        rule_win = min(rule_pool, key=lambda r: (r["E"], r["s"]))
        full_win = min(good, key=lambda r: (r["E"], r["s"]))
        rule_l1 = sum(r["nfev"] for r in rule_pool)
        full_l1 = l1_counter[0]
        # continuations
        cache = {}
        def follow(win, l1_cost):
            if win["s"] in cache:
                path, cost = cache[win["s"]]
                return path, l1_cost + cost
            counter = [l1_cost]
            # continuation counter starts at l1_cost so the cap is on the procedure total
            # but the cached continuation should be the marginal continuation cost
            c0 = [0]
            path = continue_from(target, spec, np.asarray(win["x"]), c0)
            cache[win["s"]] = (path, c0[0])
            return path, l1_cost + c0[0]
        rule_path, rule_cost = follow(rule_win, rule_l1)
        full_path, full_cost = follow(full_win, full_l1)
        executed += full_l1 + sum(v[1] for v in cache.values())
        records.append({
            "k": k, "censored": False, "cut": cut,
            "rule_s": rule_win["s"], "full_s": full_win["s"],
            "rule_cost": rule_cost, "full_cost": full_cost,
            "rule_L1": {kk: rule_win[kk] for kk in ("s", "E", "error", "F", "x", "hash")},
            "full_L1": {kk: full_win[kk] for kk in ("s", "E", "error", "F", "x", "hash")},
            "rule_path": rule_path, "full_path": full_path,
            "l1_errors": [r["error"] for r in good],
        })
        if (k + 1) % 8 == 0:
            print(f"  N{n} {k+1} executed {executed} cut {cut}", flush=True)
    return spec, records, executed


def arm_rows(records, arm, L):
    out = []
    for rec in records:
        if rec.get("censored") or "rule_path" not in rec:
            continue
        if L == 1:
            out.append(rec["rule_L1"] if arm == "rule" else rec["full_L1"])
            continue
        path = rec["rule_path"] if arm == "rule" else rec["full_path"]
        hit = [s for s in path if s.get("L") == L and not s.get("censored")]
        if hit:
            out.append(hit[0])
    return out


def summarize(n, records):
    bars = {int(k): v for k, v in PRE["bars"][str(n)].items()}
    present = [r for r in records if not r.get("censored") and "rule_path" in r]
    summary = {"n_attempted": len(records), "n_with_l1": len(present)}
    for arm, key in (("rule", "rule_cost"), ("fixed6", "full_cost")):
        costs = [r[key] for r in present]
        depths = []
        for L in range(1, 7):
            rows = arm_rows(records, "rule" if arm == "rule" else "fixed6", L)
            # unconditional denominator is 48
            wins = 0
            for rec in records:
                if rec.get("censored") or "rule_path" not in rec:
                    continue
                if L == 1:
                    row = rec["rule_L1"] if arm == "rule" else rec["full_L1"]
                else:
                    path = rec["rule_path"] if arm == "rule" else rec["full_path"]
                    hit = [s for s in path if s.get("L") == L and not s.get("censored")]
                    row = hit[0] if hit else None
                if row is not None and beats(row["E"], row["F"], bars[L]):
                    wins += 1
            errs = [r["error"] for r in rows]
            depths.append({
                "L": L, "completed": len(rows), "BOTH": wins, "of": 48,
                "wilson": wilson(wins, 48),
                "median_error": None if not errs else float(np.median(errs)),
                "q25_error": None if not errs else float(np.percentile(errs, 25)),
                "q75_error": None if not errs else float(np.percentile(errs, 75)),
                "worst_error": None if not errs else float(max(errs)),
                "best_error": None if not errs else float(min(errs)),
            })
        joint = 0
        for rec in records:
            if rec.get("censored") or "rule_path" not in rec:
                continue
            ok = True
            for L in range(1, 7):
                if L == 1:
                    row = rec["rule_L1"] if arm == "rule" else rec["full_L1"]
                else:
                    path = rec["rule_path"] if arm == "rule" else rec["full_path"]
                    hit = [s for s in path if s.get("L") == L and not s.get("censored")]
                    row = hit[0] if hit else None
                if row is None or not beats(row["E"], row["F"], bars[L]):
                    ok = False
                    break
            if ok:
                joint += 1
        rare = 0
        if n == 9:
            for rec in present:
                path = rec["rule_path"] if arm == "rule" else rec["full_path"]
                hit = [s for s in path if s.get("L") == 6 and not s.get("censored")]
                if hit and hit[0]["error"] < 0.28:
                    rare += 1
        summary[arm] = {
            "depths": depths, "joint_all_depth": joint, "joint_of": 48,
            "joint_wilson": wilson(joint, 48),
            "cost_median": float(np.median(costs)) if costs else None,
            "cost_q25": float(np.percentile(costs, 25)) if costs else None,
            "cost_q75": float(np.percentile(costs, 75)) if costs else None,
            "cost_worst": float(max(costs)) if costs else None,
            "cost_total_charged": int(sum(costs)) if costs else 0,
            "completed_L6": depths[-1]["completed"],
            "rare_L6_below_0.28": rare,
        }
    summary["cuts"] = [r["cut"] for r in present]
    summary["same_choice"] = sum(r["rule_s"] == r["full_s"] for r in present)
    return summary


def replay(n, row):
    target = h.Target(n, B)
    H = h.dense_h(target)
    x = np.asarray(row["x"])
    psi, E, g, _ = h.logical_grad(target, x)
    Ed = float(np.vdot(psi, H @ psi).real)
    return {"logged_E": row["E"], "dense_E": Ed, "delta": Ed - row["E"],
            "hash_match": h.state_hash(psi) == row["hash"], "grad": float(np.linalg.norm(g)),
            "error": float(E - h.ground_state(target)["E0"]), "F": row["F"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + 50 * 60
    all_sum = {}
    replays = {}
    for n in (9, 6, 12):
        print("size", n, flush=True)
        spec, records, executed = run_size(n, deadline)
        (OUT / f"N{n}_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        summary = summarize(n, records)
        summary["executed_union"] = executed
        all_sum[str(n)] = summary
        present = [r for r in records if not r.get("censored") and "rule_path" in r]
        finals = []
        for rec in present:
            hit = [s for s in rec["rule_path"] if s.get("L") == 6 and not s.get("censored")]
            if hit:
                finals.append(hit[0])
        if finals:
            replays[str(n)] = {
                "rule_best": replay(n, min(finals, key=lambda r: r["E"])),
                "rule_worst": replay(n, max(finals, key=lambda r: r["E"])),
            }
        print(" N", n, "rule L6", summary["rule"]["depths"][-1]["BOTH"],
              "fixed", summary["fixed6"]["depths"][-1]["BOTH"],
              "cut mean", float(np.mean(summary["cuts"])) if summary["cuts"] else None, flush=True)
    (OUT / "summary.json").write_text(json.dumps({"sizes": all_sum, "replays": replays}, indent=2) + "\n")
    lines = ["# L1 stopping rule, fresh paired test", "",
             "One shared gamma. The rule stops after two L1 starts that do not beat the best by more than 1e-4, and never draws more than six. Fixed six is the control on the same seeds. The denominator for BOTH is 48, including censored chains.",
             ""]
    for n, summary in all_sum.items():
        lines.append(f"## N={n}")
        lines.append("")
        lines.append(f"Executed union {summary['executed_union']}. Same L1 choice on {summary['same_choice']} chains. Cuts {summary['cuts'][:12]}...")
        lines.append("")
        lines.append("| arm | L6 BOTH | joint all L | L6 completed | cost median | cost worst | cost total |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in ("rule", "fixed6"):
            a = summary[arm]
            lines.append(f"| {arm} | {a['depths'][-1]['BOTH']}/48 | {a['joint_all_depth']}/48 | {a['completed_L6']}/48 | {a['cost_median']:.0f} | {a['cost_worst']:.0f} | {a['cost_total_charged']} |")
        lines.append("")
        lines.append("| arm | L | BOTH | Wilson | median | worst |")
        lines.append("|---|---|---|---|---|---|")
        for arm in ("rule", "fixed6"):
            for d in summary[arm]["depths"]:
                w = d["wilson"]
                wtxt = "—" if not w else f"[{w[0]:.3f}, {w[1]:.3f}]"
                med = "—" if d["median_error"] is None else f"{d['median_error']:.4f}"
                worst = "—" if d["worst_error"] is None else f"{d['worst_error']:.4f}"
                lines.append(f"| {arm} | {d['L']} | {d['BOTH']}/48 | {wtxt} | {med} | {worst} |")
        if n == "9":
            lines.append("")
            lines.append(f"Secondary, L6 error below 0.28: rule {summary['rule']['rare_L6_below_0.28']}, fixed6 {summary['fixed6']['rare_L6_below_0.28']}.")
        lines.append("")
    lines.append(f"Replays: {replays}.")
    lines.append("")
    lines.append("Per-layer gamma is not part of this test. N=15 and N=18 were not run.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
