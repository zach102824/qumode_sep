"""Same-budget diversity test. Reads BUDGET_PREREG.json and does not edit it."""

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
OUT = ROOT / "budget"
PRE = json.loads((ROOT / "BUDGET_PREREG.json").read_text())
B = float("-0.3444")
TAU = float(PRE["tau"])
CAP = int(PRE["per_procedure_ceiling"])
E0_N9 = -6.337088433486192


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


def both(E, F, bar):
    return bool(E < bar[0] - TAU and F > bar[1] + TAU)


def step(target, spec, x, L, maxfun, counter, ceiling):
    shells = np.asarray(x[3:], dtype=np.float64).reshape(L - 1, 3)
    x0 = np.concatenate([[x[0], x[1], x[2]], resample(shells, L).reshape(-1)])
    if counter[0] >= ceiling:
        return {"L": L, "censored": True, "reason": "ceiling"}
    def fun(z):
        _, E, g, _ = h.logical_grad(target, z)
        return E, g
    x_new, trace = lbfgs(fun, x0, maxfun, counter, ceiling)
    if x_new is None or counter[0] >= ceiling:
        return {"L": L, "censored": True, "nfev": len(trace), "reason": "ceiling"}
    psi, E, g, var = h.logical_grad(target, x_new)
    counter[0] += 1
    return {"L": L, "censored": False, "E": float(E), "energy_error": float(E - spec["E0"]),
            "F": h.fidelity(spec["gs"], psi), "var_H": float(var), "grad_norm": float(np.linalg.norm(g)),
            "nfev": len(trace) + 1, "x": x_new.tolist(), "state_hash": h.state_hash(psi)}


def run_size(n, seed_head, deadline, e0_check=None, n_chains=48):
    target = h.Target(n, B)
    spec = h.ground_state(target)
    if e0_check is not None:
        assert abs(spec["E0"] - e0_check) < 1e-9
    records = []
    executed = [0]
    for k in range(n_chains):
        if time.perf_counter() > deadline:
            records.append({"arm": "meta", "k": k, "censored": True, "reason": "wall"})
            continue
        l1 = []
        l1_counter = [0]
        for s in range(6):
            rng = np.random.default_rng(np.random.SeedSequence([seed_head, n, 0, k, s]))
            x0 = rng.normal(0.0, 0.35, size=6)
            def fun(z):
                _, E, g, _ = h.logical_grad(target, z)
                return E, g
            x, trace = lbfgs(fun, x0, 50, l1_counter, 10**9)
            psi, E, g, _ = h.logical_grad(target, x)
            l1_counter[0] += 1
            l1.append({"s": s, "E": float(E), "energy_error": float(E - spec["E0"]),
                       "F": h.fidelity(spec["gs"], psi), "grad_norm": float(np.linalg.norm(g)),
                       "nfev": len(trace) + 1, "x": x.tolist(), "state_hash": h.state_hash(psi)})
        executed[0] += l1_counter[0]
        best = min(l1, key=lambda r: (r["E"], r["s"]))
        pool = [r for r in l1 if r["E"] > best["E"] + 0.3]
        second = min(pool, key=lambda r: (r["E"], r["s"])) if pool else None
        # control: one continuation, ceiling includes the shared L1 count
        cc = [l1_counter[0]]
        cx = np.asarray(best["x"])
        control_path = []
        for L, mf in ((2, 250), (3, 250), (4, 250), (5, 250), (6, 250)):
            row = step(target, spec, cx, L, mf, cc, CAP)
            control_path.append(row)
            if row.get("censored"):
                break
            cx = np.asarray(row["x"])
        executed[0] += cc[0] - l1_counter[0]
        # treatment
        tc = [l1_counter[0]]
        gx = np.asarray(best["x"])
        greedy = []
        for L, mf in ((2, 40), (3, 70), (4, 90), (5, 80)):
            row = step(target, spec, gx, L, mf, tc, CAP)
            greedy.append(row)
            if row.get("censored"):
                break
            gx = np.asarray(row["x"])
        other = []
        sx = None if second is None else np.asarray(second["x"])
        if second is not None:
            for L, mf in ((2, 30), (3, 50), (4, 70), (5, 150)):
                row = step(target, spec, sx, L, mf, tc, CAP)
                other.append(row)
                if row.get("censored"):
                    break
                sx = np.asarray(row["x"])
        g5 = next((r for r in greedy if r.get("L") == 5 and not r.get("censored")), None)
        o5 = next((r for r in other if r.get("L") == 5 and not r.get("censored")), None)
        choose_second = o5 is not None and (g5 is None or o5["E"] < g5["E"])
        wx = np.asarray(o5["x"] if choose_second else g5["x"]) if (o5 if choose_second else g5) else None
        winner_L6 = {"L": 6, "censored": True, "reason": "no L5"} if wx is None else step(target, spec, wx, 6, 250, tc, CAP)
        executed[0] += tc[0] - l1_counter[0]
        records.append({
            "k": k, "l1": l1, "l1_evals": l1_counter[0],
            "best_s": best["s"], "second_s": None if second is None else second["s"],
            "second_L1_error": None if second is None else second["energy_error"],
            "control_path": control_path, "control_evals": cc[0],
            "greedy": greedy, "second_path": other, "choose_second": choose_second,
            "winner_L6": winner_L6, "treatment_evals": tc[0],
        })
        if (k + 1) % 8 == 0:
            print(f"  chain {k+1} executed {executed[0]}", flush=True)
    return spec, records, executed[0]


def path_at(rec, arm, L):
    if L == 1:
        best = min(rec["l1"], key=lambda r: (r["E"], r["s"]))
        return best
    if arm == "control":
        hit = [r for r in rec["control_path"] if r.get("L") == L and not r.get("censored")]
        return hit[0] if hit else None
    if L <= 4:
        hit = [r for r in rec["greedy"] if r.get("L") == L and not r.get("censored")]
        return hit[0] if hit else None
    if L == 5:
        g5 = next((r for r in rec["greedy"] if r.get("L") == 5 and not r.get("censored")), None)
        o5 = next((r for r in rec["second_path"] if r.get("L") == 5 and not r.get("censored")), None)
        if rec["choose_second"] and o5 is not None:
            return o5
        return g5
    row = rec["winner_L6"]
    return None if row.get("censored") else row


def summarize(spec, records):
    bars = PRE["qaoa_E_F"]
    out = {"control": [], "treatment": []}
    complete = [r for r in records if "l1" in r]
    for arm in ("control", "treatment"):
        for L in range(1, 7):
            rows = []
            for rec in complete:
                row = path_at(rec, arm, L)
                if row is not None:
                    rows.append(row)
            wins = sum(both(r["E"], r["F"], bars[L - 1]) for r in rows)
            errs = [r["energy_error"] for r in rows]
            out[arm].append({
                "L": L, "n": len(rows), "BOTH": wins, "wilson": wilson(wins, len(rows)),
                "median_error": None if not errs else float(np.median(errs)),
                "q25_error": None if not errs else float(np.percentile(errs, 25)),
                "q75_error": None if not errs else float(np.percentile(errs, 75)),
                "worst_error": None if not errs else float(max(errs)),
                "best_error": None if not errs else float(min(errs)),
                "worst_F": None if not rows else float(min(r["F"] for r in rows)),
                "best_F_of_best_energy": None if not rows else float(min(rows, key=lambda r: r["E"])["F"]),
            })
    raw = [item for rec in complete for item in rec["l1"]]
    raw_both = sum(both(r["E"], r["F"], bars[0]) for r in raw)
    paired = []
    rare_c = rare_t = 0
    switched = 0
    for rec in complete:
        c6 = path_at(rec, "control", 6)
        t6 = path_at(rec, "treatment", 6)
        if rec["choose_second"]:
            switched += 1
        if c6 and c6["energy_error"] < 0.28:
            rare_c += 1
        if t6 and t6["energy_error"] < 0.28:
            rare_t += 1
        if c6 and t6:
            paired.append(t6["energy_error"] - c6["energy_error"])
    paired = np.asarray(paired)
    better = int(np.sum(paired < -1e-6)) if len(paired) else 0
    return {
        "n_procedures": len(complete),
        "depths": out,
        "raw_L1": len(raw), "raw_L1_BOTH": raw_both, "raw_wilson": wilson(raw_both, len(raw)),
        "paired_n": int(len(paired)),
        "paired_median_treatment_minus_control": None if not len(paired) else float(np.median(paired)),
        "treatment_lower_energy": better,
        "switched_to_second": switched,
        "rare_control": rare_c, "rare_treatment": rare_t,
        "rare_wilson_treatment": wilson(rare_t, len(complete)),
        "control_evals": [r["control_evals"] for r in complete],
        "treatment_evals": [r["treatment_evals"] for r in complete],
    }


def replay(spec, records):
    target = h.Target(9, B)
    finals = []
    for rec in records:
        if "winner_L6" not in rec or rec["winner_L6"].get("censored"):
            continue
        finals.append(rec["winner_L6"])
    if not finals:
        return []
    picks = [min(finals, key=lambda r: r["E"]), max(finals, key=lambda r: r["E"])]
    out = []
    for row in picks:
        x = np.asarray(row["x"])
        psi, E, g, _ = h.logical_grad(target, x)
        out.append({"logged_E": row["E"], "replay_E": float(E), "delta": float(E - row["E"]),
                    "hash_match": h.state_hash(psi) == row["state_hash"],
                    "F": h.fidelity(spec["gs"], psi), "grad_norm": float(np.linalg.norm(g)),
                    "energy_error": float(E - spec["E0"])})
    return out


def render(summary, replay_rows, executed):
    lines = [
        "# Same-budget second branch",
        "",
        "Ceiling 946 evaluations per procedure, the mean of the earlier six-start run. The six L1 starts are executed once and counted in both procedures. Tau is 1e-9. The QAOA bars are the current equal-depth states and were not improved by the last rescue at N=9.",
        "",
        f"Executed optimizer evaluations, with L1 not double-counted: {executed}.",
        "",
    ]
    for arm in ("control", "treatment"):
        lines.append(f"## {arm}")
        lines.append("")
        lines.append("| L | n | BOTH | Wilson | median | worst | best | worst F |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for d in summary["depths"][arm]:
            w = d["wilson"]
            wtxt = "—" if not w else f"[{w[0]:.3f}, {w[1]:.3f}]"
            def fmt(v):
                return "—" if v is None else f"{v:.4f}"
            lines.append(f"| {d['L']} | {d['n']} | {d['BOTH']} | {wtxt} | {fmt(d['median_error'])} | {fmt(d['worst_error'])} | {fmt(d['best_error'])} | {fmt(d['worst_F'])} |")
        lines.append("")
    lines.append(f"Raw L1 starts BOTH {summary['raw_L1_BOTH']}/{summary['raw_L1']}, Wilson {summary['raw_wilson']}. These 288 starts are not 288 procedures.")
    lines.append(f"Paired L6 error, treatment minus control, median {summary['paired_median_treatment_minus_control']} over {summary['paired_n']} pairs. Treatment lower on {summary['treatment_lower_energy']} pairs.")
    lines.append(f"Treatment selected the second branch on {summary['switched_to_second']} procedures.")
    lines.append(f"L6 error below 0.28: control {summary['rare_control']}, treatment {summary['rare_treatment']}, Wilson {summary['rare_wilson_treatment']}.")
    lines.append(f"Replay of the treatment best and worst: {replay_rows}.")
    lines.append("")
    lines.append("The six-start recipe is replaced only if this treatment lowers the median L6 error without losing BOTH wins. N=15 and N=18 were not run.")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + 40 * 60
    print("n9", flush=True)
    spec, records, executed = run_size(9, 7209, deadline, E0_N9)
    summary = summarize(spec, records)
    summary["executed_evals"] = executed
    summary["mean_control_evals"] = float(np.mean(summary.pop("control_evals")))
    summary["mean_treatment_evals"] = float(np.mean(summary.pop("treatment_evals")))
    reps = replay(spec, records)
    (OUT / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (OUT / "summary.json").write_text(json.dumps({**summary, "replay": reps}, indent=2) + "\n")
    (OUT / "REPORT.md").write_text(render(summary, reps, executed))
    print("switched", summary["switched_to_second"], "paired", summary["paired_median_treatment_minus_control"], flush=True)
    if summary["switched_to_second"] > 0 and time.perf_counter() < deadline:
        print("n12", flush=True)
        spec12, rec12, ex12 = run_size(12, 7212, deadline, n_chains=8)
        switched = sum(1 for r in rec12 if r.get("choose_second"))
        (OUT / "N12_probe.json").write_text(json.dumps({
            "chains": len([r for r in rec12 if "l1" in r]),
            "executed": ex12,
            "switched_to_second": switched,
            "note": "generalization probe, not a replacement test",
        }, indent=2) + "\n")
        (OUT / "N12_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec12))
        print("n12 switched", switched, "evals", ex12, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
