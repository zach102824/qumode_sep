"""Equal-depth QAOA rescue, then the frozen N=9 two-basin check. Does not edit the prereg."""

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
OUT = ROOT / "rescue"
PREREG = json.loads((ROOT / "RESCUE_PREREG.json").read_text())
B = float("-0.3444")
TAU = float(PREREG["tau"])
SHIFT_B = 0.06888


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


def fourier(values):
    y = np.asarray(values, dtype=np.float64)
    L = len(y)
    if L < 2:
        return y.copy()
    cols = [np.sin((k - 0.5) * (np.arange(L) + 0.5) * np.pi / L) for k in (1, 2)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


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
    if n == 0:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def both(E, F, bar):
    return bool(E < bar[0] - TAU and F > bar[1] + TAU)


def run_qaoa(n, spec, deadline):
    cfg = PREREG["qaoa"][f"N{n}"]
    ceiling = int(cfg["new_ceiling"])
    depths = cfg["depths"]
    seed_head = cfg["seed"]
    counter = [0]
    records = []
    target = h.Target(n, B)
    schedules = [None] * 16

    def flush(row):
        records.append(row)

    for k in range(16):
        if time.perf_counter() > deadline or counter[0] >= ceiling:
            break
        best = None
        for s in range(6):
            if counter[0] >= ceiling:
                flush({"arm": "qaoa", "n": n, "p": 1, "k": k, "s": s, "censored": True})
                continue
            rng = np.random.default_rng(np.random.SeedSequence(seed_head + [1, k, s]))
            x0 = rng.uniform(-0.8, 0.8, size=2)
            def fun(z):
                _, E, g, _ = h.qaoa_grad(target, z)
                return E, g
            x, trace = lbfgs(fun, x0, 80, counter, ceiling)
            if x is None:
                flush({"arm": "qaoa", "n": n, "p": 1, "k": k, "s": s, "censored": True, "nfev": len(trace)})
                continue
            psi, E, _, _ = h.qaoa_grad(target, x)
            counter[0] += 1
            row = {"arm": "qaoa", "n": n, "p": 1, "k": k, "s": s, "censored": False, "E": E,
                   "F": h.fidelity(spec["gs"], psi), "energy_error": E - spec["E0"],
                   "nfev": len(trace) + 1, "x": x.tolist(), "state_hash": h.state_hash(psi)}
            flush(row)
            if best is None or E < best[0]:
                best = (E, x)
        if best is None:
            continue
        schedules[k] = best[1]
        for p in depths:
            if p == 1:
                continue
            if time.perf_counter() > deadline:
                break
            g = schedules[k][0::2]
            b = schedules[k][1::2]
            if p == 2:
                base = np.array([g[0], b[0], g[0] + 0.2, b[0] + SHIFT_B])
            else:
                base = np.empty(2 * p)
                base[0::2] = endpoint_resample(g, p)
                base[1::2] = endpoint_resample(b, p)
            rng = np.random.default_rng(np.random.SeedSequence(seed_head + [p, k, 1]))
            variants = [base, base + rng.normal(0, 0.05, size=base.size)]
            four = np.array(base, copy=True)
            four[0::2] = fourier(base[0::2])
            four[1::2] = fourier(base[1::2])
            variants.append(four)
            local = None
            for s, x0 in enumerate(variants):
                if counter[0] >= ceiling or time.perf_counter() > deadline:
                    flush({"arm": "qaoa", "n": n, "p": p, "k": k, "s": s, "censored": True})
                    continue
                def fun(z):
                    _, E, gvec, _ = h.qaoa_grad(target, z)
                    return E, gvec
                x, trace = lbfgs(fun, x0, 80, counter, ceiling)
                if x is None:
                    flush({"arm": "qaoa", "n": n, "p": p, "k": k, "s": s, "censored": True, "nfev": len(trace)})
                    continue
                psi, E, _, _ = h.qaoa_grad(target, x)
                counter[0] += 1
                row = {"arm": "qaoa", "n": n, "p": p, "k": k, "s": s, "censored": False, "E": E,
                       "F": h.fidelity(spec["gs"], psi), "energy_error": E - spec["E0"],
                       "nfev": len(trace) + 1, "x": x.tolist(), "state_hash": h.state_hash(psi)}
                flush(row)
                if local is None or E < local[0]:
                    local = (E, x)
            if local is not None:
                schedules[k] = local[1]
        print(f"  N{n} chain {k} evals {counter[0]}", flush=True)

    extra = 0
    cycle = cfg["extra_cycle"]
    while counter[0] < ceiling and time.perf_counter() < deadline:
        p = cycle[extra % len(cycle)]
        extra += 1
        pool = [r for r in records if r.get("p") == p and not r.get("censored")]
        if not pool:
            break
        parent = min(pool, key=lambda r: r["E"])
        rng = np.random.default_rng(np.random.SeedSequence(seed_head + [4, p, extra]))
        kind = extra % 3
        x_parent = np.asarray(parent["x"], dtype=np.float64)
        if kind == 0:
            x0 = np.array(x_parent, copy=True)
            x0[0::2] = fourier(x_parent[0::2])
            x0[1::2] = fourier(x_parent[1::2])
            label = "fourier"
        elif kind == 1:
            x0 = x_parent + rng.normal(0, 0.05, size=x_parent.size)
            label = "noise"
        else:
            x0 = np.empty(2 * p)
            x0[0::2] = rng.uniform(-1.2, 1.2, size=p)
            x0[1::2] = rng.uniform(-1.0, 1.0, size=p)
            label = "random"
        def fun(z):
            _, E, gvec, _ = h.qaoa_grad(target, z)
            return E, gvec
        x, trace = lbfgs(fun, x0, 80, counter, ceiling)
        if x is None:
            flush({"arm": "qaoa_extra", "n": n, "p": p, "k": extra, "label": label, "censored": True, "nfev": len(trace)})
            continue
        psi, E, _, _ = h.qaoa_grad(target, x)
        counter[0] += 1
        flush({"arm": "qaoa_extra", "n": n, "p": p, "k": extra, "label": label, "censored": False,
               "E": E, "F": h.fidelity(spec["gs"], psi), "energy_error": E - spec["E0"],
               "nfev": len(trace) + 1, "x": x.tolist(), "state_hash": h.state_hash(psi)})
        if extra % 40 == 0:
            print(f"  N{n} extra {extra} p{p} evals {counter[0]}", flush=True)
    path = OUT / f"N{n}_qaoa.jsonl"
    with path.open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    print(f"  N{n} qaoa new evals {counter[0]} ceiling {ceiling}", flush=True)
    return records, counter[0]


def compare(n, records, spec):
    hist = PREREG["historical_best_E_F"][str(n)]
    stored = [json.loads(l) for l in (ROOT / "results" / f"N{n}_records.jsonl").open() if l.strip()]
    proc = [r for r in stored if r.get("arm") == "procedure" and not r.get("censored")]
    depths = []
    for L in range(1, 7):
        fresh = [r for r in records if r.get("p") == L and not r.get("censored")]
        bar = hist[L - 1]
        source = "historical"
        fresh_best = None
        if fresh:
            fresh_best = min(fresh, key=lambda r: r["E"])
            if fresh_best["E"] < bar[0]:
                bar = [fresh_best["E"], fresh_best["F"]]
                source = "this_slice"
        rows = [r for r in proc if r.get("L") == L]
        wins = sum(both(r["E"], r["F"], bar) for r in rows)
        errs = [r["energy_error"] for r in rows]
        depths.append({
            "L": L, "n": len(rows), "BOTH": wins, "wilson": wilson(wins, len(rows)),
            "worst_error": None if not errs else float(max(errs)),
            "worst_F": None if not rows else float(min(r["F"] for r in rows)),
            "bar_E": bar[0], "bar_F": bar[1], "bar_error": bar[0] - spec["E0"], "bar_source": source,
            "fresh_starts": len(fresh),
            "fresh_best_error": None if fresh_best is None else fresh_best["energy_error"],
            "fresh_best_F": None if fresh_best is None else fresh_best["F"],
            "historical_error": hist[L - 1][0] - spec["E0"],
        })
    # replay the new best at the deepest depth this slice ran
    ran = sorted({r["p"] for r in records if not r.get("censored")})
    replay = None
    if ran:
        p = ran[-1]
        pool = [r for r in records if r.get("p") == p and not r.get("censored")]
        best = min(pool, key=lambda r: r["E"])
        target = h.Target(n, B)
        psi = h.qaoa_state(target, np.asarray(best["x"][0::2]), np.asarray(best["x"][1::2]))
        Hpsi = target.hc * psi + target.B * h.apply_sum_x(psi, n)
        E = float(np.vdot(psi, Hpsi).real)
        replay = {"p": p, "logged_E": best["E"], "replay_E": E, "delta": E - best["E"],
                  "hash_match": h.state_hash(psi) == best["state_hash"],
                  "F": h.fidelity(spec["gs"], psi)}
    return {"n": n, "E0": spec["E0"], "depths": depths, "replay": replay}


def continue_from(target, spec, x, counter, ceiling):
    rows = []
    for L in range(2, 7):
        if counter[0] >= ceiling:
            rows.append({"L": L, "censored": True, "reason": "ceiling"})
            break
        shells = x[3:].reshape(L - 1, 3)
        x0 = np.concatenate([[x[0], x[1], x[2]], resample_shells(shells, L).reshape(-1)])
        def fun(z, L=L):
            _, E, g, _ = h.logical_grad(target, z)
            return E, g
        x_new, trace = lbfgs(fun, x0, 250, counter, ceiling)
        if x_new is None:
            rows.append({"L": L, "censored": True, "nfev": len(trace)})
            break
        psi, E, g, var = h.logical_grad(target, x_new)
        counter[0] += 1
        rows.append({"L": L, "censored": False, "E": E, "energy_error": E - spec["E0"],
                     "F": h.fidelity(spec["gs"], psi), "nfev": len(trace) + 1,
                     "x": x_new.tolist(), "state_hash": h.state_hash(psi), "grad_norm": float(np.linalg.norm(g))})
        x = x_new
    return rows


def run_beam(spec, deadline):
    cfg = PREREG["beam"]
    ceiling = int(cfg["eval_ceiling"])
    counter = [0]
    target = h.Target(9, B)
    records = []
    for k in range(int(cfg["chains"])):
        if counter[0] >= ceiling or time.perf_counter() > deadline:
            records.append({"arm": "beam", "k": k, "censored": True, "reason": "ceiling"})
            continue
        raw = []
        for s in range(6):
            rng = np.random.default_rng(np.random.SeedSequence([6209, 9, 0, k, s]))
            x0 = rng.normal(0.0, 0.35, size=6)
            def fun(z):
                _, E, g, _ = h.logical_grad(target, z)
                return E, g
            x, trace = lbfgs(fun, x0, 80, counter, ceiling)
            if x is None:
                records.append({"arm": "beam_l1", "k": k, "s": s, "censored": True, "nfev": len(trace)})
                continue
            psi, E, _, _ = h.logical_grad(target, x)
            counter[0] += 1
            raw.append({"s": s, "E": E, "energy_error": E - spec["E0"], "F": h.fidelity(spec["gs"], psi),
                        "x": x.tolist(), "nfev": len(trace) + 1, "state_hash": h.state_hash(psi)})
        if not raw:
            records.append({"arm": "beam", "k": k, "censored": True, "reason": "no L1"})
            continue
        best = min(raw, key=lambda r: (r["E"], r["s"]))
        second_pool = [r for r in raw if r["E"] > best["E"] + 0.3]
        second = min(second_pool, key=lambda r: (r["E"], r["s"])) if second_pool else None
        greedy = continue_from(target, spec, np.asarray(best["x"]), counter, ceiling)
        other = None
        if second is not None:
            other = continue_from(target, spec, np.asarray(second["x"]), counter, ceiling)
        def final(path):
            done = [r for r in path if r.get("L") == 6 and not r.get("censored")]
            return done[0] if done else None
        g6, o6 = final(greedy), final(other) if other else None
        chosen = "greedy"
        winner = g6
        if o6 is not None and (g6 is None or o6["E"] < g6["E"]):
            chosen = "second"
            winner = o6
        records.append({"arm": "beam", "k": k, "censored": winner is None, "chosen": chosen,
                        "second_used": second is not None,
                        "L1_best_error": best["energy_error"],
                        "L1_second_error": None if second is None else second["energy_error"],
                        "greedy_L6_error": None if g6 is None else g6["energy_error"],
                        "greedy_L6_F": None if g6 is None else g6["F"],
                        "second_L6_error": None if o6 is None else o6["energy_error"],
                        "second_L6_F": None if o6 is None else o6["F"],
                        "winner": winner})
        if (k + 1) % 4 == 0:
            print(f"  beam {k + 1} evals {counter[0]}", flush=True)
    with (OUT / "N9_beam.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    done = [r for r in records if r.get("arm") == "beam" and not r.get("censored")]
    bar = PREREG["historical_best_E_F"]["9"][5]
    # bar may be improved by this slice's qaoa; caller replaces it
    return {"evals": counter[0], "n": len(done), "records": done, "ceiling": ceiling}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    deadline = t0 + 50 * 60
    summaries = []
    specs = {}
    for n in (6, 9, 12):
        print("qaoa", n, flush=True)
        target = h.Target(n, B)
        spec = h.ground_state(target)
        specs[n] = spec
        records, spent = run_qaoa(n, spec, deadline)
        summary = compare(n, records, spec)
        summary["new_evals"] = spent
        summary["ceiling"] = PREREG["qaoa"][f"N{n}"]["new_ceiling"]
        summary["shortfall"] = max(0, summary["ceiling"] - spent)
        summaries.append(summary)
        (OUT / f"N{n}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    beam_summary = None
    if time.perf_counter() < deadline:
        print("beam", flush=True)
        beam = run_beam(specs[9], deadline)
        # update N9 p6 bar with the rescue if it improved
        n9 = next(s for s in summaries if s["n"] == 9)
        bar_E = n9["depths"][5]["bar_E"]
        bar_F = n9["depths"][5]["bar_F"]
        wins = 0
        greedy_wins = 0
        second_better = 0
        margins = []
        for r in beam["records"]:
            w = r["winner"]
            if w and both(w["E"], w["F"], [bar_E, bar_F]):
                wins += 1
            if r["greedy_L6_error"] is not None:
                # reconstruct greedy E from error
                gE = r["greedy_L6_error"] + specs[9]["E0"]
                if both(gE, r["greedy_L6_F"], [bar_E, bar_F]):
                    greedy_wins += 1
            if r["second_L6_error"] is not None and r["greedy_L6_error"] is not None:
                if r["second_L6_error"] < r["greedy_L6_error"] - TAU:
                    second_better += 1
                    margins.append(r["greedy_L6_error"] - r["second_L6_error"])
        beam_summary = {
            "evals": beam["evals"], "ceiling": beam["ceiling"], "n": beam["n"],
            "BOTH_vs_rescued_p6": wins, "wilson": wilson(wins, beam["n"]),
            "greedy_BOTH": greedy_wins,
            "second_better_energy_count": second_better,
            "second_energy_gain": margins,
            "shortfall": max(0, beam["ceiling"] - beam["evals"]),
        }
        (OUT / "N9_beam_summary.json").write_text(json.dumps(beam_summary, indent=2) + "\n")
    (OUT / "REPORT.md").write_text(render(summaries, beam_summary))
    print("finished", flush=True)


def render(summaries, beam):
    lines = [
        "# Equal-depth QAOA rescue",
        "",
        "The 48 stored procedures were not retrained. Tau is 1e-9. The bar at each depth is the lower energy of the historical state and this slice, and the fidelity of that same state.",
        "",
    ]
    for s in summaries:
        lines.append(f"## N={s['n']}")
        lines.append("")
        lines.append(f"New optimizer evaluations {s['new_evals']} of ceiling {s['ceiling']}. Shortfall {s['shortfall']}.")
        lines.append("")
        lines.append("| L | stored BOTH | Wilson | worst error | worst F | bar error | bar F | source | fresh best error |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for d in s["depths"]:
            w = d["wilson"]
            wtxt = "—" if not w else f"[{w[0]:.3f}, {w[1]:.3f}]"
            fresh = "—" if d["fresh_best_error"] is None else f"{d['fresh_best_error']:.4f}"
            worst = "—" if d["worst_error"] is None else f"{d['worst_error']:.4f}"
            wf = "—" if d["worst_F"] is None else f"{d['worst_F']:.4f}"
            lines.append(f"| {d['L']} | {d['BOTH']}/{d['n']} | {wtxt} | {worst} | {wf} | {d['bar_error']:.4f} | {d['bar_F']:.4f} | {d['bar_source']} | {fresh} |")
        lines.append("")
        lines.append(f"Replay of the deepest new QAOA state: {s['replay']}.")
        lines.append("")
    if beam:
        lines.append("## N=9 two-basin continuation")
        lines.append("")
        lines.append(f"Fresh chains {beam['n']}. Evaluations {beam['evals']} of ceiling {beam['ceiling']}. Shortfall {beam['shortfall']}.")
        lines.append(f"Beam BOTH against the rescued p=6 bar: {beam['BOTH_vs_rescued_p6']}/{beam['n']}, Wilson {beam['wilson']}.")
        lines.append(f"The single-basin arm of the same chains: {beam['greedy_BOTH']}/{beam['n']}.")
        lines.append(f"Second basin lower at L6 in {beam['second_better_energy_count']} chains. Energy gains: {beam['second_energy_gain']}.")
        lines.append("")
    lines.append("No search here is a global QAOA optimum. N=15 and N=18 were not run.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
