"""48-chain held-out of per-layer gamma at L=6. Reads the prereg and does not edit it."""

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
OUT = ROOT / "layer_gamma_confirm"
PRE = json.loads((ROOT / "LAYER_GAMMA_PREREG.json").read_text())
B = float("-0.3444")
TAU = float(PRE["tau"])


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


def grow(x, L_new):
    L = L_new - 1
    gammas, thk, thm = x[:L], x[L], x[L + 1]
    shells = x[L + 2:].reshape(L, 3)
    gnew = np.concatenate([gammas, [gammas[-1]]])
    return np.concatenate([gnew, [thk, thm], resample(shells, L_new).ravel()])


def grad_layer_gamma(target, x):
    L = (len(x) - 2) // 4
    gammas = x[:L]
    thk, thm = float(x[L]), float(x[L + 1])
    shells = np.asarray(x[L + 2:], dtype=np.float64).reshape(L, 3)
    G = target.g_diag(thk, thm)
    psi = h.plus(target.n)
    frames = []
    for ell, row in enumerate(shells):
        psi = psi * np.exp(-1j * float(gammas[ell]) * G)
        frames.append(("d", psi, ell, float(gammas[ell])))
        U, dUs = h.mixer(float(row[0]), float(row[1]), float(row[2]), True)
        for index in target.cav:
            psi = h.apply_cav(psi, U, index)
        frames.append(("s", psi, U, dUs, ell))
    Hpsi = target.hc * psi + target.B * h.apply_sum_x(psi, target.n)
    E = float(np.vdot(psi, Hpsi).real)
    lam = Hpsi
    grad = np.zeros(4 * L + 2)
    for kind, *rest in reversed(frames):
        if kind == "d":
            psi_d, ell, gamma = rest
            im = np.imag(np.conj(lam) * psi_d)
            grad[ell] += 2.0 * float(np.sum(G * im))
            grad[L] += 2.0 * float(np.sum((gamma * target.khat) * im))
            grad[L + 1] += 2.0 * float(np.sum((gamma * target.mhat) * im))
            lam = np.exp(1j * gamma * G) * lam
        else:
            psi_s, U, dUs, ell = rest
            for k in range(3):
                acc = np.zeros_like(psi_s)
                Va = dUs[k] @ np.conjugate(U).T
                for index in target.cav:
                    acc += h.apply_cav(psi_s, Va, index)
                grad[L + 2 + 3 * ell + k] += 2.0 * float(np.real(np.vdot(lam, acc)))
            for index in target.cav:
                lam = h.apply_cav(lam, np.conjugate(U).T, index)
    return E, grad, psi


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


def run_chains(n, n_chains, seed_head, maxfun, ceiling_each, deadline):
    target = h.Target(n, B)
    spec = h.ground_state(target)
    records = []
    executed = 0
    for k in range(n_chains):
        if time.perf_counter() > deadline:
            records.append({"n": n, "k": k, "censored": True, "reason": "wall"})
            continue
        rng = np.random.default_rng(np.random.SeedSequence(seed_head + [k]))
        x = rng.normal(0.0, 0.35, size=6)
        counter = [0]
        chain = []
        for ell in range(1, 7):
            if ell > 1:
                x = grow(x, ell)
            if len(x) != 4 * ell + 2:
                raise SystemExit(f"parameter count {len(x)} at L={ell}")
            xn, trace = lbfgs(lambda z: grad_layer_gamma(target, z)[:2], x, maxfun, counter, ceiling_each)
            if xn is None or counter[0] >= ceiling_each:
                chain.append({"L": ell, "censored": True, "nfev": len(trace)})
                break
            E, g, psi = grad_layer_gamma(target, xn)
            counter[0] += 1
            chain.append({
                "L": ell, "censored": False, "E": float(E), "error": float(E - spec["E0"]),
                "F": h.fidelity(spec["gs"], psi), "grad": float(np.linalg.norm(g)),
                "x": xn.tolist(), "hash": h.state_hash(psi), "nfev": len(trace) + 1,
                "seed": seed_head + [k],
            })
            x = xn
        executed += counter[0]
        records.append({"n": n, "k": k, "censored": False, "evals": counter[0], "path": chain})
        if (k + 1) % 8 == 0:
            print(f"  N{n} {k+1} executed {executed}", flush=True)
    return spec, records, executed


def summarize(records, bars, spec):
    done = [r for r in records if r.get("path")]
    out = []
    for L, key in ((1, None), (6, "p6")):
        pass
    for L in range(1, 7):
        rows = []
        for rec in done:
            hit = [s for s in rec["path"] if s.get("L") == L and not s.get("censored")]
            if hit:
                rows.append(hit[0])
        entry = {"L": L, "n": len(rows)}
        if rows:
            errs = np.array([r["error"] for r in rows])
            entry.update({
                "median_error": float(np.median(errs)),
                "q25_error": float(np.percentile(errs, 25)),
                "q75_error": float(np.percentile(errs, 75)),
                "best_error": float(errs.min()),
                "worst_error": float(errs.max()),
                "best_F_at_best_E": float(min(rows, key=lambda r: r["E"])["F"]),
                "worst_F": float(min(r["F"] for r in rows)),
            })
        for name, bar in bars.items():
            wins = sum(beats(r["E"], r["F"], bar) for r in rows)
            entry[name] = {"BOTH": wins, "wilson": wilson(wins, len(rows)), "of": len(rows)}
        out.append(entry)
    return out


def replay(n, row):
    target = h.Target(n, B)
    H = h.dense_h(target)
    x = np.asarray(row["x"])
    _, _, psi = grad_layer_gamma(target, x)
    Ed = float(np.vdot(psi, H @ psi).real)
    return {"logged_E": row["E"], "dense_E": Ed, "delta": Ed - row["E"],
            "hash_match": h.state_hash(psi) == row["hash"], "F": row["F"], "error": row["error"],
            "grad": row["grad"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + 40 * 60
    target = h.Target(9, B)
    z = np.random.default_rng(1).normal(0, 0.2, size=26)
    _, g, _ = grad_layer_gamma(target, z)
    e = np.zeros(26)
    e[0] = 1e-6
    Ep = grad_layer_gamma(target, z + e)[0]
    Em = grad_layer_gamma(target, z - e)[0]
    if abs(g[0] - (Ep - Em) / 2e-6) > 1e-4:
        raise SystemExit("gradient check failed")
    print("n9", flush=True)
    spec, records, executed = run_chains(9, 48, [7709, 9, 1], 120, int(PRE["ceiling_per_chain"]), deadline)
    bars = {k: PRE["bars_N9"][k] for k in ("p6", "p11", "p12")}
    summary = summarize(records, bars, spec)
    l6 = []
    for rec in records:
        hit = [s for s in rec.get("path", []) if s.get("L") == 6 and not s.get("censored")]
        if hit:
            l6.append(hit[0])
    replays = {}
    if l6:
        replays["best"] = replay(9, min(l6, key=lambda r: r["E"]))
        replays["worst"] = replay(9, max(l6, key=lambda r: r["E"]))
    p11_wins = next(d["p11"]["BOTH"] for d in summary if d["L"] == 6)
    transfer = None
    if p11_wins >= 24 and time.perf_counter() < deadline:
        transfer = {}
        for n, bars_n, seed in (
            (6, PRE["transfer"]["bars_N6"], [7706, 6, 1]),
            (12, PRE["transfer"]["bars_N12"], [7712, 12, 1]),
        ):
            print("transfer", n, flush=True)
            sp, rec, ev = run_chains(n, 8, seed, 80, 8 * 81, deadline)
            sm = summarize(rec, bars_n, sp)
            transfer[str(n)] = {"evals": ev, "summary": sm}
            (OUT / f"N{n}_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec))
    (OUT / "N9_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    payload = {"executed": executed, "summary": summary, "replays": replays, "transfer": transfer,
               "p11_wins_L6": p11_wins}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Per-layer gamma, L=6 held out", "",
             "This is not the one-gamma six-start recipe. P = 4L+2. L=6 has 26 parameters. One L=1 draw, then continuation. Energy only.",
             "", f"Executed evaluations {executed}. Chains {len(records)}.", ""]
    lines.append("| L | n | median | worst | best | BOTH p6 | BOTH p11 | BOTH p12 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in summary:
        def cell(name):
            item = d.get(name)
            if not item:
                return "—"
            return f"{item['BOTH']}/{item['of']}"
        med = "—" if "median_error" not in d else f"{d['median_error']:.4f}"
        worst = "—" if "worst_error" not in d else f"{d['worst_error']:.4f}"
        best = "—" if "best_error" not in d else f"{d['best_error']:.4f}"
        lines.append(f"| {d['L']} | {d['n']} | {med} | {worst} | {best} | {cell('p6')} | {cell('p11')} | {cell('p12')} |")
    lines.append("")
    l6s = next(d for d in summary if d["L"] == 6)
    lines.append(f"L6 Wilson versus p11: {l6s['p11']['wilson']}. Versus p6: {l6s['p6']['wilson']}. Versus p12: {l6s['p12']['wilson']}.")
    lines.append("There is one L=1 start per chain, so the L=1 rate is the raw-start rate. It is not a six-start screen.")
    lines.append(f"Dense replay: {replays}.")
    if transfer:
        lines.append("")
        lines.append("## Small transfer")
        for n, block in transfer.items():
            row = next(d for d in block["summary"] if d["L"] == 6)
            lines.append(f"N={n}: {block['evals']} evaluations. L6 BOTH p6 {row['p6']['BOTH']}/{row['p6']['of']}, p11 {row['p11']['BOTH']}/{row['p11']['of']}, p12 {row['p12']['BOTH']}/{row['p12']['of']}. Median error {row.get('median_error')}.")
    lines.append("")
    lines.append("The one-gamma equal-depth recipe is unchanged. N=15 and N=18 were not run.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("p11", p11_wins, "executed", executed, flush=True)


if __name__ == "__main__":
    main()
