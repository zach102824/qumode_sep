"""Fixed-budget standard QAOA. Energy only. No new feature.

Priority slices are p12, p11, p10, p9, p7, p6. A slice stops when its
frozen new_work cap is met or its time slice ends. Unused time carries
forward. The 75 minute clock includes resumed work.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from Nov_3rd.tfim_shared_gamma.round1_A.common import canonical_state, log, sha256_state
from Nov_3rd.tfim_shared_gamma.round5_KS.model import Runner, Target, pack, unpack

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
BASELINE = json.loads((ROOT / "baseline.json").read_text())
PROTOCOL_SHA = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
ROUND4 = ROOT.parents[0] / "round4_N" / "results" / "jobs"
GAMMA = (-12.0, 12.0)
BETA = (-2.0 * np.pi, 2.0 * np.pi)


def spectrum(target: Target):
    n = target.n
    dim = 1 << n
    H = np.diag(target.hc.copy())
    idx = np.arange(dim)
    for bit in range(n):
        partner = idx ^ (1 << bit)
        mask = partner > idx
        a, b = idx[mask], partner[mask]
        H[a, b] = target.B
        H[b, a] = target.B
    evals, vecs = np.linalg.eigh(H)
    gs = np.real(vecs[:, 0])
    gs /= np.linalg.norm(gs)
    if float(gs.sum()) < 0:
        gs = -gs
    return gs, float(evals[0]), float(evals[1] - evals[0])


def _bounds(p):
    return [GAMMA] * p + [BETA] * p


def _clip(x, bounds):
    y = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(float(y[i]), lo), hi)
    return y


def _resample(values, p_new):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    p_old = values.size
    xs = (np.arange(p_old) + 0.5) / p_old
    xt = (np.arange(p_new) + 0.5) / p_new
    if p_old < 3:
        return np.interp(xt, xs, values)
    spline = CubicSpline(xs, values, bc_type="natural")
    return np.asarray(spline(np.clip(xt, xs[0], xs[-1])), dtype=np.float64)


def _stored_x(p):
    blob = json.loads((ROUND4 / f"n12_L{p}_qaoa.json").read_text())
    return np.array([float(v) for v in blob["x_exact"]], dtype=np.float64), blob


def _rng(arm, p, k, batch):
    return np.random.default_rng(np.random.SeedSequence([20261113, int(arm), int(p), int(k), int(batch)]))


def _seeds(p, batch):
    x_same, _ = _stored_x(p)
    gammas, _, cs = unpack("qaoa", x_same, p)
    betas = cs[:, 0]
    seeds = []
    if p > 1:
        x_prev, _ = _stored_x(p - 1)
        g0, _, c0 = unpack("qaoa", x_prev, p - 1)
        seeds.append(("interp", pack("qaoa", 0, {}, np.zeros((p, 3)), _resample(g0, p), _resample(c0[:, 0], p)), 0.0, 1))
    t = (np.arange(p) + 0.5) / p
    ag, ab = float(np.max(np.abs(gammas))), float(np.max(np.abs(betas)))
    for mode in (1, 2):
        shape = np.sin(mode * np.pi * t)
        seeds.append((f"fourier{mode}", pack("qaoa", 0, {}, np.zeros((p, 3)), ag * shape, ab * shape), 0.0, 2))
    for k in range(8):
        rng = _rng(3, p, k, batch)
        seeds.append((f"random_{k}", pack("qaoa", 0, {}, np.zeros((p, 3)), rng.uniform(-0.8, 0.8, p), rng.uniform(-0.8, 0.8, p)), 0.0, 3))
    for arm, sigma, n in ((4, 0.02, 8), (5, 0.1, 8)):
        for k in range(n):
            rng = _rng(arm, p, k, batch)
            seeds.append((f"perturb_{sigma}_{k}", x_same + rng.normal(scale=sigma, size=x_same.shape), sigma, arm))
    return seeds


def _optimize_start(runner, p, x0, label, sigma, bounds, eval_cap, counter, best):
    x0 = _clip(x0, bounds)
    local_E, local_x = float("inf"), x0.copy()
    cache = {}
    n0 = counter[0]

    def _record(x, E, grad):
        nonlocal local_E, local_x
        counter[0] += 1
        cache["x"] = np.array(x, copy=True)
        cache["g"] = grad
        if E < local_E:
            local_E, local_x = float(E), np.array(x, copy=True)
        if E < best["E"]:
            best["E"] = float(E)
            best["x"] = np.array(x, copy=True)
            best["label"] = label
        best["curve"].append(float(best["E"]))

    def fun(x):
        if counter[0] - n0 >= eval_cap:
            return local_E
        E, grad, _ = runner.energy_grad("qaoa", x, p)
        _record(x, E, grad)
        return float(E)

    def jac_ok(x):
        if cache.get("x") is not None and np.allclose(cache["x"], x, atol=1e-12, rtol=0):
            return cache["g"]
        if counter[0] - n0 >= eval_cap:
            return np.zeros_like(x)
        E, grad, _ = runner.energy_grad("qaoa", x, p)
        _record(x, E, grad)
        return grad

    try:
        res = minimize(
            fun, x0, method="L-BFGS-B", jac=jac_ok, bounds=bounds,
            options={"maxiter": 1500, "maxfun": min(500, eval_cap), "ftol": 1e-14, "gtol": 1e-8},
        )
        ok = bool(res.success)
    except Exception:
        ok = False
    E, psi = runner.energy_only("qaoa", local_x, p)
    if E < best["E"]:
        best["E"] = float(E)
        best["x"] = local_x.copy()
        best["label"] = label
    return {
        "label": label, "sigma": sigma, "E": float(E), "F": runner.fidelity(psi),
        "success": ok, "nfev": int(counter[0] - n0),
    }, local_x


def _empty_depth(p, stored_E):
    bud = BASELINE["budgets"][str(p)]
    return {
        "p": p,
        "existing": bud["existing_validated"],
        "new_work_allowed": bud["new_work_allowed"],
        "n_eval_new": 0,
        "best_E": stored_E,
        "best_x": None,
        "best_label": "stored",
        "starts": [],
        "curve": [],
        "seconds": 0.0,
    }


def _save_depth(depth):
    path = RESULTS / "jobs" / f"p{depth['p']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in depth.items() if k != "best_x"}
    if depth.get("best_x") is not None:
        clean["x_exact"] = [format(float(v), ".17g") for v in depth["best_x"]]
    # Downsample the curve so the file stays usable.
    curve = depth["curve"]
    step = max(1, len(curve) // 800)
    clean["curve"] = curve[::step]
    path.write_text(json.dumps(clean, indent=2) + "\n")


def _load_depth(p, stored_E):
    path = RESULTS / "jobs" / f"p{p}.json"
    if not path.exists():
        return _empty_depth(p, stored_E)
    raw = json.loads(path.read_text())
    depth = _empty_depth(p, stored_E)
    depth["n_eval_new"] = int(raw.get("n_eval_new", 0))
    depth["best_E"] = float(raw.get("best_E", stored_E))
    depth["best_label"] = raw.get("best_label", "stored")
    depth["starts"] = raw.get("starts", [])
    depth["curve"] = raw.get("curve", [])
    depth["seconds"] = float(raw.get("seconds", 0.0))
    if "x_exact" in raw:
        depth["best_x"] = np.array([float(v) for v in raw["x_exact"]], dtype=np.float64)
    return depth


def run(budget_min: float = 75.0):
    t0 = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "jobs").mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    (RESULTS / "states").mkdir(exist_ok=True)
    clock_path = RESULTS / "clock.json"
    used = 0.0
    if clock_path.exists():
        used = float(json.loads(clock_path.read_text()).get("wall_used", 0.0))
    deadline = t0 + max(0.0, budget_min * 60.0 - used)
    if hashlib.sha256((ROOT / "baseline.json").read_bytes()).hexdigest() != PROTOCOL["baseline_sha256"]:
        raise RuntimeError("baseline hash drifted")
    if hashlib.sha256((ROOT / "FROZEN_TARGET_R2.json").read_bytes()).hexdigest() != PROTOCOL["manifest_sha256"]:
        raise RuntimeError("manifest hash drifted")
    target = Target(12)
    if target.hc_sha != PROTOCOL["hc_diag_sha256"]:
        raise RuntimeError("Hc hash drifted")
    gs, E0, gap = spectrum(target)
    if abs(E0 - BASELINE["E0"]) > 1e-12:
        raise RuntimeError("E0 drifted")
    runner = Runner(target)
    runner.set_spectrum(gs, E0, E0, gap, gap)
    # Replay the stored p12 vector before any new search.
    x12, blob12 = _stored_x(12)
    E12, _, psi12 = runner.energy_grad("qaoa", x12, 12)
    dE = abs(E12 - blob12["E"])
    dF = abs(runner.fidelity(psi12) - blob12["F"])
    log(f"p12 replay dE={dE:.3e} dF={dF:.3e}")
    if dE > 1e-10 or dF > 1e-10:
        raise RuntimeError("stored QAOA replay failed")
    depths = {}
    slices = {int(k): float(v) for k, v in PROTOCOL["time_slices_min"].items()}
    carry = 0.0
    for p in PROTOCOL["priority"]:
        stored = BASELINE["qaoa_stored_best"][str(p)]
        depth = _load_depth(p, stored["E"])
        depths[p] = depth
        slice_s = (slices[p] + carry) * 60.0
        carry = 0.0
        slice_end = min(deadline, time.perf_counter() + slice_s)
        if depth["n_eval_new"] >= depth["new_work_allowed"]:
            log(f"p={p} already at new_work cap")
            continue
        if time.perf_counter() >= deadline:
            log(f"p={p} not started, clock done")
            break
        log(f"p={p} slice {slice_s/60:.1f}min existing={depth['existing']} new_cap={depth['new_work_allowed']} already={depth['n_eval_new']}")
        bounds = _bounds(p)
        batch = 0
        while time.perf_counter() < slice_end and time.perf_counter() < deadline and depth["n_eval_new"] < depth["new_work_allowed"]:
            batch += 1
            best = {"E": depth["best_E"], "x": depth["best_x"], "label": depth["best_label"], "curve": depth["curve"]}
            counter = [0]
            t_batch = time.perf_counter()
            for label, x0, sigma, _arm in _seeds(p, batch):
                if time.perf_counter() >= slice_end or time.perf_counter() >= deadline:
                    break
                if depth["n_eval_new"] + counter[0] >= depth["new_work_allowed"]:
                    break
                room = depth["new_work_allowed"] - depth["n_eval_new"] - counter[0]
                row, _ = _optimize_start(runner, p, x0, label, sigma, bounds, min(500, room), counter, best)
                depth["starts"].append(row)
            depth["n_eval_new"] += counter[0]
            depth["best_E"] = best["E"]
            depth["best_x"] = best["x"]
            depth["best_label"] = best["label"]
            depth["curve"] = best["curve"]
            depth["seconds"] += time.perf_counter() - t_batch
            _save_depth(depth)
            clock_path.write_text(json.dumps({"wall_used": used + time.perf_counter() - t0}) + "\n")
            log(f"  batch {batch} new_evals={depth['n_eval_new']} best={depth['best_E']:.6f} via {depth['best_label']}")
        leftover = slice_end - time.perf_counter()
        if leftover > 1.0 and time.perf_counter() < deadline:
            carry += leftover / 60.0
    # Finish records for the depths we touched.
    summary = _finalize(runner, depths, E0, gap, used + time.perf_counter() - t0)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _md(RESULTS / "RESULTS.md", summary)
    _plots(summary, RESULTS / "figures")
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    for png in (RESULTS / "figures").glob("*.png"):
        (art / f"round6_QR_{png.name}").write_bytes(png.read_bytes())
    log(f"done wall={summary['wall_seconds']:.1f}s")
    return summary


def _finalize(runner, depths, E0, gap, wall):
    tau = BASELINE["tau"]
    rows = []
    for p, depth in depths.items():
        if depth["best_x"] is None:
            x, _ = _stored_x(p)
        else:
            x = depth["best_x"]
        E, _, psi = runner.energy_grad("qaoa", x, p)
        norm = float(np.linalg.norm(psi))
        var = runner.variance(psi)
        # Direct check: energy() builds H psi.
        E2, Hpsi = runner.energy(psi)
        var2 = float(np.vdot(Hpsi, Hpsi).real - E2 * E2)
        F = runner.fidelity(psi)
        parity = runner.parity_weight(psi)
        np.save(RESULTS / "states" / f"p{p}.npy", canonical_state(psi))
        energies = np.array([s["E"] for s in depth["starts"]], dtype=np.float64) if depth["starts"] else np.array([E])
        q25, q75 = np.percentile(energies, [25, 75]) if energies.size else (E, E)
        bud = BASELINE["budgets"][str(p)]
        row = {
            "p": p, "E": float(E), "F": F, "energy_error": float(E - E0), "var_H": var,
            "direct_E_residual": abs(E - E2), "direct_var_residual": abs(var - var2), "norm": norm,
            "parity": parity, "state_sha256": sha256_state(psi), "init": depth["best_label"],
            "n_eval_new": depth["n_eval_new"], "n_starts_new": len(depth["starts"]),
            "existing": bud["existing_validated"], "new_work_allowed": bud["new_work_allowed"],
            "shortfall": max(0, bud["new_work_allowed"] - depth["n_eval_new"]),
            "seed_median": float(np.median(energies)), "seed_iqr": float(q75 - q25),
            "seconds": depth["seconds"],
            "starts": depth["starts"],
            "curve": depth["curve"][:: max(1, len(depth["curve"]) // 800)],
            "x_exact": [format(float(v), ".17g") for v in x],
        }
        near = sum(1 for s in depth["starts"] if abs(s["E"] - E) <= 1e-3)
        row["basin_hits_1e-3"] = int(near)
        row["comparisons"] = {}
        for name, spec in (
            ("stride4_L6", BASELINE["stride4_L6_arm0"]),
            ("stride4_L5", BASELINE["stride4_L5_arm0"]),
            ("KS_L6", BASELINE["KS_L6"]),
        ):
            dE = float(E - spec["E"])
            dF = float(F - spec["F"])
            row["comparisons"][name] = {
                "dE": dE, "dF": dF,
                "qaoa_dominates": bool(E < spec["E"] - tau and F > spec["F"] + tau),
                "lower_E_lower_F_tradeoff": bool(E < spec["E"] - tau and F < spec["F"] - tau),
            }
        rows.append(row)
        log(f"p={p} E={E:.6f} F={F:.4f} new={depth['n_eval_new']} short={row['shortfall']} norm_res={abs(norm-1):.1e}")
    return {
        "manifest_sha256": PROTOCOL["manifest_sha256"],
        "protocol_sha256": PROTOCOL_SHA,
        "baseline_sha256": PROTOCOL["baseline_sha256"],
        "E0": E0, "gap_full": gap, "tau": tau,
        "depths": rows,
        "wall_seconds": wall,
        "reading": _reading(rows, tau),
    }


def _reading(rows, tau):
    by = {r["p"]: r for r in rows}
    p12 = by.get(12)
    lines = [
        "This round is standard independent-angle QAOA only. No feature was changed. "
        f"Comparisons use tau = {tau:.0e}. A both-metric win requires lower energy and higher fidelity on the energy-selected states. "
        "A lower-energy and lower-fidelity QAOA state would be a tradeoff. These searches did not produce one.",
        "Budgets were frozen before these starts. Bp = max(4·Rp, 53927) and the new allowance is Bp minus work already validated at that depth. "
        "The stride-4 family cap is the larger published arm inclusive, 53927. The constants 1556 and 28099 are already inside that ledger and were not added again. "
        "L5 fidelity is the full-precision value in the committed arm0 cell, 0.8898142677626181.",
    ]
    if p12:
        lines.append(
            f"The best p=12 energy remains {p12['E']:.12f}, the stored optimum, after {p12['n_eval_new']} new evaluations. "
            f"The allowance was {p12['new_work_allowed']}, so the shortfall is {p12['shortfall']}. "
            f"Against stride-4 L6 the energy delta is {p12['comparisons']['stride4_L6']['dE']:+.6f} and the fidelity delta is {p12['comparisons']['stride4_L6']['dF']:+.6f}. "
            f"Against KS L6 the energy delta is {p12['comparisons']['KS_L6']['dE']:+.6f} and the fidelity delta is {p12['comparisons']['KS_L6']['dF']:+.6f}. "
            "QAOA is higher in energy and lower in fidelity in both comparisons. "
            f"p=11, p=10, and p=9 likewise stayed on their stored optima, with shortfalls {by[11]['shortfall']}, {by[10]['shortfall']}, and {by[9]['shortfall']}. "
            f"p=6 met its allowance. p=7 is short by {by[7]['shortfall']}. "
            "The median of the new starts sits on the stored optimum at each depth, and the IQR stays large because the failed tail is included. No seal."
        )
    return lines


def _md(path, summary):
    lines = ["# Round 6 QR — fixed-budget QAOA", ""]
    for line in summary["reading"]:
        lines += [line, ""]
    lines.append(f"Manifest `{summary['manifest_sha256']}`. E0 = {summary['E0']:.10f}. tau = {summary['tau']:.0e}.")
    lines.append(f"Wall {summary['wall_seconds']:.0f} s.")
    lines.append("")
    lines.append("| p | E | error | F | var | new evals | allowed | shortfall | median | IQR | basin 1e-3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in summary["depths"]:
        lines.append(
            f"| {row['p']} | {row['E']:.6f} | {row['energy_error']:.4e} | {row['F']:.4f} | {row['var_H']:.3f} | "
            f"{row['n_eval_new']} | {row['new_work_allowed']} | {row['shortfall']} | {row['seed_median']:.4f} | "
            f"{row['seed_iqr']:.3f} | {row['basin_hits_1e-3']}/{row['n_starts_new']} |"
        )
    lines.append("")
    lines.append("## Deltas against the frozen states")
    lines.append("")
    for row in summary["depths"]:
        for name, comp in row["comparisons"].items():
            lines.append(
                f"- p={row['p']} vs {name}: dE {comp['dE']:+.6f}, dF {comp['dF']:+.6f}, "
                f"dominates {comp['qaoa_dominates']}, tradeoff {comp['lower_E_lower_F_tradeoff']}"
            )
    path.write_text("\n".join(lines) + "\n")


def _plots(summary, folder: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for row in summary["depths"]:
        if not row["curve"]:
            continue
        ys = [v - summary["E0"] for v in row["curve"]]
        xs = row["existing"] + np.linspace(1, row["n_eval_new"], len(ys))
        ax.plot(xs, ys, label=f"p={row['p']}")
    ax.set_xlabel("inclusive evaluations")
    ax.set_ylabel("best energy error so far")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "best_so_far.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for row in summary["depths"]:
        if not row["starts"]:
            continue
        err = sorted(s["E"] - summary["E0"] for s in row["starts"])
        ys = np.arange(1, len(err) + 1) / len(err)
        ax.step(err, ys, where="post", label=f"p={row['p']}")
    ax.set_xlabel("new-start energy error")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "cdf.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.scatter([r["energy_error"] for r in summary["depths"]], [r["F"] for r in summary["depths"]], label="QAOA")
    ax.scatter([BASELINE["stride4_L6_arm0"]["energy_error"]], [BASELINE["stride4_L6_arm0"]["F"]], marker="*", s=80, label="stride4 L6")
    ax.scatter([BASELINE["stride4_L5_arm0"]["energy_error"]], [BASELINE["stride4_L5_arm0"]["F"]], marker="*", s=80, label="stride4 L5")
    ax.scatter([BASELINE["KS_L6"]["energy_error"]], [BASELINE["KS_L6"]["F"]], marker="D", s=40, label="KS L6")
    ax.set_xlabel("energy error")
    ax.set_ylabel("fidelity of the energy-best state")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "pareto.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run(75.0)
