"""Round 4 R: three frozen initialization arms for the same sublattice+M generator.

No new G. Energy training only. Stored angles are seeds for the transfer arm
and are charged on the inclusive ledger.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round3_R"))
import circuit
import model

import schedules

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCE = ROOT.parent / "round3_R" / "results" / "cells"
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
ARMS = ("plain", "transfer", "chebyshev")


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2) + "\n")


def clip(x, bounds):
    y = np.array(x, dtype=np.float64, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(float(y[i]), lo), hi)
    return y


def load_source(L):
    cell = json.loads((SOURCE / f"N12_L{L}_A_sub_M.json").read_text())
    if cell["manifest_sha256"] != PROTOCOL["manifest_sha256"]:
        raise RuntimeError("source manifest mismatch")
    if cell["packing_sha256"] != PROTOCOL["packing_sha256"]:
        raise RuntimeError("source packing mismatch")
    ang = cell["angles"]
    x = [ang["gamma"], ang["theta_k"], ang["theta_m"]]
    for i in range(L):
        x.extend([ang["c1"][i], ang["c2"][i], ang["c3"][i]])
    return np.asarray(x, dtype=np.float64), cell


def source_chain_nfev(max_L):
    total = 0
    for L in range(1, max_L + 1):
        total += int(load_source(L)[1]["nfev_sum"])
    return total


def lbfgs(objective, x0, bounds, maxfun):
    trace = []

    def fun(x):
        E, g = objective(np.asarray(x, dtype=np.float64))
        trace.append(float(E))
        return float(E), np.asarray(g, dtype=np.float64)

    res = minimize(
        fun, np.asarray(x0, dtype=np.float64), method="L-BFGS-B", jac=True, bounds=bounds,
        options={"maxiter": int(maxfun), "maxfun": int(maxfun), "ftol": 1e-14, "gtol": 1e-8},
    )
    return {
        "x": np.asarray(res.x, dtype=np.float64),
        "E": float(res.fun),
        "nfev": int(res.nfev),
        "nit": int(res.nit),
        "success": bool(res.success),
        "trace": trace,
    }


def full_objective(x, ctx):
    _, E, g, _ = circuit.simulate("M", x, ctx["L"], ctx["hc"], ctx["fk"], ctx["fm"], ctx["B"], 12, ctx["indices"], True)
    return E, g


def chebyshev_objective(coeff, degree, ctx):
    a = np.asarray(coeff[3:], dtype=np.float64).reshape(3, degree + 1)
    shells = schedules.shells_from_chebyshev(a, ctx["L"])
    x = schedules.pack_full(coeff[0], coeff[1], coeff[2], shells)
    _, E, g, _ = circuit.simulate("M", x, ctx["L"], ctx["hc"], ctx["fk"], ctx["fm"], ctx["B"], 12, ctx["indices"], True)
    basis = schedules.chebyshev_basis(ctx["L"], degree)
    out = np.zeros_like(coeff, dtype=np.float64)
    out[:3] = g[:3]
    gs = g[3:].reshape(ctx["L"], 3)
    for k in range(3):
        out[3 + k * (degree + 1): 3 + (k + 1) * (degree + 1)] = basis.T @ gs[:, k]
    return E, out


def coeff_bounds(degree):
    g = (-2.0 * np.pi, 2.0 * np.pi)
    th = (-5.0, 5.0)
    a = (-2.0 * np.pi, 2.0 * np.pi)
    return [g, th, th] + [a] * (3 * (degree + 1))


def make_plain(L, n_starts, rng, previous, B, bounds):
    starts = []
    dim = circuit.n_params("M", L)
    if previous is None:
        x = np.zeros(dim)
        x[0], x[1], x[2] = 0.35, -0.45, 0.2
        starts.append({"name": "structured_negK", "x0": clip(x, bounds), "role": "structured"})
        x2 = np.zeros(dim)
        x2[0] = -0.3
        starts.append({"name": "structured_neg_gamma", "x0": clip(x2, bounds), "role": "structured"})
    else:
        g, tk, tm, shells = schedules.unpack_full(previous, L - 1)
        auto = schedules.resample_shells(shells, L, "auto")
        starts.append({"name": "bootstrap_centers", "x0": clip(schedules.pack_full(g, tk, tm, auto), bounds), "role": "bootstrap"})
        if L >= 3:
            lin = schedules.resample_shells(shells, L, "linear")
            starts.append({"name": "bootstrap_linear", "x0": clip(schedules.pack_full(g, tk, tm, lin), bounds), "role": "bootstrap"})
        if L == 2:
            shifted = auto.copy()
            shifted[1, 0] = shifted[0, 0] - 0.2 * B
            starts.append({"name": "pagano_c1_shift", "x0": clip(schedules.pack_full(g, tk, tm, shifted), bounds), "role": "bootstrap"})
    while len(starts) < n_starts:
        starts.append({
            "name": f"random{len(starts)}",
            "x0": clip(rng.normal(0.0, 0.35, size=dim), bounds),
            "role": "random",
        })
    return starts[:n_starts]


def make_transfer(L, n_starts, rng, bounds):
    neighbors = []
    if L > 1:
        neighbors.append(L - 1)
    if L < 6:
        neighbors.append(L + 1)
    bases = []
    for src in neighbors:
        xsrc, _ = load_source(src)
        g, tk, tm, shells = schedules.unpack_full(xsrc, src)
        resampled = schedules.resample_shells(shells, L, "auto")
        bases.append((src, schedules.pack_full(g, tk, tm, resampled)))
    starts = []
    for src, x in bases:
        starts.append({"name": f"exact_from_L{src}", "x0": clip(x, bounds), "sigma": 0.0, "source_L": src, "role": "transfer"})
    remaining = n_starts - len(starts)
    n_lo = remaining // 2
    for i in range(remaining):
        sigma = 0.02 if i < n_lo else 0.1
        src, x = bases[i % len(bases)]
        starts.append({
            "name": f"sigma{sigma}_from_L{src}_{i}",
            "x0": clip(x + rng.normal(0.0, sigma, size=x.size), bounds),
            "sigma": sigma,
            "source_L": src,
            "role": "transfer",
        })
    return starts, max(neighbors)


def summarize(records, E0, tau, f_base, marginal, inclusive):
    errors = np.array([r["energy_error"] for r in records], dtype=np.float64)
    fids = np.array([r["F"] for r in records], dtype=np.float64)
    q1, q3 = np.percentile(errors, [25, 75])
    n_E = int(np.sum(errors <= tau)) if tau is not None else None
    n_both = int(np.sum((errors <= tau) & (fids >= f_base))) if tau is not None else None
    n = len(records)
    def rate(k):
        return None if k is None else float(k) / float(n)
    def per(k):
        if k is None:
            return None
        if k == 0:
            return None
        return {"marginal": marginal / k, "inclusive": inclusive / k}
    return {
        "n_starts": n,
        "energy_error_best": float(np.min(errors)),
        "energy_error_median": float(np.median(errors)),
        "energy_error_iqr": float(q3 - q1),
        "F_of_best_energy": float(records[int(np.argmin(errors))]["F"]),
        "F_median": float(np.median(fids)),
        "n_success_energy": n_E,
        "p_success_energy": rate(n_E),
        "n_success_energy_and_F": n_both,
        "p_success_energy_and_F": rate(n_both),
        "evals_per_energy_success": per(n_E),
        "evals_per_joint_success": per(n_both),
        "energy_success_censored": bool(n_E == 0) if n_E is not None else False,
        "joint_success_censored": bool(n_both == 0) if n_both is not None else False,
        "marginal_nfev": int(marginal),
        "inclusive_nfev": int(inclusive),
    }


def cluster_states(psis, threshold=0.99):
    n = len(psis)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    conj_only = 0
    for i in range(n):
        for j in range(i + 1, n):
            direct = abs(np.vdot(psis[i], psis[j]))
            conj = abs(np.vdot(psis[i], np.conj(psis[j])))
            if direct > threshold:
                parent[find(j)] = find(i)
            elif conj > threshold:
                conj_only += 1
    roots = [find(i) for i in range(n)]
    labels = {r: k for k, r in enumerate(sorted(set(roots)))}
    ids = [labels[r] for r in roots]
    return {"n_basins": len(set(ids)), "labels": ids, "conjugate_only_pairs": conj_only, "threshold": threshold}


def best_so_far(energies, E0):
    best = []
    running = np.inf
    for i, E in enumerate(energies, start=1):
        running = min(running, float(E) - E0)
        best.append(running)
    return best


def roughness(shells):
    s = np.asarray(shells, dtype=np.float64)
    if len(s) < 3:
        return 0.0
    acc = 0.0
    for i in range(1, len(s) - 1):
        acc += float(np.sum((s[i + 1] - 2 * s[i] + s[i - 1]) ** 2))
    return acc


def evaluate_point(x, ctx, E0, gs):
    psi, E, g, var = circuit.simulate("M", x, ctx["L"], ctx["hc"], ctx["fk"], ctx["fm"], ctx["B"], 12, ctx["indices"], True)
    return {
        "E": float(E),
        "energy_error": float(E - E0),
        "F": circuit.fidelity(psi, gs),
        "var_H": float(var),
        "grad_norm": float(np.linalg.norm(g)),
        "parity": circuit.parity_expectation(psi),
        "reflection": circuit.reflection_expectation(psi, ctx["ridx"]),
        "psi": psi,
        "state_hash": hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest(),
    }


def run_arm(arm, L, ctx, rng, previous, deadline):
    if time.perf_counter() > deadline:
        return {"arm": arm, "L": L, "skipped": True, "reason": "time budget"}
    bounds = ctx["bounds"]
    n_starts = int(PROTOCOL["n_starts"][str(L)])
    maxfun = int(PROTOCOL["full_maxfun"][str(L)])
    pre_fun = int(PROTOCOL["pretrain_maxfun_per_stage"][str(L)])
    source_nfev = 0
    source_note = "no stored seed"
    if arm == "plain":
        specs = make_plain(L, n_starts, rng, previous, ctx["B"], bounds)
    elif arm == "transfer":
        specs, max_neighbor = make_transfer(L, n_starts, rng, bounds)
        source_nfev = source_chain_nfev(max_neighbor)
        source_note = f"stored round-3 angles from neighbors up to L={max_neighbor}; ancestor nfev L1..L{max_neighbor} charged; not a fresh discovery"
    else:
        d1 = min(1, L - 1)
        d2 = min(3, L - 1)
        degrees = [d1] if d2 == d1 else [d1, d2]
        specs = []
        for i in range(n_starts):
            specs.append({
                "name": f"chebyshev{i}",
                "coeff0": rng.normal(0.0, 0.3, size=3 + 3 * (degrees[0] + 1)),
                "degrees": degrees,
                "role": "pretrain",
            })
    records = []
    eval_energies = []
    t0 = time.perf_counter()
    for spec in specs:
        if time.perf_counter() > deadline:
            break
        local_trace = []
        if arm == "chebyshev":
            coeff = np.asarray(spec["coeff0"], dtype=np.float64)
            degree_now = spec["degrees"][0]
            pre_nfev = 0
            for degree in spec["degrees"]:
                if degree != degree_now:
                    old = coeff[3:].reshape(3, degree_now + 1)
                    grown = np.zeros((3, degree + 1))
                    grown[:, : degree_now + 1] = old
                    coeff = np.concatenate([coeff[:3], grown.reshape(-1)])
                    degree_now = degree
                stage = lbfgs(
                    lambda c, degree=degree: chebyshev_objective(c, degree, ctx),
                    clip(coeff, coeff_bounds(degree)),
                    coeff_bounds(degree),
                    pre_fun,
                )
                coeff = stage["x"]
                degree_now = degree
                pre_nfev += stage["nfev"]
                local_trace.extend(stage["trace"])
            a = coeff[3:].reshape(3, degree_now + 1)
            shells = schedules.shells_from_chebyshev(a, L)
            x0 = clip(schedules.pack_full(coeff[0], coeff[1], coeff[2], shells), bounds)
            solved = lbfgs(lambda x: full_objective(x, ctx), x0, bounds, maxfun)
            local_trace.extend(solved["trace"])
            solved["nfev"] += pre_nfev
            solved["pretrain_nfev"] = pre_nfev
            spec_name = spec["name"]
            x_seed = spec["coeff0"]
        else:
            solved = lbfgs(lambda x: full_objective(x, ctx), spec["x0"], bounds, maxfun)
            local_trace.extend(solved["trace"])
            spec_name = spec["name"]
            x_seed = spec["x0"]
            solved["pretrain_nfev"] = 0
        diag = evaluate_point(solved["x"], ctx, ctx["E0"], ctx["gs"])
        local_trace.append(diag["E"])
        eval_energies.extend(local_trace)
        g, tk, tm, shells = schedules.unpack_full(solved["x"], L)
        records.append({
            "name": spec_name,
            "role": spec.get("role"),
            "sigma": spec.get("sigma"),
            "source_L": spec.get("source_L"),
            "seed": [float(v) for v in np.asarray(x_seed).reshape(-1)],
            "x": [float(v) for v in solved["x"]],
            "angles": {"gamma": g, "theta_k": tk, "theta_m": tm,
                       "c1": shells[:, 0].tolist(), "c2": shells[:, 1].tolist(), "c3": shells[:, 2].tolist()},
            "nfev": int(solved["nfev"] + 1),
            "pretrain_nfev": int(solved["pretrain_nfev"]),
            "nit": int(solved["nit"]),
            "success_flag": bool(solved["success"]),
            "roughness": roughness(shells),
            **{k: diag[k] for k in ("E", "energy_error", "F", "var_H", "grad_norm", "parity", "reflection", "state_hash")},
            "_psi": diag["psi"],
        })
    if not records:
        return {"arm": arm, "L": L, "skipped": True, "reason": "time budget before a start finished"}
    marginal = int(sum(r["nfev"] for r in records))
    inclusive = marginal + int(source_nfev)
    tau = f_base = None
    if str(L) in PROTOCOL["thresholds"]:
        tau = PROTOCOL["thresholds"][str(L)]["tau_energy_error"]
        f_base = PROTOCOL["thresholds"][str(L)]["F_baseline"]
    stats = summarize(records, ctx["E0"], tau, f_base, marginal, inclusive)
    psis = [r.pop("_psi") for r in records]
    basins = cluster_states(psis)
    curve = best_so_far(eval_energies, ctx["E0"])
    best_i = int(np.argmin([r["energy_error"] for r in records]))
    cell = {
        "arm": arm,
        "L": L,
        "skipped": False,
        "P": 3 * L + 3,
        "manifest_sha256": PROTOCOL["manifest_sha256"],
        "packing_sha256": PROTOCOL["packing_sha256"],
        "source_note": source_note,
        "source_nfev_charged": int(source_nfev),
        "stats": stats,
        "basins": {k: basins[k] for k in ("n_basins", "conjugate_only_pairs", "threshold")},
        "basin_labels": basins["labels"],
        "best_so_far_error": curve[:: max(1, len(curve) // 400)],
        "best_so_far_nfev": list(range(1, len(curve) + 1))[:: max(1, len(curve) // 400)],
        "best": {k: records[best_i][k] for k in records[best_i] if k != "seed"},
        "starts": records,
        "wall_time_s": time.perf_counter() - t0,
        "partial": len(records) < n_starts,
        "n_finished": len(records),
        "n_planned": n_starts,
    }
    state_path = RESULTS / "states" / f"L{L}_{arm}.npz"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_path,
        psis=np.stack(psis),
        x=np.stack([r["x"] for r in records]),
        best_so_far=np.asarray(curve),
    )
    # Drop bulky seeds from the human table but keep them in the cell.
    save_json(RESULTS / "cells" / f"L{L}_{arm}.json", cell)
    print(
        f"  {arm} L{L} best {stats['energy_error_best']:.4f} F {stats['F_of_best_energy']:.3f} "
        f"med {stats['energy_error_median']:.4f} IQR {stats['energy_error_iqr']:.4f} "
        f"pE {stats['p_success_energy']} both {stats['p_success_energy_and_F']} "
        f"basins {basins['n_basins']} conj_only {basins['conjugate_only_pairs']} "
        f"marginal {marginal} inclusive {inclusive}",
        flush=True,
    )
    return cell


def write_report(cells, replay, meta):
    lines = []
    lines.append("# Round 4 R — initialization of the same sublattice + block-mean generator")
    lines.append("")
    lines.append(
        "No new generator. G = Hc + thetaK Khat + thetaM Mhat, P = 3L+3, sublattice packing "
        f"`{PROTOCOL['packing_sha256']}`, manifest `{PROTOCOL['manifest_sha256']}`. "
        "Three arms were frozen in `protocol.json` before this round's training evaluations. "
        "Gamma and theta are shared and are not wrapped. New shells are not padded with zeros."
    )
    lines.append("")
    lines.append(
        f"Stored-angle replay of round-3 L6: energy error {replay['energy_error']:.6f} "
        f"(stored {replay['stored_error']:.6f}), fidelity {replay['F']:.6f}, state overlap {replay['overlap']:.6f}."
    )
    lines.append("")
    lines.append("## Interpolation")
    lines.append("")
    lines.append(
        "Knots are layer centers x_l = (l-0.5)/L. Interior interpolation is linear below 3 source layers and a natural cubic otherwise. "
        "Outside the source knots the value is the nearest endpoint. "
        "Shell coefficients are unwrapped with period pi after a numerical global-phase check. Gamma and theta are copied, not unwrapped."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| L | arm | finished | best E−E0 | F | median | IQR | energy successes | joint successes | marginal nfev | inclusive nfev | basins | conjugate-only pairs |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for L in range(1, 7):
        for arm in ARMS:
            cell = cells.get((L, arm))
            if not cell or cell.get("skipped"):
                lines.append(f"| {L} | {arm} | skipped | | | | | | | | | | |")
                continue
            st = cell["stats"]
            lines.append(
                f"| {L} | {arm} | {cell['n_finished']}/{cell['n_planned']} | {st['energy_error_best']:.4f} | "
                f"{st['F_of_best_energy']:.4f} | {st['energy_error_median']:.4f} | {st['energy_error_iqr']:.4f} | "
                f"{st['n_success_energy']} | {st['n_success_energy_and_F']} | {st['marginal_nfev']} | "
                f"{st['inclusive_nfev']} | {cell['basins']['n_basins']} | {cell['basins']['conjugate_only_pairs']} |"
            )
    lines.append("")
    lines.append("Energy success uses the frozen thresholds: L4 error ≤ 0.435669, L5 ≤ 0.390, L6 ≤ 0.315. ")
    lines.append("Joint success also requires fidelity at least 0.3728, 0.412, and 0.485 on that same start. ")
    lines.append("A zero success count is censored: evals per success is not a number.")
    lines.append("")
    lines.append(_screen(cells))
    lines.append("")
    lines.append("## Ledgers and the stored optimum")
    lines.append("")
    lines.append(
        "Plain and Chebyshev inclusive counts equal their marginal counts. "
        "Transfer adds the round-3 A_sub_M evaluations from L=1 through the highest neighbor used. "
        "Seeding L=5 from stored L=6, or any neighbor from an ascending round-3 chain, is not an independent fresh discovery. "
        "The stored L=6 point itself is not copied onto an L=6 start."
    )
    lines.append("")
    lines.append(
        "Round-3 QAOA after its own rescue remains the comparison: p=9 error 0.3903 F 0.412, "
        "p=11 error 0.3150 F 0.485, p=12 error 0.2878 F 0.515. "
        "The p=12 energy edge of about 0.004 over the stored block-mean state is a different state, not a noise bar on that state's energy. "
        "p=12 does not have both error ≤ 0.292 and F ≥ 0.610, so it does not remove the fidelity side of that comparison. "
        "Equal-layer results and parameter-matched results stay separate. P=21 versus p=11's 22 parameters is a rounded match, not an equality of optima."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- wall {meta.get('wall_time_s')} s")
    lines.append(f"- protocol sha256 `{meta.get('protocol_sha256')}`")
    lines.append(f"- partial: {meta.get('partial')}")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _screen(cells):
    lines = ["## Screening", ""]
    lines.append(
        "The finite-budget screen asks whether transfer or Chebyshev raises energy-success rate by 0.25 "
        "and cuts inclusive evals per energy success to at most half of plain, at L5 and L6. "
        "Missing the screen is not a proof that the basin is a single point, and a numerical miss is not a ceiling."
    )
    for L in (5, 6):
        plain = cells.get((L, "plain"))
        if not plain or plain.get("skipped"):
            lines.append(f"L{L}: plain did not finish.")
            continue
        p0 = plain["stats"]["p_success_energy"]
        t0 = plain["stats"]["evals_per_energy_success"]
        for arm in ("transfer", "chebyshev"):
            cell = cells.get((L, arm))
            if not cell or cell.get("skipped"):
                lines.append(f"L{L} {arm}: not finished.")
                continue
            p = cell["stats"]["p_success_energy"]
            t = cell["stats"]["evals_per_energy_success"]
            if p is None or t0 is None or t is None or t0["inclusive"] == 0:
                lines.append(f"L{L} {arm}: success count censored on one of the arms, screen not computed.")
                continue
            dp = p - p0
            ratio = t["inclusive"] / t0["inclusive"]
            met = dp >= 0.25 and ratio <= 0.5
            lines.append(
                f"L{L} {arm}: energy p_success {p:.3f} versus plain {p0:.3f} (difference {dp:.3f}); "
                f"inclusive evals/success {t['inclusive']:.1f} versus plain {t0['inclusive']:.1f} (ratio {ratio:.3f}); "
                f"screen {'met' if met else 'not met'}."
            )
        # sigma 0.1 reaching tau
        cell = cells.get((L, "transfer"))
        if cell and not cell.get("skipped") and L == 6:
            tau = PROTOCOL["thresholds"]["6"]["tau_energy_error"]
            hits = [s for s in cell["starts"] if s.get("sigma") == 0.1 and s["energy_error"] <= tau]
            lines.append(
                f"L6 transfer starts with sigma 0.1 that reached error ≤ {tau}: {len(hits)} of "
                f"{sum(1 for s in cell['starts'] if s.get('sigma') == 0.1)}. "
                "Zero would mean this budget did not show a neighborhood, not that the basin is only the stored point."
            )
    return "\n".join(lines)


def make_plots(cells):
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"plain": "C0", "transfer": "C1", "chebyshev": "C2"}
    # CDF at L4, L5, L6
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, L in zip(axes, (4, 5, 6)):
        tau = PROTOCOL["thresholds"][str(L)]["tau_energy_error"]
        for arm in ARMS:
            cell = cells.get((L, arm))
            if not cell or cell.get("skipped"):
                continue
            err = np.sort([s["energy_error"] for s in cell["starts"]])
            y = np.arange(1, len(err) + 1) / len(err)
            ax.step(err, y, where="post", color=colors[arm], label=arm)
        ax.axvline(tau, color="k", lw=0.8, ls="--")
        ax.set_title(f"L={L}")
        ax.set_xlabel("energy error of every finished start")
    axes[0].set_ylabel("empirical CDF")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cdf_L4_L6.png", dpi=140)
    plt.close(fig)
    # Best so far, inclusive and marginal
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    qaoa_lines = {5: 0.3903282882, 6: 0.3149868740}
    for col, L in enumerate((4, 5, 6)):
        for arm in ARMS:
            cell = cells.get((L, arm))
            if not cell or cell.get("skipped"):
                continue
            path = RESULTS / "states" / f"L{L}_{arm}.npz"
            curve = np.load(path)["best_so_far"]
            x_m = np.arange(1, len(curve) + 1)
            x_i = x_m + cell["source_nfev_charged"]
            axes[0, col].plot(x_i, curve, color=colors[arm], label=arm, lw=1.2)
            axes[1, col].plot(x_m, curve, color=colors[arm], label=arm, lw=1.2)
        if L in qaoa_lines:
            axes[0, col].axhline(qaoa_lines[L], color="k", lw=0.7, ls="--")
            axes[1, col].axhline(qaoa_lines[L], color="k", lw=0.7, ls="--")
        axes[0, col].set_title(f"L={L} inclusive")
        axes[1, col].set_title(f"L={L} marginal")
        axes[1, col].set_xlabel("evaluations")
    axes[0, 0].set_ylabel("best energy error so far")
    axes[1, 0].set_ylabel("best energy error so far")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_so_far_L4_L6.png", dpi=140)
    plt.close(fig)
    # Pareto of best energy-selected states
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arm in ARMS:
        xs, ys, ss = [], [], []
        for L in range(1, 7):
            cell = cells.get((L, arm))
            if not cell or cell.get("skipped"):
                continue
            xs.append(3 * L + 3)
            ys.append(cell["stats"]["energy_error_best"])
            ss.append(40 + 80 * cell["stats"]["F_of_best_energy"])
        ax.scatter(xs, ys, s=ss, color=colors[arm], label=arm)
    qaoa = [(18, 0.3903282882, 0.412), (22, 0.3149868740, 0.485), (24, 0.2877949092, 0.515)]
    ax.scatter([p[0] for p in qaoa], [p[1] for p in qaoa], marker="x", color="k", label="QAOA p9, p11, p12")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("best energy error")
    ax.set_title("Energy-selected states; marker size grows with F")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pareto_energy_selected.png", dpi=140)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    deadline = t0 + 70 * 60
    protocol_hash = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    man = hashlib.sha256((ROOT / "FROZEN_TARGET_R2.json").read_bytes()).hexdigest()
    pak = hashlib.sha256((ROOT / "packings.json").read_bytes()).hexdigest()
    if man != PROTOCOL["manifest_sha256"] or pak != PROTOCOL["packing_sha256"]:
        raise RuntimeError(f"hash mismatch {man} {pak}")
    print("protocol", protocol_hash, flush=True)
    sys12 = model.system(12)
    spec = circuit.dense_spectrum(sys12["hc"], sys12["B"], 12)
    trips, _ = model.packing("sublattice", 12)
    ctx = {
        "L": None,
        "hc": sys12["hc"],
        "fk": sys12["fk"],
        "fm": model.block_mean_diag(sys12["J"], trips, sys12["z"])[0],
        "B": sys12["B"],
        "indices": model.precompute(trips, 12),
        "E0": spec["E0"],
        "gs": spec["gs"],
        "ridx": spec["ridx"],
        "bounds": None,
    }
    # Replay stored L6. One forward, not a training start.
    x6, cell6 = load_source(6)
    psi, E, _, _ = circuit.simulate("M", x6, 6, ctx["hc"], ctx["fk"], ctx["fm"], ctx["B"], 12, ctx["indices"], False)
    replay = {
        "energy_error": float(E - spec["E0"]),
        "stored_error": float(cell6["energy_error"]),
        "F": circuit.fidelity(psi, spec["gs"]),
        "overlap": abs(np.vdot(psi, np.load(ROOT.parent / "round3_R" / "results" / "states" / "N12_L6_A_sub_M.npz")["psi"])),
    }
    print(f"replay L6 err {replay['energy_error']:.6f} stored {replay['stored_error']:.6f} overlap {replay['overlap']:.6f}", flush=True)
    if abs(replay["energy_error"] - replay["stored_error"]) > 1e-5:
        raise RuntimeError("stored L6 replay does not match")
    cells = {}
    previous = None
    partial = False
    for L in range(1, 7):
        ctx["L"] = L
        ctx["bounds"] = circuit.bounds_for("M", L)
        for arm_i, arm in enumerate(ARMS):
            rng = np.random.default_rng(20261107 + 1000 * L + arm_i)
            print(f"optimize L={L} {arm}", flush=True)
            cell = run_arm(arm, L, ctx, rng, previous if arm == "plain" else None, deadline)
            cells[(L, arm)] = cell
            if cell.get("skipped") or cell.get("partial"):
                partial = True
            if arm == "plain" and not cell.get("skipped"):
                previous = cell["best"]["x"]
        # Checkpoint a compact summary after every depth.
        save_json(RESULTS / "summary_partial.json", {
            "finished_through_attempted_L": L,
            "partial": partial,
            "rows": [
                {"L": c["L"], "arm": c["arm"], "skipped": c.get("skipped", False),
                 "best": None if c.get("skipped") else c["stats"]["energy_error_best"]}
                for c in cells.values()
            ],
        })
    meta = {
        "wall_time_s": time.perf_counter() - t0,
        "protocol_sha256": protocol_hash,
        "partial": partial,
        "E0": spec["E0"],
        "replay": replay,
    }
    save_json(RESULTS / "run_meta.json", meta)
    try:
        make_plots(cells)
    except Exception as exc:
        print("plot failure", exc, flush=True)
        meta["plot_failure"] = str(exc)
    write_report(cells, replay, meta)
    print(f"done {meta['wall_time_s']:.1f}s partial={partial}", flush=True)


if __name__ == "__main__":
    main()
