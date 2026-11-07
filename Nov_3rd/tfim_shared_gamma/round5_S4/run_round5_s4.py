"""Round 5 S4. Replay is already required to pass; these arms train energy only."""

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

import schedules
import sim

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
STRIDE4 = PROTOCOL["stride4_cavities"]
BASE = ROOT.parents[0]


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


def bounds(L):
    g = (-2.0 * np.pi, 2.0 * np.pi)
    th = (-5.0, 5.0)
    return [g, th, th] + [g] * (3 * L)


def clip(x, L):
    y = np.array(x, dtype=np.float64, copy=True)
    for i, (lo, hi) in enumerate(bounds(L)):
        y[i] = min(max(float(y[i]), lo), hi)
    return y


def load_vec(path, L):
    cell = json.loads(Path(path).read_text())
    if "x" in cell and len(cell["x"]) == 3 + 3 * L:
        return np.asarray(cell["x"], dtype=np.float64), cell
    ang = cell["angles"] if "angles" in cell else cell["best"]["angles"]
    x = [ang["gamma"], ang["theta_k"], ang["theta_m"]]
    for i in range(L):
        x.extend([ang["c1"][i], ang["c2"][i], ang["c3"][i]])
    return np.asarray(x, dtype=np.float64), cell


def sublattice_sources():
    r3 = BASE / "round3_R" / "results" / "cells"
    r4 = BASE / "round4_R" / "results" / "cells"
    out = {}
    for L, path in {
        3: r4 / "L3_plain.json",
        4: r4 / "L4_plain.json",
        5: r3 / "N12_L5_A_sub_M.json",
        6: r3 / "N12_L6_A_sub_M.json",
    }.items():
        x, cell = load_vec(path, L)
        if "stats" in cell:
            x = np.asarray(cell["best"]["x"], dtype=np.float64)
        out[L] = x
    return out


def lbfgs(sys, L, x0):
    trace = []

    def fun(x):
        _, E, g, _ = sim.logical_grad(sys, L, x)
        trace.append(float(E))
        return float(E), np.asarray(g, dtype=np.float64)

    res = minimize(
        fun, np.asarray(x0, dtype=np.float64), method="L-BFGS-B", jac=True, bounds=bounds(L),
        options={"maxiter": 1500, "maxfun": 600, "ftol": 1e-14, "gtol": 1e-8},
    )
    return np.asarray(res.x, dtype=np.float64), trace, int(res.nfev), bool(res.success)


def plain_center(L, previous):
    if previous is None:
        return None
    g, tk, tm, shells = sim.unpack(previous, L - 1)
    return schedules.pack(g, tk, tm, schedules.resample_shells(shells, L, "auto"))


def make_starts(arm, L, center_list):
    n = 12 if L < 4 and arm == 0 else 24
    starts = []
    for k in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([20261107, arm, L, k]))
        if arm == 0:
            x = rng.normal(0.0, 0.35, size=3 + 3 * L)
            role = "random"
            if k == 0 and center_list:
                x = center_list[0]
                role = "bootstrap"
            elif k == 1 and L >= 3 and len(center_list) > 1:
                x = center_list[1]
                role = "linear"
            starts.append({"k": k, "role": role, "sigma": None, "x0": clip(x, L)})
        else:
            sigma = 0.02 if k < 12 else 0.1
            center = center_list[k % len(center_list)]
            x = center + rng.normal(0.0, sigma, size=center.size)
            starts.append({"k": k, "role": "perturb", "sigma": sigma, "x0": clip(x, L)})
    return starts


def centers_for(arm, L, previous_plain, arm1_L5, sources):
    if arm == 0:
        if previous_plain is None:
            return []
        g, tk, tm, shells = sim.unpack(previous_plain, L - 1)
        auto = schedules.pack(g, tk, tm, schedules.resample_shells(shells, L, "auto"))
        lin = schedules.pack(g, tk, tm, schedules.resample_shells(shells, L, "linear"))
        return [auto, lin]
    if arm == 1:
        if L == 6:
            g, tk, tm, shells = sim.unpack(arm1_L5, 5)
            return [schedules.pack(g, tk, tm, schedules.resample_shells(shells, 6, "auto"))]
        stored = np.load(ROOT / "source" / "control_stride4.npz")["x"]
        g, tk, tm, shells = sim.unpack(stored, 6)
        return [schedules.pack(g, tk, tm, schedules.resample_shells(shells, L, "auto"))]
    neighbors = [d for d in (L - 1, L + 1) if d in sources]
    built = []
    for d in neighbors:
        g, tk, tm, shells = sim.unpack(sources[d], d)
        built.append(schedules.pack(g, tk, tm, schedules.resample_shells(shells, L, "auto")))
    return built


def cluster(psis):
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
            if direct > 0.99:
                parent[find(j)] = find(i)
            elif conj > 0.99:
                conj_only += 1
    roots = [find(i) for i in range(n)]
    return len(set(roots)), conj_only


def wilson(k, n):
    if n == 0:
        return None
    z = 1.96
    p = k / n
    den = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def run_cell(arm, L, sys, spec, center_list, deadline):
    if time.perf_counter() > deadline:
        return {"arm": arm, "L": L, "skipped": True}
    starts = make_starts(arm, L, center_list)
    records = []
    energies = []
    psis = []
    t0 = time.perf_counter()
    for spec_s in starts:
        if time.perf_counter() > deadline:
            break
        x, trace, nfev, ok = lbfgs(sys, L, spec_s["x0"])
        psi, E, g, var = sim.logical_grad(sys, L, x)
        energies.extend(trace)
        energies.append(E)
        gma, tk, tm, shells = sim.unpack(x, L)
        rec = {
            "k": spec_s["k"],
            "role": spec_s["role"],
            "sigma": spec_s["sigma"],
            "x0": spec_s["x0"].tolist(),
            "x": x.tolist(),
            "angles": {"gamma": gma, "theta_k": tk, "theta_m": tm,
                       "c1": shells[:, 0].tolist(), "c2": shells[:, 1].tolist(), "c3": shells[:, 2].tolist()},
            "E": E,
            "energy_error": E - spec["E0"],
            "F": sim.fidelity(spec["gs"], psi),
            "var_H": var,
            "grad_norm": float(np.linalg.norm(g)),
            "nfev": nfev + 1,
            "optimizer_success": ok,
            "state_hash": hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest(),
        }
        records.append(rec)
        psis.append(psi)
    if not records:
        return {"arm": arm, "L": L, "skipped": True}
    err = np.array([r["energy_error"] for r in records])
    fids = np.array([r["F"] for r in records])
    q1, q3 = np.percentile(err, [25, 75])
    best = int(np.argmin(err))
    qaoa = PROTOCOL["qaoa_floor"]
    matched = {4: "9", 5: "9", 6: "11"}[L] if L >= 4 else None
    # L4 matched depth is p=8. D's floor starts at p=9, the next depth. Both are recorded.
    def count(thr_e, thr_f):
        return int(np.sum((err <= thr_e) & (fids + 0 >= thr_f))) if thr_f is not None else int(np.sum(err <= thr_e))
    hist = PROTOCOL["historical_screens_not_a_seal"].get(str(L))
    n_hist = int(np.sum(err <= hist)) if hist is not None else None
    ref = qaoa.get(matched) if matched else None
    n_matched = None
    if ref is not None:
        n_matched = int(np.sum((err <= ref["energy_error"] + 1e-8) & (fids >= ref["F"] - 1e-8)))
    basins, conj_only = cluster(psis)
    curve = []
    running = np.inf
    for i, E in enumerate(energies, start=1):
        running = min(running, E - spec["E0"])
        curve.append(running)
    cell = {
        "arm": arm,
        "L": L,
        "skipped": False,
        "n_finished": len(records),
        "n_planned": len(starts),
        "partial": len(records) < len(starts),
        "marginal_nfev": int(sum(r["nfev"] for r in records)),
        "energy_error_best": float(err[best]),
        "F_best_energy": float(fids[best]),
        "var_best": float(records[best]["var_H"]),
        "median": float(np.median(err)),
        "iqr": float(q3 - q1),
        "n_hist_energy": n_hist,
        "hist_interval": wilson(n_hist, len(records)) if n_hist is not None else None,
        "n_joint_vs_matched_qaoa": n_matched,
        "joint_interval": wilson(n_matched, len(records)) if n_matched is not None else None,
        "basins": basins,
        "conjugate_only_pairs": conj_only,
        "best": records[best],
        "starts": records,
        "curve": curve,
        "wall_s": time.perf_counter() - t0,
    }
    path = RESULTS / "states" / f"arm{arm}_L{L}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, psis=np.stack(psis), x=np.asarray([r["x"] for r in records]), curve=np.asarray(curve))
    save_json(RESULTS / "cells" / f"arm{arm}_L{L}.json", cell)
    print(
        f"  arm{arm} L{L} best {cell['energy_error_best']:.4f} F {cell['F_best_energy']:.3f} "
        f"med {cell['median']:.4f} IQR {cell['iqr']:.4f} n {cell['n_finished']} nfev {cell['marginal_nfev']}",
        flush=True,
    )
    return cell


def arm_inclusive(cells, arm):
    marginal = sum(c["marginal_nfev"] for (a, L), c in cells.items() if a == arm and not c.get("skipped"))
    source = PROTOCOL["source_costs"]["arm1_stride4_control"] if arm == 1 else (
        PROTOCOL["source_costs"]["arm2_sublattice"] if arm == 2 else 0
    )
    return marginal + source


def write_report(cells, replay, meta):
    lines = [
        "# Round 5 S4 — stride-4 replay and transfer",
        "",
        "The logical target is the frozen manifest "
        f"`{PROTOCOL['manifest_sha256']}`. Stride-4 cavities are "
        "`[[0,4,8],[1,5,9],[2,6,10],[3,7,11]]`. "
        f"Canonical packing hash `{meta['stride4_packing_sha256']}`. "
        "M and rM were recomputed for this packing. "
        f"rM = {meta['rM']:.12f}, rK = {meta['rK']:.12f}.",
        "",
        f"Independent replay of D's energy-selected L6 state, commit `5ebca82a`, "
        f"file sha256 `{meta['stride4_npz_sha256']}`: "
        f"logical/packed overlap {replay['logical_packed']:.3e} below 1, "
        f"source overlap {replay['source']:.3e} below 1, "
        f"energy residual {replay['dE']:.3e}, fidelity residual {replay['dF']:.3e}.",
        "",
        "Within-cavity bit reversal leaves this symmetric shell unchanged. "
        "Inverting the physical-to-logical basis map does change the state, and that direction was rejected. "
        "A π shift of one shell coefficient is a global phase. "
        "Negating gamma and the shells conjugates the state and does not make it the same ray.",
        "",
        "## Arms",
        "",
        "| arm | L | best E−E0 | F | var | median | IQR | marginal | basins | conjugate-only |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in (0, 1, 2):
        for L in range(1, 7):
            cell = cells.get((arm, L))
            if not cell or cell.get("skipped"):
                continue
            lines.append(
                f"| {arm} | {L} | {cell['energy_error_best']:.4f} | {cell['F_best_energy']:.4f} | "
                f"{cell['var_best']:.3f} | {cell['median']:.4f} | {cell['iqr']:.4f} | "
                f"{cell['marginal_nfev']} | {cell['basins']} | {cell['conjugate_only_pairs']} |"
            )
    lines += [
        "",
        f"Inclusive totals, source charged once: arm 0 = {meta['inclusive']['0']}, "
        f"arm 1 = {meta['inclusive']['1']} (includes 1556), "
        f"arm 2 = {meta['inclusive']['2']} (includes 28099).",
        "",
        "## Screen",
        "",
        meta["screen_text"],
        "",
        "QAOA p=12 from the same manifest, 89272 evaluations, has energy error 0.2878 and fidelity 0.515. "
        "A stride-4 state with higher fidelity and higher energy is a tradeoff, not joint dominance. "
        "p=11 is the ceil-rounded match to P=21, not an equal parameter count. "
        "Historical thresholds 0.390 and 0.315 are optimizer diagnostics only.",
        "",
        f"Wall {meta['wall_s']:.1f} s. Partial: {meta['partial']}.",
    ]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


def plots(cells):
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: "C0", 1: "C1", 2: "C2"}
    names = {0: "plain", 1: "descend", 2: "cross"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, L in zip(axes, (4, 5, 6)):
        for arm in (0, 1, 2):
            cell = cells.get((arm, L))
            if not cell or cell.get("skipped"):
                continue
            err = np.sort([s["energy_error"] for s in cell["starts"]])
            ax.step(err, np.arange(1, len(err) + 1) / len(err), where="post", color=colors[arm], label=names[arm])
        ax.set_title(f"L={L}")
        ax.set_xlabel("energy error")
    axes[0].set_ylabel("CDF of every finished start")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cdf_L4_L6.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    source_share = {0: 0, 1: 1556 / 3, 2: 28099 / 3}
    for ax, L in zip(axes, (4, 5, 6)):
        for arm in (0, 1, 2):
            cell = cells.get((arm, L))
            if not cell or cell.get("skipped"):
                continue
            parent = 0
            if arm == 0:
                parent = sum(cells[(0, d)]["marginal_nfev"] for d in range(1, L) if (0, d) in cells and not cells[(0, d)].get("skipped"))
            x = np.arange(1, len(cell["curve"]) + 1) + parent + source_share[arm]
            ax.plot(x, cell["curve"], color=colors[arm], label=names[arm], lw=1.1)
        ax.axhline(0.2917196118876131, color="k", lw=0.7, ls="--")
        ax.set_title(f"L={L}")
        ax.set_xlabel("inclusive evaluations")
    axes[0].set_ylabel("best energy error so far")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_so_far_inclusive.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arm in (0, 1, 2):
        xs, ys, ss = [], [], []
        for L, cell in cells.items():
            if L[0] != arm or cell.get("skipped"):
                continue
            xs.append(3 * L[1] + 3)
            ys.append(cell["energy_error_best"])
            ss.append(30 + 90 * cell["F_best_energy"])
        ax.scatter(xs, ys, s=ss, color=colors[arm], label=names[arm])
    ax.scatter([18, 22, 24], [0.39032828820639587, 0.3149868740398478, 0.28779490915225203], marker="x", color="k", label="QAOA p9, p11, p12")
    ax.scatter([21], [0.2923991307066025], marker="D", color="C3", label="stored stride4 L6")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("best energy error")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pareto_energy_selected.png", dpi=140)
    plt.close(fig)


def screen_text(cells):
    med_sub = PROTOCOL["sublattice_transfer_medians"]
    best_sub = {"5": 0.31834106335331214, "6": 0.2917196118876131}
    worse = []
    for L in ("5", "6"):
        cell = cells.get((1, int(L)))
        if not cell or cell.get("skipped"):
            return "Arm 1 did not finish at L5 and L6, so the retirement screen is not evaluated."
        gap = cell["median"] - med_sub[L]
        worse.append(gap > 0.01)
    beat = False
    for arm in (0, 1, 2):
        for L, ref in ((5, best_sub["5"]), (6, best_sub["6"])):
            cell = cells.get((arm, L))
            if cell and not cell.get("skipped") and cell["energy_error_best"] < ref - 1e-8:
                beat = True
    if all(worse) and not beat:
        return (
            "Arm 1 median error is more than 0.01 above the sublattice transfer median at both L5 and L6, "
            "and no energy-selected stride-4 state beats the stored sublattice energy at either depth. "
            "Stride-4 is retired as a candidate and kept as a control. Its fidelity tradeoff is not erased."
        )
    return (
        "The retirement screen is not met. "
        f"Median gaps versus the sublattice transfer median are recorded in the table. "
        f"A stored-sublattice energy improvement was {'found' if beat else 'not found'}."
    )


def main():
    t0 = time.perf_counter()
    deadline = t0 + 70 * 60
    if sim.sha256_file(sim.MANIFEST) != PROTOCOL["manifest_sha256"]:
        raise RuntimeError("manifest")
    sys = sim.parse_system(STRIDE4)
    spec = sim.spectrum(sys)
    stored = np.load(ROOT / "source" / "control_stride4.npz")
    psi_l = sim.logical_state(sys, 6, stored["x"])
    psi_p = sim.physical_state(sys, 6, stored["x"])
    E, _ = sim.energy_var(psi_l, sys)
    F = sim.fidelity(spec["gs"], psi_l)
    replay = {
        "logical_packed": 1 - abs(np.vdot(psi_l, psi_p)),
        "source": 1 - abs(np.vdot(psi_l, stored["psi"])),
        "dE": (E - spec["E0"]) - 0.2923991307066025,
        "dF": F - 0.8579519623570816,
    }
    print("replay", replay, "E0", spec["E0"], flush=True)
    if replay["source"] > 1e-12 or abs(replay["dE"]) > 1e-10 or abs(replay["dF"]) > 1e-10:
        raise RuntimeError("replay failed; training blocked")
    packing_bytes = json.dumps({"name": "stride4", "n": 12, "cavities": STRIDE4, "low_bit": "first"}, separators=(",", ":")).encode()
    sources = sublattice_sources()
    cells = {}
    previous = None
    arm1_L5 = None
    partial = False
    print("arm 0 ladder", flush=True)
    for L in range(1, 7):
        centers = centers_for(0, L, previous, None, sources)
        cell = run_cell(0, L, sys, spec, centers, deadline)
        cells[(0, L)] = cell
        partial = partial or cell.get("skipped") or cell.get("partial")
        if not cell.get("skipped"):
            previous = cell["best"]["x"]
    print("arm 1 descending", flush=True)
    for L in (5, 4, 6):
        if L == 6 and (arm1_L5 is None):
            cells[(1, 6)] = {"arm": 1, "L": 6, "skipped": True}
            partial = True
            continue
        centers = centers_for(1, L, None, arm1_L5, sources)
        cell = run_cell(1, L, sys, spec, centers, deadline)
        cells[(1, L)] = cell
        partial = partial or cell.get("skipped") or cell.get("partial")
        if L == 5 and not cell.get("skipped"):
            arm1_L5 = cell["best"]["x"]
    print("arm 2 cross packing", flush=True)
    for L in (6, 5, 4):
        centers = centers_for(2, L, None, None, sources)
        cell = run_cell(2, L, sys, spec, centers, deadline)
        cells[(2, L)] = cell
        partial = partial or cell.get("skipped") or cell.get("partial")
    # Equalize inclusive totals if time remains.
    extras = {(arm, L): 0 for arm in (0, 1, 2) for L in (4, 5, 6)}
    cycle = [(0, 6), (1, 6), (2, 6), (0, 5), (1, 5), (2, 5), (0, 4), (1, 4), (2, 4)]
    while time.perf_counter() < deadline - 30:
        totals = {arm: arm_inclusive(cells, arm) for arm in (0, 1, 2)}
        if max(totals.values()) - min(totals.values()) < 600:
            break
        arm = min(totals, key=totals.get)
        placed = False
        for a, L in cycle:
            if a != arm or extras[(arm, L)] >= 24:
                continue
            if (arm, L) not in cells or cells[(arm, L)].get("skipped"):
                continue
            print(f"equalize arm {arm} L{L}", flush=True)
            k = 100 + extras[(arm, L)]
            rng = np.random.default_rng(np.random.SeedSequence([20261107, arm, L, k]))
            if arm == 0:
                x0 = clip(rng.normal(0.0, 0.35, size=3 + 3 * L), L)
            else:
                # one more perturbation around the same frozen center
                centers = centers_for(arm, L, None, arm1_L5, sources)
                sigma = 0.02 if extras[(arm, L)] % 2 == 0 else 0.1
                x0 = clip(centers[0] + rng.normal(0.0, sigma, size=centers[0].shape), L)
            x, trace, nfev, ok = lbfgs(sys, L, x0)
            psi, E, g, var = sim.logical_grad(sys, L, x)
            rec = {
                "k": k, "role": "equalize", "sigma": None, "x0": x0.tolist(), "x": x.tolist(),
                "E": E, "energy_error": E - spec["E0"], "F": sim.fidelity(spec["gs"], psi),
                "var_H": var, "grad_norm": float(np.linalg.norm(g)), "nfev": nfev + 1,
                "optimizer_success": ok,
                "state_hash": hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest(),
            }
            cell = cells[(arm, L)]
            cell["starts"].append(rec)
            cell["marginal_nfev"] += rec["nfev"]
            cell["n_finished"] += 1
            err = np.array([s["energy_error"] for s in cell["starts"]])
            cell["energy_error_best"] = float(np.min(err))
            cell["median"] = float(np.median(err))
            q1, q3 = np.percentile(err, [25, 75])
            cell["iqr"] = float(q3 - q1)
            if rec["energy_error"] <= cell["energy_error_best"] + 1e-15:
                cell["F_best_energy"] = rec["F"]
                cell["var_best"] = rec["var_H"]
                cell["best"] = rec
            cell["curve"].extend([min(cell["curve"][-1], E - spec["E0"])] * len(trace))
            extras[(arm, L)] += 1
            save_json(RESULTS / "cells" / f"arm{arm}_L{L}.json", cell)
            placed = True
            break
        if not placed:
            break
    inclusive = {str(arm): arm_inclusive(cells, arm) for arm in (0, 1, 2)}
    text = screen_text(cells)
    meta = {
        "wall_s": time.perf_counter() - t0,
        "partial": partial,
        "inclusive": inclusive,
        "screen_text": text,
        "rM": sys["rM"],
        "rK": sys["rK"],
        "hc_sha256": sys["hc_sha256"],
        "stride4_packing_sha256": hashlib.sha256(packing_bytes).hexdigest(),
        "stride4_npz_sha256": hashlib.sha256((ROOT / "source" / "control_stride4.npz").read_bytes()).hexdigest(),
        "E0": spec["E0"],
        "replay": replay,
    }
    save_json(RESULTS / "run_meta.json", meta)
    (ROOT / "stride4_packing.json").write_bytes(packing_bytes)
    try:
        plots(cells)
    except Exception as exc:
        print("plot", exc, flush=True)
    write_report(cells, replay, meta)
    print("done", meta["wall_s"], inclusive, text, flush=True)


if __name__ == "__main__":
    main()
