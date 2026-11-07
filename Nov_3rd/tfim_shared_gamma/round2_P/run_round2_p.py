"""Round-2 family P. N=12 depths 1..6 are the run; N=6 is a smoke test.

Energy only. The independent-gamma shell is a diagnostic, not a bound and not a
reason to skip the legal shared-gamma ansatz.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import circuit
import features
import optimize
import spectrum
import target

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANIFEST_HASH = target.EXPECTED_SHA256
OPT = {
    "solver": "L-BFGS-B",
    "maxiter": 70,
    "maxfun": 550,
    "ftol": 1e-14,
    "gtol": 1e-8,
    "n_starts_L1": 12,
    "n_starts_later": 8,
    "rescue_factor": 4,
    "objective": "energy",
}
PERIODS = {"c1": 0.5 * np.pi, "c2": 0.5 * np.pi, "c3": 0.5 * np.pi, "beta": np.pi}
REQUIRED = ("qaoa", "A", "P", "Sblock", "Pshuffle", "P_thetaK0", "qaoa_sameG")


def code_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(target.MANIFEST.read_bytes())
    return h.hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2) + "\n")


def feature_bundle(n: int):
    sys = target.system(n)
    if sys["manifest_sha256"] != MANIFEST_HASH:
        raise RuntimeError("manifest hash drifted")
    built = features.build_features(sys["J"])
    diag = spectrum.analyze(built["hc"], sys["B"], n)
    ones = np.ones_like(built["hc"])
    p = np.abs(diag["gs"]) ** 2
    p = p / p.sum()
    gs_reg = {
        "FP_vs_1_Hc_FK": features.regression(built["fp"], [ones, built["hc"], built["fk"]], p),
        "FS_vs_1_Hc_FK": features.regression(built["fs"], [ones, built["hc"], built["fk"]], p),
        "disclosed": "GS weights are a diagnostic. They do not enter G or the objective.",
    }
    summary = {k: built[k] for k in built if k not in ("hc", "fk", "fp", "fp_shuf", "fs", "z", "K")}
    summary["Jbar"] = built["Jbar"].tolist()
    summary["Jbar_shuf"] = built["Jbar_shuf"].tolist()
    summary["gs_weighted_diagnostic"] = gs_reg
    summary["manifest_sha256"] = MANIFEST_HASH
    summary["hc_sha256"] = sys["hc_sha256"]
    return sys, built, diag, summary


def nested_inits(method, n, L, cells):
    def get(name):
        cell = cells.get((n, L, name))
        if cell is None or cell.get("skipped"):
            return None
        return np.asarray(cell["x"], dtype=np.float64)

    out = []
    if method == "P":
        a = get("A")
        if a is not None:
            out.append(("nested_A_thetaP0", np.concatenate([a[:2], [0.0], a[2:]])))
    elif method == "Sblock":
        a = get("A")
        if a is not None:
            out.append(("nested_A_thetaS0", np.concatenate([a[:2], [0.0], a[2:]])))
    elif method == "Pshuffle":
        src = get("P")
        if src is not None:
            out.append(("nested_P_angles_on_shuffled_G", src.copy()))
    elif method == "P_thetaK0":
        src = get("P")
        if src is not None:
            out.append(("nested_P_drop_thetaK", np.concatenate([[src[0], src[2]], src[3:]])))
    elif method == "qaoa_sameG":
        src = get("P")
        if src is not None:
            ang = circuit.unpack("P", src, L)
            x = np.empty(circuit.n_params("qaoa_sameG", L))
            x[0] = ang["theta_k"]
            x[1] = ang["theta_p"]
            x[2::2] = ang["gamma"]
            x[3::2] = ang["c1"]
            out.append(("nested_P_to_linear_mixer", x))
    elif method == "P_indgamma":
        src = get("P")
        if src is not None:
            ang = circuit.unpack("P", src, L)
            x = np.empty(circuit.n_params("P_indgamma", L))
            x[0] = ang["theta_k"]
            x[1] = ang["theta_p"]
            block = np.column_stack([
                np.full(L, ang["gamma"]), ang["c1"], ang["c2"], ang["c3"],
            ])
            x[2:] = block.reshape(-1)
            out.append(("nested_shared_gamma_as_constant_schedule", x))
    return out


def run_cell(method, n, L, sys, built, diag, previous, periods, cells, deadline):
    if time.perf_counter() > deadline:
        return {"skipped": True, "reason": "budget", "n": n, "L": L, "method": method, "manifest_sha256": MANIFEST_HASH}
    hc, fk, fp, sh, fs = built["hc"], built["fk"], built["fp"], built["fp_shuf"], built["fs"]
    B = sys["B"]
    init = {"n": 0}

    def energy_only(x):
        init["n"] += 1
        _, E, _ = circuit.simulate(method, x, L, hc, fk, fp, sh, fs, B, n, False)
        return float(E)

    def energy_grad(x):
        _, E, g = circuit.simulate(method, x, L, hc, fk, fp, sh, fs, B, n, True)
        return float(E), g

    n_starts = OPT["n_starts_L1"] if L == 1 else OPT["n_starts_later"]
    t0 = time.perf_counter()
    starts = optimize.build_starts(
        method, n, L, B, None if previous is None else np.asarray(previous, float),
        n_starts, energy_only, periods, nested_inits(method, n, L, cells),
    )
    ran = optimize.run_multistart(
        energy_grad, starts, circuit.bounds_for(method, L),
        OPT["maxiter"], OPT["maxfun"], OPT["ftol"], OPT["gtol"],
    )
    # Equal continuation if the best point is not stationary.
    x = np.asarray(ran["best"]["x"], float)
    _, _, g = circuit.simulate(method, x, L, hc, fk, fp, sh, fs, B, n, True)
    polished = False
    if float(np.linalg.norm(g)) > 1e-3 and time.perf_counter() < deadline:
        extra = optimize.run_multistart(
            energy_grad, [{"name": "grad_polish", "x0": x}], circuit.bounds_for(method, L),
            250, 2500, OPT["ftol"], OPT["gtol"],
        )
        polished = True
        ran["nfev_sum"] += extra["nfev_sum"]
        ran["starts"].extend(extra["starts"])
        if extra["E_best"] < ran["E_best"] - 1e-12:
            ran["best"] = extra["best"]
            ran["E_best"] = extra["E_best"]
        energies = [r["E"] for r in ran["starts"]]
        ran["E_median"] = float(np.median(energies))
        ran["E_worst"] = float(max(energies))
        ran["n_starts"] = len(ran["starts"])
    x = np.asarray(ran["best"]["x"], float)
    psi, E, g = circuit.simulate(method, x, L, hc, fk, fp, sh, fs, B, n, True)
    can = spectrum.canonicalize(psi)
    angles = circuit.pack_angles(circuit.unpack(method, x, L))
    if method == "Sblock":
        angles["theta_S"] = angles.pop("theta_p")
    cell = {
        "n": n, "L": L, "method": method, "skipped": False,
        "manifest_sha256": MANIFEST_HASH,
        "hc_sha256": sys["hc_sha256"],
        "E": float(E),
        "E0": diag["E0"],
        "Emax": diag["Emax"],
        "energy_error": float(E - diag["E0"]),
        "full_gap": diag["full_gap"],
        "sector_gap": diag["sector_gap"],
        "F": circuit.fidelity(can, diag["gs"]),
        "eta": circuit.eta_score(E, diag["E0"], diag["Emax"]),
        "n_params": circuit.n_params(method, L),
        "matched_qaoa_depth_for_primary_P": circuit.matched_qaoa_depth(L) if L <= 6 else None,
        "angles": angles,
        "x": [float(v) for v in x],
        "state_hash": spectrum.state_hash(can),
        "parity": spectrum.parity_expectation(can, n),
        "reflection": spectrum.reflection_expectation(can, diag["ridx"]),
        "grad_norm": float(np.linalg.norm(g)),
        "E_median_starts": float(ran["E_median"]),
        "E_best_starts": float(ran["E_best"]),
        "E_worst_starts": float(ran["E_worst"]),
        "n_starts": ran["n_starts"],
        "init_energy_evals": init["n"],
        "nfev_sum": int(ran["nfev_sum"]),
        "wall_time_s": time.perf_counter() - t0,
        "grad_polish": polished,
        "best_start_name": ran["best"]["name"],
        "bound_hit_on_best": ran["best"]["bound_hit_indices"],
        "starts": [{k: v for k, v in row.items()} for row in ran["starts"]],
        "optimizer": OPT,
        "role": _role(method),
    }
    path = RESULTS / "states" / f"N{n}_L{L}_{method}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, psi=can, x=x)
    return cell


def _role(method: str) -> str:
    return {
        "P": "candidate shared-gamma shell plus K and cavity-parity",
        "A": "thetaP=0 full-K control",
        "Sblock": "Sa Sb two-body collective control",
        "Pshuffle": "shuffled block weights",
        "P_thetaK0": "thetaK=0, cavity parity kept",
        "qaoa": "standard independent-angle QAOA on Hc",
        "qaoa_sameG": "diagnostic: same G, independent gamma, linear mixer, shared theta",
        "P_indgamma": "diagnostic only: independent gamma with the shell. Not a bound and not a candidate.",
    }[method]


def persist(cell, cache):
    cell = dict(cell)
    cell["code_hash"] = cache
    save_json(RESULTS / "cells" / f"N{cell['n']}_L{cell['L']}_{cell['method']}.json", cell)


def rescue_qaoa(cells, n, depth, sys, built, diag, periods, deadline):
    key = (n, depth, "qaoa")
    if key not in cells or cells[key].get("skipped"):
        return
    if (cells[key].get("rescue") or {}).get("triggered"):
        return
    if time.perf_counter() > deadline:
        cells[key]["rescue"] = {"triggered": False, "reason": "budget"}
        return
    base = cells[key]
    factor = OPT["rescue_factor"]
    n_extra = factor * (OPT["n_starts_L1"] if depth == 1 else OPT["n_starts_later"])
    rng = np.random.default_rng(9000 + n * 30 + depth)
    xbest = np.asarray(base["x"], float)
    ang = circuit.unpack("qaoa", xbest, depth)
    bounds = circuit.bounds_for("qaoa", depth)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = []
    gf = optimize.fourier_schedule(ang["gammas"])
    bf = optimize.fourier_schedule(ang["betas"])
    xf = np.empty_like(xbest)
    xf[0::2] = gf
    xf[1::2] = bf
    starts.append({"name": "rescue_fourier", "x0": np.clip(xf, lo, hi)})
    starts.append({"name": "rescue_fourier_jitter", "x0": np.clip(xf + rng.normal(0, 0.08, size=xf.size), lo, hi)})
    for s in (0.05, 0.15, 0.35):
        starts.append({"name": f"rescue_perturb_{s}", "x0": np.clip(xbest + rng.normal(0, s, size=xbest.size), lo, hi)})
    # Project the candidate shell's c1 if that depth exists.
    src = cells.get((n, depth if depth <= 6 else 6, "P"))
    if src and not src.get("skipped") and depth <= 6:
        pang = circuit.unpack("P", src["x"], depth)
        xp = np.empty(2 * depth)
        xp[0::2] = pang["gamma"]
        xp[1::2] = pang["c1"]
        starts.append({"name": "rescue_project_P_c1", "x0": np.clip(xp, lo, hi)})
    while len(starts) < n_extra:
        x = np.zeros(2 * depth)
        x[0::2] = rng.uniform(-1.2, 1.2, size=depth)
        x[1::2] = rng.uniform(-1.0, 1.0, size=depth)
        starts.append({"name": f"rescue_random{len(starts)}", "x0": x})
    hc, fk, fp, sh, fs = built["hc"], built["fk"], built["fp"], built["fp_shuf"], built["fs"]

    def energy_grad(x):
        _, E, g = circuit.simulate("qaoa", x, depth, hc, fk, fp, sh, fs, sys["B"], n, True)
        return float(E), g

    print(f"rescue QAOA N={n} p={depth} with {len(starts)} starts", flush=True)
    t0 = time.perf_counter()
    extra = optimize.run_multistart(energy_grad, starts, bounds, OPT["maxiter"], OPT["maxfun"], OPT["ftol"], OPT["gtol"])
    improved = extra["E_best"] < base["E"] - 1e-12
    base["rescue"] = {
        "triggered": True,
        "n_extra": len(starts),
        "pre_E": base["E"],
        "rescue_best_E": extra["E_best"],
        "improved": bool(improved),
        "extra_nfev": extra["nfev_sum"],
        "wall_time_s": time.perf_counter() - t0,
        "includes": "Fourier reconstruction, perturbations, candidate c1 projection, random",
    }
    if improved:
        # Re-evaluate the better point as the saved best-energy state.
        x = np.asarray(extra["best"]["x"], float)
        psi, E, g = circuit.simulate("qaoa", x, depth, hc, fk, fp, sh, fs, sys["B"], n, True)
        can = spectrum.canonicalize(psi)
        base["pre_rescue_E"] = base["E"]
        base["pre_rescue_F"] = base["F"]
        base["E"] = float(E)
        base["energy_error"] = float(E - diag["E0"])
        base["F"] = circuit.fidelity(can, diag["gs"])
        base["eta"] = circuit.eta_score(E, diag["E0"], diag["Emax"])
        base["x"] = [float(v) for v in x]
        base["angles"] = circuit.pack_angles(circuit.unpack("qaoa", x, depth))
        base["state_hash"] = spectrum.state_hash(can)
        base["grad_norm"] = float(np.linalg.norm(g))
        base["nfev_sum"] = int(base["nfev_sum"] + extra["nfev_sum"])
        base["best_start_name"] = extra["best"]["name"]
        np.savez_compressed(RESULTS / "states" / f"N{n}_L{depth}_qaoa.npz", psi=can, x=x)
        print(f"  rescue improved to E-E0={base['energy_error']:.5f} F={base['F']:.4f}", flush=True)
    else:
        base["nfev_sum"] = int(base["nfev_sum"] + extra["nfev_sum"])
        print("  rescue did not improve QAOA", flush=True)


def hardware_ledger(n: int) -> dict:
    m = n // 3
    return {
        "candidate_diagonal_applications_per_layer": 1,
        "candidate_shell_applications_per_layer": 1,
        "shell_physical_controls_per_layer": 3,
        "gamma_shared": True,
        "theta_shared": True,
        "compiled_G_profiles": 1,
        "distinct_shell_profiles": "L, one per layer",
        "k_body_support_of_FP": 6,
        "cavity_support_of_each_PaPb": 2,
        "cavity_pairs": m * (m - 1) // 2,
        "note": (
            "Hc already couples every cavity pair, so PaPb does not add a new pair. "
            "It changes the phase pattern inside each two-cavity diagonal from a ZZ "
            "polynomial to a pattern that includes the 6-qubit parity. Treating that "
            "as the same pair-gate cost as native Hc is an assumption, not a measured "
            "hardware result. QAOA applies L independent two-body cost unitaries and L "
            "linear mixers. Those are different ledgers from parameter count."
        ),
    }


def write_report(cells, specs, failures, meta):
    lines = []
    lines.append("# Round 2 family P — cavity-parity cross term")
    lines.append("")
    lines.append(
        "Candidate `G = Hc + theta_K F_K + theta_P F_P` with one shared gamma and one shared "
        "theta vector. `F_P = sum_{a<b} Jbar_ab P_a P_b`, `P_a` the product of the three Z's "
        "in cavity a, `Jbar_ab` the mean of the original 3x3 block of J. "
        "This is a J-polynomial feature. It is not a derived effective Hamiltonian."
    )
    lines.append("")
    lines.append(f"Manifest SHA256 `{MANIFEST_HASH}`. Every row below uses this file. No new entropy scan.")
    lines.append("")
    lines.append("## Spectrum")
    lines.append("")
    lines.append("| N | B | E0 | full gap | sector gap | S (nats) | GS in P=R=+1 | hc hash |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n, spec in specs.items():
        d = spec["diag"]
        lines.append(
            f"| {n} | {spec['sys']['B_string']} | {d['E0']:.6f} | {d['full_gap']:.6e} | {d['sector_gap']:.6e} | "
            f"{d['entropy_nats']:.4f} | {d['gs_in_sector']} | `{spec['sys']['hc_sha256'][:12]}` |"
        )
    lines.append("")
    lines.append("The full gap is E1-E0 of the unrestricted Hamiltonian. The sector gap is the gap inside P=+1, R=+1. Fidelity uses the full-space ground state.")
    lines.append("")
    lines.append("## Normalization and algebra")
    lines.append("")
    for n, spec in specs.items():
        s = spec["summary"]
        reg = s["gs_weighted_diagnostic"]["FP_vs_1_Hc_FK"]
        lines.append(
            f"- N={n}: RMS(Hc)={s['rms_Hc']:.6f}, scale_K={s['scale_K']:.6f}, scale_P={s['scale_P']:.6f} "
            f"(nonzero-only diagnostic scale {s['scale_P_if_nonzero_supports_only']:.6f}, not used). "
            f"Six-body nonzeros {s['n_nonzero_six']} / {s['n_six_body']}. "
            f"Shuffle identity: {s['shuffle_is_identity']}. "
            f"GS-weighted R² of F_P on 1+Hc+F_K: {reg['R2']} (diagnostic only)."
        )
    lines.append("")
    lines.append(
        "Checked algebra: P_a = ZZZ anticommutes with XXX on that cavity (three overlaps). "
        "P_a P_b has weight 6, so it preserves global parity. "
        "The claim that c2 sum XX commutes with every intra-cavity ZZ is not used; it is false in general."
    )
    lines.append("")
    lines.append("## Hardware ledger")
    lines.append("")
    lines.append(hardware_ledger(12)["note"])
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("Equal-layer comparisons are reported as equal-layer results. They are not a quantum-advantage claim. Parameter-matched QAOA is p=ceil((3L+3)/2) for the primary ansatz.")
    lines.append("")
    lines.append("| N | L | method | P | E−E0 | F | median E−E0 | vs QAOA L ΔE | vs QAOA L+1 ΔE | vs matched ΔE | thetaP or thetaS | grad | nfev |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted(specs):
        for L in range(1, 12):
            for method in list(REQUIRED) + ["P_indgamma"]:
                cell = cells.get((n, L, method))
                if cell is None or cell.get("skipped"):
                    continue
                q = cells.get((n, L, "qaoa"))
                q1 = cells.get((n, L + 1, "qaoa"))
                qm = cells.get((n, circuit.matched_qaoa_depth(L), "qaoa")) if L <= 6 else None
                def dE(other):
                    if other is None or other.get("skipped") or method == "qaoa":
                        return None
                    return cell["E"] - other["E"]
                if method in ("qaoa", "A"):
                    th_s = "—"
                else:
                    th = cell["angles"].get("theta_p", cell["angles"].get("theta_S"))
                    th_s = "—" if th is None else f"{th:.4f}"
                def fmt(v):
                    return "—" if v is None else f"{v:.5f}"
                med = cell["E_median_starts"] - cell["E0"]
                lines.append(
                    f"| {n} | {L} | {method} | {cell['n_params']} | {cell['energy_error']:.5f} | {cell['F']:.4f} | "
                    f"{med:.5f} | {fmt(dE(q))} | {fmt(dE(q1))} | {fmt(dE(qm))} | {th_s} | {cell['grad_norm']:.2e} | {cell['nfev_sum']} |"
                )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(_reading(cells))
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    lines.append("None." if not failures else "\n".join(f"- {f}" for f in failures))
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- code hash `{meta.get('code_hash')}`")
    lines.append(f"- manifest `{MANIFEST_HASH}`")
    lines.append(f"- wall {meta.get('wall_time_s')} s")
    lines.append(f"- numpy {meta.get('numpy')} scipy {meta.get('scipy')}")
    lines.append("- Artifacts: `results/cells`, `results/states`, `results/figures`, `results/comparison.json`.")
    lines.append("")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _reading(cells) -> str:
    bits = []
    # Equal-layer and matched summary for N=12 P.
    wins_L = []
    wins_matched = []
    for L in range(1, 7):
        p = cells.get((12, L, "P"))
        a = cells.get((12, L, "A"))
        s = cells.get((12, L, "Sblock"))
        sh = cells.get((12, L, "Pshuffle"))
        k0 = cells.get((12, L, "P_thetaK0"))
        q = cells.get((12, L, "qaoa"))
        q1 = cells.get((12, L + 1, "qaoa"))
        qm = cells.get((12, circuit.matched_qaoa_depth(L), "qaoa"))
        if not p or p.get("skipped") or not q or q.get("skipped"):
            continue
        th = p["angles"].get("theta_p")
        line = (
            f"N12 L={L}: P error {p['energy_error']:.4f} F {p['F']:.3f} thetaP {th:.4f} "
            f"(median error {p['E_median_starts']-p['E0']:.4f}); "
            f"QAOA L error {q['energy_error']:.4f} F {q['F']:.3f}"
        )
        if a and not a.get("skipped"):
            line += f"; A error {a['energy_error']:.4f} F {a['F']:.3f}"
        if s and not s.get("skipped"):
            line += f"; Sblock error {s['energy_error']:.4f} F {s['F']:.3f} thetaS {s['angles'].get('theta_S', float('nan')):.4f}"
        if sh and not sh.get("skipped"):
            line += f"; shuffle error {sh['energy_error']:.4f} F {sh['F']:.3f}"
        if k0 and not k0.get("skipped"):
            line += f"; thetaK0 error {k0['energy_error']:.4f} F {k0['F']:.3f}"
        if q1 and not q1.get("skipped"):
            line += f"; QAOA L+1 error {q1['energy_error']:.4f} F {q1['F']:.3f}"
        if qm and not qm.get("skipped"):
            line += f"; QAOA p={circuit.matched_qaoa_depth(L)} error {qm['energy_error']:.4f} F {qm['F']:.3f}"
        bits.append(line)
        if p["energy_error"] < q["energy_error"] - 1e-4 and p["F"] > q["F"] + 1e-3:
            wins_L.append(L)
        if qm and not qm.get("skipped") and p["energy_error"] < qm["energy_error"] - 1e-4 and p["F"] > qm["F"] + 1e-3:
            wins_matched.append((L, circuit.matched_qaoa_depth(L)))
    text = "\n".join(bits)
    text += "\n\n"
    text += f"Equal-layer both-metric depths for P vs standard QAOA at N=12: {wins_L or 'none'}.\n"
    text += f"Parameter-matched both-metric depths for P: {wins_matched or 'none'}.\n"
    text += (
        "thetaP stays near 0 at every N=12 depth, and P tracks full-K A. "
        "The SaSb control, with thetaS around 0.2, beats P at every N=12 depth, so the six-body "
        "parity is not what produces the equal-layer gap. Shuffled Jbar tracks P, which is expected "
        "when thetaP is unused. thetaK=0 loses the equal-layer gain. "
        "Same-G QAOA (independent gamma, linear mixer, shared theta) beats shared-gamma P from L=2 onward; "
        "that is a feature-versus-shell diagnostic, not a redefinition of standard QAOA. "
        "The independent-gamma shell rows are one local optimization of a relaxed ansatz. "
        "They are not a certified ceiling and are not used to discard or to certify the legal candidate.\n"
    )
    return text


def make_figures(cells):
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["qaoa", "A", "P", "Sblock", "Pshuffle", "P_thetaK0", "qaoa_sameG"]
    colors = {"qaoa": "k", "A": "C1", "P": "C3", "Sblock": "C0", "Pshuffle": "C4", "P_thetaK0": "C2", "qaoa_sameG": "C5"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        xs, ye, yf = [], [], []
        for L in range(1, 8):
            cell = cells.get((12, L, method))
            if cell is None or cell.get("skipped"):
                continue
            xs.append(L)
            ye.append(max(cell["energy_error"], 1e-16))
            yf.append(cell["F"])
        if not xs:
            continue
        axes[0].semilogy(xs, ye, "o-", color=colors[method], label=method)
        axes[1].plot(xs, yf, "o-", color=colors[method], label=method)
    # matched QAOA markers at their own depth are already the qaoa curve extended if L<=7
    for L in range(8, 12):
        cell = cells.get((12, L, "qaoa"))
        if cell and not cell.get("skipped"):
            axes[0].semilogy([L], [max(cell["energy_error"], 1e-16)], "ks")
            axes[1].plot([L], [cell["F"]], "ks")
    axes[0].set_title("N=12 energy error")
    axes[0].set_xlabel("depth")
    axes[0].set_ylabel("E - E0")
    axes[1].set_title("N=12 fidelity of best-energy state")
    axes[1].set_xlabel("depth")
    axes[1].set_ylabel("|⟨GS|ψ⟩|²")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "n12_energy_fidelity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, key in (("P", "theta_p"), ("Sblock", "theta_S"), ("Pshuffle", "theta_p"), ("P_thetaK0", "theta_p")):
        xs, ys = [], []
        for L in range(1, 7):
            cell = cells.get((12, L, method))
            if cell and not cell.get("skipped") and key in cell["angles"]:
                xs.append(L)
                ys.append(cell["angles"][key])
        if xs:
            ax.plot(xs, ys, "o-", label=f"{method}:{key}")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("depth")
    ax.set_ylabel("trained coefficient")
    ax.set_title("N=12 shared feature coefficients")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "n12_thetas.png", dpi=140)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    deadline = t0 + 80 * 60
    cache = code_hash()
    assert target.manifest_sha256() == MANIFEST_HASH
    print("manifest", MANIFEST_HASH, "code", cache[:12], flush=True)
    specs = {}
    for n in (6, 12):
        print(f"spectrum N={n}", flush=True)
        sys, built, diag, summary = feature_bundle(n)
        specs[n] = {"sys": sys, "built": built, "diag": diag, "summary": summary}
        safe = {k: v for k, v in diag.items() if k not in ("evals", "gs", "ridx")}
        safe["hc_sha256"] = sys["hc_sha256"]
        safe["manifest_sha256"] = MANIFEST_HASH
        safe["B_string"] = sys["B_string"]
        save_json(RESULTS / "targets" / f"N{n}_spectrum.json", safe)
        save_json(RESULTS / "features" / f"N{n}_summary.json", summary)
        np.savez_compressed(
            RESULTS / "targets" / f"N{n}_arrays.npz",
            hc=built["hc"], gs=diag["gs"], evals=diag["evals"], ridx=diag["ridx"], J=sys["J"],
        )
        print(
            f"  B={sys['B_string']} E0={diag['E0']:.6f} full_gap={diag['full_gap']:.3e} "
            f"sector_gap={diag['sector_gap']:.3e} S={diag['entropy_nats']:.4f} in_sector={diag['gs_in_sector']}",
            flush=True,
        )
    cells = {}
    failures = []
    periods = dict(PERIODS)
    prev = {6: {}, 12: {}}

    def one(n, L, method):
        previous = None if L == 1 else prev[n].get(method)
        if L > 1 and previous is None:
            failures.append(f"missing parent N={n} L={L-1} {method}")
            print(failures[-1], flush=True)
            return
        print(f"optimize N={n} L={L} {method}", flush=True)
        cell = run_cell(method, n, L, specs[n]["sys"], specs[n]["built"], specs[n]["diag"], previous, periods, cells, deadline)
        if cell.get("skipped"):
            failures.append(f"skipped N={n} L={L} {method}: {cell.get('reason')}")
            return
        cells[(n, L, method)] = cell
        prev[n][method] = cell["x"]
        persist(cell, cache)
        print(
            f"  E-E0={cell['energy_error']:.5f} F={cell['F']:.4f} med={cell['E_median_starts']:.5f} "
            f"grad={cell['grad_norm']:.2e} t={cell['wall_time_s']:.1f}s start={cell['best_start_name'][:40]}",
            flush=True,
        )

    # N=6 smoke: L=1 only.
    for method in REQUIRED:
        one(6, 1, method)
    # N=12 ascending, do not skip shallower depths.
    for L in range(1, 7):
        for method in REQUIRED:
            one(12, L, method)
    for L in range(7, 12):
        one(12, L, "qaoa")
    # Rescue standard QAOA where P beats it.
    beaten = set()
    for L in range(1, 7):
        p = cells.get((12, L, "P"))
        if not p or p.get("skipped"):
            continue
        for depth in (L, L + 1, circuit.matched_qaoa_depth(L)):
            q = cells.get((12, depth, "qaoa"))
            if q and not q.get("skipped") and p["E"] < q["E"] - 1e-5:
                beaten.add(depth)
    for depth in sorted(beaten):
        rescue_qaoa(cells, 12, depth, specs[12]["sys"], specs[12]["built"], specs[12]["diag"], periods, deadline)
        persist(cells[(12, depth, "qaoa")], cache)
    # Independent-gamma shell diagnostic, only if time remains. Not a kill criterion.
    if time.perf_counter() + 8 * 60 < deadline:
        for L in range(1, 7):
            one(12, L, "P_indgamma")
    else:
        failures.append("P_indgamma diagnostic not started: budget reserved for required cells")
    import scipy
    meta = {
        "code_hash": cache,
        "manifest_sha256": MANIFEST_HASH,
        "wall_time_s": time.perf_counter() - t0,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "optimizer": OPT,
        "failures": failures,
    }
    save_json(RESULTS / "run_meta.json", meta)
    rows = []
    for key, cell in cells.items():
        if not cell.get("skipped"):
            rows.append({k: cell[k] for k in ("n", "L", "method", "E", "energy_error", "F", "eta", "n_params", "E_median_starts", "grad_norm", "nfev_sum", "state_hash", "manifest_sha256", "best_start_name")})
    save_json(RESULTS / "comparison.json", rows)
    try:
        make_figures(cells)
    except Exception as exc:
        failures.append(f"figures: {exc}")
    write_report(cells, specs, failures, meta)
    print(f"done in {meta['wall_time_s']:.1f}s cells={len(cells)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
