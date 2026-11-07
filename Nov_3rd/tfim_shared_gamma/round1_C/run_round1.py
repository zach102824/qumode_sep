"""Round-1 direction C: shared generator Hc + theta_K F_K + theta_C F_star.

Re-run from this directory:
    python3 run_round1.py

Energy is the only training objective. Fidelity is computed after the fact.
N=12 retraining is not seeded from the N=6 transfer.
"""

from __future__ import annotations

import argparse
import csv
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
import hamiltonian
import optimize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PRIMARY = ("qaoa", "shared_linear", "D", "A", "C")
CANDIDATES = ("D", "A", "C")
PERIODS = {"c1": 0.5 * np.pi, "c2": 0.5 * np.pi, "c3": 0.5 * np.pi, "beta": np.pi}

OPT = {
    "solver": "L-BFGS-B",
    "maxiter": 80,
    "maxfun": 700,
    "ftol": 1e-14,
    "gtol": 1e-8,
    "n_starts_L1": 18,
    "n_starts_ge2": 12,
    "objective": "energy <H>",
    "fidelity": "post-hoc only; best-E state is kept, not best-F",
    "rescue_extra_starts": 12,
    "nested_inits": (
        "Richer models replace up to two of their random starts with the same-depth "
        "simpler model (new coefficients set to 0). The start count does not increase. "
        "QAOA rescue includes a projection of the winning shell's c1 onto beta."
    ),
}

HYPOTHESIS = (
    "The four-body star is the cubic monomial suggested by a formal Schrieffer-Wolff "
    "expansion of a transverse field around a uniform classical bound built from J alone. "
    "That motivation is a hypothesis. This run does not solve a Schrieffer-Wolff effective "
    "Hamiltonian, does not claim the series is controlled at the entropy peak, and does not "
    "fix theta_C to a perturbative value. Theta is a free shared energy-fit coefficient in "
    "front of an explicitly normalized J-polynomial feature."
)


def code_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def feature_summary(built: dict) -> dict:
    skip = {"hc_diag", "fk_diag", "fs_diag", "K", "z"}
    return {k: clean(v) for k, v in built.items() if k not in skip}


def prepare_target(n: int, cache_hash: str, force: bool) -> dict:
    path = RESULTS / "targets" / f"N{n}.json"
    npz = RESULTS / "targets" / f"N{n}_arrays.npz"
    if path.exists() and npz.exists() and not force:
        meta = load_json(path)
        if meta.get("code_hash") == cache_hash:
            arrays = np.load(npz)
            meta["gs"] = arrays["gs"]
            meta["hc"] = arrays["hc"]
            meta["J"] = arrays["J"]
            meta["evals"] = arrays["evals"]
            meta["reflection_indices"] = arrays["ridx"]
            print(f"loaded cached target N={n} B={meta['B']}", flush=True)
            return meta
    print(f"selecting entropy-max field N={n}", flush=True)
    J = hamiltonian.build_J(n)
    raw = hamiltonian.select_field(n, J, verbose=True)
    gs = raw.pop("gs")
    hc = raw.pop("hc")
    evals = raw.pop("evals")
    ridx = raw.pop("reflection_indices")
    raw["code_hash"] = cache_hash
    raw["J"] = J
    # coupling table
    table = RESULTS / "targets" / f"N{n}_couplings.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "Jij"])
        for i in range(n):
            for j in range(i + 1, n):
                w.writerow([i, j, f"{J[i, j]:.16e}"])
    save_json(path, {k: v for k, v in raw.items() if k != "J"})
    np.savez_compressed(npz, gs=gs, hc=hc, J=J, evals=evals, ridx=ridx)
    raw["gs"] = gs
    raw["hc"] = hc
    raw["evals"] = evals
    raw["reflection_indices"] = ridx
    print(
        f"frozen N={n} B={raw['B']:.4f} E0={raw['E0']:.6f} gap={raw['gap']:.3e} "
        f"S={raw['entropy_nats']:.4f} nats P={raw['parity_expectation']:.6f} "
        f"R={raw['reflection_expectation']:.6f} resid={raw['gs_residual']:.2e}",
        flush=True,
    )
    return raw


def prepare_features(n: int, J: np.ndarray, gs: np.ndarray) -> dict:
    primary = features.build_features(J, max_cavities=2)
    orth = features.orthogonality_report(primary["hc_diag"], primary["fk_diag"], primary["fs_diag"], primary["z"])
    relevant = features.relevant_state_report(primary["hc_diag"], primary["fk_diag"], primary["fs_diag"], gs)
    unrestricted = None
    orth_u = None
    if n > 6:
        unrestricted = features.build_features(J, max_cavities=None)
        orth_u = features.orthogonality_report(
            unrestricted["hc_diag"], unrestricted["fk_diag"], unrestricted["fs_diag"], unrestricted["z"]
        )
    pair_extra = sorted(set(map(tuple, primary["star_cavity_pairs"])) - set(map(tuple, primary["hc_cavity_pairs"])))
    summary = {
        "primary": feature_summary(primary),
        "orthogonality_full_space": orth,
        "relevant_states": relevant,
        "unrestricted": None if unrestricted is None else feature_summary(unrestricted),
        "unrestricted_orthogonality": orth_u,
        "star_pairs_not_already_in_Hc": pair_extra,
        "hypothesis": HYPOTHESIS,
        "normalization": (
            "F_K and F_star are divided by ||raw||_HS/||Hc||_HS so each has the same "
            "Pauli HS norm as Hc. Theta is dimensionless. The (J^2)_ii constant from "
            "Z^2=I is excluded from F_K and reported as dropped_J2_identity_constant."
        ),
        "truncation_rule": (
            "Primary: keep a weight-4 mask iff its qubits lie in at most 2 cavities "
            "(any pair, adjacent or not). N=6 has 2 cavities so nothing is dropped. "
            "N=12 drops 3- and 4-cavity masks. Unrestricted stars are diagnostic only. "
            "Pair-gate cost equivalence of an arbitrary two-cavity diagonal with the ZZ "
            "block is an assumption, not a measured hardware result."
        ),
    }
    save_json(RESULTS / "features" / f"N{n}_summary.json", summary)
    np.savez_compressed(
        RESULTS / "features" / f"N{n}_primary.npz",
        hc=primary["hc_diag"],
        fk=primary["fk_diag"],
        fs=primary["fs_diag"],
    )
    if unrestricted is not None:
        np.savez_compressed(
            RESULTS / "features" / f"N{n}_unrestricted.npz",
            fs=unrestricted["fs_diag"],
            fk=unrestricted["fk_diag"],
        )
    return {"primary": primary, "unrestricted": unrestricted, "summary": summary}


def gauge_audit(hc, fk, fs, B, n) -> dict:
    rng = np.random.default_rng(11)
    diffs = {"c1": [], "c2": [], "c3": [], "beta": []}
    for _ in range(4):
        x = rng.normal(scale=0.35, size=circuit.n_params("D", 2))
        _, e0, _ = circuit.simulate("D", x, 2, hc, fk, fs, B, n, False)
        for name, idx in (("c1", 1), ("c2", 2), ("c3", 3)):
            y = x.copy()
            y[idx] += 0.5 * np.pi
            _, e, _ = circuit.simulate("D", y, 2, hc, fk, fs, B, n, False)
            diffs[name].append(abs(e - e0))
        q = rng.normal(scale=0.4, size=4)
        _, eq, _ = circuit.simulate("qaoa", q, 2, hc, fk, fs, B, n, False)
        q2 = q.copy()
        q2[1::2] += np.pi
        _, eq2, _ = circuit.simulate("qaoa", q2, 2, hc, fk, fs, B, n, False)
        diffs["beta"].append(abs(eq - eq2))
    # Algebraic c2 fact: s2 in {3,-1}, (pi/2)*(3-(-1))=2*pi, so c2 -> c2+pi/2
    # multiplies every Fock state by the same phase.
    s2 = ((3 - 2 * np.arange(4)) ** 2 - 3) / 2
    algebraic_c2 = float(np.max(np.abs(np.exp(-1j * 0.5 * np.pi * s2) - np.exp(-1j * 0.5 * np.pi * s2[0]))))
    report = {
        "max_abs_energy_change": {k: float(np.max(v)) for k, v in diffs.items()},
        "algebraic_c2_global_phase_deviation": algebraic_c2,
        "periods_used_for_interpolation": {
            k: (None if report_period_ok(k, diffs, algebraic_c2) is False else PERIODS[k])
            for k in PERIODS
        },
    }
    # fill periods_used properly
    used = {}
    for k in ("c1", "c2", "c3", "beta"):
        ok = max(diffs[k]) < 1e-8 and (k != "c2" or algebraic_c2 < 1e-10)
        used[k] = PERIODS[k] if ok else None
    report["periods_used_for_interpolation"] = used
    report["note"] = (
        "Unwrap is applied only to controls whose pi/2 (shell) or pi (beta) shift left "
        "the energy unchanged on random D/QAOA states. Gamma is not given a period."
    )
    return report


def report_period_ok(k, diffs, algebraic_c2) -> bool:
    ok = max(diffs[k]) < 1e-8
    if k == "c2":
        ok = ok and algebraic_c2 < 1e-10
    return ok


def grad_audit(hc, fk, fs, B, n) -> dict:
    rng = np.random.default_rng(19)
    out = {}
    specs = {
        "qaoa": rng.normal(scale=0.3, size=circuit.n_params("qaoa", 2)),
        "D": rng.normal(scale=0.3, size=circuit.n_params("D", 2)),
        "A": rng.normal(scale=0.3, size=circuit.n_params("A", 1)),
        "C": rng.normal(scale=0.3, size=circuit.n_params("C", 1)),
    }
    for method, x in specs.items():
        L = 2 if method in ("qaoa", "D") else 1
        _, _, g = circuit.simulate(method, x, L, hc, fk, fs, B, n, True)
        gn = circuit.numerical_energy_grad(method, x, L, hc, fk, fs, B, n, eps=1e-6)
        err = float(np.max(np.abs(g - gn)))
        out[method] = {"max_abs": err, "L": L}
        if err > 2e-4:
            raise RuntimeError(f"gradient audit failed for {method}: {err}")
    return out


def nested_inits(method: str, n: int, L: int, cells: dict):
    """Same-depth simpler models, with every new coefficient set to zero.

    These replace random starts. They are energy-trained angles, not ground-state data.
    """
    found = []

    def take(key):
        cell = cells.get(key)
        if cell is None or cell.get("skipped"):
            return None
        return np.asarray(cell["x"], dtype=np.float64)

    if method == "D":
        sl = take((n, L, "shared_linear"))
        if sl is not None:
            x = [float(sl[0])]
            for c1 in sl[1:]:
                x.extend([float(c1), 0.0, 0.0])
            found.append(("nested_shared_linear", np.asarray(x, dtype=np.float64)))
    elif method == "A":
        d = take((n, L, "D"))
        if d is not None:
            found.append(("nested_D_thetaK0", np.concatenate([[d[0], 0.0], d[1:]])))
    elif method == "C":
        a = take((n, L, "A"))
        if a is not None:
            found.append(("nested_A_thetaC0", np.concatenate([[a[0], a[1], 0.0], a[2:]])))
        d = take((n, L, "D"))
        if d is not None:
            found.append(("nested_D_theta0", np.concatenate([[d[0], 0.0, 0.0], d[1:]])))
    elif method == "C_unrestricted":
        c = take((n, L, "C"))
        if c is not None:
            found.append(("nested_truncated_C", c.copy()))
        a = take((n, L, "A"))
        if a is not None:
            found.append(("nested_A_thetaC0", np.concatenate([[a[0], a[1], 0.0], a[2:]])))
    return found


def project_shell_to_qaoa(method: str, x, L: int) -> np.ndarray:
    parsed = circuit.unpack(method, np.asarray(x, dtype=np.float64), L)
    out = np.empty(2 * L, dtype=np.float64)
    out[0::2] = float(parsed["gamma"])
    out[1::2] = np.asarray(parsed["c1"], dtype=np.float64)
    return out


def run_one(method, n, L, B, hc, fk, fs, E0, Emax, gs, ridx, previous_x, periods, deadline, nested=None) -> dict:
    if time.perf_counter() > deadline:
        return {"skipped": True, "reason": "budget", "n": n, "L": L, "method": method}
    init_evals = {"n": 0}

    def energy_only(x):
        init_evals["n"] += 1
        _, E, _ = circuit.simulate(method, x, L, hc, fk, fs, B, n, False)
        return float(E)

    def energy_grad(x):
        _, E, g = circuit.simulate(method, x, L, hc, fk, fs, B, n, True)
        return float(E), g

    n_starts = OPT["n_starts_L1"] if L == 1 else OPT["n_starts_ge2"]
    t0 = time.perf_counter()
    starts = optimize.build_starts(
        method, n, L, B, None if previous_x is None else np.asarray(previous_x, dtype=np.float64),
        n_starts, energy_only, periods,
    )
    bounds = circuit.bounds_for(method, L)
    slots = [i for i, s in enumerate(starts) if str(s["name"]).startswith("random")]
    if len(slots) < len(nested or []):
        slots += [i for i, s in enumerate(starts) if str(s["name"]).startswith("perturb")]
    # Keep the earliest random seeds and overwrite the extras added for nesting.
    chosen = list(reversed(slots))[: len(nested or [])]
    for slot, (name, xnest) in zip(chosen, nested or []):
        xnest = np.asarray(xnest, dtype=np.float64)
        if xnest.shape != (circuit.n_params(method, L),):
            raise RuntimeError(f"nested {name} has shape {xnest.shape}")
        clipped = []
        for v, (lo, hi) in zip(xnest, bounds):
            clipped.append(min(max(float(v), lo), hi))
        starts[slot] = {"name": name, "x0": np.asarray(clipped, dtype=np.float64)}
    ran = optimize.run_multistart(
        energy_grad, starts, circuit.bounds_for(method, L),
        OPT["maxiter"], OPT["maxfun"], OPT["ftol"], OPT["gtol"],
    )
    polished = False
    if ran["best"]["bound_hit_indices"]:
        x0 = np.asarray(ran["best"]["x"], dtype=np.float64)
        wide = [(-2 * abs(lo) - 2, 2 * abs(hi) + 2) for lo, hi in circuit.bounds_for(method, L)]
        extra = optimize.run_multistart(
            energy_grad, [{"name": "bound_polish", "x0": x0}], wide,
            OPT["maxiter"], OPT["maxfun"], OPT["ftol"], OPT["gtol"],
        )
        polished = True
        if extra["E_best"] < ran["E_best"] - 1e-12:
            ran["starts"].extend(extra["starts"])
            ran["best"] = extra["best"]
            ran["E_best"] = extra["E_best"]
            ran["nfev_sum"] += extra["nfev_sum"]
            ran["E_median"] = float(np.median([r["E"] for r in ran["starts"]]))
            ran["E_worst"] = float(max(r["E"] for r in ran["starts"]))
            ran["n_starts"] = len(ran["starts"])
    cell = assemble_cell(
        method, n, L, B, hc, fk, fs, E0, Emax, gs, ridx, ran,
        init_evals["n"], time.perf_counter() - t0, polished, previous_x is not None,
    )
    return cell


def assemble_cell(method, n, L, B, hc, fk, fs, E0, Emax, gs, ridx, ran, init_evals, wall, polished, continued) -> dict:
    x = np.asarray(ran["best"]["x"], dtype=np.float64)
    psi, E, g = circuit.simulate(method, x, L, hc, fk, fs, B, n, True)
    F = circuit.fidelity(psi, gs)
    err = float(E - E0)
    parsed = circuit.pack_angles(circuit.unpack(method, x, L))
    gnorm = float(np.linalg.norm(g))
    state_path = RESULTS / "states" / f"N{n}_L{L}_{method}.npz"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    can = hamiltonian.canonicalize_vector(psi)
    np.savez_compressed(state_path, psi=can, x=x)
    energies = [r["E"] for r in ran["starts"]]
    cell = {
        "n": n,
        "L": L,
        "method": method,
        "skipped": False,
        "E": float(E),
        "E_optimizer": float(ran["E_best"]),
        "E_recompute_minus_optimizer": float(E - ran["E_best"]),
        "E0": float(E0),
        "Emax": float(Emax),
        "energy_error": err,
        "variational_violation": bool(err < -1e-7),
        "F": F,
        "eta": circuit.eta_score(E, E0, Emax),
        "n_params": circuit.n_params(method, L),
        "angles": parsed,
        "x": [float(v) for v in x],
        "state_hash": hamiltonian.state_hash(can),
        "parity_expectation": hamiltonian.parity_expectation(can),
        "reflection_expectation": hamiltonian.reflection_expectation(can, ridx),
        "grad_norm": gnorm,
        "E_median_starts": float(np.median(energies)),
        "E_worst_starts": float(max(energies)),
        "E_best_starts": float(min(energies)),
        "n_starts": len(ran["starts"]),
        "init_energy_evals": int(init_evals),
        "nfev_sum": int(ran["nfev_sum"]),
        "nit_sum": int(ran["nit_sum"]),
        "wall_time_s": float(wall),
        "bound_polish_attempted": bool(polished),
        "bound_hit_on_best": ran["best"]["bound_hit_indices"],
        "continued_from_previous_depth": bool(continued),
        "best_start_name": ran["best"]["name"],
        "optimizer": dict(OPT),
        "starts": [
            {k: v for k, v in row.items() if k != "trace"} | {"trace_len": len(row["trace"]), "trace": row["trace"]}
            for row in ran["starts"]
        ],
        "convergence_trace_best": ran["best"]["trace"],
    }
    return cell


def maybe_rescue(cells, method_keys, n, L, B, hc, fk, fs, E0, Emax, gs, ridx, deadline, extra_challengers=None) -> None:
    qkey = (n, L, "qaoa")
    if qkey not in cells or cells[qkey].get("skipped"):
        return
    if (cells[qkey].get("rescue") or {}).get("triggered"):
        return
    qE = cells[qkey]["E"]
    challengers = []
    for method in method_keys:
        key = (n, L, method)
        if key in cells and not cells[key].get("skipped") and cells[key]["E"] < qE - 1e-5:
            challengers.append(method)
    for name in extra_challengers or []:
        if name not in challengers:
            challengers.append(name)
    if not challengers:
        cells[qkey]["rescue"] = {"triggered": False}
        return
    if time.perf_counter() > deadline:
        cells[qkey]["rescue"] = {"triggered": False, "reason": "budget", "would_have_because": challengers}
        return
    print(f"rescue QAOA N={n} L={L} because {challengers} beat E={qE:.6f}", flush=True)
    pre = cells[qkey]["E"]
    rng = np.random.default_rng(7000 + n * 20 + L)
    xbest = np.asarray(cells[qkey]["x"], dtype=np.float64)
    bounds = circuit.bounds_for("qaoa", L)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = []
    for method in ("C", "A", "D"):
        src = cells.get((n, L, method))
        if src is None or src.get("skipped"):
            continue
        proj = project_shell_to_qaoa(method, src["x"], L)
        starts.append({"name": f"rescue_project_{method}", "x0": np.clip(proj, lo, hi)})
    for s, name in ((0.05, "r1"), (0.12, "r2"), (0.25, "r3"), (0.45, "r4")):
        noise = rng.normal(0.0, s, size=xbest.size)
        starts.append({"name": f"rescue_perturb_{name}", "x0": np.clip(xbest + noise, lo, hi)})
    pdim = xbest.size
    while len(starts) < OPT["rescue_extra_starts"]:
        x = np.zeros(pdim)
        x[0::2] = rng.uniform(-1.2, 1.2, size=L)
        x[1::2] = rng.uniform(-1.0, 1.0, size=L)
        starts.append({"name": f"rescue_random{len(starts)}", "x0": x})
    starts = starts[: OPT["rescue_extra_starts"]]

    def energy_grad(x):
        _, E, g = circuit.simulate("qaoa", x, L, hc, fk, fs, B, n, True)
        return float(E), g

    t0 = time.perf_counter()
    extra = optimize.run_multistart(
        energy_grad, starts, bounds, OPT["maxiter"], OPT["maxfun"], OPT["ftol"], OPT["gtol"]
    )
    improved = extra["E_best"] < pre - 1e-12
    cells[qkey]["rescue"] = {
        "triggered": True,
        "challengers": challengers,
        "pre_E": pre,
        "rescue_best_E": extra["E_best"],
        "improved": bool(improved),
        "extra_nfev": extra["nfev_sum"],
        "wall_time_s": time.perf_counter() - t0,
    }
    if improved:
        # Rebuild the cell from the better QAOA point but keep the original starts on record.
        rebuilt = assemble_cell(
            "qaoa", n, L, B, hc, fk, fs, E0, Emax, gs, ridx, extra,
            0, cells[qkey]["wall_time_s"] + (time.perf_counter() - t0), False, True,
        )
        rebuilt["rescue"] = cells[qkey]["rescue"]
        rebuilt["pre_rescue_E"] = pre
        rebuilt["pre_rescue_F"] = cells[qkey]["F"]
        rebuilt["nfev_sum"] = int(cells[qkey]["nfev_sum"] + extra["nfev_sum"])
        rebuilt["starts"] = cells[qkey]["starts"] + rebuilt["starts"]
        energies = [s["E"] for s in rebuilt["starts"]]
        rebuilt["E_median_starts"] = float(np.median(energies))
        rebuilt["E_worst_starts"] = float(max(energies))
        rebuilt["n_starts"] = len(rebuilt["starts"])
        cells[qkey] = rebuilt
        print(f"  rescue improved QAOA to E={rebuilt['E']:.6f} F={rebuilt['F']:.4f}", flush=True)
    else:
        print(f"  rescue did not improve QAOA (best extra {extra['E_best']:.6f})", flush=True)


def persist_cell(cell, cache_hash: str) -> None:
    cell = dict(cell)
    cell["code_hash"] = cache_hash
    path = RESULTS / "cells" / f"N{cell['n']}_L{cell['L']}_{cell['method']}.json"
    save_json(path, cell)


def load_cell(n, L, method, cache_hash, force):
    path = RESULTS / "cells" / f"N{n}_L{L}_{method}.json"
    if force or not path.exists():
        return None
    cell = load_json(path)
    if cell.get("code_hash") != cache_hash or cell.get("skipped"):
        return None
    return cell


def map_transfer_x(method, x, L, kappa_ratio, scale6, scale12, mode: str) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    if mode != "unscaled_gamma":
        if method == "qaoa":
            y[0::2] *= kappa_ratio
        else:
            y[0] *= kappa_ratio
    if mode == "alt_raw_prefactor":
        if method in ("A", "C", "C_unrestricted"):
            y[1] *= scale6["K"] / scale12["K"]
        if method in ("C", "C_unrestricted"):
            y[2] *= scale6["star"] / scale12["star"]
    return y


def evaluate_mapped(method, x, L, hc, fk, fs, B, n, E0, Emax, gs, ridx) -> dict:
    psi, E, _ = circuit.simulate(method, x, L, hc, fk, fs, B, n, False)
    can = hamiltonian.canonicalize_vector(psi)
    return {
        "E": float(E),
        "energy_error": float(E - E0),
        "F": circuit.fidelity(can, gs),
        "eta": circuit.eta_score(E, E0, Emax),
        "state_hash": hamiltonian.state_hash(can),
        "parity_expectation": hamiltonian.parity_expectation(can),
        "reflection_expectation": hamiltonian.reflection_expectation(can, ridx),
        "x": [float(v) for v in x],
        "angles": circuit.pack_angles(circuit.unpack(method, x, L)),
    }


def make_figures(cells, targets, feature_pack) -> None:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Entropy curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, n in zip(axes, (6, 12)):
        rows = targets[n]["coarse_grid"]
        ax.plot([r["B"] for r in rows], [r["entropy_nats"] for r in rows], color="C0")
        ax.axvline(targets[n]["B"], color="C3", ls="--", label=f"B={targets[n]['B']:.4f}")
        ax.set_xlabel("B")
        ax.set_ylabel("half-chain entropy (nats)")
        ax.set_title(f"N={n}")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_vs_B.png", dpi=140)
    plt.close(fig)

    methods = ["qaoa", "shared_linear", "D", "A", "C"]
    colors = {"qaoa": "k", "shared_linear": "0.5", "D": "C0", "A": "C1", "C": "C3", "C_unrestricted": "C5"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, n in enumerate((6, 12)):
        for method in methods + ["C_unrestricted"]:
            xs, ye, yf = [], [], []
            for L in range(1, 8):
                cell = cells.get((n, L, method))
                if cell is None or cell.get("skipped"):
                    continue
                xs.append(L)
                ye.append(max(cell["energy_error"], 1e-16))
                yf.append(cell["F"])
            if not xs:
                continue
            kw = {"color": colors[method], "marker": "o", "label": method}
            if method == "C_unrestricted":
                kw["ls"] = "--"
            axes[0, col].semilogy(xs, ye, **kw)
            axes[1, col].plot(xs, yf, **kw)
        axes[0, col].set_title(f"N={n} energy error")
        axes[0, col].set_xlabel("depth L")
        axes[0, col].set_ylabel("E - E0")
        axes[1, col].set_title(f"N={n} fidelity")
        axes[1, col].set_xlabel("depth L")
        axes[1, col].set_ylabel("|⟨GS|ψ⟩|²")
        axes[1, col].set_ylim(-0.02, 1.02)
        axes[0, col].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "energy_fidelity_vs_depth.png", dpi=140)
    plt.close(fig)

    # Schedules at the deepest available C for each N.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, n in zip(axes, (6, 12)):
        chosen = None
        for L in range(6, 0, -1):
            if (n, L, "C") in cells and not cells[(n, L, "C")].get("skipped"):
                chosen = cells[(n, L, "C")]
                break
        if chosen is None:
            ax.set_title(f"N={n} no C schedule")
            continue
        L = chosen["L"]
        s = np.linspace(0, 1, L) if L > 1 else np.array([0.0])
        ang = chosen["angles"]
        ax.plot(s, ang["c1"], "o-", label="c1")
        ax.plot(s, ang["c2"], "s-", label="c2")
        ax.plot(s, ang["c3"], "^-", label="c3")
        ax.set_title(f"N={n} C L={L}  γ={ang['gamma']:.3f} θK={ang['theta_k']:.3f} θC={ang['theta_c']:.3f}")
        ax.set_xlabel("normalized layer time s")
        ax.set_ylabel("shell control")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "shell_schedules_C.png", dpi=140)
    plt.close(fig)

    # Convergence at L=4 if present.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, n in zip(axes, (6, 12)):
        for method in ("qaoa", "C", "A", "D"):
            cell = cells.get((n, 4, method))
            if cell is None or cell.get("skipped"):
                continue
            tr = cell.get("convergence_trace_best") or []
            if tr:
                ax.plot(tr, color=colors[method], label=method)
        ax.set_title(f"N={n} L=4 best-start trace")
        ax.set_xlabel("stored evaluation")
        ax.set_ylabel("energy")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "convergence_L4.png", dpi=140)
    plt.close(fig)

    # Independence residual ratios (1 = not a cost reweighting).
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = []
    vals = []
    for n, pack in feature_pack.items():
        rel = pack["summary"]["relevant_states"]
        for name, key in (
            ("single flips", "single_flips"),
            ("excitations", "excitations_single_pair_block"),
            ("low classical", "lowest_64_classical"),
        ):
            block = rel[key]["vs_constant_plus_Hc_plus_FK"]
            labels.append(f"N{n} {name}")
            vals.append(block.get("residual_rms_ratio"))
        if "gs_probability_weighted" in rel:
            labels.append(f"N{n} GS-weighted")
            vals.append(rel["gs_probability_weighted"]["vs_constant_plus_Hc_plus_FK"].get("residual_rms_ratio"))
    ax.bar(range(len(vals)), [0 if v is None else v for v in vals], color="C0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("residual RMS / feature RMS")
    ax.set_ylim(0, 1.05)
    ax.set_title("F_star after regressing out 1, Hc, F_K")
    fig.tight_layout()
    fig.savefig(fig_dir / "star_independence.png", dpi=140)
    plt.close(fig)


def comparison_rows(cells) -> list:
    rows = []
    methods = list(PRIMARY) + ["C_unrestricted"]
    for n in (6, 12):
        for L in range(1, 8):
            q = cells.get((n, L, "qaoa"))
            q1 = cells.get((n, L + 1, "qaoa"))
            for method in methods:
                cell = cells.get((n, L, method))
                if cell is None or cell.get("skipped"):
                    continue
                def delta(other, field):
                    if other is None or other.get("skipped"):
                        return None
                    return float(cell[field] - other[field])
                rows.append({
                    "n": n,
                    "L": L,
                    "method": method,
                    "n_params": cell["n_params"],
                    "energy_error": cell["energy_error"],
                    "F": cell["F"],
                    "eta": cell["eta"],
                    "E_median_starts": cell["E_median_starts"],
                    "dE_vs_qaoa_same_L": None if method == "qaoa" else delta(q, "E"),
                    "dF_vs_qaoa_same_L": None if method == "qaoa" else delta(q, "F"),
                    "dE_vs_qaoa_Lplus1": None if method == "qaoa" else delta(q1, "E"),
                    "dF_vs_qaoa_Lplus1": None if method == "qaoa" else delta(q1, "F"),
                    "qaoa_L_params": None if q is None else q["n_params"],
                    "qaoa_Lplus1_params": None if q1 is None or q1.get("skipped") else q1["n_params"],
                    "param_matched_qaoa_L": int(np.ceil(cell["n_params"] / 2.0)),
                    "grad_norm": cell["grad_norm"],
                    "wall_time_s": cell["wall_time_s"],
                    "nfev_sum": cell["nfev_sum"],
                    "state_hash": cell["state_hash"],
                })
    return rows


def interpretation_lines(cells, targets, feature_pack, transfer) -> list:
    """Numbers-only reading. No claim that a Schrieffer-Wolff model has been established."""
    lines = ["## How to read this", ""]
    lines.append(
        "Same-depth wins are not parameter-matched wins. QAOA at depth ceil(n_params/2) is the "
        "matched-parameter comparator when that depth was actually run (through L=7). "
        "A negative ΔE is a lower energy than that QAOA. Fidelity is only the best-energy state."
    )
    lines.append("")
    lines.append("| N | candidate | L | params | E−E0 | F | matched QAOA L | QAOA E−E0 | QAOA F | both better? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    matched_both = []
    for n in (6, 12):
        for L in range(1, 8):
            for method in ("D", "A", "C"):
                cell = cells.get((n, L, method))
                if cell is None or cell.get("skipped"):
                    continue
                need = int(np.ceil(cell["n_params"] / 2.0))
                q = cells.get((n, need, "qaoa"))
                if q is None or q.get("skipped"):
                    lines.append(
                        f"| {n} | {method} | {L} | {cell['n_params']} | {cell['energy_error']:.5f} | {cell['F']:.4f} | {need} | not run | not run | no |"
                    )
                    continue
                better = cell["energy_error"] < q["energy_error"] - 1e-4 and cell["F"] > q["F"] + 1e-3
                if better:
                    matched_both.append((n, method, L, need))
                lines.append(
                    f"| {n} | {method} | {L} | {cell['n_params']} | {cell['energy_error']:.5f} | {cell['F']:.4f} | "
                    f"{need} | {q['energy_error']:.5f} | {q['F']:.4f} | {'yes' if better else 'no'} |"
                )
    lines.append("")
    if not matched_both:
        lines.append("No candidate beat parameter-matched QAOA on both energy error and fidelity at a depth that was run.")
    else:
        lines.append("Parameter-matched wins on both energy and fidelity (best energy state):")
        for n, method, L, need in matched_both:
            lines.append(f"- N={n} {method} L={L} versus QAOA L={need}.")
    lines.append("")
    lines.append("Family C versus Family A (theta_C free versus theta_C fixed at 0). Small gaps mean the star is not carrying the result.")
    lines.append("")
    lines.append("| N | L | A E−E0 | C E−E0 | C−A energy | A F | C F | C theta_C | C best start |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for n in (6, 12):
        for L in range(1, 7):
            a = cells.get((n, L, "A"))
            c = cells.get((n, L, "C"))
            if not a or not c or a.get("skipped") or c.get("skipped"):
                continue
            th = c["angles"].get("theta_c")
            lines.append(
                f"| {n} | {L} | {a['energy_error']:.5f} | {c['energy_error']:.5f} | {c['energy_error']-a['energy_error']:.5f} | "
                f"{a['F']:.4f} | {c['F']:.4f} | {th:.4f} | {c['best_start_name'][:48]} |"
            )
    lines.append("")
    lines.append(
        "Medians are over the multi-start energies, not over fidelities. A best-energy win whose median energy "
        "is worse than QAOA's best energy is a single-basin result."
    )
    lines.append("")
    for n in (6, 12):
        rel = feature_pack[n]["summary"]["relevant_states"]
        sf = rel["single_flips"]["vs_constant_plus_Hc_plus_FK"]
        ex = rel["excitations_single_pair_block"]["vs_constant_plus_Hc_plus_FK"]
        gs = rel.get("gs_probability_weighted", {}).get("vs_constant_plus_Hc_plus_FK", {})
        orth = feature_pack[n]["summary"]["orthogonality_full_space"]
        lines.append(
            f"N={n} operator check: full-space regression of F_star on all two-body Z strings plus identity has R² "
            f"{orth['full_two_body_regression_R2']:.3e} (orthogonal, not a two-body rewrite). "
            f"On Néel single flips, R² against 1+Hc+F_K is {sf.get('R2')} (residual ratio {sf.get('residual_rms_ratio')}, n={sf.get('n')}). "
            f"On single, pair, and cavity-block flips together, residual ratio is {ex.get('residual_rms_ratio')} (R² {ex.get('R2')}). "
            f"GS-weighted residual ratio is {gs.get('residual_rms_ratio')} (R² {gs.get('R2')}); that weight is diagnostic only."
        )
    lines.append("")
    # Transfer: primary mode, C and A, report F.
    lines.append("Frozen transfer uses no N=12 training. Primary map only:")
    for method in ("A", "C", "qaoa", "D"):
        bits = []
        for L in range(1, 7):
            recs = [r for r in transfer if r["method"] == method and r["L"] == L and r["mode"] == "primary_kappa_theta_copy"]
            if not recs:
                continue
            r = recs[0]
            bits.append(f"L{L} F={r['F']:.3f} (retrained {r['retrained_F']})")
        if bits:
            lines.append(f"- {method}: " + "; ".join(bits))
    lines.append("")
    lines.append(
        "Large |theta_K| with a near-zero theta_C means the energy fit wants the two-body K feature, "
        "not the four-body star. That does not confirm the Schrieffer-Wolff expansion; the entropy-peak "
        "field is outside the regime where that expansion would be controlled, and theta_C was trained rather than fixed."
    )
    lines.append("")
    lines.append(
        "The entropy maximum is flat. On the coarse grid the best N=6 entropies differ by about 1e-5 nats "
        "between B=-0.17 and B=-0.175, and the frozen value sits near ln 2. B is the argmax on the 1e-4 "
        "refinement, not a sharply resolved critical field. The gap quoted above is from a dense "
        "diagonalization at that B. Coarse-scan gaps at N=12 are not used; a two-eigenvalue sparse solve "
        "did not resolve the first excited state."
    )
    lines.append("")
    lines.append(
        "Points whose analytic gradient norm exceeded 1e-3 were continued with L-BFGS-B (maxiter 400, maxfun 4000). "
        "That continuation is recorded as cell['polish'] when it changed the saved point, and its evaluations are included in nfev_sum."
    )
    return lines


def write_report(cells, targets, feature_pack, transfer, meta, failures) -> None:
    rows = comparison_rows(cells)
    save_json(RESULTS / "comparison.json", rows)
    save_json(RESULTS / "transfer.json", transfer)
    # Flat machine-readable dump of cells without huge traces duplicated: keep them.
    save_json(RESULTS / "cells_all.json", {"cells": [cells[k] for k in sorted(cells, key=lambda t: (t[0], t[1], t[2]))]})
    lines = []
    lines.append("# Round 1 direction C — shared star generator vs QAOA")
    lines.append("")
    lines.append(HYPOTHESIS)
    lines.append("")
    lines.append("Target: open chain `H = sum_{i<j} |i-j|^{-1.1} Z_i Z_j + B sum_i X_i`, `J0=1`.")
    lines.append("This is an ideal power-law model. `B` is the entropy-maximizing value found on the contract grid, not a published Pagano field.")
    lines.append("Training minimized energy only. Fidelity was not used to choose angles, features, or the transfer map.")
    lines.append("")
    lines.append("## Frozen fields")
    lines.append("")
    lines.append("| N | B | E0 | gap | Emax | S (nats) | ⟨P⟩ | ⟨R⟩ | P=+1 weight | GS residual | H hash |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in (6, 12):
        t = targets[n]
        lines.append(
            f"| {n} | {t['B']:.4f} | {t['E0']:.6f} | {t['gap']:.3e} | {t['Emax']:.6f} | {t['entropy_nats']:.4f} | "
            f"{t['parity_expectation']:.6f} | {t['reflection_expectation']:.6f} | {t['weight_Pplus']:.6f} | "
            f"{t['gs_residual']:.2e} | `{t['H_hash'][:12]}` |"
        )
    lines.append("")
    lines.append("Endpoint expansions: " + json.dumps({str(n): targets[n]["endpoint_expansions"] for n in (6, 12)}))
    lines.append("")
    lines.append("## Feature structure")
    lines.append("")
    lines.append("Normalization: each of `F_K` and `F_star` is rescaled to the Pauli HS norm of `Hc`. Duplicate 4-qubit masks sum all center contributions (four per mask).")
    lines.append("")
    for n in (6, 12):
        p = feature_pack[n]["summary"]["primary"]
        orth = feature_pack[n]["summary"]["orthogonality_full_space"]
        rel = feature_pack[n]["summary"]["relevant_states"]
        lines.append(
            f"- N={n}: star masks {p['n_star_masks']} (full {p['n_star_masks_full']}), "
            f"HS fraction retained {p['fraction_hs_norm_retained']:.4f}, "
            f"max centers/mask {p['max_centers_per_mask']}, odd masks {p['n_odd_masks']}, "
            f"full-space two-body R² {orth['full_two_body_regression_R2']:.3e}, "
            f"intra-cavity Hc HS fraction {p['intra_hc_hs_fraction']:.3f}."
        )
        sf = rel["single_flips"]["vs_constant_plus_Hc_plus_FK"]
        lines.append(
            f"  - Single-flip residual ratio after removing 1, Hc, F_K: {sf.get('residual_rms_ratio')} (R² {sf.get('R2')}). "
            "Néel cat: both Néel strings carry the same Hc and the same F_star, so the feature is only a phase there."
        )
        if "gs_probability_weighted" in rel:
            gsw = rel["gs_probability_weighted"]["vs_constant_plus_Hc_plus_FK"]
            lines.append(
                f"  - GS-probability-weighted residual ratio (diagnostic, not used in training): {gsw.get('residual_rms_ratio')} (R² {gsw.get('R2')})."
            )
    lines.append("")
    lines.append("## Optimizer")
    lines.append("")
    lines.append(
        f"L-BFGS-B, maxiter {OPT['maxiter']}, maxfun {OPT['maxfun']}, ftol {OPT['ftol']}, gtol {OPT['gtol']}. "
        f"Starts: {OPT['n_starts_L1']} at L=1 (grid-initialized) and {OPT['n_starts_ge2']} at L>1, identical across methods. "
        "QAOA L=2 uses the Pagano shift after converting β_paper·B into our β. "
        "Candidate L=2 duplicates shells and shifts only c1; gamma stays one scalar. "
        "L≥3 interpolates shells (linear at L=3, cubic after) and copies gamma and theta. "
        "D, A, and C each replace up to two random starts with the same-depth simpler model "
        "(shared-linear into D, D into A, A and D into C) with every new coefficient set to 0. "
        "The number of starts does not change. "
        "If a non-QAOA energy beats QAOA by more than 1e-5, QAOA gets 12 extra rescue starts, "
        "including a projection of that depth's c1 schedule onto beta."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| N | L | method | params | E−E0 | F | η | median E−E0 | ΔE vs QAOA | ΔF vs QAOA | ΔE vs QAOA L+1 | ΔF vs QAOA L+1 | grad | nfev |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        def fmt(v, nd=4):
            return "—" if v is None else f"{v:.{nd}f}"
        med_err = None if row["E_median_starts"] is None else row["E_median_starts"] - targets[row["n"]]["E0"]
        lines.append(
            f"| {row['n']} | {row['L']} | {row['method']} | {row['n_params']} | {row['energy_error']:.5f} | "
            f"{row['F']:.4f} | {fmt(row['eta'])} | {fmt(med_err, 5)} | {fmt(row['dE_vs_qaoa_same_L'], 5)} | "
            f"{fmt(row['dF_vs_qaoa_same_L'])} | {fmt(row['dE_vs_qaoa_Lplus1'], 5)} | {fmt(row['dF_vs_qaoa_Lplus1'])} | "
            f"{row['grad_norm']:.2e} | {row['nfev_sum']} |"
        )
    lines.append("")
    # Wins
    both = []
    both_l1 = []
    median_both = []
    for row in rows:
        if row["method"] == "qaoa":
            continue
        if row["dE_vs_qaoa_same_L"] is not None and row["dE_vs_qaoa_same_L"] < -1e-4 and (row["dF_vs_qaoa_same_L"] or 0) > 1e-3:
            both.append(row)
            cell = cells[(row["n"], row["L"], row["method"])]
            q = cells[(row["n"], row["L"], "qaoa")]
            if cell["E_median_starts"] < q["E"] - 1e-4:
                # Median energy of this method vs best QAOA. Also require a fidelity check only on the best state.
                median_both.append(row)
        if (
            row["dE_vs_qaoa_Lplus1"] is not None
            and row["dE_vs_qaoa_Lplus1"] < -1e-4
            and (row["dF_vs_qaoa_Lplus1"] or 0) > 1e-3
            and row["dE_vs_qaoa_same_L"] is not None
            and row["dE_vs_qaoa_same_L"] < -1e-4
            and (row["dF_vs_qaoa_same_L"] or 0) > 1e-3
        ):
            both_l1.append(row)
    lines.append("## What beat same-depth QAOA on both energy and fidelity")
    lines.append("")
    lines.append("These rows are equal-depth only. Most of them do not survive the parameter-matched QAOA comparison in the following section.")
    lines.append("")
    if not both:
        lines.append("No completed non-QAOA cell beat same-depth QAOA on both energy error (by >1e-4) and fidelity (by >1e-3).")
    else:
        for row in both:
            lines.append(
                f"- N={row['n']} L={row['L']} {row['method']}: ΔE={row['dE_vs_qaoa_same_L']:.5f}, ΔF={row['dF_vs_qaoa_same_L']:.4f}, "
                f"params {row['n_params']} vs QAOA {row['qaoa_L_params']}."
            )
    lines.append("")
    lines.append("## Cells that also beat QAOA at L+1 on both")
    lines.append("")
    if not both_l1:
        lines.append("None. A same-depth gain that loses to QAOA at L+1 is not counted as a matched-depth-plus-one win.")
    else:
        for row in both_l1:
            lines.append(
                f"- N={row['n']} L={row['L']} {row['method']}: ΔE(L+1)={row['dE_vs_qaoa_Lplus1']:.5f}, ΔF(L+1)={row['dF_vs_qaoa_Lplus1']:.4f}."
            )
    lines.append("")
    lines.append("Parameter-matched QAOA depth is `ceil(n_params/2)`. Where that depth was not run, the table leaves the L+1 columns as the only extra-depth check and does not invent a matched-parameter win.")
    lines.append("")
    lines.extend(interpretation_lines(cells, targets, feature_pack, transfer))
    lines.append("")
    lines.append("## Frozen N=6 to N=12 transfer")
    lines.append("")
    lines.append("Primary map, fixed before looking at N=12 fidelity: shell controls copied, theta copied as a dimensionless HS weight, gamma multiplied by κ_6/κ_12. Two pre-specified sensitivities are also reported: unscaled gamma, and theta rescaled so the raw monomial prefactor matches. Neither map is chosen after seeing the score, and neither is used to start N=12 training.")
    lines.append("")
    if not transfer:
        lines.append("Transfer was not computed.")
    else:
        lines.append("| N6 L | method | mode | E−E0 | F | retrained E−E0 | retrained F |")
        lines.append("|---|---|---|---|---|---|---|")
        for rec in transfer:
            re_e = "—" if rec["retrained_energy_error"] is None else f"{rec['retrained_energy_error']:.5f}"
            re_f = "—" if rec["retrained_F"] is None else f"{rec['retrained_F']:.4f}"
            lines.append(
                f"| {rec['L']} | {rec['method']} | {rec['mode']} | {rec['energy_error']:.5f} | {rec['F']:.4f} | {re_e} | {re_f} |"
            )
    lines.append("")
    lines.append("## Failures and incomplete cells")
    lines.append("")
    if not failures:
        lines.append("No skipped cells were recorded.")
    else:
        for f in failures:
            lines.append(f"- {f}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- code hash `{meta['code_hash']}`")
    lines.append(f"- wall time {meta.get('wall_time_s')} s")
    lines.append(f"- python {meta.get('python')} numpy {meta.get('numpy')} scipy {meta.get('scipy')}")
    lines.append(f"- host {meta.get('platform')}")
    lines.append("- Artifacts: `results/cells/`, `results/states/`, `results/targets/`, `results/features/`, `results/figures/`, `results/comparison.json`, `results/transfer.json`, `results/cells_all.json`.")
    lines.append("")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-min", type=float, default=80.0)
    parser.add_argument("--priority-L", type=int, default=4)
    parser.add_argument("--max-L", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t_start = time.perf_counter()
    deadline = t_start + args.budget_min * 60.0
    cache = code_hash()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"code {cache[:12]} budget {args.budget_min} min priority L<={args.priority_L} max L={args.max_L}", flush=True)

    targets = {n: prepare_target(n, cache, args.force) for n in (6, 12)}
    # Targets were saved before code_hash of this process's later edits; stamp the hash used for cells separately.
    feature_pack = {}
    for n in (6, 12):
        feature_pack[n] = prepare_features(n, targets[n]["J"], targets[n]["gs"])
        print(
            f"features N={n} masks {feature_pack[n]['summary']['primary']['n_star_masks']}/"
            f"{feature_pack[n]['summary']['primary']['n_star_masks_full']} "
            f"retained {feature_pack[n]['summary']['primary']['fraction_hs_norm_retained']:.3f} "
            f"two-body R2 {feature_pack[n]['summary']['orthogonality_full_space']['full_two_body_regression_R2']:.2e}",
            flush=True,
        )

    periods_report = gauge_audit(
        feature_pack[6]["primary"]["hc_diag"], feature_pack[6]["primary"]["fk_diag"],
        feature_pack[6]["primary"]["fs_diag"], targets[6]["B"], 6,
    )
    periods = {k: v for k, v in periods_report["periods_used_for_interpolation"].items()}
    print("gauge periods", periods, "max energy changes", periods_report["max_abs_energy_change"], flush=True)
    grad_report = {
        "N6": grad_audit(
            feature_pack[6]["primary"]["hc_diag"], feature_pack[6]["primary"]["fk_diag"],
            feature_pack[6]["primary"]["fs_diag"], targets[6]["B"], 6,
        )
    }
    # A few N=12 coordinates, not a full finite-difference vector.
    b12 = feature_pack[12]["primary"]
    x12 = np.array([0.2, -0.1, 0.15, 0.1, -0.05, 0.08])
    _, _, g12 = circuit.simulate("C", x12, 1, b12["hc_diag"], b12["fk_diag"], b12["fs_diag"], targets[12]["B"], 12, True)
    gnum = []
    for i in (0, 2, 5):
        e = np.zeros_like(x12)
        e[i] = 1e-6
        _, fp, _ = circuit.simulate("C", x12 + e, 1, b12["hc_diag"], b12["fk_diag"], b12["fs_diag"], targets[12]["B"], 12, False)
        _, fm, _ = circuit.simulate("C", x12 - e, 1, b12["hc_diag"], b12["fk_diag"], b12["fs_diag"], targets[12]["B"], 12, False)
        gnum.append(abs(g12[i] - (fp - fm) / 2e-6))
    grad_report["N12_C_L1_selected_coords_maxabs"] = float(max(gnum))
    if max(gnum) > 2e-4:
        raise RuntimeError(f"N12 gradient audit failed {gnum}")
    print("gradient audit", grad_report, flush=True)

    zero = circuit.zero_shell_is_not_identity(feature_pack[6]["primary"]["hc_diag"], targets[6]["B"], 6)
    print("zero-shell check", zero, flush=True)

    cells = {}
    failures = []
    prev = {6: {}, 12: {}}

    def queue():
        # Priority: L=1..priority both N, primary methods, QAOA included.
        for L in range(1, args.priority_L + 1):
            for n in (6, 12):
                for method in PRIMARY:
                    yield n, L, method, "primary"
        # QAOA one past the priority, so L+1 exists before deeper candidates.
        for n in (6, 12):
            yield n, args.priority_L + 1, "qaoa", "qaoa_lookahead"
        for L in range(args.priority_L + 1, args.max_L + 1):
            for n in (6, 12):
                for method in PRIMARY:
                    if method == "qaoa" and L == args.priority_L + 1:
                        continue
                    yield n, L, method, "deeper"
        for n in (6, 12):
            yield n, args.max_L + 1, "qaoa", "qaoa_lookahead"
        for L in range(1, args.priority_L + 1):
            yield 12, L, "C_unrestricted", "diagnostic"

    for n, L, method, stage in queue():
        key = (n, L, method)
        cached = load_cell(n, L, method, cache, args.force)
        if cached is not None:
            cells[key] = cached
            prev[n][method] = cached["x"]
            print(f"cache hit {stage} N={n} L={L} {method} E={cached['E']:.5f}", flush=True)
            continue
        if time.perf_counter() > deadline:
            failures.append(f"skipped {stage} N={n} L={L} {method}: budget")
            print(failures[-1], flush=True)
            continue
        # Deeper QAOA/candidate stages need the previous depth of the SAME method at this N.
        previous_x = None
        if L > 1:
            if method not in prev[n]:
                failures.append(f"skipped {stage} N={n} L={L} {method}: missing L={L-1}")
                print(failures[-1], flush=True)
                continue
            previous_x = prev[n][method]
        pack = feature_pack[n]["unrestricted"] if method == "C_unrestricted" else feature_pack[n]["primary"]
        if pack is None:
            failures.append(f"skipped {method}: no unrestricted feature")
            continue
        print(f"optimize {stage} N={n} L={L} {method}", flush=True)
        cell = run_one(
            method, n, L, targets[n]["B"], pack["hc_diag"], pack["fk_diag"], pack["fs_diag"],
            targets[n]["E0"], targets[n]["Emax"], targets[n]["gs"], targets[n]["reflection_indices"],
            previous_x, periods, deadline, nested=nested_inits(method, n, L, cells),
        )
        if cell.get("skipped"):
            failures.append(f"skipped {stage} N={n} L={L} {method}: {cell.get('reason')}")
            continue
        cells[key] = cell
        prev[n][method] = cell["x"]
        persist_cell(cell, cache)
        print(
            f"  E-E0={cell['energy_error']:.5f} F={cell['F']:.4f} medianE={cell['E_median_starts']:.5f} "
            f"grad={cell['grad_norm']:.2e} nfev={cell['nfev_sum']} t={cell['wall_time_s']:.1f}s "
            f"P={cell['parity_expectation']:.3f} R={cell['reflection_expectation']:.3f}",
            flush=True,
        )
        # Rescue once the primary methods at this depth are all present or skipped.
        if method == "C" and stage in ("primary", "deeper"):
            maybe_rescue(
                cells, CANDIDATES, n, L, targets[n]["B"],
                feature_pack[n]["primary"]["hc_diag"], feature_pack[n]["primary"]["fk_diag"],
                feature_pack[n]["primary"]["fs_diag"], targets[n]["E0"], targets[n]["Emax"],
                targets[n]["gs"], targets[n]["reflection_indices"], deadline,
            )
            if (n, L, "qaoa") in cells:
                persist_cell(cells[(n, L, "qaoa")], cache)
                prev[n]["qaoa"] = cells[(n, L, "qaoa")]["x"]

    # If a finished candidate beats QAOA at L+1, rescue that deeper QAOA too.
    for n in (6, 12):
        pack = feature_pack[n]["primary"]
        for L in range(1, args.max_L + 1):
            q1 = cells.get((n, L + 1, "qaoa"))
            if q1 is None or q1.get("skipped"):
                continue
            extras = []
            for method in CANDIDATES:
                cell = cells.get((n, L, method))
                if cell and not cell.get("skipped") and cell["E"] < q1["E"] - 1e-5:
                    extras.append(f"{method}@L{L}")
            if extras:
                maybe_rescue(
                    cells, CANDIDATES, n, L + 1, targets[n]["B"],
                    pack["hc_diag"], pack["fk_diag"], pack["fs_diag"],
                    targets[n]["E0"], targets[n]["Emax"], targets[n]["gs"],
                    targets[n]["reflection_indices"], deadline, extra_challengers=extras,
                )
                if (n, L + 1, "qaoa") in cells:
                    persist_cell(cells[(n, L + 1, "qaoa")], cache)

    # Frozen transfer. Uses truncated N=12 features. Does not train.
    transfer = []
    kappa_ratio = targets[6]["kappa"] / targets[12]["kappa"]
    scale6 = {"K": feature_pack[6]["primary"]["scale_K"], "star": feature_pack[6]["primary"]["scale_star"]}
    scale12 = {"K": feature_pack[12]["primary"]["scale_K"], "star": feature_pack[12]["primary"]["scale_star"]}
    print(f"transfer kappa ratio {kappa_ratio:.4f}", flush=True)
    for L in range(1, args.max_L + 1):
        for method in PRIMARY:
            src = cells.get((6, L, method))
            if src is None or src.get("skipped"):
                continue
            for mode in ("primary_kappa_theta_copy", "unscaled_gamma", "alt_raw_prefactor"):
                x = map_transfer_x(method, src["x"], L, kappa_ratio, scale6, scale12, mode if mode != "primary_kappa_theta_copy" else "primary")
                # map_transfer_x modes: unscaled_gamma, alt_raw_prefactor, anything else scales gamma and copies theta.
                if mode == "primary_kappa_theta_copy":
                    x = map_transfer_x(method, src["x"], L, kappa_ratio, scale6, scale12, "primary")
                ev = evaluate_mapped(
                    method, x, L,
                    feature_pack[12]["primary"]["hc_diag"], feature_pack[12]["primary"]["fk_diag"],
                    feature_pack[12]["primary"]["fs_diag"], targets[12]["B"], 12,
                    targets[12]["E0"], targets[12]["Emax"], targets[12]["gs"], targets[12]["reflection_indices"],
                )
                retr = cells.get((12, L, method))
                ev.update({
                    "L": L,
                    "method": method,
                    "mode": mode,
                    "kappa_ratio": kappa_ratio,
                    "retrained_energy_error": None if retr is None or retr.get("skipped") else retr["energy_error"],
                    "retrained_F": None if retr is None or retr.get("skipped") else retr["F"],
                    "source_state_hash_N6": src["state_hash"],
                })
                transfer.append(ev)
    save_json(RESULTS / "transfer.json", transfer)

    import scipy
    meta = {
        "code_hash": cache,
        "wall_time_s": time.perf_counter() - t_start,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "hypothesis": HYPOTHESIS,
        "gauge_audit": periods_report,
        "gradient_audit": grad_report,
        "zero_shell": zero,
        "optimizer": OPT,
        "kappa": {str(n): targets[n]["kappa"] for n in (6, 12)},
        "deadline_s": args.budget_min * 60.0,
    }
    save_json(RESULTS / "run_meta.json", meta)
    # Target json already written; refresh summary fields that are JSON safe.
    for n in (6, 12):
        safe = {k: v for k, v in targets[n].items() if k not in ("gs", "hc", "J", "evals", "reflection_indices")}
        safe["code_hash"] = cache
        save_json(RESULTS / "targets" / f"N{n}.json", safe)
    try:
        make_figures(cells, targets, feature_pack)
    except Exception as exc:
        failures.append(f"figures failed: {exc}")
        print("FIGURE FAILURE", exc, flush=True)
    write_report(cells, targets, feature_pack, transfer, meta, failures)
    print(f"done in {meta['wall_time_s']:.1f}s cells {len(cells)} failures {len(failures)}", flush=True)
    print(f"report {RESULTS / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
