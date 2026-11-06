"""Round-3 matched-Gamma ladders for the staggered Baxter–Wu strips.

Phases:
    python3 run_round3.py --phase export
    python3 run_round3.py --phase qaoa

Export writes canonical geometry hashes, periods, and full-space ground
states before the ladders. QAOA does not overwrite round-2 files.

Schedules are chosen by energy. A drop in fidelity is kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

import geometry as geo
import model as M
import qaoa
from round3_common import HASH_NOTE, MATCHED_GAMMAS, REPLAY_ONLY, STRIP_GEOMS

OUT = THIS / "runs" / "round3"


def canon_payload(lx: int, ly: int) -> dict:
    geom = geo.build_triangles(lx, ly)
    return {
        "H": "-sum_up ZZZ + sum_down ZZZ - Gamma * sum X",
        "bit": "index x+Lx*y is bit q, LSB = qubit 0",
        "down": [list(t) for t in geom["down"]],
        "down_corners": "(x,y),(x,y+1),(x+1,y+1) mod periods",
        "lambda": 1.0,
        "lx": lx,
        "ly": ly,
        "n": lx * ly,
        "normalization": "triangle coefficients exactly -1 (up) and +1 (down); not divided by N",
        "up": [list(t) for t in geom["up"]],
        "up_corners": "(x,y),(x+1,y),(x+1,y+1) mod periods",
    }


def sha256_json(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sv_hashes(psi) -> dict:
    a = np.asarray(psi, dtype=np.complex128)
    real = np.ascontiguousarray(a.real.astype(np.float64))
    imag = np.ascontiguousarray(a.imag.astype(np.float64))
    exact = hashlib.sha256(real.tobytes() + imag.tobytes()).hexdigest()
    rr = np.ascontiguousarray(np.round(real, 8))
    ii = np.ascontiguousarray(np.round(imag, 8))
    rounded = hashlib.sha256(rr.tobytes() + ii.tobytes()).hexdigest()
    mag = np.abs(a)
    top = np.argsort(-mag)[:8]
    return {
        "sha256_f64_reim": exact,
        "sha256_round8_reim": rounded,
        "norm": float(np.linalg.norm(a)),
        "largest": [{"index": int(i), "re": float(a[i].real), "im": float(a[i].imag)} for i in top],
    }


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def say(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as f:
        f.write(line + "\n")


def period_check(hc: np.ndarray, n: int) -> dict:
    info = M.energy_step_and_period(hc)
    T = info["gamma_period"]
    rng = np.random.default_rng(123 + n)
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    a = qaoa.apply_cost(psi, hc, 0.17)
    full = float(abs(np.vdot(a, qaoa.apply_cost(psi, hc, 0.17 + T))) ** 2)
    half = float(abs(np.vdot(a, qaoa.apply_cost(psi, hc, 0.17 + 0.5 * T))) ** 2)
    ket0 = np.zeros(1 << n, dtype=np.complex128)
    ket0[0] = 1.0
    b0 = qaoa.apply_mixer(ket0, 0.31, n)
    beta_pi = float(abs(np.vdot(b0, qaoa.apply_mixer(ket0, 0.31 + np.pi, n))) ** 2)
    beta_half = float(abs(np.vdot(b0, qaoa.apply_mixer(ket0, 0.31 + 0.5 * np.pi, n))) ** 2)
    return {
        **info,
        "gamma_shift_overlap": full,
        "gamma_half_shift_overlap": half,
        "gamma_period_verified": full > 1.0 - 1e-9 and half < 0.999,
        "beta_period_pi_overlap": beta_pi,
        "beta_half_overlap": beta_half,
        "beta_period_verified": beta_pi > 1.0 - 1e-9 and beta_half < 0.999,
    }


def gs_record(lx, ly, gamma, hc, geom, second_start=False, dense=False):
    n = lx * ly
    spec = M.ground_state(hc, gamma, n)
    extra = None
    if second_start or spec["gap"] < 0.05:
        # Independent start via a sign-flip of a random vector is inside eigsh.
        # Re-run and compare the two Ritz values.
        spec2 = M.ground_state(hc, gamma, n)
        extra = {
            "E0_repeat": spec2["E0"],
            "gap_repeat": spec2["gap"],
            "abs_E0_diff": abs(spec["E0"] - spec2["E0"]),
            "gs_overlap": float(abs(np.dot(spec["gs"], spec2["gs"])) ** 2),
        }
    dense_info = None
    if dense and n <= 12:
        den = M.ground_state_dense(hc, gamma, n)
        dense_info = {
            "E0": den["E0"],
            "gap": den["gap"],
            "residual": den["residual"],
            "overlap_with_lanczos": float(np.dot(den["gs"], spec["gs"]) ** 2),
            "abs_E0_diff": abs(den["E0"] - spec["E0"]),
        }
    cut = M.spatial_cut(lx, ly)
    ent = M.half_cut_entropy(spec["gs"], cut["keep"])
    prod = M.best_product_overlap(spec["gs"], n, seed=lx * 100 + ly + int(gamma * 1000))
    man = M.classical_manifold_overlaps(spec["gs"], hc)
    clauses = M.clause_means(spec["gs"], n, geom)
    mz = float(np.mean(M.z_expectations(spec["gs"], n)))
    hashes = sv_hashes(spec["gs"])
    gap_uncertain = bool(spec["gap"] < max(1e-3, 50.0 * spec["residual"]))
    return {
        "lx": lx,
        "ly": ly,
        "n": n,
        "gamma": float(gamma),
        "E0": spec["E0"],
        "E1": spec["E1"],
        "Emax": spec["Emax"],
        "gap": spec["gap"],
        "gap_uncertain": gap_uncertain,
        "residual": spec["residual"],
        "emax_plus_e0": spec["emax_plus_e0"],
        "n_negative": spec["n_negative"],
        "degenerate": spec["degenerate"],
        "dim": spec["dim"],
        "S_bits": ent["S_bits"],
        "cut_kind": cut["kind"],
        "cut_keep": cut["keep"],
        "F_product_optimized": prod["F_product"],
        "F_product_note": "L-BFGS lower bound on the best real-product overlap, not a certified upper bound",
        "F_plus": prod["F_plus"],
        "F_best_bit": prod["F_best_bit"],
        "F_classical_subspace": man["F_classical_subspace"],
        "F_equal_positive_cat": man["F_equal_positive_cat"],
        "cat_note": "equal positive cat of the classical ground bitstrings; overlap is not an upper bound on every cat",
        "classical_degeneracy": man["degeneracy"],
        "m_z": mz,
        "clause_up": clauses["up"],
        "clause_down": clauses["down"],
        "gs_hash": hashes,
        "repeat": extra,
        "dense": dense_info,
        "gs": spec["gs"],
    }


def export_one(job):
    lx, ly, gammas, dense_gammas = job
    geom = geo.build_triangles(lx, ly)
    payload = canon_payload(lx, ly)
    n = lx * ly
    tris, coeffs = M.faces_and_coeffs(geom, 1.0)
    hc = M.cost_diagonal(n, tris, coeffs)
    per = period_check(hc, n)
    clas = M.classical_summary(hc)
    rows = []
    for g in gammas:
        rec = gs_record(
            lx, ly, g, hc, geom,
            dense=any(abs(float(g) - float(d)) < 1e-8 for d in dense_gammas),
        )
        rec.pop("gs")
        rows.append(rec)
    return {
        "payload": payload,
        "sha256": sha256_json(payload),
        "period": per,
        "classical": clas,
        "gs": rows,
    }


def phase_export():
    say("export canonical geometry, periods, ground states")
    # Gap scan plus the matched fields and the replay field 0.7.
    scan = [round(float(x), 2) for x in np.arange(0.4, 2.01, 0.2)]
    matched = list(MATCHED_GAMMAS) + [0.7]
    jobs = []
    for lx, ly in STRIP_GEOMS:
        gammas = sorted(set(scan + matched))
        dense = {0.8, 1.0} if lx * ly <= 12 else set()
        jobs.append((lx, ly, gammas, dense))
    results = []
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=4) as pool:
        for rec in pool.imap_unordered(export_one, jobs):
            results.append(rec)
            say(f"  exported {rec['payload']['lx']}x{rec['payload']['ly']} hash={rec['sha256'][:12]} period_ok={rec['period']['gamma_period_verified']}")
    results.sort(key=lambda r: (r["payload"]["n"], r["payload"]["lx"]))
    # Gamma* from the scan (not from a single matched field).
    gamma_star = []
    for rec in results:
        scan_rows = [g for g in rec["gs"] if abs(g["gamma"] * 5 - round(g["gamma"] * 5)) < 1e-9 or True]
        # use every computed gamma on the 0.2 grid, which is all except 0.7 if 0.7 isn't on the grid.
        grid = [g for g in rec["gs"] if any(abs(g["gamma"] - s) < 1e-9 for s in scan)]
        if not grid:
            grid = rec["gs"]
        best = min(grid, key=lambda g: g["gap"])
        gamma_star.append({
            "lx": rec["payload"]["lx"],
            "ly": rec["payload"]["ly"],
            "n": rec["payload"]["n"],
            "gamma_star": best["gamma"],
            "gap_star": best["gap"],
            "gap_uncertain": best["gap_uncertain"],
            "definition": "argmin gap on Gamma in {0.4,0.6,...,2.0}; not a rescaling of H",
        })
    bundle = {
        "hash_note": HASH_NOTE,
        "normalization": "same ±1 triangle coefficients at every N; Gamma is not divided by N or by Gamma*",
        "geometries": [{k: v for k, v in rec.items()} for rec in results],
        "gamma_star": gamma_star,
    }
    # Drop nothing; gs vectors already removed.
    json_dump(OUT / "canonical_geometry.json", bundle)
    # Flat GS table for quick reads.
    flat = []
    for rec in results:
        for g in rec["gs"]:
            flat.append({
                "lx": g["lx"], "ly": g["ly"], "n": g["n"], "gamma": g["gamma"],
                "E0": g["E0"], "gap": g["gap"], "residual": g["residual"],
                "gap_uncertain": g["gap_uncertain"],
                "gamma_star": next(s["gamma_star"] for s in gamma_star if s["n"] == g["n"]),
                "gamma_over_gamma_star": g["gamma"] / next(s["gamma_star"] for s in gamma_star if s["n"] == g["n"]),
                "S_bits": g["S_bits"],
                "F_plus": g["F_plus"],
                "F_product_optimized": g["F_product_optimized"],
                "F_classical_subspace": g["F_classical_subspace"],
                "F_equal_positive_cat": g["F_equal_positive_cat"],
                "sha256_geometry": rec["sha256"],
                "gs_sha256_f64": g["gs_hash"]["sha256_f64_reim"],
                "gs_sha256_round8": g["gs_hash"]["sha256_round8_reim"],
                "dense_overlap": None if not g["dense"] else g["dense"]["overlap_with_lanczos"],
                "dense_E0_diff": None if not g["dense"] else g["dense"]["abs_E0_diff"],
            })
    _write_csv(OUT / "gs_energies.csv", flat)
    json_dump(OUT / "periods.json", [{
        "lx": r["payload"]["lx"], "ly": r["payload"]["ly"], "n": r["payload"]["n"],
        "sha256": r["sha256"], **r["period"], "classical_Emin": r["classical"]["Emin"],
        "classical_degeneracy": r["classical"]["degeneracy"],
    } for r in results])
    say(f"export wrote {OUT}")
    return bundle


def _eta(E, E0, Emax):
    den = float(Emax - E0)
    return float((Emax - E) / den) if abs(den) > 1e-15 else float("nan")


def _eval_angles(psi0, hc, n, gamma, gammas, betas, gs):
    psi = qaoa.apply_qaoa(psi0, hc, n, gammas, betas)
    E = qaoa.energy_of(psi, hc, n, gamma)[0]
    F = float(abs(np.vdot(gs, psi)) ** 2)
    return psi, E, F


def optimize_logged(psi0, hc, n, gamma, starts, maxiter, gs, gamma_period, objective="energy"):
    """Return every basin. The selected basin is the lowest energy, even if fidelity is lower."""
    gs = np.asarray(gs, dtype=np.complex128)
    Lbound = len(np.asarray(starts[0][1])) // 2
    bounds = [(-float(gamma_period), 2 * float(gamma_period))] * Lbound + [(-qaoa.BETA_PERIOD, 2 * qaoa.BETA_PERIOD)] * Lbound

    def obj_of(x, which):
        L = len(x) // 2
        psi = qaoa.apply_qaoa(psi0, hc, n, x[:L], x[L:])
        E = qaoa.energy_of(psi, hc, n, gamma)[0]
        F = float(abs(np.vdot(gs, psi)) ** 2)
        if which == "energy":
            return E, E, F, psi
        return -F, E, F, psi

    basins = []
    nfev = 0
    for name, x0 in starts:
        x0 = np.asarray(x0, dtype=np.float64)
        seed_val, seed_E, seed_F, _ = obj_of(x0, objective)
        nfev += 1
        x_opt, val, n_loc = qaoa._lbfgs(lambda x: obj_of(x, objective)[0], x0, maxiter, bounds)
        nfev += n_loc
        opt_val, opt_E, opt_F, _ = obj_of(x_opt, objective)
        nfev += 1
        if seed_val <= opt_val:
            used, E, F, tag = x0, seed_E, seed_F, name + ":seed"
        else:
            used, E, F, tag = x_opt, opt_E, opt_F, name
        L = len(used) // 2
        basins.append({
            "name": tag,
            "energy": float(E),
            "F": float(F),
            "gammas": [float(v) for v in used[:L]],
            "betas": [float(v) for v in used[L:]],
            "nfev_local": int(n_loc),
        })
    basins.sort(key=lambda b: b["energy"] if objective == "energy" else -b["F"])
    best = basins[0]
    # Polish the selected point.
    x_best = np.asarray(best["gammas"] + best["betas"], dtype=np.float64)
    x_pol, _, n_pol = qaoa._lbfgs(lambda x: obj_of(x, objective)[0], x_best, maxiter, bounds)
    nfev += n_pol
    _, pol_E, pol_F, psi = obj_of(x_pol, objective)
    nfev += 1
    polish_dE = float(pol_E - best["energy"])
    moved = abs(polish_dE) > 1e-4 and pol_E < best["energy"] - 1e-6
    take_polish = (objective == "energy" and pol_E < best["energy"]) or (
        objective == "fidelity" and pol_F > best["F"] + 1e-8
    )
    if take_polish:
        L = len(x_pol) // 2
        best = {
            "name": best["name"] + "+polish",
            "energy": float(pol_E),
            "F": float(pol_F),
            "gammas": [float(v) for v in x_pol[:L]],
            "betas": [float(v) for v in x_pol[L:]],
        }
        psi = qaoa.apply_qaoa(psi0, hc, n, best["gammas"], best["betas"])
    else:
        psi = qaoa.apply_qaoa(psi0, hc, n, best["gammas"], best["betas"])
        best["energy"] = qaoa.energy_of(psi, hc, n, gamma)[0]
        best["F"] = float(abs(np.vdot(gs, psi)) ** 2)
    second = basins[1] if len(basins) > 1 else None
    return {
        "best": best,
        "second": second,
        "basin_gap_energy": None if second is None else float(second["energy"] - best["energy"]),
        "n_starts": len(basins),
        "nfev": int(nfev),
        "polish_dE": polish_dE,
        "converged": (not moved),
        "psi": psi,
        "top_basins": [{"name": b["name"], "energy": b["energy"], "F": b["F"]} for b in basins[:6]],
    }


def higher_starts(L, prev_g, prev_b, rng, n_random, gamma_period, warm):
    starts = []
    g_prev = [float(v) for v in prev_g]
    b_prev = [float(v) for v in prev_b]
    if L == 2 and len(g_prev) == 1:
        g1, b1 = g_prev[0], b_prev[0]
        starts.append(("pagano_p2", qaoa._pack([g1, g1 + 0.2], [b1, b1 - 0.2])))
        starts.append(("p2_repeat", qaoa._pack([g1, g1], [b1, b1])))
    if L >= 3 and len(g_prev) == L - 1:
        gi = qaoa.interpolate_schedule(g_prev, L)
        bi = qaoa.interpolate_schedule(b_prev, L)
        starts.append(("pagano_interp", qaoa._pack(gi, bi)))
        starts.append(("interp_pert", qaoa._pack(gi + rng.normal(0, 0.04, L), bi + rng.normal(0, 0.06, L))))
    if L >= 2 and len(g_prev) == L - 1:
        starts.append(("zeropad_append", qaoa._pack(g_prev + [0.0], b_prev + [0.0])))
        starts.append(("zeropad_prepend", qaoa._pack([0.0] + g_prev, [0.0] + b_prev)))
    if isinstance(warm, dict) and len(warm.get("gammas", [])) == L:
        starts.append(("warm_other_gamma", qaoa._pack(warm["gammas"], warm["betas"])))
    for i in range(n_random):
        starts.append((
            f"random_{i}",
            qaoa._pack(rng.uniform(0.0, gamma_period, L), rng.uniform(0.0, qaoa.BETA_PERIOD, L)),
        ))
    return starts


def run_one_ladder(task):
    try:
        return _run_one_ladder(task)
    except Exception:
        return {"ok": False, "error": traceback.format_exc(), "task": {"lx": task.get("lx"), "ly": task.get("ly"), "gamma": task.get("gamma")}}


def _run_one_ladder(task):
    t0 = time.time()
    lx, ly = int(task["lx"]), int(task["ly"])
    gamma = float(task["gamma"])
    n = lx * ly
    geom = geo.build_triangles(lx, ly)
    tris, coeffs = M.faces_and_coeffs(geom, 1.0)
    hc = M.cost_diagonal(n, tris, coeffs)
    per = M.energy_step_and_period(hc)
    gamma_period = float(per["gamma_period"])
    spec = M.ground_state(hc, gamma, n)
    gs = spec["gs"]
    psi0 = M.plus_state(n)
    max_L = int(task["max_L"])
    n_random = int(task["n_random"])
    n_grid = int(task["n_grid"])
    maxiter = int(task["maxiter"])
    seed = int(task["seed"])
    rng = np.random.default_rng(np.random.PCG64(seed))
    warm_by_L = task.get("warm") or {}
    rows = []
    e0 = qaoa.energy_of(psi0, hc, n, gamma)[0]
    f0 = float(abs(np.vdot(gs, psi0)) ** 2)
    rows.append({
        "L": 0, "F": f0, "energy": e0, "eta": _eta(e0, spec["E0"], spec["Emax"]),
        "winner": "plus", "gammas": [], "betas": [], "n_restarts": 0, "nfev": 0,
        "energy_nested": True, "fidelity_monotone": True, "selected_by": "initial_state",
        "sv": sv_hashes(psi0),
    })
    # L=1 strong grid.
    g_axis, b_axis, Es, Fs = qaoa.l1_grid(psi0, hc, n, gamma, gs, gamma_period, n_grid)
    e_coords = qaoa.top_indices(Es, 8, largest=False)
    f_coords = qaoa.top_indices(Fs, 6, largest=True)

    def coord_starts(coords, prefix):
        out = []
        for rank, (ib, ig) in enumerate(coords):
            out.append((f"{prefix}{rank}", qaoa._pack([g_axis[ig]], [b_axis[ib]])))
        return out

    e_starts = coord_starts(e_coords, "gridE_")
    for i in range(n_random):
        e_starts.append((f"random_{i}", qaoa._pack(rng.uniform(0, gamma_period, 1), rng.uniform(0, qaoa.BETA_PERIOD, 1))))
    warm1 = warm_by_L.get("1") if isinstance(warm_by_L, dict) else None
    if isinstance(warm1, dict) and warm1.get("gammas"):
        e_starts.append(("warm_other_gamma", qaoa._pack(warm1["gammas"], warm1["betas"])))
    optE = optimize_logged(psi0, hc, n, gamma, e_starts, maxiter, gs, gamma_period, "energy")
    f_starts = coord_starts(f_coords, "gridF_")
    for i in range(min(8, n_random)):
        f_starts.append((f"frand_{i}", qaoa._pack(rng.uniform(0, gamma_period, 1), rng.uniform(0, qaoa.BETA_PERIOD, 1))))
    optF = optimize_logged(psi0, hc, n, gamma, f_starts, max(30, maxiter // 2), gs, gamma_period, "fidelity")
    prev_g, prev_b = optE["best"]["gammas"], optE["best"]["betas"]
    prev_E, prev_F = e0, f0
    reports = []
    for L in range(1, max_L + 1):
        if L == 1:
            chosen = optE
        else:
            starts = higher_starts(L, prev_g, prev_b, rng, n_random, gamma_period, warm_by_L.get(str(L)))
            chosen = optimize_logged(psi0, hc, n, gamma, starts, maxiter, gs, gamma_period, "energy")
        best = chosen["best"]
        psi = chosen["psi"]
        rep = qaoa.branch_report(best["gammas"], best["betas"], gamma_period)
        psi_p = qaoa.apply_qaoa(psi0, hc, n, rep["gamma_principal"], rep["beta_principal"])
        prin = float(abs(np.vdot(psi, psi_p)) ** 2)
        E = float(best["energy"])
        F = float(best["F"])
        row = {
            "L": L,
            "F": F,
            "energy": E,
            "eta": _eta(E, spec["E0"], spec["Emax"]),
            "r_over_gap": float((E - spec["E0"]) / spec["gap"]) if spec["gap"] > 1e-12 else None,
            "winner": best["name"],
            "second_energy": None if chosen["second"] is None else chosen["second"]["energy"],
            "second_F": None if chosen["second"] is None else chosen["second"]["F"],
            "basin_gap_energy": chosen["basin_gap_energy"],
            "n_restarts": chosen["n_starts"],
            "nfev": chosen["nfev"],
            "polish_dE": chosen["polish_dE"],
            "converged": chosen["converged"],
            "gammas": best["gammas"],
            "betas": best["betas"],
            "energy_nested": bool(E <= prev_E + 1e-6),
            "fidelity_monotone": bool(F + 1e-7 >= prev_F),
            "selected_by": "energy",
            "principal_state_overlap": prin,
            "top_basins": chosen["top_basins"],
            "sv": sv_hashes(psi),
            **rep,
        }
        if L == 1:
            row.update({
                "l1_grid_min_energy": float(Es.min()),
                "l1_grid_max_fidelity": float(Fs.max()),
                "l1_energy_opt_beats_grid": bool(E <= float(Es.min()) + 1e-6),
                "l1_fidelity_max_F": float(optF["best"]["F"]),
                "l1_fidelity_max_energy": float(optF["best"]["energy"]),
                "l1_F_gap_fidopt_minus_energyopt": float(optF["best"]["F"] - F),
                "l1_false_hard_diagnostic": bool(float(Fs.max()) > F + 0.2 and float(Fs.max()) > 0.5),
            })
        # Shared-gamma diagnostic at L=1 and the deepest L, and at L=3 when cheap.
        if L == 1 or L == max_L or (L == 3 and n <= 15):
            shared = _shared(psi0, hc, n, gamma, gs, best["gammas"], best["betas"], rng, gamma_period)
            row["shared_F"] = shared["F"]
            row["shared_energy"] = shared["energy"]
        g_a, b_a = qaoa.anneal_angles(L, gamma, rep["duration_principal"])
        if g_a:
            _, Ea, Fa = _eval_angles(psi0, hc, n, gamma, g_a, b_a, gs)
            row["anneal_F"] = Fa
            row["anneal_energy"] = Ea
        reports.append(row)
        prev_g, prev_b = best["gammas"], best["betas"]
        prev_E, prev_F = E, F
    # Observables of the deepest energy-selected state and of the GS are attached once.
    cut = M.spatial_cut(lx, ly)
    deep_psi = qaoa.apply_qaoa(psi0, hc, n, prev_g, prev_b)
    deep_ent = M.half_cut_entropy(deep_psi, cut["keep"])
    deep_cl = M.clause_means(deep_psi, n, geom)
    return {
        "ok": True,
        "seconds": time.time() - t0,
        "task": {"lx": lx, "ly": ly, "n": n, "gamma": gamma, "role": task.get("role", "matched"), "max_L": max_L, "n_random": n_random},
        "spectrum": {
            "E0": spec["E0"], "E1": spec["E1"], "Emax": spec["Emax"], "gap": spec["gap"],
            "residual": spec["residual"], "emax_plus_e0": spec["emax_plus_e0"],
            "gap_uncertain": bool(spec["gap"] < max(1e-3, 50.0 * spec["residual"])),
            "n_negative": spec["n_negative"], "dim": spec["dim"],
            "gamma_period": gamma_period, "cost_delta": float(M.energy_step_and_period(hc)["delta"]),
        },
        "deep_observables": {
            "S_bits": deep_ent["S_bits"],
            "cut_kind": cut["kind"],
            "clause_up": deep_cl["up"],
            "clause_down": deep_cl["down"],
            "m_z": float(np.mean(M.z_expectations(deep_psi, n))),
        },
        "ladder": reports,
        "angles_for_transfer": {str(r["L"]): {"gammas": r["gammas"], "betas": r["betas"], "F": r["F"]} for r in reports if r["L"] > 0},
    }


def _shared(psi0, hc, n, gamma, gs, gammas, betas, rng, gamma_period):
    L = len(gammas)
    starts = [("mean", np.concatenate([[float(np.mean(gammas))], np.asarray(betas, dtype=np.float64)]))]
    for i in range(3):
        starts.append(("r" + str(i), np.concatenate([
            rng.uniform(0.0, gamma_period, 1),
            rng.uniform(0.0, qaoa.BETA_PERIOD, L),
        ])))
    bounds = [(-float(gamma_period), 2 * float(gamma_period))] + [(-qaoa.BETA_PERIOD, 2 * qaoa.BETA_PERIOD)] * L

    def obj(x):
        psi = qaoa.apply_qaoa(psi0, hc, n, [float(x[0])] * L, x[1:])
        return qaoa.energy_of(psi, hc, n, gamma)[0]

    best_x, best_E = None, np.inf
    for _, x0 in starts:
        e_seed = float(obj(x0))
        x_opt, val, _ = qaoa._lbfgs(obj, x0, 25, bounds)
        if e_seed < val:
            x_opt, val = np.asarray(x0, dtype=np.float64), e_seed
        if val < best_E:
            best_x, best_E = x_opt, val
    psi = qaoa.apply_qaoa(psi0, hc, n, [float(best_x[0])] * L, best_x[1:])
    return {"F": float(abs(np.vdot(gs, psi)) ** 2), "energy": float(best_E)}


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv(r.get(k)) for k in keys})


def _csv(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, bool):
        return int(v)
    return v


def flatten(cell, gamma_star):
    task = cell["task"]
    spec = cell["spectrum"]
    gstar = gamma_star.get((task["lx"], task["ly"]))
    rows = []
    for r in cell["ladder"]:
        flat = {
            "lx": task["lx"], "ly": task["ly"], "n": task["n"], "gamma": task["gamma"],
            "role": task["role"],
            "gamma_star": None if gstar is None else gstar,
            "gamma_over_gamma_star": None if not gstar else task["gamma"] / gstar,
            "L": r["L"], "F": r["F"], "energy": r["energy"], "eta": r["eta"],
            "E0": spec["E0"], "gap": spec["gap"], "residual": spec["residual"],
            "gap_uncertain": spec["gap_uncertain"],
            "r_over_gap": r.get("r_over_gap"),
            "winner": r.get("winner"),
            "second_energy": r.get("second_energy"),
            "second_F": r.get("second_F"),
            "basin_gap_energy": r.get("basin_gap_energy"),
            "n_restarts": r.get("n_restarts"),
            "nfev": r.get("nfev"),
            "polish_dE": r.get("polish_dE"),
            "converged": r.get("converged"),
            "energy_nested": r.get("energy_nested"),
            "fidelity_monotone": r.get("fidelity_monotone"),
            "selected_by": r.get("selected_by"),
            "duration_principal": r.get("duration_principal"),
            "sum_abs_gamma_principal": r.get("sum_abs_gamma_principal"),
            "sum_abs_beta_principal": r.get("sum_abs_beta_principal"),
            "n_gamma_wrapped": r.get("n_gamma_wrapped"),
            "principal_state_overlap": r.get("principal_state_overlap"),
            "gamma_period": r.get("gamma_period", spec["gamma_period"]),
            "sv_sha256_round8": None if "sv" not in r else r["sv"]["sha256_round8_reim"],
            "shared_F": r.get("shared_F"),
            "anneal_F": r.get("anneal_F"),
            "l1_grid_max_fidelity": r.get("l1_grid_max_fidelity"),
            "l1_fidelity_max_F": r.get("l1_fidelity_max_F"),
            "l1_false_hard_diagnostic": r.get("l1_false_hard_diagnostic"),
            "gammas": r.get("gammas"),
            "betas": r.get("betas"),
        }
        rows.append(flat)
    return rows


def transfer_eval(src_cell, lx, ly, gamma):
    """Apply src angles at the same Gamma on another strip. No reoptimization."""
    n = lx * ly
    geom = geo.build_triangles(lx, ly)
    tris, coeffs = M.faces_and_coeffs(geom, 1.0)
    hc = M.cost_diagonal(n, tris, coeffs)
    spec = M.ground_state(hc, gamma, n)
    psi0 = M.plus_state(n)
    rows = []
    for Lstr, ang in src_cell["angles_for_transfer"].items():
        _, E, F = _eval_angles(psi0, hc, n, gamma, ang["gammas"], ang["betas"], spec["gs"])
        rows.append({
            "src_lx": src_cell["task"]["lx"], "src_ly": src_cell["task"]["ly"],
            "dst_lx": lx, "dst_ly": ly, "dst_n": n, "gamma": gamma, "L": int(Lstr),
            "F_transfer": F, "E_transfer": E,
            "F_src": ang["F"],
            "delta_F": F - ang["F"],
            "E0_dst": spec["E0"], "gap_dst": spec["gap"], "residual_dst": spec["residual"],
        })
    return rows


def phase_qaoa():
    bundle = json.loads((OUT / "canonical_geometry.json").read_text())
    gamma_star = {(g["lx"], g["ly"]): g["gamma_star"] for g in bundle["gamma_star"]}
    say("qaoa matched grid")
    # Settings: 20 restarts on N<=15. N=18 uses 8 because a 20-restart ladder was not affordable.
    def settings(n, role):
        if n <= 9:
            return dict(max_L=4, n_grid=31, n_random=20, maxiter=40)
        if n <= 15:
            return dict(max_L=6, n_grid=29, n_random=20, maxiter=36)
        return dict(max_L=5, n_grid=17, n_random=8, maxiter=22)

    cells = []
    jsonl = OUT / "cells.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    def store(cell):
        # Drop bulky psi; hashes already inside.
        pub = {k: v for k, v in cell.items()}
        cells.append(pub)
        with jsonl.open("a") as f:
            f.write(json.dumps(pub) + "\n")
        t = pub["task"]
        l1 = next(r for r in pub["ladder"] if r["L"] == 1)
        top = pub["ladder"][-1]
        say(
            f"  {t['lx']}x{t['ly']} G={t['gamma']} {t['role']} "
            f"L1={l1['F']:.3f} L1max={l1.get('l1_grid_max_fidelity', float('nan')):.3f} "
            f"Ffid={l1.get('l1_fidelity_max_F', float('nan')):.3f} "
            f"L{top['L']}={top['F']:.3f} mono={top['fidelity_monotone']} "
            f"conv={l1.get('converged')} {pub['seconds']:.0f}s"
        )

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    # Gamma waves so the next field can warm-start from the previous strip solution.
    waves = [0.6, 0.8, 1.0, 1.2]
    solved = {}  # (lx, ly, gamma) -> cell
    for gamma in waves:
        tasks = []
        for lx, ly in ((3, 3), (3, 4), (3, 5)):
            warm = {}
            prev_g = {0.8: 0.6, 1.0: 0.8, 1.2: 1.0}.get(gamma)
            if prev_g is not None and (lx, ly, prev_g) in solved:
                warm = solved[(lx, ly, prev_g)]["angles_for_transfer"]
            cfg = settings(lx * ly, "matched")
            tasks.append({
                "lx": lx, "ly": ly, "gamma": gamma, "role": "matched",
                "seed": 300_000 + lx * 1000 + ly * 10 + int(gamma * 100),
                "warm": warm, **cfg,
            })
        say(f"wave Gamma={gamma} tasks={len(tasks)}")
        with ctx.Pool(processes=3) as pool:
            for cell in pool.imap_unordered(run_one_ladder, tasks):
                if not cell.get("ok", True) and "error" in cell:
                    say("FAIL " + cell["error"][-300:])
                    continue
                store(cell)
                solved[(cell["task"]["lx"], cell["task"]["ly"], cell["task"]["gamma"])] = cell
        # Angle transfer from 3×4 onto 3×5 and 3×6 at this Gamma.
        src = solved.get((3, 4, gamma))
        if src:
            for lx, ly in ((3, 5), (3, 6)):
                rows = transfer_eval(src, lx, ly, gamma)
                with (OUT / "angle_transfer.jsonl").open("a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                deep = [r for r in rows if r["L"] == src["task"]["max_L"]]
                if deep:
                    say(f"  transfer 3x4-> {lx}x{ly} G={gamma} L={deep[0]['L']} F={deep[0]['F_transfer']:.3f} dF={deep[0]['delta_F']:.3f}")

    # Replay-only 3×4 @ 0.7, warm from 0.6 and noted separately.
    say("replay 3x4 Gamma=0.7")
    warm = {}
    if (3, 4, 0.6) in solved:
        warm = solved[(3, 4, 0.6)]["angles_for_transfer"]
    cfg = settings(12, "replay")
    cell = run_one_ladder({
        "lx": 3, "ly": 4, "gamma": 0.7, "role": "replay_round2_point",
        "seed": 77_007, "warm": warm, **cfg,
    })
    store(cell)

    # N=18 promising matched rows: 0.8 and 1.0. 0.6 and 1.2 only if the wall clock allows.
    say("N=18 promising rows Gamma=0.8 and 1.0")
    n18_tasks = []
    for gamma in (0.8, 1.0):
        n18_tasks.append({
            "lx": 3, "ly": 6, "gamma": gamma,
            "role": "matched" if gamma != 1.0 else "matched_and_round2_point",
            "seed": 180_000 + int(gamma * 100),
            "warm": {},
            **settings(18, "matched"),
        })
    with ctx.Pool(processes=2) as pool:
        for cell in pool.imap_unordered(run_one_ladder, n18_tasks):
            store(cell)
            solved[(3, 6, cell["task"]["gamma"])] = cell

    # Flat tables.
    flat = []
    for c in cells:
        flat.extend(flatten(c, gamma_star))
    _write_csv(OUT / "qaoa_matched.csv", flat)
    angles = []
    for c in cells:
        for r in c["ladder"]:
            if r["L"] == 0:
                continue
            angles.append({
                "lx": c["task"]["lx"], "ly": c["task"]["ly"], "n": c["task"]["n"],
                "gamma": c["task"]["gamma"], "role": c["task"]["role"], "L": r["L"],
                "gammas": r["gammas"], "betas": r["betas"],
                "gammas_principal": r.get("gamma_principal"),
                "betas_principal": r.get("beta_principal"),
                "gamma_branch_k": r.get("gamma_branch_k"),
                "beta_branch_k": r.get("beta_branch_k"),
                "F": r["F"], "energy": r["energy"], "winner": r.get("winner"),
                "sv_sha256_round8": r["sv"]["sha256_round8_reim"],
                "sv_sha256_f64": r["sv"]["sha256_f64_reim"],
            })
    json_dump(OUT / "angles.json", angles)
    say(f"qaoa done cells={len(cells)}")
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("export", "qaoa", "all"), default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("export", "all"):
        phase_export()
    if args.phase in ("qaoa", "all"):
        phase_qaoa()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("FATAL\n" + traceback.format_exc())
        raise
