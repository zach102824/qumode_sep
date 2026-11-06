"""Round-1 QAOA scan for Family C pure distinct-index p=4.

Primary ansatz: independent (gamma_l, beta_l), cost then mixer, initial |+>,
energy objective. Shared-gamma, L=1 fidelity maximization, and a digitized
linear anneal are diagnostics / baselines and are stored in separate methods.

Gamma list is fixed before looking at QAOA fidelities. Which Gamma is drawn
in the main F-vs-1/N panel is chosen from ground-state properties only
(overlap with |+>, cat overlap, half-cut entropy, gap), by score_gamma().

Usage:
  python run_round1.py
"""
from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

import validate
from model import (
    J,
    a_N,
    canonical_angles,
    cat_overlap,
    best_basis_population,
    classical_optimum,
    eta_score,
    full_spectrum_edges,
    gamma_period,
    half_cut_entropy_dicke,
    l1_duration,
    lower_body_delta,
    moments_from_dicke,
    plus_dicke,
    subspace_fidelity,
    symmetric_dicke_spectrum,
    symmetric_product_overlap,
    DickeModel,
)
from qaoa import (
    ANNEAL_T_GRID,
    EvalCounter,
    evaluate_anneal,
    fidelity_of,
    optimize_fidelity_l1,
    optimize_independent,
    optimize_shared_gamma,
)

SEED = 20261106
# Fixed before any QAOA fidelity is computed. Includes the shallow-AF region
# (Gamma << 1) and the ferro crossover near Gamma ~ 1. Not a fitted critical point.
GAMMAS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.10, 1.25, 1.50, 2.00)
# Odd sizes are spectrum-only (parity bookkeeping). Even sizes get QAOA.
SPECTRUM_NS = (6, 7, 8, 9, 10, 11, 12)
QAOA_NS = (6, 8, 10, 12)
EXTRA_NS = (14, 16, 20)  # symmetric-block QAOA only after the full-H check
DEPTHS = (1, 2, 3, 4, 5)
SIGNS = (("ferro", -1), ("af", +1))
TOL_E = 1e-6


def score_gamma(F0, F_cat, S_bits, F_product, gap) -> float:
    """Higher is a better *finite-size target*. Uses GS properties only.

    Hard rejects (score -1), which the panel logic treats as "do not feature":
      GHZ cat, already-large |+> overlap, near-product paramagnet, unresolved gap.
    """
    if F_cat > 0.85:
        return -1.0
    if F0 > 0.45:
        return -1.0
    if F_product > 0.80 and S_bits < 0.50:
        return -1.0
    if not (gap == gap) or gap < 1e-4:
        return -1.0
    return float((1.0 - F0) * S_bits * min(gap, 1.0))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _dump(path: Path, obj):
    path.write_text(json.dumps(_jsonable(obj), indent=2))


def _fmt(xs) -> str:
    return ";".join(f"{float(v):.8g}" for v in xs)


def analyze_spectrum(n: int, s: int, gamma: float, kind: str) -> dict:
    t0 = time.perf_counter()
    spec = symmetric_dicke_spectrum(n, s, gamma, kind)
    ed_error = ""
    try:
        full = full_spectrum_edges(n, s, gamma, kind, k_bottom=20)
    except Exception as exc:  # noqa: BLE001 — record and keep the Dicke numbers
        full = None
        ed_error = f"{type(exc).__name__}: {exc}"
    even_e = spec["even_evals"]
    even_v = spec["even_vecs"]
    m_even = even_e <= spec["E0_even"] + 1e-8
    gs_even = even_v[:, m_even]
    v0 = even_v[:, 0]
    c0 = plus_dicke(n)
    F0 = subspace_fidelity(c0, gs_even)
    ent = half_cut_entropy_dicke(v0, n)
    prod = symmetric_product_overlap(v0, n)
    mom = moments_from_dicke(v0, n, s, kind)
    clas = classical_optimum(n, s, kind)
    if full is None:
        E0 = E1 = Emax = gap = float("nan")
        mult = -1
        truncated = True
        method = "failed"
        d_even = float("nan")
        sym_ok = False
        even_global = False
    else:
        E0 = full["E0"]
        E1 = full["E1"]
        Emax = full["Emax"]
        gap = full["gap"]
        mult = full["multiplicity"]
        truncated = bool(full["truncated"])
        method = full["method"]
        d_even = float(E0 - spec["E0_even"])
        # Symmetric even block contains a global GS vector when its energy
        # matches the full-space bottom. Odd-sector equality is reported
        # separately and does not make that vector reachable from |+>.
        sym_ok = abs(d_even) <= TOL_E or abs(E0 - spec["E0_odd"]) <= TOL_E
        even_global = abs(d_even) <= TOL_E and spec["E0_even"] <= spec["E0_odd"] + TOL_E
    period = gamma_period(n) if kind == "pure" else {"T_gamma": spec and DickeModel.build(n, s, gamma, kind).T_gamma}
    # moments already computed; T from a cheap call only if needed
    if kind != "pure":
        period = {"T_gamma": DickeModel.build(n, s, gamma, kind).T_gamma}
    row = {
        "model": kind,
        "sign_name": "ferro" if s < 0 else "af",
        "s": int(s),
        "N": int(n),
        "Gamma": float(gamma),
        "a_N": a_N(n) if kind == "pure" else float("nan"),
        "J": J,
        "E0_full": E0,
        "E1_full": E1,
        "gap_full": gap,
        "Emax_full": Emax,
        "mult_full": mult,
        "truncated": truncated,
        "ed_method": method,
        "ed_error": ed_error,
        "E0_even": spec["E0_even"],
        "E1_even": spec["E1_even"],
        "gap_even": float(spec["E1_even"] - spec["E0_even"]),
        "mult_even": int(m_even.sum()),
        "E0_odd": spec["E0_odd"],
        "parity_gap_odd_minus_even": float(spec["E0_odd"] - spec["E0_even"]),
        "delta_E_full_minus_even": d_even,
        "sym_achieves_E0": bool(sym_ok),
        "even_is_global_gs": bool(even_global),
        "F0": F0,
        "F_cat": cat_overlap(v0),
        "F_product": prod["F_product_symmetric"],
        "F_best_basis": best_basis_population(v0, n),
        "S_half_bits": ent["S_bits"],
        "S_half_nats": ent["S_nats"],
        "subsystem": ent["subsystem"],
        "m1": mom["m1"],
        "m2": mom["m2"],
        "m4": mom["m4"],
        "C2": mom["C2"],
        "C4": mom["C4"],
        "Hc": mom["Hc"],
        "sumX": mom["sumX"],
        "S_M_nats": mom["S_M_nats"],
        "M_classical": ",".join(str(m) for m in clas["M_list"]),
        "E_classical": clas["E_classical"],
        "classical_degeneracy": clas["degeneracy"],
        "T_gamma": float(period["T_gamma"]),
        "gs_score": score_gamma(F0, cat_overlap(v0), ent["S_bits"], prod["F_product_symmetric"], gap if gap == gap else spec["E1_even"] - spec["E0_even"]),
        "wall_s": time.perf_counter() - t0,
        "_gs_even": gs_even,
        "_v0": v0,
        "_E0": E0,
        "_Emax": Emax,
    }
    return row


def _obs_fields(model: DickeModel, c: np.ndarray) -> dict:
    o = model.observables(c)
    return {
        "S_half_bits": o["S_half_bits"],
        "F_product": o["F_product"],
        "F_cat": o["F_cat"],
        "m1": o["m1"],
        "m2": o["m2"],
        "m4": o["m4"],
        "C2": o["C2"],
        "C4": o["C4"],
        "Hc": o["Hc"],
        "sumX": o["sumX"],
        "parity": o["parity"],
    }


def _base_row(spec, method, L, E, F_global, F_reach, extra):
    E0 = spec["_E0"]
    Emax = spec["_Emax"]
    row = {
        "model": spec["model"],
        "sign_name": spec["sign_name"],
        "s": spec["s"],
        "N": spec["N"],
        "Gamma": spec["Gamma"],
        "L": int(L),
        "method": method,
        "E": float(E),
        "eta_full": eta_score(float(E), float(E0), float(Emax)) if E0 == E0 else float("nan"),
        "F_global": float(F_global),
        "F_reach": float(F_reach),
        "F_equal": bool(abs(float(F_global) - float(F_reach)) < 1e-8),
        "E0_full": E0,
        "Emax_full": Emax,
        "gap_full": spec["gap_full"],
        "F0_ref": spec["F0"],
        "even_is_global_gs": spec["even_is_global_gs"],
        "T_gamma": spec["T_gamma"],
    }
    row.update(extra)
    return row


def run_qaoa(spec_row: dict) -> tuple[list[dict], dict]:
    n = spec_row["N"]
    s = spec_row["s"]
    gamma = spec_row["Gamma"]
    kind = spec_row["model"]
    model = DickeModel.build(n, s, gamma, kind)
    gs = spec_row["_gs_even"]
    # Global-in-ansatz target: even eigenstates sitting at the full-space E0.
    # When even_is_global_gs, this is the even ground subspace and F_global=F_reach
    # for every ansatz state. When the global GS is outside, the projector is empty
    # and every QAOA fidelity to the global GS is 0.
    if spec_row["even_is_global_gs"]:
        tgt_global = gs
    else:
        tgt_global = np.zeros((n + 1, 0), dtype=np.complex128)
    tgt_reach = gs
    c0 = model.c0
    F0g = fidelity_of(c0, tgt_global) if tgt_global.shape[1] else 0.0
    F0r = fidelity_of(c0, tgt_reach)
    E0_state = model.energy(c0)
    rows = []
    obs0 = _obs_fields(model, c0)
    rows.append(_base_row(spec_row, "init", 0, E0_state, F0g, F0r, {
        "winner_tag": "plus",
        "angle_l1_raw": 0.0,
        "angle_l1_canonical": 0.0,
        "betas_raw": "",
        "gammas_raw": "",
        "betas_canonical": "",
        "gammas_canonical": "",
        "gamma_shared": "",
        "T_anneal": "",
        "n_evals": 0,
        "wall_s": 0.0,
        **obs0,
    }))
    rng = np.random.default_rng(np.random.SeedSequence([
        SEED, int(n), int(s + 2), int(round(gamma * 10000)), 1 if kind == "pure" else 2,
    ]))
    counter = EvalCounter()
    angle_blob = {"N": n, "s": s, "Gamma": gamma, "model": kind, "T_gamma": model.T_gamma, "depths": []}
    prev = None
    t_all = time.perf_counter()
    for L in DEPTHS:
        t1 = time.perf_counter()
        rec = optimize_independent(model, L, tgt_global if tgt_global.shape[1] else tgt_reach, tgt_reach, prev, rng, counter)
        # If global target was empty, optimize_independent scored F_global against
        # the reachable target because we passed tgt_reach as a stand-in.
        # Overwrite with the honest value.
        if tgt_global.shape[1] == 0:
            rec["F_global"] = 0.0
        elif spec_row["even_is_global_gs"]:
            rec["F_global"] = rec["F_reach"]
        b_can = rec["betas_canonical"]
        g_can = rec["gammas_canonical"]
        obs = _obs_fields(model, rec["state"])
        rows.append(_base_row(spec_row, "qaoa_energy", L, rec["E"], rec["F_global"], rec["F_reach"], {
            "winner_tag": rec["winner_tag"],
            "angle_l1_raw": rec["angle_l1_duration"],
            "angle_l1_canonical": l1_duration(b_can, g_can),
            "betas_raw": _fmt(rec["betas_raw"]),
            "gammas_raw": _fmt(rec["gammas_raw"]),
            "betas_canonical": _fmt(b_can),
            "gammas_canonical": _fmt(g_can),
            "gamma_shared": "",
            "T_anneal": "",
            "n_evals": counter.n,
            "wall_s": time.perf_counter() - t1,
            **obs,
        }))
        angle_blob["depths"].append({
            "L": L,
            "method": "qaoa_energy",
            "winner_tag": rec["winner_tag"],
            "E": rec["E"],
            "F_global": rec["F_global"],
            "F_reach": rec["F_reach"],
            "betas_raw": rec["betas_raw"],
            "gammas_raw": rec["gammas_raw"],
            "betas_canonical": b_can,
            "gammas_canonical": g_can,
            "angle_l1_raw": rec["angle_l1_duration"],
            "angle_l1_canonical": l1_duration(b_can, g_can),
            "tried": rec["tried"],
            "grid_minima": None if not rec["grid"] else rec["grid"]["separated_minima"],
        })
        prev = rec

        # shared-gamma diagnostic at every depth
        t2 = time.perf_counter()
        sh = optimize_shared_gamma(model, L, tgt_global if tgt_global.shape[1] else tgt_reach, tgt_reach, rec, rng, counter)
        if tgt_global.shape[1] == 0:
            sh["F_global"] = 0.0
        elif spec_row["even_is_global_gs"]:
            sh["F_global"] = sh["F_reach"]
        bcan, gcan = canonical_angles(sh["betas_raw"], sh["gammas_raw"], model.T_gamma)
        obs = _obs_fields(model, sh["state"])
        rows.append(_base_row(spec_row, "shared_gamma", L, sh["E"], sh["F_global"], sh["F_reach"], {
            "winner_tag": sh["winner_tag"],
            "angle_l1_raw": sh["angle_l1_duration"],
            "angle_l1_canonical": l1_duration(bcan, gcan),
            "betas_raw": _fmt(sh["betas_raw"]),
            "gammas_raw": _fmt(sh["gammas_raw"]),
            "betas_canonical": _fmt(bcan),
            "gammas_canonical": _fmt(gcan),
            "gamma_shared": sh["gamma_shared"],
            "T_anneal": "",
            "n_evals": counter.n,
            "wall_s": time.perf_counter() - t2,
            **obs,
        }))

        ann = evaluate_anneal(
            model, L,
            tgt_global if tgt_global.shape[1] else tgt_reach,
            tgt_reach,
            rec["angle_l1_duration"],
        )
        for key, method in (("best_grid", "anneal_bestT"), ("matched", "anneal_matched")):
            a = ann[key]
            if a is None:
                continue
            if tgt_global.shape[1] == 0:
                a["F_global"] = 0.0
            elif spec_row["even_is_global_gs"]:
                a["F_global"] = a["F_reach"]
            bcan, gcan = canonical_angles(a["betas_raw"], a["gammas_raw"], model.T_gamma)
            # observables need the state; re-evolve (cheap)
            c = model.evolve(a["betas_raw"], a["gammas_raw"])
            obs = _obs_fields(model, c)
            rows.append(_base_row(spec_row, method, L, a["E"], a["F_global"], a["F_reach"], {
                "winner_tag": "midpoint_linear",
                "angle_l1_raw": a["angle_l1_duration"],
                "angle_l1_canonical": l1_duration(bcan, gcan),
                "betas_raw": _fmt(a["betas_raw"]),
                "gammas_raw": _fmt(a["gammas_raw"]),
                "betas_canonical": _fmt(bcan),
                "gammas_canonical": _fmt(gcan),
                "gamma_shared": "",
                "T_anneal": a["T"],
                "n_evals": 0,
                "wall_s": 0.0,
                **obs,
            }))

    # L=1 fidelity maximization diagnostic (separate objective)
    tF = time.perf_counter()
    label = "global_gs" if spec_row["even_is_global_gs"] else "reachable_sector_not_global"
    target = tgt_reach if spec_row["even_is_global_gs"] or tgt_global.shape[1] == 0 else tgt_global
    if target.shape[1] == 0:
        target = tgt_reach
        label = "reachable_sector_not_global"
    fid = optimize_fidelity_l1(model, target, rng, counter, label)
    Fg = fid["F_target"] if spec_row["even_is_global_gs"] else fidelity_of(fid["state"], tgt_global) if tgt_global.shape[1] else 0.0
    Fr = fidelity_of(fid["state"], tgt_reach)
    bcan, gcan = canonical_angles(fid["betas_raw"], fid["gammas_raw"], model.T_gamma)
    obs = _obs_fields(model, fid["state"])
    rows.append(_base_row(spec_row, "qaoa_fidelity_L1", 1, fid["E"], Fg, Fr, {
        "winner_tag": fid["winner_tag"] + "|" + label,
        "angle_l1_raw": fid["angle_l1_duration"],
        "angle_l1_canonical": l1_duration(bcan, gcan),
        "betas_raw": _fmt(fid["betas_raw"]),
        "gammas_raw": _fmt(fid["gammas_raw"]),
        "betas_canonical": _fmt(bcan),
        "gammas_canonical": _fmt(gcan),
        "gamma_shared": "",
        "T_anneal": "",
        "n_evals": counter.n,
        "wall_s": time.perf_counter() - tF,
        **obs,
    }))
    angle_blob["fidelity_L1"] = {
        "target": label,
        "F_target": fid["F_target"],
        "E": fid["E"],
        "betas_raw": fid["betas_raw"],
        "gammas_raw": fid["gammas_raw"],
        "betas_canonical": bcan.tolist(),
        "gammas_canonical": gcan.tolist(),
    }
    angle_blob["wall_s"] = time.perf_counter() - t_all
    angle_blob["n_evals"] = counter.n
    return rows, angle_blob


SPEC_FIELDS = [
    "model", "sign_name", "s", "N", "Gamma", "a_N", "J",
    "E0_full", "E1_full", "gap_full", "Emax_full", "mult_full", "truncated", "ed_method", "ed_error",
    "E0_even", "E1_even", "gap_even", "mult_even", "E0_odd", "parity_gap_odd_minus_even",
    "delta_E_full_minus_even", "sym_achieves_E0", "even_is_global_gs",
    "F0", "F_cat", "F_product", "F_best_basis", "S_half_bits", "S_half_nats", "subsystem",
    "m1", "m2", "m4", "C2", "C4", "Hc", "sumX", "S_M_nats",
    "M_classical", "E_classical", "classical_degeneracy", "T_gamma", "gs_score", "wall_s",
]
QAOA_FIELDS = [
    "model", "sign_name", "s", "N", "Gamma", "L", "method",
    "E", "eta_full", "F_global", "F_reach", "F_equal",
    "E0_full", "Emax_full", "gap_full", "F0_ref", "even_is_global_gs", "T_gamma",
    "winner_tag", "angle_l1_raw", "angle_l1_canonical",
    "betas_raw", "gammas_raw", "betas_canonical", "gammas_canonical",
    "gamma_shared", "T_anneal", "n_evals", "wall_s",
    "S_half_bits", "F_product", "F_cat", "m1", "m2", "m4", "C2", "C4", "Hc", "sumX", "parity",
]


def _write_csv(path: Path, fields, rows):
    import csv
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, float):
                    out[k] = f"{v:.10g}"
                elif isinstance(v, (bool, np.bool_)):
                    out[k] = "1" if v else "0"
                else:
                    out[k] = v
            w.writerow(out)


def lower_body_table(ns) -> list[dict]:
    rows = []
    for n in ns:
        for M in range(-n, n + 1, 2):
            d = float(lower_body_delta(np.array(float(M)), n))
            rows.append({
                "N": n,
                "M": M,
                "m": M / n,
                "pure_cost_s+1": float(__import__("model").pure_cost_from_M(np.array(float(M)), n, 1)),
                "ctrl_Nm4": float(M**4) / float(n**3),
                "delta_pure_minus_ctrl": d,
            })
    return rows


def select_gamma(spec_rows, sign_name: str) -> dict:
    """One Gamma per sign, from GS scores averaged over the even QAOA sizes."""
    by_g = {}
    for r in spec_rows:
        if r["model"] != "pure" or r["sign_name"] != sign_name or r["N"] not in QAOA_NS:
            continue
        by_g.setdefault(r["Gamma"], []).append(r)
    ranked = []
    for g, rows in by_g.items():
        scores = [float(r["gs_score"]) for r in rows]
        mean = float(np.mean(scores))
        n_pass = int(sum(s > 0 for s in scores))
        ranked.append({"Gamma": g, "mean_score": mean, "n_pass": n_pass, "scores": scores})
    ranked.sort(key=lambda d: (d["n_pass"], d["mean_score"]), reverse=True)
    featured = ranked[0] if ranked else None
    passed = featured is not None and featured["n_pass"] >= 1 and featured["mean_score"] > 0
    return {"sign": sign_name, "featured": featured, "passed_filter": bool(passed), "ranked": ranked}


def main():
    t_run = time.perf_counter()
    out = Path(__file__).resolve().parent / "results" / "round1"
    out.mkdir(parents=True, exist_ok=True)
    print("validation...", flush=True)
    # validation writes its own folder; call the tests inline via the module main
    import sys
    sys.argv = ["validate.py", "--outdir", str(out.parent / "validation")]
    validate.main()

    spec_rows = []
    print("spectrum", flush=True)
    for kind in ("pure", "collective"):
        for n in list(SPECTRUM_NS) + [14, 16]:
            for sign_name, s in SIGNS:
                for g in GAMMAS:
                    row = analyze_spectrum(n, s, g, kind)
                    # N=14,16 only need the sector check + GS diagnostics; still stored
                    spec_rows.append(row)
                    flag = "EVEN=GS" if row["even_is_global_gs"] else "EVEN!=GS"
                    print(
                        f"  {kind:11s} {sign_name:5s} N={n:2d} G={g:.2f} "
                        f"E0={row['E0_full']:.4f} F0={row['F0']:.3f} cat={row['F_cat']:.3f} "
                        f"S={row['S_half_bits']:.2f} gap={row['gap_full']:.3e} {flag} "
                        f"dE={row['delta_E_full_minus_even']:+.1e} {row['wall_s']:.2f}s",
                        flush=True,
                    )
    # N=20 full-space check only at three Gammas per sign (budget), pure model.
    for sign_name, s in SIGNS:
        for g in (0.05, 1.00, 1.10):
            try:
                row = analyze_spectrum(20, s, g, "pure")
            except Exception:
                traceback.print_exc()
                continue
            spec_rows.append(row)
            print(
                f"  pure        {sign_name:5s} N=20 G={g:.2f} E0={row['E0_full']:.4f} "
                f"even_gs={row['even_is_global_gs']} dE={row['delta_E_full_minus_even']:+.1e} {row['wall_s']:.1f}s",
                flush=True,
            )

    public_spec = [{k: r[k] for k in SPEC_FIELDS} for r in spec_rows]
    _write_csv(out / "spectrum.csv", SPEC_FIELDS, public_spec)
    _dump(out / "lower_body_delta.json", lower_body_table((6, 8, 10, 12)))

    # pure vs collective GS overlap on the even ground vector (N in spectrum list)
    overlaps = []
    index = {(r["model"], r["sign_name"], r["N"], r["Gamma"]): r for r in spec_rows}
    for n in list(SPECTRUM_NS) + [14, 16]:
        for sign_name, _s in SIGNS:
            for g in GAMMAS:
                a = index.get(("pure", sign_name, n, g))
                b = index.get(("collective", sign_name, n, g))
                if a is None or b is None:
                    continue
                ov = subspace_fidelity(a["_v0"], b["_gs_even"])
                overlaps.append({
                    "sign_name": sign_name, "N": n, "Gamma": g,
                    "F_pure_vs_ctrl_gs": ov,
                    "E0_pure": a["E0_full"], "E0_ctrl": b["E0_full"],
                    "F0_pure": a["F0"], "F0_ctrl": b["F0"],
                    "F_cat_pure": a["F_cat"], "F_cat_ctrl": b["F_cat"],
                    "even_pure": a["even_is_global_gs"], "even_ctrl": b["even_is_global_gs"],
                })
    _dump(out / "pure_vs_collective_gs.json", overlaps)

    selections = {name: select_gamma(public_spec, name) for name, _ in SIGNS}
    _dump(out / "gamma_selection.json", selections)
    for name, sel in selections.items():
        print(f"featured {name}: {sel['featured']} passed={sel['passed_filter']}", flush=True)

    # QAOA: all even N<=12, all Gammas, pure. Then extra N only where the
    # full-H check says the even symmetric GS is the global GS, at every Gamma
    # that passed for that sign; if none passed, still run the featured Gamma
    # and label it.
    qaoa_rows = []
    angles = []
    jobs = []
    for sign_name, s in SIGNS:
        for n in QAOA_NS:
            for g in GAMMAS:
                jobs.append((n, s, g, "pure"))
        sel = selections[sign_name]
        gammas_extra = []
        if sel["passed_filter"]:
            gammas_extra.append(sel["featured"]["Gamma"])
        # always include the top-ranked Gamma so the larger-N curve exists,
        # plus neighbors 0.05 (AF-like) and 1.10 (ferro crossover) if the
        # sector check passes at that point.
        for g in (sel["featured"]["Gamma"], 0.05, 1.10):
            if g not in gammas_extra:
                gammas_extra.append(g)
        for n in EXTRA_NS:
            for g in gammas_extra:
                key = ("pure", sign_name, n, g)
                src = index.get(key)
                if src is None:
                    # N=20 was only checked at 0.05, 1.00, 1.10. Run the check now if missing.
                    print(f"  full-H check N={n} {sign_name} G={g}", flush=True)
                    try:
                        src = analyze_spectrum(n, s, g, "pure")
                    except Exception:
                        traceback.print_exc()
                        continue
                    spec_rows.append(src)
                    index[key] = src
                if not src["even_is_global_gs"]:
                    print(f"  SKIP QAOA N={n} {sign_name} G={g}: even sector is not the global GS", flush=True)
                    continue
                jobs.append((n, s, g, "pure"))

    # Control QAOA is intentionally narrow: N=8, both signs, three Gammas, L via the
    # same routine but the rows are tagged model=collective so plots can drop them.
    for sign_name, s in SIGNS:
        for g in (0.05, 1.00, 1.10):
            key = ("collective", sign_name, 8, g)
            if key not in index:
                index[key] = analyze_spectrum(8, s, g, "collective")
                spec_rows.append(index[key])
            if index[key]["even_is_global_gs"]:
                jobs.append((8, s, g, "collective"))

    print(f"QAOA jobs: {len(jobs)}", flush=True)
    for i, (n, s, g, kind) in enumerate(jobs, start=1):
        spec = index[(kind, "ferro" if s < 0 else "af", n, g)]
        print(f"[{i}/{len(jobs)}] QAOA {kind} {spec['sign_name']} N={n} G={g}", flush=True)
        rows, blob = run_qaoa(spec)
        qaoa_rows.extend(rows)
        angles.append(blob)
        # checkpoint
        if i % 4 == 0:
            _write_csv(out / "qaoa.csv", QAOA_FIELDS, qaoa_rows)
    _write_csv(out / "qaoa.csv", QAOA_FIELDS, qaoa_rows)
    _dump(out / "angles.json", angles)
    # refresh spectrum csv in case N=20 rows were added late
    public_spec = [{k: r[k] for k in SPEC_FIELDS} for r in spec_rows]
    _write_csv(out / "spectrum.csv", SPEC_FIELDS, public_spec)

    meta = {
        "seed": SEED,
        "J": J,
        "a_N": "N/binom(N,4) = 24/((N-1)(N-2)(N-3))",
        "H": "H = s*J*a_N*sum_{i<j<k<l} Zi Zj Zk Zl - Gamma*sum_i X_i",
        "s": {"ferro": -1, "af": +1},
        "control": "H_C^ctrl = s*J*N*m_z^4, same s, not used as the primary QAOA family",
        "gammas": list(GAMMAS),
        "qaoa_Ns": list(QAOA_NS),
        "circuit": "layer k=1..L acts first-to-last; within layer exp(-i gamma H_C) then exp(-i beta sum X); init |+>",
        "objective": "energy of full H; fidelity is a separate L=1 diagnostic",
        "eta": "(E - Emax_full)/(E0_full - Emax_full)",
        "beta_period": "pi (projective); operator equality up to global phase (-1)^N",
        "anneal_T_grid": list(ANNEAL_T_GRID),
        "selections": selections,
        "wall_s": time.perf_counter() - t_run,
        "versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    _dump(out / "run_meta.json", meta)
    print(f"done in {meta['wall_s']:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
