"""Correctness follow-up for the N9-FT pilot. Does not modify original cell files.

Classifies restricted-displacement norm aborts, checks the query ledger, and
records exact-state finite differences around the cold Nov9 initialization.
Finite differences are diagnostics. They are not an optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Nov9_SK_SPSA.experiments.FT.params import (
    METHOD_ID,
    TRAIN_SEEDS,
    initial_x,
    make_rng,
    spec_for,
    unpack,
)
from Nov9_SK_SPSA.experiments.FT.run_ft import (
    RESULT,
    _BatchCursor,
    claim_block,
    evaluate,
    load,
)
from Nov9_SK_SPSA.experiments.FT.sim import (
    FockTables,
    displacement_matrix,
    generator,
    get_instance,
    run_nov9,
)
from Nov9_SK_SPSA.experiments.FT.spsa import spsa_start

AUDIT = RESULT / "audit_followup"


def _conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=_conv, indent=2))


def catalog() -> dict:
    by_family = {"qaoa": 0, "nov3": 0, "nov9b": 0, "nov9": 0}
    rows = []
    ledger_bad = []
    for path in sorted((RESULT / "cells").glob("*.json")) + sorted((RESULT / "train").glob("*.json")):
        doc = load(path)
        fam = doc.get("family") or ("qaoa" if "qaoa" in doc["method"] else "nov9")
        if "nov9b" in doc["method"]:
            fam = "nov9b"
        elif "nov3" in doc["method"]:
            fam = "nov3"
        elif "qaoa" in doc["method"]:
            fam = "qaoa"
        elif "nov9" in doc["method"]:
            fam = "nov9"
        nfail = int(doc.get("failed_starts") or 0)
        by_family[fam] = by_family.get(fam, 0) + nfail
        starts = doc.get("starts") or []
        if starts:
            q = sum(int(s["queries"]) for s in starts)
            if int(doc.get("queries_counted") or -1) != q:
                ledger_bad.append({"path": path.name, "queries_counted": doc.get("queries_counted"), "sum_starts": q})
            energies = [float(s["E"]) for s in starts]
            winner = min(range(len(starts)), key=lambda i: (energies[i], starts[i]["start_id"]))
            if int(doc.get("selected_start")) != winner:
                ledger_bad.append({"path": path.name, "selected": doc.get("selected_start"), "argmin": winner})
            for s in starts:
                if s.get("failed") or s.get("scored_as_x0"):
                    e0 = float(s["init_report_only"]["E"])
                    if abs(float(s["E"]) - e0) > 1e-12:
                        ledger_bad.append({"path": path.name, "start": s["start_id"], "E": s["E"], "E_init": e0})
                    if int(s["queries"]) <= 0:
                        ledger_bad.append({"path": path.name, "start": s["start_id"], "queries": s["queries"]})
        if nfail:
            rows.append({
                "path": str(path.relative_to(RESULT)),
                "kind": doc.get("kind"),
                "method": doc["method"],
                "family": fam,
                "N": doc.get("N"),
                "b": doc.get("b"),
                "L": doc.get("L"),
                "seed": doc.get("seed"),
                "cap": doc.get("cap"),
                "nf": doc.get("nf"),
                "alpha_sigma_scale": doc.get("alpha_sigma_scale"),
                "failed_starts": nfail,
                "selected_start": doc.get("selected_start"),
                "selected_was_failed": bool(starts and starts[int(doc["selected_start"])].get("failed")),
                "queries_counted": doc.get("queries_counted"),
                "starts": [
                    {
                        "start_id": s["start_id"],
                        "failed": bool(s.get("failed")),
                        "stage": s.get("failure_stage"),
                        "queries": s.get("queries"),
                        "E_scored": s.get("E"),
                        "E_init": s.get("init_report_only", {}).get("E"),
                        "which": s.get("which"),
                    }
                    for s in starts
                ],
            })
    return {
        "n_records_with_a_failed_start": len(rows),
        "failed_start_counts_by_family": by_family,
        "ledger_mismatches": ledger_bad,
        "asymmetry": {
            "nov9_restricted_D_failed_starts": by_family.get("nov9", 0),
            "qaoa_failed_starts": by_family.get("qaoa", 0),
            "nov3_failed_starts": by_family.get("nov3", 0),
            "nov9b_unitary_D8_failed_starts": by_family.get("nov9b", 0),
            "note": "QAOA and Nov3rd are exactly unitary on the logical space. Nov9th-B exponentiates the projected 8-level generator, which is unitary on that space. Only restricted true-D can lose norm.",
        },
        "rows": rows,
    }


def _population_ge(tables: FockTables, psi: np.ndarray, level: int) -> float:
    mask = np.any(tables.fs >= int(level), axis=1)
    return float(np.dot(np.abs(psi) ** 2, mask))


def reference_at(N, seed, b, role, spec, x) -> dict:
    """Compare the restricted NF=16 state with larger-cutoff oscillator states. No renormalization."""
    un = unpack(spec, x)
    alphas = [float(a) for a in un["alphas"]]
    out = {"alphas": alphas, "max_abs_alpha": max((abs(a) for a in alphas), default=0.0)}
    # Matrix-level: restricted slice versus a unitary expm on the same box, and versus a deeper oscillator.
    mats = []
    for alpha in sorted(set(np.round(alphas, 8))):
        R = displacement_matrix(float(alpha), 16, "restricted")
        Ubox = generator(16).expm_reference(float(alpha))
        deep = generator(16 + 64).exp(float(alpha))[:16, :16]
        col = np.linalg.norm(R, axis=0)
        mats.append({
            "alpha": float(alpha),
            "restricted_col_norm_min": float(col.min()),
            "restricted_col_deficit_max": float(np.max(1.0 - col)),
            "restricted_vs_unitary_box_max_abs": float(np.max(np.abs(R - Ubox))),
            "restricted_32pad_vs_64pad_max_abs": float(np.max(np.abs(R - deep))),
            "unitary_box_col_deficit_max": float(np.max(1.0 - np.linalg.norm(Ubox, axis=0))),
        })
    out["matrices"] = mats
    states = {}
    for nf in (16, 32, 48):
        if N >= 9 and nf == 48:
            # 48^3 = 110592 is acceptable; 48^2 is trivial.
            pass
        inst = get_instance(N, seed, b, role)
        tables = FockTables(inst, nf)
        met = run_nov9(tables, un, "u0" if "u0" in spec.method else "uz", "restricted", return_state=True)
        psi = met.pop("_psi")
        states[str(nf)] = {
            "norm": float(met["norm"]),
            "norm_deficit": float(1.0 - met["norm"]),
            "p_code": float(met["p_code"]),
            "p_leak_unconditional": float(met["p_leak"]),
            "simulated_leak_8_to_nf": float(met["norm"] - met["p_code"]),
            "E": float(met["E"]),
            "F": float(met["F"]),
            "E_zz": float(met["E_zz"]),
            "p_ge_16": _population_ge(tables, psi, 16) if nf > 16 else None,
            "p_ge_32": _population_ge(tables, psi, 32) if nf > 32 else None,
        }
    s16 = states["16"]
    s48 = states["48"]
    deficit = s16["norm_deficit"]
    tail = s48["p_ge_16"]
    states["match"] = {
        "norm_deficit_16": deficit,
        "population_ge_16_at_48": tail,
        "abs_difference": abs(deficit - tail) if tail is not None else None,
        "relative_difference": abs(deficit - tail) / max(abs(tail), 1e-30) if tail is not None else None,
        "norm_deficit_48": s48["norm_deficit"],
        "population_ge_32_at_48": s48["p_ge_32"],
        "dE_16_vs_48": abs(s16["E"] - s48["E"]),
        "dF_16_vs_48": abs(s16["F"] - s48["F"]),
        "d_pleak_16_vs_48": abs(s16["p_leak_unconditional"] - s48["p_leak_unconditional"]),
    }
    out["states"] = states
    return out


def _capture_objective_factory(spec, nf, role, b, bad: dict, single_seed=None):
    def objective(x, ctx):
        seeds = [single_seed] if single_seed is not None else list(ctx or [])
        if single_seed is not None:
            met = evaluate(bad["N"], single_seed, b, role, spec, x, nf)
            if not met["ok"] and "x" not in bad:
                bad["x"] = np.asarray(x, dtype=np.float64).copy()
                bad["side_metrics"] = met
                bad["seed"] = single_seed
            return met
        mets = [evaluate(bad["N"], int(seed), b, role, spec, x, nf) for seed in seeds]
        if (not all(m["ok"] for m in mets)) and "x" not in bad:
            bad["x"] = np.asarray(x, dtype=np.float64).copy()
            bad["per_seed"] = [
                {"seed": int(seed), "ok": m["ok"], "norm": m["norm"], "p_leak": m["p_leak"], "E": m["E"]}
                for seed, m in zip(seeds, mets)
            ]
        return {
            "E": float(np.mean([m["E"] for m in mets])),
            "F": float(np.mean([m["F"] for m in mets])),
            "p_leak": float(np.mean([m["p_leak"] for m in mets])),
            "parity_code_conditional": float(np.nanmean([m["parity_code_conditional"] for m in mets])),
            "norm": float(np.min([m["norm"] for m in mets])),
            "ok": all(m["ok"] for m in mets),
            "n_queries": len(mets),
            "p_code": float(np.mean([m["p_code"] for m in mets])),
            "E_zz": float(np.mean([m["E_zz"] for m in mets])),
            "E_xpen": float(np.mean([m["E_xpen"] for m in mets])),
        }

    return objective


def replay_failures(cat_rows: list[dict]) -> list[dict]:
    found = []
    for row in cat_rows:
        doc = load(RESULT / row["path"])
        spec = spec_for(doc["method"], int(doc["L"]), float(doc["alpha_sigma_scale"]))
        role = "train" if doc.get("kind") == "train" else "pilot"
        for s in doc.get("starts") or []:
            if not s.get("failed"):
                continue
            bad = {"N": int(doc["N"]), "method": doc["method"], "start_id": s["start_id"]}
            if doc.get("kind") == "train":
                objective = _capture_objective_factory(spec, int(doc["nf"]), "train", float(doc["b"]), bad)
                ctx = _BatchCursor(TRAIN_SEEDS, offset=17 * int(s["start_id"]))
                rng_i = make_rng(TRAIN_SEEDS[0], int(doc["N"]), float(doc["b"]), int(doc["L"]), METHOD_ID[doc["method"]], int(s["start_id"]), 2)
                inst = get_instance(int(doc["N"]), TRAIN_SEEDS[0], float(doc["b"]), "train")
                x0 = initial_x(spec, int(doc["L"]), int(s["start_id"]), rng_i, inst.theta_K_prior)
                rng = make_rng(TRAIN_SEEDS[0], int(doc["N"]), float(doc["b"]), int(doc["L"]), METHOD_ID[doc["method"]], int(s["start_id"]), 0)
                result = spsa_start(objective, x0, spec, rng, int(doc["n_iter_per_start"]), make_context=ctx)
            else:
                objective = _capture_objective_factory(
                    spec, int(doc["nf"]), "pilot", float(doc["b"]), bad, single_seed=int(doc["seed"])
                )
                rng_i = make_rng(int(doc["seed"]), int(doc["N"]), float(doc["b"]), int(doc["L"]), METHOD_ID[doc["method"]], int(s["start_id"]), 2)
                inst = get_instance(int(doc["N"]), int(doc["seed"]), float(doc["b"]), "pilot")
                x0 = initial_x(spec, int(doc["L"]), int(s["start_id"]), rng_i if int(s["start_id"]) else rng_i, inst.theta_K_prior)
                if doc.get("kind") == "finetune":
                    x0 = np.asarray(doc["x_star"], dtype=np.float64)
                rng = make_rng(int(doc["seed"]), int(doc["N"]), float(doc["b"]), int(doc["L"]), METHOD_ID[doc["method"]], int(s["start_id"]), 0)
                result = spsa_start(objective, x0, spec, rng, int(doc["n_iter_per_start"]))
            ref = None
            if "x" in bad:
                # Reference the instance that actually lost norm. Training may fail on one seed of the batch.
                ref_seed = bad.get("seed")
                if ref_seed is None and bad.get("per_seed"):
                    losers = [p for p in bad["per_seed"] if not p["ok"]]
                    ref_seed = losers[0]["seed"] if losers else bad["per_seed"][0]["seed"]
                ref = reference_at(int(doc["N"]), int(ref_seed), float(doc["b"]), role, spec, bad["x"])
            found.append({
                "path": row["path"],
                "method": doc["method"],
                "N": doc["N"],
                "b": doc["b"],
                "L": doc["L"],
                "seed": doc.get("seed"),
                "start_id": s["start_id"],
                "replay_failed": bool(result["failed"]),
                "replay_stage": result["failure_stage"],
                "replay_queries": result["queries"],
                "original_queries": s["queries"],
                "queries_match": int(result["queries"]) == int(s["queries"]),
                "captured": "x" in bad,
                "failing_norm": None if "side_metrics" not in bad and "per_seed" not in bad else (
                    bad.get("side_metrics", {}).get("norm") if "side_metrics" in bad else min(p["norm"] for p in bad["per_seed"])
                ),
                "reference": ref,
            })
            print(
                f"replay {doc['method']} N={doc['N']} b={doc['b']} L={doc['L']} seed={doc.get('seed')} "
                f"start={s['start_id']} q={result['queries']} captured={('x' in bad)}",
                flush=True,
            )
    return found


def finite_differences() -> dict:
    """Exact-state central differences around the deterministic cold init. Not an optimizer."""
    hs = (1e-4, 1e-3, 1e-2, 5e-2)
    blocks = []
    for N in (6, 9):
        for b in (1.0, 1.5):
            for L in (1, 2, 3):
                for seed in (10000, 10004):
                    spec = spec_for("nov9_kmr1_uz", L, 1.0)
                    inst = get_instance(N, seed, b, "pilot")
                    x0 = initial_x(spec, L, 0, None, inst.theta_K_prior)
                    base = evaluate(N, seed, b, "pilot", spec, x0, 16)
                    plus = -float(b) * N
                    coord_rows = []
                    for i, name in enumerate(spec.names):
                        for h in hs:
                            d = np.zeros(spec.n)
                            d[i] = h
                            # physical step: alpha coordinates are shifted in alpha, not in u
                            from Nov9_SK_SPSA.experiments.FT.params import shift_physical
                            Ep = evaluate(N, seed, b, "pilot", spec, shift_physical(x0, d, spec), 16)
                            Em = evaluate(N, seed, b, "pilot", spec, shift_physical(x0, -d, spec), 16)
                            slope = (Ep["E"] - Em["E"]) / (2.0 * h)
                            curv = (Ep["E"] - 2.0 * base["E"] + Em["E"]) / (h * h)
                            coord_rows.append({
                                "coord": name,
                                "h": h,
                                "slope_dE": slope,
                                "curvature": curv,
                                "E_plus": Ep["E"],
                                "E_minus": Em["E"],
                                "dE_plus": Ep["E"] - base["E"],
                                "dE_minus": Em["E"] - base["E"],
                                "dEzz_plus": Ep["E_zz"] - base["E_zz"],
                                "p_leak_plus": Ep["p_leak"],
                                "below_product": bool(min(Ep["E"], Em["E"]) < plus - 1e-6),
                                "below_base": bool(min(Ep["E"], Em["E"]) < base["E"] - 1e-8),
                            })
                    # A few joint small moves, still diagnostic samples, not a search.
                    joints = []
                    rng = np.random.default_rng([N, int(round(10 * b)), L, seed, 7])
                    from Nov9_SK_SPSA.experiments.FT.params import shift_physical
                    for scale, label in ((1e-3, "joint_1e-3"), (1e-2, "joint_1e-2"), (5e-2, "joint_5e-2")):
                        d = rng.normal(size=spec.n)
                        d = scale * d / max(np.max(np.abs(d)), 1e-15)
                        both = []
                        for sign in (1.0, -1.0):
                            met = evaluate(N, seed, b, "pilot", spec, shift_physical(x0, sign * d, spec), 16)
                            both.append(met["E"])
                        joints.append({
                            "label": label,
                            "E": both,
                            "best_minus_base": min(both) - base["E"],
                            "best_minus_product": min(both) - plus,
                        })
                    slopes = [abs(r["slope_dE"]) for r in coord_rows if r["h"] == 1e-3]
                    blocks.append({
                        "N": N,
                        "b": b,
                        "L": L,
                        "seed": seed,
                        "E_init": base["E"],
                        "F_init": base["F"],
                        "E_zz_init": base["E_zz"],
                        "p_leak_init": base["p_leak"],
                        "product_energy": plus,
                        "E0": inst.E0,
                        "gap_product_to_E0": plus - inst.E0,
                        "max_abs_slope_at_1e-3": max(slopes) if slopes else None,
                        "any_coord_step_below_product": any(r["below_product"] for r in coord_rows),
                        "any_coord_step_below_base": any(r["below_base"] for r in coord_rows),
                        "coords_at_1e-3": [r for r in coord_rows if r["h"] == 1e-3],
                        "coords_at_5e-2": [r for r in coord_rows if r["h"] == 5e-2],
                        "joints": joints,
                    })
                    print(f"fd N={N} b={b} L={L} seed={seed} Einit={base['E']:.6f} product={plus:.3f} maxslope={max(slopes):.4e}", flush=True)
    return {"hs": list(hs), "blocks": blocks, "optimizer": "none; central differences of the exact state energy"}


def classify(found: list[dict]) -> dict:
    refs = [r for r in found if r.get("reference")]
    phys = 0
    numerical = 0
    mixed = 0
    for r in refs:
        m = r["reference"]["states"]["match"]
        deficit = abs(m["norm_deficit_16"])
        tail = m["population_ge_16_at_48"]
        deep = abs(m["norm_deficit_48"])
        agree = abs(deficit - tail) <= max(1e-12, 0.05 * max(deficit, tail, 1e-30))
        if agree and deep < 1e-12 and deficit > 5e-10:
            phys += 1
        elif deficit > 5e-10 and tail < deficit * 0.1:
            numerical += 1
        else:
            mixed += 1
    return {
        "n_captured": len(refs),
        "physical_projection_loss": phys,
        "numerical_matrix_error": numerical,
        "other": mixed,
        "threshold": (
            "The 1e-9 gate is applied to the norm of a knowingly contractive restriction. "
            "A physical tail between 1e-9 and the 1e-8 cutoff budget aborts Nov9 only. "
            "QAOA, Nov3rd, and unitary D8 never see that contraction."
        ),
    }


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    cat = catalog()
    dump(cat, AUDIT / "ledger_and_counts.json")
    print("catalog", cat["n_records_with_a_failed_start"], "ledger mismatches", len(cat["ledger_mismatches"]), flush=True)
    print("by family", cat["failed_start_counts_by_family"], flush=True)
    found = replay_failures(cat["rows"])
    dump(found, AUDIT / "abort_references.json")
    verdict = classify(found)
    fd = finite_differences()
    dump(fd, AUDIT / "finite_difference.json")
    doc = {
        **claim_block(),
        "original_pilot_retained": True,
        "original_cells_modified": False,
        "renormalized_states": False,
        "threshold_changed": False,
        "validation_final_touched": False,
        "classification": verdict,
        "ledger_mismatch_count": len(cat["ledger_mismatches"]),
        "asymmetry": cat["asymmetry"],
    }
    dump(doc, AUDIT / "VERDICT.json")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
