"""Round-3 family N on the frozen manifest.

N=12 depths 1..6 are the study. N=6 is a smoke test. Energy is the only
training objective.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.common import canonical_state, log, sha256_state
from Nov_3rd.tfim_shared_gamma.round2_S.frozen import MANIFEST_SHA256

from .circuit import (
    energy_and_grad,
    energy_only,
    eta,
    fidelity,
    fock_occupations,
    layer_rows,
    param_count,
    unpack,
    variance_H,
)
from .features import level_spacings, load_features, low_eigenpairs, sha256_hc
from .optimize import optimize_method

RESULTS = Path(__file__).resolve().parent / "results"
ROUND2_JOBS = Path(__file__).resolve().parents[1] / "round2_S" / "results" / "jobs"
KEY = ("thetaN0", "A", "thetaM0", "full_intra", "wrong_sel")
SECONDARY = ("shuf_jbar", "thetaK0")


def _cap(kind: str) -> int:
    if kind == "main":
        return 5200
    if kind == "qaoa":
        return 5200
    if kind == "smoke":
        return 400
    return 2800


def _fresh(kind: str) -> int:
    if kind == "main":
        return 12
    if kind == "qaoa":
        return 16
    if kind == "smoke":
        return 2
    return 6


def _matched(L: int) -> int:
    return int(np.ceil((3 * L + 4) / 2.0))


def _floor_qaoa(L: int):
    path = ROUND2_JOBS / f"n12_L{L}_qaoa.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("manifest_sha256") != MANIFEST_SHA256:
        return None
    rec["x"] = np.array([float(v) for v in rec["x_exact"]], dtype=np.float64)
    return rec


def _record(feat, out, hc_hash, evals_low=None):
    x = np.asarray(out["x"], dtype=np.float64)
    E2, _g, psi = energy_and_grad(feat, out["method"], x, out["L"])
    if abs(E2 - out["E"]) > 1e-6:
        raise RuntimeError(f"replay {out['method']} {E2} vs {out['E']}")
    if out["method"] not in ("qaoa", "same_G_qaoa"):
        if max(out["gamma"]) - min(out["gamma"]) > 1e-12:
            raise RuntimeError("shared gamma split")
    sys = feat.system
    var = variance_H(sys, psi)
    occ = fock_occupations(psi, sys.n)
    weights = []
    if evals_low is not None:
        ev, vecs = evals_low
        for i in range(vecs.shape[1]):
            weights.append({"index": i, "E": float(ev[i]), "weight": float(abs(np.vdot(vecs[:, i], psi)) ** 2)})
    rec = {
        "method": out["method"],
        "n": sys.n,
        "L": out["L"],
        "role": out["method"],
        "param_count": param_count(out["method"], out["L"]),
        "B": sys.B_string,
        "E0": sys.E0,
        "Emax": sys.Emax,
        "gap_full": sys.spectrum["gap_full"],
        "gap_accessible_PR_plus": sys.spectrum["gap_accessible_PR_plus"],
        "E": float(E2),
        "energy_error": float(E2 - sys.E0),
        "F": float(out["F"]),
        "eta": eta(E2, sys.E0, sys.Emax),
        "variance_H": var,
        "variational_ok": bool(E2 >= sys.E0 - 1e-7),
        "x_exact": [format(float(v), ".17g") for v in x],
        "gamma": out["gamma"],
        "theta": out["theta"],
        "shell": out["shell"],
        "init_label": out["init_label"],
        "n_eval": out["n_eval"],
        "n_starts": out["n_starts"],
        "seconds": out["seconds"],
        "seed_best": out["seed_best"],
        "seed_median": out["seed_median"],
        "seed_iqr": out["seed_iqr"],
        "seed_q25": out["seed_q25"],
        "seed_q75": out["seed_q75"],
        "seed_std": out["seed_std"],
        "trace_best_start": out["trace_best_start"],
        "starts": out["starts"],
        "state_sha256_f64": sha256_state(psi),
        "state_sha256_round10": sha256_state(psi, decimals=10),
        "layers": layer_rows(feat, out["method"], x, out["L"]),
        "neel_occupation": occ,
        "low_state_weights": weights,
        "manifest_sha256": MANIFEST_SHA256,
        "hc_diag_sha256": hc_hash,
        "fidelity_used_for_selection": False,
        "objective": "energy",
        "compile": _compile(sys.n),
    }
    if rec["manifest_sha256"] != MANIFEST_SHA256:
        raise RuntimeError("hash dropped")
    np.save(RESULTS / "states" / f"n{sys.n}_L{out['L']}_{out['method']}.npy", canonical_state(psi))
    return rec


def _compile(n: int) -> dict:
    m = n // 3
    return {
        "packing": "contiguous triples (0,1,2),(3,4,5),...",
        "k_body_max": 2,
        "n_pair_slots": n * (n - 1) // 2,
        "n_cavities": m,
        "n_cavity_pairs": m * (m - 1) // 2,
        "shell": "3 relative Hamming phases per layer, shared across cavities",
        "shared_scalars": ["gamma", "thetaK", "thetaM", "thetaN"],
        "level_resolved_intra": (
            "N and the intra part of Hc are not constant on Hamming weight. "
            "They need an 8-level diagonal on each cavity, not the 4-phase shell. "
            "M and K and Hc_inter are two-body and can share one compiled inter profile "
            "once the shared thetas are fixed. This is not claimed to cost the same as native Hc."
        ),
        "applications_per_layer": {
            "shared_gamma_global_diagonal": 1,
            "level_resolved_cavity_diagonals": m,
            "shared_hamming_shell": 1,
        },
    }


def _save(rec):
    path = RESULTS / "jobs" / f"n{rec['n']}_L{rec['L']}_{rec['method']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")


def _load(n, L, method, hc_hash):
    path = RESULTS / "jobs" / f"n{n}_L{L}_{method}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("manifest_sha256") != MANIFEST_SHA256 or rec.get("hc_diag_sha256") != hc_hash:
        return None
    rec["x"] = np.array([float(v) for v in rec["x_exact"]], dtype=np.float64)
    return rec


def _alias(feat12):
    """Show that a one-gap 2π shift is not an alias of the whole intra spectrum."""
    from Nov_3rd.tfim_shared_gamma.round2_S.circuit import energy_only as e2
    from Nov_3rd.tfim_shared_gamma.round2_S.circuit import unpack as u2

    path = ROUND2_JOBS / "n12_L6_family_S.json"
    info = level_spacings(feat12)
    if not path.exists():
        info["round2_angles"] = None
        return info
    rec = json.loads(path.read_text())
    x = np.array([float(v) for v in rec["x_exact"]], dtype=np.float64)
    gammas, tin, tk, cs = u2("family_S", x, 6)
    gamma = float(gammas[0])
    gap1 = info["gap_neel_vs_other_weight1"]
    gap2 = info["gap_vacuum_vs_other_weight1"]
    # Shift that adds 2π to gamma*theta*gap1. Not applied as a true symmetry.
    shift = (2.0 * np.pi) / (gamma * gap1) if abs(gamma * gap1) > 1e-12 else float("nan")
    x2 = x.copy()
    x2[1] = x2[1] + shift
    sys = feat12.system
    E1, p1 = e2(sys, "family_S", x, 6)
    E2, p2 = e2(sys, "family_S", x2, 6)
    phase_slip = float((gamma * shift * gap2) / (2.0 * np.pi))
    info.update(
        {
            "round2_L6_gamma": gamma,
            "round2_L6_thetaIn": float(tin),
            "round2_L6_E": float(E1),
            "shift_thetaIn_for_neel_gap": float(shift),
            "phase_slip_of_other_gap_in_units_of_2pi": phase_slip,
            "shifted_state_overlap": float(abs(np.vdot(p1, p2)) ** 2),
            "shifted_energy": float(E2),
            "energy_change": float(E2 - E1),
            "charged_forward_evals": 2,
            "conclusion": (
                "The shift that puts one intra spacing back by 2π moves the other "
                "spacing by a non-integer number of turns, and the prepared state changes. "
                "It is not an exact alias of the intra generator."
            ),
        }
    )
    return info


def _grad_check(feat, methods):
    rng = np.random.default_rng(11 + feat.n)
    worst = 0.0
    for method in methods:
        L = 1 if feat.n == 12 else 2
        x = rng.normal(scale=0.2, size=param_count(method, L))
        _E, g, _p = energy_and_grad(feat, method, x, L)
        idxs = range(x.size) if feat.n == 6 else rng.choice(x.size, size=min(4, x.size), replace=False)
        for i in idxs:
            step = np.zeros_like(x)
            step[int(i)] = 1e-6
            Ep, _ = energy_only(feat, method, x + step, L)
            Em, _ = energy_only(feat, method, x - step, L)
            worst = max(worst, abs((Ep - Em) / 2e-6 - g[int(i)]))
    return worst


def run(budget_min: float = 80.0):
    t0 = time.perf_counter()
    deadline = t0 + budget_min * 60.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "jobs").mkdir(exist_ok=True)
    (RESULTS / "states").mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    failures = []
    log("loading frozen systems")
    sys6, feat6 = load_features(6)
    sys12, feat12 = load_features(12)
    hc_hash = sha256_hc(sys12)
    if sha256_hc(sys6) == hc_hash:
        raise RuntimeError("N=6 and N=12 Hc hashes collided")
    log(
        f"N12 full gap {sys12.spectrum['gap_full']:.6e} accessible {sys12.spectrum['gap_accessible_PR_plus']:.6e}"
    )
    alias = _alias(feat12)
    log(
        f"alias slip {alias.get('phase_slip_of_other_gap_in_units_of_2pi')} "
        f"overlap {alias.get('shifted_state_overlap')}"
    )
    for feat, methods in (
        (feat6, ("family_N", "A", "qaoa", "wrong_sel")),
        (feat12, ("family_N", "qaoa", "same_G_qaoa")),
    ):
        err = _grad_check(feat, methods)
        log(f"grad N={feat.n} maxabs {err:.3e}")
        if err > 5e-5:
            raise RuntimeError(f"gradient N={feat.n} {err}")
    low12 = low_eigenpairs(sys12, 8)
    gs_occ = fock_occupations(sys12.gs, 12)

    def remaining():
        return deadline - time.perf_counter()

    mem = {}

    def run_one(feat, L, method, kind, extra=None, seed_tag=1):
        sys = feat.system
        loaded = None if kind == "smoke" else _load(sys.n, L, method, hc_hash if sys.n == 12 else sha256_hc(sys6))
        if loaded is not None:
            log(f"resume N={sys.n} L={L} {method} E={loaded['E']:.6f}")
            mem[(sys.n, L, method)] = loaded
            return loaded
        if remaining() < 30:
            failures.append(f"skipped N={sys.n} L={L} {method}")
            return None
        prev = mem.get((sys.n, L - 1, method))
        prev_x = None if prev is None else prev["x"]
        cap = _cap(kind)
        if method == "qaoa":
            sib = mem.get((sys.n, L, "family_N"))
            if sib is not None:
                cap = max(cap, int(sib["n_eval"]))
        log(f"optimize N={sys.n} L={L} {method} cap={cap} fresh={_fresh(kind)} left={remaining()/60:.1f}min")
        out = optimize_method(
            feat, method, L, prev_x, cap, _fresh(kind), seed_tag, extra_starts=extra, do_screen=True,
        )
        rec = _record(feat, out, hc_hash if sys.n == 12 else sha256_hc(sys), low12 if sys.n == 12 else None)
        rec["x"] = out["x"]
        if kind != "smoke":
            _save({k: v for k, v in rec.items() if k != "x"})
        mem[(sys.n, L, method)] = rec
        log(
            f"  E={rec['E']:.6f} err={rec['energy_error']:.4e} F={rec['F']:.4f} "
            f"med={rec['seed_median']:.4f} iqr={rec['seed_iqr']:.4f} "
            f"theta={ {k: round(v, 4) for k, v in rec['theta'].items() if abs(v) > 0} } "
            f"var={rec['variance_H']:.4f} neel={rec['neel_occupation']['mean_p_neel']:.3f} evals={rec['n_eval']}"
        )
        return rec

    smoke = []
    for method in ("family_N", "A", "thetaN0", "qaoa"):
        rec = run_one(feat6, 1, method, "smoke", seed_tag=2)
        if rec:
            smoke.append({k: v for k, v in rec.items() if k not in ("x", "starts", "trace_best_start", "layers")})

    for L in (1, 2, 3, 4, 5, 6):
        run_one(feat12, L, "family_N", "main", seed_tag=1)
        for method in KEY:
            if remaining() < 40:
                failures.append(f"skipped control N=12 L={L} {method}")
                break
            run_one(feat12, L, method, "ablation", seed_tag=1)
        if L in (2, 4, 6) and remaining() > 180:
            for method in SECONDARY:
                run_one(feat12, L, method, "ablation", seed_tag=1)
        elif L in (2, 4, 6):
            failures.append(f"skipped secondary L={L}")
        floor = _floor_qaoa(L)
        extra = [("round2_floor", floor["x"])] if floor is not None else None
        run_one(feat12, L, "qaoa", "qaoa", extra=extra, seed_tag=4)
        if remaining() > 90:
            run_one(feat12, L, "same_G_qaoa", "ablation", seed_tag=4)
        else:
            failures.append(f"skipped same_G L={L}")

    for L in range(7, 12):
        floor = _floor_qaoa(L)
        extra = [("round2_floor", floor["x"])] if floor is not None else None
        run_one(feat12, L, "qaoa", "qaoa", extra=extra, seed_tag=5)

    # QAOA must receive at least as many evaluations as the main candidate,
    # including initialization. Early convergence does not spend that budget.
    for L in range(1, 12):
        q = mem.get((12, L, "qaoa"))
        srcL = min(L, 6)
        src = mem.get((12, srcL, "family_N"))
        if q is None or src is None or q["n_eval"] >= src["n_eval"] or remaining() < 40:
            if q is not None and src is not None and q["n_eval"] < src["n_eval"]:
                failures.append(f"QAOA L={L} evals {q['n_eval']} remain below candidate {src['n_eval']}")
            continue
        guard = 0
        while q["n_eval"] < src["n_eval"] and guard < 6 and remaining() > 30:
            guard += 1
            need = int(src["n_eval"] - q["n_eval"])
            log(f"top up QAOA L={L} by about {need} evals")
            out = optimize_method(
                feat12, "qaoa", L, None, need + 50, max(12, need // 40), 80 + 10 * L + guard,
                extra_starts=[("kept", q["x"])], do_screen=False,
            )
            out["n_eval"] = int(q["n_eval"] + out["n_eval"])
            out["seconds"] = float(q["seconds"] + out["seconds"])
            out["starts"] = list(q["starts"]) + list(out["starts"])
            energies = np.array([r["E"] for r in out["starts"] if "E" in r], dtype=np.float64)
            stats = {
                "n_eval": out["n_eval"],
                "seconds": out["seconds"],
                "starts": out["starts"],
                "n_starts": int(energies.size),
                "seed_median": float(np.median(energies)),
                "seed_best": float(energies.min()),
                "seed_std": float(energies.std()),
            }
            q25, q75 = np.percentile(energies, [25, 75])
            stats["seed_q25"] = float(q25)
            stats["seed_q75"] = float(q75)
            stats["seed_iqr"] = float(q75 - q25)
            if out["E"] < q["E"] - 1e-10:
                rec = _record(feat12, out, hc_hash, low12)
                rec["x"] = out["x"]
                rec.update(stats)
                rec["eval_topped"] = True
                _save({k: v for k, v in rec.items() if k != "x"})
                mem[(12, L, "qaoa")] = rec
                q = rec
                log(f"  improved to {q['E']:.6f} after {q['n_eval']} evals")
            else:
                q.update(stats)
                q["eval_topped"] = True
                _save({k: v for k, v in q.items() if k != "x"})
                log(f"  kept {q['E']:.6f} after {q['n_eval']} evals")

    def both(a, b):
        return a is not None and b is not None and a["E"] < b["E"] - 1e-8 and a["F"] > b["F"] + 1e-6

    for L in (4, 5, 6):
        cand = mem.get((12, L, "family_N"))
        q = mem.get((12, _matched(L), "qaoa"))
        if not both(cand, q):
            continue
        if q.get("rescue") or q.get("rescue_attempted"):
            continue
        if remaining() < 60:
            failures.append(f"no time to rescue matched QAOA L={_matched(L)}")
            continue
        log(f"rescue matched QAOA p={_matched(L)} and p+1")
        for Lp, tag in ((_matched(L), 90), (_matched(L) + 1, 91)):
            base = mem.get((12, Lp, "qaoa"))
            if base is None or remaining() < 40:
                continue
            extra = [("kept", base["x"])]
            out = optimize_method(
                feat12, "qaoa", Lp, None, max(2000, 3 * base["n_starts"] * 80),
                max(8, 3 * base["n_starts"]), tag, extra_starts=extra, do_screen=False,
            )
            out["n_eval"] = int(base["n_eval"] + out["n_eval"])
            out["seconds"] = float(base["seconds"] + out["seconds"])
            if out["E"] < base["E"] - 1e-10:
                rec = _record(feat12, out, hc_hash, low12)
                rec["x"] = out["x"]
                rec["rescue"] = True
                rec["prerescue_E"] = base["E"]
                _save({k: v for k, v in rec.items() if k != "x"})
                mem[(12, Lp, "qaoa")] = rec
                log(f"  L={Lp} rescue {base['E']:.6f} -> {rec['E']:.6f}")
            else:
                base["rescue_attempted"] = True
                base["rescue_E"] = out["E"]
                base["n_eval"] = out["n_eval"]
                _save({k: v for k, v in base.items() if k != "x"})
                log(f"  L={Lp} rescue did not improve")

    jobs = [{k: v for k, v in rec.items() if k != "x"} for (n, _L, _m), rec in mem.items() if n == 12]
    summary = {
        "manifest_sha256": MANIFEST_SHA256,
        "hc_diag_sha256": hc_hash,
        "identity": feat12.identity,
        "selector_eigenvalues": feat12.eigs,
        "normalization": {
            "sK": feat12.sK,
            "sM": feat12.sM,
            "sN": feat12.sN,
            "s_wrong": feat12.sW,
            "s_intra": feat12.sI,
            "rms_J": feat12.rms_J,
            "rule": "RMS over all i<j slots, zeros included; hat = raw * RMS(J) / RMS(raw)",
        },
        "shuffle_jbar": {k: v for k, v in feat12.shuffle.items() if k != "jbar"},
        "alias": alias,
        "gs_neel_occupation": gs_occ,
        "gap": {
            "N12_full": sys12.spectrum["gap_full"],
            "N12_accessible": sys12.spectrum["gap_accessible_PR_plus"],
            "N6_full": sys6.spectrum["gap_full"],
            "N6_accessible": sys6.spectrum["gap_accessible_PR_plus"],
        },
        "smoke_n6": smoke,
        "jobs": jobs,
        "comparisons": _comparisons(jobs),
        "failures": failures,
        "wall_seconds": time.perf_counter() - t0,
        "budget_min": budget_min,
        "not_nested_in_S": (
            "Family N uses full K, a normalized block-mean, and the Neel selector. "
            "Round-2 S uses inter-only K and a multiplicative weight on the entire raw intra matrix. "
            "Setting the new thetas does not recover S."
        ),
    }
    summary["reading"] = _reading(summary)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(RESULTS / "RESULTS.md", summary)
    _plots(summary, RESULTS / "figures")
    log(f"wrote round3 in {summary['wall_seconds']:.1f}s failures={len(failures)}")
    return summary


def _comparisons(jobs):
    index = {(j["L"], j["method"]): j for j in jobs}
    rows = []
    for L in range(1, 7):
        N = index.get((L, "family_N"))
        Q = index.get((L, "qaoa"))
        if N is None or Q is None:
            continue
        md = _matched(L)
        Q1 = index.get((L + 1, "qaoa"))
        Qm = index.get((md, "qaoa"))

        def pack(other, name):
            if other is None:
                return {}
            iqr = max(N["seed_iqr"], other["seed_iqr"])
            return {
                f"E_{name}": other["E"],
                f"F_{name}": other["F"],
                f"err_{name}": other["energy_error"],
                f"med_{name}": other["seed_median"],
                f"iqr_{name}": other["seed_iqr"],
                f"dE_{name}": N["E"] - other["E"],
                f"dF_{name}": N["F"] - other["F"],
                f"dE_over_iqr_{name}": None if iqr <= 0 else (other["E"] - N["E"]) / iqr,
            }

        def both(other):
            if other is None:
                return None
            return bool(N["E"] < other["E"] - 1e-8 and N["F"] > other["F"] + 1e-6)

        def half_iqr(job):
            if job is None:
                return None
            es = sorted(r["E"] for r in job.get("starts", []) if "E" in r)
            if not es:
                return None
            half = es[: max(1, len(es) // 2)]
            if len(half) < 2:
                return 0.0
            return float(np.percentile(half, 75) - np.percentile(half, 25))

        row = {
            "L": L,
            "iqr_best_half_N": half_iqr(N),
            "iqr_best_half_qaoa": half_iqr(Q),
            "P": N["param_count"],
            "E_N": N["E"],
            "F_N": N["F"],
            "err_N": N["energy_error"],
            "med_N": N["seed_median"],
            "iqr_N": N["seed_iqr"],
            "var_N": N["variance_H"],
            "neel_N": N["neel_occupation"]["mean_p_neel"],
            "theta": N["theta"],
            "evals_N": N["n_eval"],
            "evals_qaoa": Q["n_eval"],
            "matched_depth": md,
            "both_equal": both(Q),
            "both_Lplus1": both(Q1),
            "both_matched": both(Qm),
            **pack(Q, "qaoa"),
            **pack(Q1, "L1"),
            **pack(Qm, "matched"),
            **pack(index.get((L, "A")), "A"),
            **pack(index.get((L, "thetaN0")), "sa"),
            **pack(index.get((L, "thetaM0")), "m0"),
            **pack(index.get((L, "full_intra")), "intra"),
            **pack(index.get((L, "wrong_sel")), "wrong"),
            **pack(index.get((L, "shuf_jbar")), "shuf"),
            **pack(index.get((L, "thetaK0")), "k0"),
            **pack(index.get((L, "same_G_qaoa")), "sameG"),
        }
        rows.append(row)
    return rows


def _reading(summary):
    rows = summary["comparisons"]
    eq = [r for r in rows if r["both_equal"]]
    matched_win = [f"L{r['L']}" for r in rows if r["both_matched"]]
    l1_win = [f"L{r['L']}" for r in rows if r["both_Lplus1"]]
    lines = [
        (
            f"Full-space N=12 gap is {summary['gap']['N12_full']:.6e}. "
            f"The accessible P=R=+1 gap is {summary['gap']['N12_accessible']:.6e}. "
            "They are not the same number."
        ),
        (
            "Family N is Hc plus normalized full K, block-mean M, and the contiguous Neel selector N. "
            "One gamma and the three thetas are shared by every layer. "
            "This is not a special case of round-2 S: S used inter-only K and a weight on all of Hintra."
        ),
        summary["alias"].get("conclusion", ""),
        (
            f"Equal depth, both best energy and fidelity versus standard QAOA: {len(eq)} of {len(rows)} depths. "
            "That comparison gives N 3L+4 parameters against QAOA's 2L, and the selector is a level-resolved phase, not a shell SNAP."
        ),
        (
            "Parameter-matched QAOA, depth ceil((3L+4)/2): "
            + ("ahead at " + ", ".join(matched_win) if matched_win else "ahead at no depth")
            + "."
        ),
        (
            "Versus QAOA at L+1, both metrics hold at "
            + (", ".join(l1_win) if l1_win else "no depth")
            + "."
        ),
        (
            "A one-gap 2π shift of the old intra angle is not an exact alias. "
            "The ratio of the two intra spacings is not an integer, and the shifted circuit state is different. "
            "Raw consultant confidence is not a proof."
        ),
        "The all-start IQR includes failed seeds and is often several energy units. "
        "The best-half IQR is also stored. A gap smaller than that spread is not treated as material. No seal is claimed.",
    ]
    return lines


def _write_md(path: Path, summary: dict):
    lines = ["# Round 3 family N", ""]
    for line in summary["reading"]:
        if line:
            lines.append(line)
            lines.append("")
    g = summary["gap"]
    lines.append("## Spectra")
    lines.append("")
    lines.append(f"N=12 full gap {g['N12_full']:.6e}, accessible gap {g['N12_accessible']:.6e}.")
    lines.append(f"N=6 full gap {g['N6_full']:.6e}, accessible gap {g['N6_accessible']:.6e}.")
    lines.append(f"Manifest `{summary['manifest_sha256']}`.")
    lines.append("")
    ident = summary["identity"]
    lines.append(
        f"Intra identity max abs error {ident['max_abs_error']:.3e} "
        f"(alpha={ident['alpha']:.6f}, c={ident['c']:.6f}). "
        f"Selector eigenvalue check passed: {summary['selector_eigenvalues']['neel_matches_minus4_on_2_and_5']}."
    )
    lines.append("")
    lines.append("## N=12")
    lines.append("")
    lines.append("| L | method | P | E | E-E0 | F | varH | median | IQR | neel | evals |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for j in sorted(summary["jobs"], key=lambda r: (r["L"], r["method"])):
        if j["L"] > 6 and j["method"] != "qaoa":
            continue
        if j["L"] > 11:
            continue
        ne = j.get("neel_occupation", {}).get("mean_p_neel", float("nan"))
        lines.append(
            f"| {j['L']} | {j['method']} | {j['param_count']} | {j['E']:.5f} | {j['energy_error']:.4e} | "
            f"{j['F']:.4f} | {j['variance_H']:.4f} | {j['seed_median']:.4f} | {j['seed_iqr']:.4f} | "
            f"{ne:.3f} | {j['n_eval']} |"
        )
    lines.append("")
    lines.append("| L | dE QAOA | dF | both L | both L+1 | both matched | dE A | dE SaSb | dE wrong | dE intra |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["comparisons"]:
        def f(k):
            v = r.get(k)
            return "" if v is None else f"{v:.4e}"
        lines.append(
            f"| {r['L']} | {f('dE_qaoa')} | {f('dF_qaoa')} | {r['both_equal']} | {r['both_Lplus1']} | "
            f"{r['both_matched']} | {f('dE_A')} | {f('dE_sa')} | {f('dE_wrong')} | {f('dE_intra')} |"
        )
    lines.append("")
    lines.append("Negative dE means family N has lower energy.")
    lines.append("")
    if summary["failures"]:
        lines.append("## Skips")
        for item in summary["failures"]:
            lines.append(f"- {item}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _plots(summary, folder: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = summary["jobs"]
    methods = ["family_N", "A", "thetaN0", "qaoa", "wrong_sel", "full_intra"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        xs, ye, yf = [], [], []
        for L in range(1, 7):
            hit = [j for j in jobs if j["L"] == L and j["method"] == method]
            if not hit:
                continue
            xs.append(L)
            ye.append(hit[0]["energy_error"])
            yf.append(hit[0]["F"])
        if xs:
            axes[0].plot(xs, ye, marker="o", label=method)
            axes[1].plot(xs, yf, marker="o", label=method)
    # matched QAOA energy error at the matched depth, plotted against candidate L
    mx, my, mf = [], [], []
    for r in summary["comparisons"]:
        if r.get("err_matched") is None:
            continue
        mx.append(r["L"])
        my.append(r["err_matched"])
        mf.append(r["F_matched"])
    if mx:
        axes[0].plot(mx, my, marker="s", linestyle="--", label="QAOA matched")
        axes[1].plot(mx, mf, marker="s", linestyle="--", label="QAOA matched")
    axes[0].set_xlabel("depth L")
    axes[0].set_ylabel("energy error")
    axes[1].set_xlabel("depth L")
    axes[1].set_ylabel("full-space fidelity")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "n12_error_fidelity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, style in (("K", "-"), ("M", "--"), ("N", ":")):
        xs, ys = [], []
        for L in range(1, 7):
            hit = [j for j in jobs if j["L"] == L and j["method"] == "family_N"]
            if not hit:
                continue
            xs.append(L)
            ys.append(hit[0]["theta"].get(name, 0.0))
        ax.plot(xs, ys, linestyle=style, marker="o", label=name)
    ax.set_xlabel("depth L")
    ax.set_ylabel("shared coefficient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "n12_thetas.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run(80.0)
