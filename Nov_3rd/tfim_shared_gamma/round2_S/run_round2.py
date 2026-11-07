"""Round-2 family S on the frozen manifest.

N=12 is the study, depths 1..6 in ascending order. N=6 is a smoke test.
Standard QAOA is continued through the parameter-matched depth ceil((3L+3)/2).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.common import canonical_state, log, sha256_state

from .algebra import assert_algebra, normalization_checks
from .circuit import energy_and_grad, energy_only, eta, layer_rows, param_count
from .frozen import MANIFEST_SHA256, build_system
from .optimize import optimize_method, top_up

RESULTS = Path(__file__).resolve().parent / "results"
STUDY = ("thetaK0", "inter_K_A", "full_K_A", "shuffled_intra", "family_S")
PERIODS = [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]


def _cap(L: int, smoke: bool) -> int:
    if smoke:
        return 450
    return {1: 2600, 2: 4000, 3: 4600, 4: 5000, 5: 5400, 6: 5800}.get(L, 5000)


def _fresh(method: str, smoke: bool) -> int:
    if smoke:
        return 2
    if method in ("qaoa", "same_G_qaoa"):
        return 12
    return 6


def _record(system, out) -> dict:
    x = np.asarray(out["x"], dtype=np.float64)
    # Independent in-process replay. Not a second-agent replay.
    E2, _g, psi = energy_and_grad(system, out["method"], x, out["L"])
    if abs(E2 - out["E"]) > 1e-6:
        raise RuntimeError(f"replay mismatch {out['method']} {E2} vs {out['E']}")
    if out["method"] not in ("qaoa", "same_G_qaoa"):
        if max(out["gamma"]) - min(out["gamma"]) > 1e-12:
            raise RuntimeError("shared gamma split")
    rec = {
        "method": out["method"],
        "n": system.n,
        "L": out["L"],
        "role": _role(out["method"]),
        "param_count": param_count(out["method"], out["L"]),
        "B": system.B_string,
        "E0": system.E0,
        "Emax": system.Emax,
        "gap_full": system.spectrum["gap_full"],
        "gap_accessible_PR_plus": system.spectrum["gap_accessible_PR_plus"],
        "E": float(E2),
        "energy_error": float(E2 - system.E0),
        "F": float(out["F"]),
        "eta": eta(E2, system.E0, system.Emax),
        "variational_ok": bool(E2 >= system.E0 - 1e-7),
        "x_exact": [format(float(v), ".17g") for v in x],
        "gamma": out["gamma"],
        "thetaIn": out["thetaIn"],
        "thetaK": out["thetaK"],
        "shell": out["shell"],
        "init_label": out["init_label"],
        "n_eval": out["n_eval"],
        "eval_cap": out["eval_cap"],
        "n_starts": out["n_starts"],
        "seconds": out["seconds"],
        "seed_median": out["seed_median"],
        "seed_median_best_half": out["seed_median_best_half"],
        "seed_best": out["seed_best"],
        "seed_std": out["seed_std"],
        "seed_spread_median_minus_best": out["seed_spread_median_minus_best"],
        "trace_best_start": out["trace_best_start"],
        "starts": out["starts"],
        "state_sha256_f64": sha256_state(psi),
        "state_sha256_round10": sha256_state(psi, decimals=10),
        "layers": layer_rows(system, out["method"], x, out["L"]),
        "manifest_sha256": MANIFEST_SHA256,
        "hc_diag_sha256": system.hc_diag_sha256,
        "ledger_hardware": system.ledger,
        "ledger_params": {
            "P": param_count(out["method"], out["L"]),
            "n_eval": out["n_eval"],
            "n_starts": out["n_starts"],
            "seed_median": out["seed_median"],
            "seed_best": out["seed_best"],
        },
        "in_process_replay": True,
        "fidelity_used_for_selection": False,
        "objective": "energy",
        "topped_up": bool(out.get("topped_up", False)),
        "rescue": bool(out.get("rescue", False)),
    }
    if rec["manifest_sha256"] != MANIFEST_SHA256:
        raise RuntimeError("result row lost the manifest hash")
    np.save(RESULTS / "states" / f"n{system.n}_L{out['L']}_{out['method']}.npy", canonical_state(psi))
    return rec


def _role(method: str) -> str:
    return {
        "family_S": "candidate",
        "qaoa": "standard_qaoa",
        "same_G_qaoa": "same_G_diagnostic_not_standard_qaoa",
        "full_K_A": "control_full_K_A",
        "inter_K_A": "control_inter_K_A_thetaIn_fixed_1",
        "thetaK0": "control_thetaK_0",
        "shuffled_intra": "control_shuffled_intra",
    }[method]


def _save(rec: dict) -> None:
    path = RESULTS / "jobs" / f"n{rec['n']}_L{rec['L']}_{rec['method']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")


def _load(n, L, method, manifest, hc_hash):
    path = RESULTS / "jobs" / f"n{n}_L{L}_{method}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("manifest_sha256") != manifest or rec.get("hc_diag_sha256") != hc_hash:
        return None
    rec["x"] = np.array([float(v) for v in rec["x_exact"]])
    return rec


def _public_spectrum(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k != "gs"}


def run(budget_min: float = 80.0) -> dict:
    t0 = time.perf_counter()
    deadline = t0 + budget_min * 60.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "states").mkdir(exist_ok=True)
    (RESULTS / "jobs").mkdir(exist_ok=True)
    failures = []
    algebra = assert_algebra()
    log("algebra: c2 sum XX does not commute with intra ZZ; c3 XXX does, and anticommutes with a touching inter ZZ")

    systems = {}
    for n in (6, 12):
        log(f"dense spectrum N={n}")
        systems[n] = build_system(n)
        sp = systems[n].spectrum
        log(
            f"  B={systems[n].B_string} E0={sp['E0']:.6f} full_gap={sp['gap_full']:.6e} "
            f"accessible_gap={sp['gap_accessible_PR_plus']:.6e} "
            f"P={sp['gs_parity']:.3f} R={sp['gs_reflection']:.3f}"
        )
        if sp["gs_parity"] < 0.5 or sp["gs_reflection"] < 0.5:
            failures.append(f"N={n} full GS is outside P=R=+1")
        if abs(sp["gap_full"] - (sp["gap_accessible_PR_plus"] or -1)) < 1e-6:
            failures.append(f"N={n} full gap and accessible gap numerically equal; check the labeling")

    # Gradient spot checks. N=6 all coordinates, N=12 a handful.
    for n, coords in ((6, None), (12, 4)):
        sys = systems[n]
        rng = np.random.default_rng(3 + n)
        for method in ("qaoa", "family_S", "same_G_qaoa", "full_K_A"):
            L = 2 if n == 6 else 1
            x = rng.normal(scale=0.2, size=param_count(method, L))
            _E, grad, _psi = energy_and_grad(sys, method, x, L)
            idxs = range(x.size) if coords is None else rng.choice(x.size, size=min(coords, x.size), replace=False)
            err = 0.0
            for i in idxs:
                step = np.zeros_like(x)
                step[int(i)] = 1e-6
                Ep, _ = energy_only(sys, method, x + step, L)
                Em, _ = energy_only(sys, method, x - step, L)
                err = max(err, abs((Ep - Em) / 2e-6 - grad[int(i)]))
            log(f"  grad {method} N={n} maxabs {err:.3e}")
            if err > 5e-5:
                raise RuntimeError(f"gradient failed {method} N={n} {err}")

    mem: dict[tuple, dict] = {}

    def remaining():
        return deadline - time.perf_counter()

    def run_one(n, L, method, smoke=False, eval_cap=None, n_fresh=None, seed_tag=1):
        sys = systems[n]
        loaded = _load(n, L, method, sys.manifest_sha256, sys.hc_diag_sha256)
        if loaded is not None and not smoke:
            log(f"resume N={n} L={L} {method} E={loaded['E']:.6f}")
            mem[(n, L, method)] = loaded
            return loaded
        if remaining() < 40:
            failures.append(f"skipped N={n} L={L} {method} for time")
            return None
        prev = mem.get((n, L - 1, method))
        prev_x = None if prev is None else prev["x"]
        cap = int(_cap(L, smoke) if eval_cap is None else eval_cap)
        fresh = int(_fresh(method, smoke) if n_fresh is None else n_fresh)
        log(f"optimize N={n} L={L} {method} cap={cap} remaining={remaining()/60:.1f}min")
        out = optimize_method(sys, method, L, prev_x, PERIODS, cap, fresh, seed_tag)
        rec = _record(sys, out)
        rec["x"] = out["x"]
        if not smoke:
            _save({k: v for k, v in rec.items() if k != "x"})
        mem[(n, L, method)] = rec
        log(
            f"  E={rec['E']:.6f} err={rec['energy_error']:.4e} F={rec['F']:.4f} "
            f"med={rec['seed_median']:.4f} thetaIn={rec['thetaIn']} thetaK={rec['thetaK']} "
            f"evals={rec['n_eval']} init={rec['init_label']}"
        )
        if not rec["variational_ok"]:
            failures.append(f"variational N={n} L={L} {method}")
        return rec

    # N=6 smoke. Not a ladder and not a claim.
    smoke_rows = []
    for method in list(STUDY) + ["qaoa", "same_G_qaoa"]:
        rec = run_one(6, 1, method, smoke=True, seed_tag=3)
        if rec:
            smoke_rows.append({k: v for k, v in rec.items() if k != "x"})

    # N=12 ascending. Do not skip a shallower depth.
    for L in (1, 2, 3, 4, 5, 6):
        for method in STUDY:
            run_one(12, L, method)
        srec = mem.get((12, L, "family_S"))
        target = 0 if srec is None else int(srec["n_eval"])
        q = run_one(12, L, "qaoa", eval_cap=max(_cap(L, False), target))
        if q is not None and target and q["n_eval"] < target and remaining() > 60:
            log(f"top up QAOA N=12 L={L} from {q['n_eval']} toward {target}")
            more = top_up(systems[12], "qaoa", L, q, PERIODS, target, seed_tag=40 + L)
            more_rec = _record(systems[12], more)
            more_rec["x"] = more["x"]
            _save({k: v for k, v in more_rec.items() if k != "x"})
            mem[(12, L, "qaoa")] = more_rec
            q = more_rec
        g = run_one(12, L, "same_G_qaoa", eval_cap=max(_cap(L, False), target))
        if g is not None and target and g["n_eval"] < target and remaining() > 60:
            log(f"top up same-G QAOA N=12 L={L}")
            more = top_up(systems[12], "same_G_qaoa", L, g, PERIODS, target, seed_tag=60 + L)
            more_rec = _record(systems[12], more)
            more_rec["x"] = more["x"]
            _save({k: v for k, v in more_rec.items() if k != "x"})
            mem[(12, L, "same_G_qaoa")] = more_rec

    # Standard QAOA through matched depth of L=6, which is 11, and L+1 which is 7.
    for L in range(7, 12):
        run_one(12, L, "qaoa", eval_cap=4500, n_fresh=10, seed_tag=5)

    def both_better(a, b):
        if a is None or b is None:
            return False
        return a["E"] < b["E"] - 1e-8 and a["F"] > b["F"] + 1e-6

    # Rescue only where the candidate is ahead of equal-depth QAOA on both metrics.
    for L in (1, 2, 3, 4, 5, 6):
        S = mem.get((12, L, "family_S"))
        Q = mem.get((12, L, "qaoa"))
        if not both_better(S, Q):
            continue
        if Q.get("rescue_attempted") or Q.get("rescue"):
            continue
        if remaining() < 90:
            failures.append(f"no time to rescue QAOA L={L}")
            continue
        log(f"rescue QAOA N=12 L={L} with about 4x starts")
        extra_starts = max(8, 3 * int(Q["n_starts"]))
        extra = optimize_method(
            systems[12], "qaoa", L, Q["x"], PERIODS,
            eval_cap=extra_starts * 140, n_fresh=extra_starts, seed_tag=90 + L,
            do_screen=False, fresh_only=True,
        )
        extra["n_eval"] = int(Q["n_eval"] + extra["n_eval"])
        extra["seconds"] = float(Q["seconds"] + extra["seconds"])
        extra["rescue"] = True
        if extra["E"] < Q["E"] - 1e-10:
            rec = _record(systems[12], extra)
            rec["x"] = extra["x"]
            rec["rescue"] = True
            rec["prerescue_E"] = Q["E"]
            rec["prerescue_F"] = Q["F"]
            _save({k: v for k, v in rec.items() if k != "x"})
            mem[(12, L, "qaoa")] = rec
            log(f"  rescue improved {Q['E']:.6f} -> {rec['E']:.6f}")
        else:
            Q["rescue_attempted"] = True
            Q["rescue_E"] = extra["E"]
            Q["n_eval"] = extra["n_eval"]
            _save({k: v for k, v in Q.items() if k != "x"})
            log(f"  rescue did not improve ({extra['E']:.6f} vs {Q['E']:.6f})")

    jobs = []
    for rec in mem.values():
        if rec["n"] == 12:
            jobs.append({k: v for k, v in rec.items() if k != "x"})
    summary = {
        "manifest_sha256": MANIFEST_SHA256,
        "algebra": algebra,
        "normalization": {str(n): normalization_checks(systems[n]) for n in systems},
        "spectra": {
            str(n): {
                **_public_spectrum(systems[n].spectrum),
                "B": systems[n].B_string,
                "hc_diag_sha256": systems[n].hc_diag_sha256,
                "shuffle": {k: v for k, v in systems[n].shuffle.items() if k != "matrix"},
                "sK_full": systems[n].sK_full,
                "sK_inter": systems[n].sK_inter,
                "ledger": systems[n].ledger,
            }
            for n in systems
        },
        "gap_correction": {
            "round1_error": (
                "Round 1 reported the N=12 gap 0.433 as the full-space gap. "
                "That eigensolve started from the symmetric product state, so it "
                "stayed in P=R=+1. The number was the accessible-sector gap."
            ),
            "frozen_N12_full_gap": systems[12].spectrum["gap_full"],
            "frozen_N12_accessible_gap": systems[12].spectrum["gap_accessible_PR_plus"],
            "frozen_N6_full_gap": systems[6].spectrum["gap_full"],
            "frozen_N6_accessible_gap": systems[6].spectrum["gap_accessible_PR_plus"],
            "partner": "The state at the full gap has P=R=-1. The ground state itself is P=R=+1 and is the fidelity target.",
        },
        "smoke_n6_L1": smoke_rows,
        "jobs": jobs,
        "failures": failures,
        "comparisons": _comparisons(jobs),
        "wall_seconds": time.perf_counter() - t0,
        "budget_min": budget_min,
        "note": (
            "same_G_qaoa is an independent-gamma linear-mixer diagnostic with the "
            "candidate's shared thetas. It is not standard QAOA and it is not a "
            "certified ceiling. A failed local solve is not a proof that shared gamma cannot win."
        ),
    }
    summary["reading"] = _reading(summary)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(RESULTS / "RESULTS.md", summary)
    log(f"wrote results in {summary['wall_seconds']:.1f}s failures={len(failures)}")
    return summary


def _comparisons(jobs: list[dict]) -> list[dict]:
    index = {(j["L"], j["method"]): j for j in jobs}
    rows = []
    for L in range(1, 7):
        S = index.get((L, "family_S"))
        Q = index.get((L, "qaoa"))
        if S is None or Q is None:
            continue
        P = S["param_count"]
        matched = int(np.ceil(P / 2))
        Q1 = index.get((L + 1, "qaoa"))
        Qm = index.get((matched, "qaoa"))
        G = index.get((L, "same_G_qaoa"))

        def pack(other, name):
            if other is None:
                return {}
            return {
                f"E_{name}": other["E"],
                f"F_{name}": other["F"],
                f"err_{name}": other["energy_error"],
                f"med_{name}": other["seed_median"],
                f"dE_{name}": S["E"] - other["E"],
                f"dF_{name}": S["F"] - other["F"],
            }

        row = {
            "L": L,
            "P": P,
            "E_S": S["E"],
            "F_S": S["F"],
            "err_S": S["energy_error"],
            "med_S": S["seed_median"],
            "spread_S": S["seed_spread_median_minus_best"],
            "thetaIn": S["thetaIn"],
            "thetaK": S["thetaK"],
            "evals_S": S["n_eval"],
            "evals_qaoa": Q["n_eval"],
            "matched_depth": matched,
            **pack(Q, "qaoa"),
            **pack(Q1, "qaoa_Lplus1"),
            **pack(Qm, "qaoa_matched"),
            **pack(G, "sameG"),
            **pack(index.get((L, "full_K_A")), "fullK"),
            **pack(index.get((L, "inter_K_A")), "interK"),
            **pack(index.get((L, "thetaK0")), "theta0"),
            **pack(index.get((L, "shuffled_intra")), "shuf"),
        }
        row["both_vs_equal_L"] = bool(S["E"] < Q["E"] - 1e-8 and S["F"] > Q["F"] + 1e-6)
        row["both_vs_Lplus1"] = None if Q1 is None else bool(S["E"] < Q1["E"] - 1e-8 and S["F"] > Q1["F"] + 1e-6)
        row["both_vs_matched"] = None if Qm is None else bool(S["E"] < Qm["E"] - 1e-8 and S["F"] > Qm["F"] + 1e-6)
        row["both_vs_sameG"] = None if G is None else bool(S["E"] < G["E"] - 1e-8 and S["F"] > G["F"] + 1e-6)
        rows.append(row)
    return rows


def _reading(summary: dict) -> list[str]:
    rows = summary["comparisons"]
    g = summary["gap_correction"]
    lines = [
        (
            f"Gap correction: on the frozen N=12 target the dense full-space gap is "
            f"{g['frozen_N12_full_gap']:.6e} and the P=R=+1 gap is {g['frozen_N12_accessible_gap']:.6e}. "
            f"Round 1's N=12 figure 0.433 was the accessible gap. The partner at the full gap has P=R=-1. "
            f"Fidelity uses the full-space ground state, which is in P=R=+1."
        ),
        (
            "Family S trains one gamma, one thetaIn and one thetaK. thetaIn=1 is inter-K A, "
            "Hc plus a normalized inter-only FK, not the original full-K A. Both controls are in the table."
        ),
    ]
    both = [r for r in rows if r["both_vs_equal_L"]]
    lines.append(
        f"Equal depth, both energy and fidelity versus standard QAOA: {len(both)} of {len(rows)} N=12 depths. "
        "That ledger gives S 3L+3 parameters against QAOA's 2L, and the intra phases are level-resolved SNAP rather than the shared Hamming shell."
    )
    matched_loss = [f"L{r['L']}" for r in rows if r["both_vs_matched"] is False]
    matched_win = [f"L{r['L']}" for r in rows if r["both_vs_matched"]]
    lines.append(
        "Parameter-matched standard QAOA, depth ceil((3L+3)/2): "
        + ("ahead at " + ", ".join(matched_win) if matched_win else "ahead at none")
        + "; not ahead at "
        + (", ".join(matched_loss) if matched_loss else "none")
        + "."
    )
    l1_loss = [f"L{r['L']}" for r in rows if r["both_vs_Lplus1"] is False]
    l1_win = [f"L{r['L']}" for r in rows if r["both_vs_Lplus1"]]
    lines.append(
        "Versus QAOA at L+1, the both-metric edge fails at "
        + (", ".join(l1_loss) if l1_loss else "no computed depth")
        + (
            " and still holds at " + ", ".join(l1_win) + ". That depth still loses the matched-depth energy comparison."
            if l1_win
            else "."
        )
    )
    lines.append(
        "same-G QAOA uses independent gamma, a linear mixer, and the same shared thetaIn and thetaK. "
        "It is a feature-versus-shell diagnostic. It is not standard QAOA and not a rigorous ceiling; "
        "a local optimizer miss would not prove that shared gamma cannot win."
    )
    lines.append(
        "thetaIn is the trained ratio of intra couplings to the fixed inter Hc coefficient. "
        "Near 1, family S has collapsed to inter-K A. A large value is an intra-heavy profile, not a free scale gauge. "
        "The reflection-symmetric shuffled intra control loses to the physical intra pattern. "
        "Full-K A is a separate control; thetaIn=1 does not recover it."
    )
    lines.append(
        "The c2 commutation claim is false. One XX and one ZZ anticommute when they share a single site, "
        "so the intra-cavity sum of XX does not commute with intra ZZ. XXX does commute with intra ZZ "
        "(two overlaps) and anticommutes with an inter ZZ that touches the cavity once."
    )
    lines.append(
        "Seed medians include failed starts and are not an error bar on the best energy. "
        "Best and median are both stored. No seal is claimed."
    )
    return lines


def _write_md(path: Path, summary: dict) -> None:
    lines = ["# Round 2 family S", ""]
    for line in summary["reading"]:
        lines.append(line)
        lines.append("")
    lines.append("## Frozen spectra")
    lines.append("")
    lines.append("| N | B | E0 | full gap | accessible gap | P | R | residual |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n, sp in summary["spectra"].items():
        lines.append(
            f"| {n} | {sp['B']} | {sp['E0']:.6f} | {sp['gap_full']:.6e} | "
            f"{sp['gap_accessible_PR_plus']:.6e} | {sp['gs_parity']:.3f} | {sp['gs_reflection']:.3f} | "
            f"{sp['gs_residual']:.2e} |"
        )
    lines.append("")
    lines.append(f"Manifest SHA-256 `{summary['manifest_sha256']}`.")
    lines.append("")
    lines.append("## N=12 depth table")
    lines.append("")
    lines.append("| L | method | P | E | E-E0 | F | median | best | thetaIn | thetaK | evals |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for j in sorted(summary["jobs"], key=lambda r: (r["L"], r["method"])):
        tin = "" if j["thetaIn"] is None else f"{j['thetaIn']:.3f}"
        tk = "" if j["thetaK"] is None else f"{j['thetaK']:.3f}"
        lines.append(
            f"| {j['L']} | {j['method']} | {j['param_count']} | {j['E']:.5f} | {j['energy_error']:.4e} | "
            f"{j['F']:.4f} | {j['seed_median']:.4f} | {j['seed_best']:.4f} | {tin} | {tk} | {j['n_eval']} |"
        )
    lines.append("")
    lines.append("## Candidate versus baselines")
    lines.append("")
    lines.append("| L | dE QAOA | dF QAOA | both L | both L+1 | both matched | dE fullK | dE interK | dE theta0 | dE shuf | dE sameG |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["comparisons"]:
        def f(key):
            v = r.get(key)
            return "" if v is None else f"{v:.4e}"

        lines.append(
            f"| {r['L']} | {f('dE_qaoa')} | {f('dF_qaoa')} | {r['both_vs_equal_L']} | {r['both_vs_Lplus1']} | "
            f"{r['both_vs_matched']} | {f('dE_fullK')} | {f('dE_interK')} | {f('dE_theta0')} | {f('dE_shuf')} | {f('dE_sameG')} |"
        )
    lines.append("")
    lines.append("Negative dE means family S has lower energy than that baseline.")
    lines.append("")
    lines.append("## Hardware ledger")
    lines.append("")
    led = summary["spectra"]["12"]["ledger"]
    lines.append(
        f"All generators here are two-body. N=12 has {led['n_intra_bonds']} intra bonds and "
        f"{led['n_inter_bonds']} inter bonds, and {led['n_cavity_pairs']} cavity pairs. "
        "Each layer applies one shared-gamma inter diagonal (Hc_inter and FK_inter are one compiled inter profile "
        "once thetaK is fixed), plus one level-resolved 8-phase diagonal on each cavity for the intra part, "
        "plus one Hamming shell shared across cavities. The intra phases are not that shell: equal Hamming weight "
        "states in a cavity do not share an intra energy. This is not claimed to cost the same as native Hc."
    )
    lines.append("")
    if summary["failures"]:
        lines.append("## Failures")
        lines.append("")
        for item in summary["failures"]:
            lines.append(f"- {item}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run(80.0)
