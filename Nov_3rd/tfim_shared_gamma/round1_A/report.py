"""Tables and machine-readable summary for round-1 A."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import PRIMARY_TRANSFER_SCALE, json_dump


def matched_qaoa_depth(L: int) -> int:
    """Smallest independent-angle depth whose parameter count is at least 2+3L."""
    return int(np.ceil((2 + 3 * L) / 2.0))


def comparisons(jobs: list[dict]) -> list[dict]:
    index = {(j["n"], j["L"], j["method"]): j for j in jobs}
    rows = []
    keys = sorted({(j["n"], j["L"]) for j in jobs if j["method"] == "family_A"})
    for n, L in keys:
        A = index.get((n, L, "family_A"))
        Q = index.get((n, L, "qaoa"))
        if A is None or Q is None:
            continue
        D = index.get((n, L, "native_D"))
        R = index.get((n, L, "random_K"))
        S = index.get((n, L, "refl_K"))
        G = index.get((n, L, "shared_qaoa"))
        Q1 = index.get((n, L + 1, "qaoa"))
        Qm = index.get((n, matched_qaoa_depth(L), "qaoa"))
        spread = max(
            float(A.get("restart_E_std") or 0.0),
            float((R or {}).get("restart_E_std") or 0.0),
        )

        def better_E(other):
            if other is None:
                return None
            return bool(A["E"] < other["E"] - 1e-8)

        def better_F(other):
            if other is None:
                return None
            return bool(A["F"] > other["F"] + 1e-6)

        row = {
            "n": n,
            "L": L,
            "params_A": A["param_count"],
            "params_qaoa": Q["param_count"],
            "E_A": A["E"],
            "E_qaoa": Q["E"],
            "dE_A_minus_qaoa": A["E"] - Q["E"],
            "err_A": A["energy_error"],
            "err_qaoa": Q["energy_error"],
            "F_A": A["F"],
            "F_qaoa": Q["F"],
            "dF_A_minus_qaoa": A["F"] - Q["F"],
            "eta_A": A["eta"],
            "eta_qaoa": Q["eta"],
            "thetaK": A.get("theta"),
            "gamma_A": A["gamma"][0] if A.get("gamma") else None,
            "energy_better_than_qaoa": better_E(Q),
            "fidelity_better_than_qaoa": better_F(Q),
            "both_better_than_qaoa": bool(better_E(Q) and better_F(Q)),
            "energy_better_than_native": better_E(D),
            "fidelity_better_than_native": better_F(D),
            "energy_better_than_random_K": better_E(R),
            "energy_better_than_refl_K": better_E(S),
            "energy_better_than_shared_qaoa": better_E(G),
            "dE_vs_native": None if D is None else A["E"] - D["E"],
            "dE_vs_random_K": None if R is None else A["E"] - R["E"],
            "dE_vs_refl_K": None if S is None else A["E"] - S["E"],
            "dE_vs_shared": None if G is None else A["E"] - G["E"],
            "random_margin_vs_restart_std": None
            if R is None
            else (R["E"] - A["E"]) / spread
            if spread > 0
            else None,
            "qaoa_Lplus1_E": None if Q1 is None else Q1["E"],
            "qaoa_Lplus1_F": None if Q1 is None else Q1["F"],
            "qaoa_Lplus1_params": None if Q1 is None else Q1["param_count"],
            "energy_better_than_qaoa_Lplus1": better_E(Q1),
            "fidelity_better_than_qaoa_Lplus1": better_F(Q1),
            "both_better_than_qaoa_Lplus1": None
            if Q1 is None
            else bool(better_E(Q1) and better_F(Q1)),
            "matched_qaoa_depth": matched_qaoa_depth(L),
            "matched_qaoa_E": None if Qm is None else Qm["E"],
            "matched_qaoa_F": None if Qm is None else Qm["F"],
            "energy_better_than_matched_qaoa": better_E(Qm),
            "fidelity_better_than_matched_qaoa": better_F(Qm),
            "A_init": A.get("init_label"),
            "qaoa_init": Q.get("init_label"),
            "qaoa_was_rescued": bool(Q.get("rescue")),
        }
        rows.append(row)
    return rows


def gamma_ramp_section(jobs: list[dict], targets: dict) -> dict:
    """Measurements around the ramp heuristic. Not a derivation of it."""
    by_depth = []
    for job in jobs:
        if job["method"] != "family_A":
            continue
        by_depth.append(
            {
                "n": job["n"],
                "L": job["L"],
                "thetaK": job.get("theta"),
                "gamma": job["gamma"][0],
                "E": job["E"],
                "F": job["F"],
                "layers": job.get("layers"),
            }
        )
    return {
        "statement": (
            "An effective gamma ramp from thetaK<0 was a design heuristic, not a "
            "result. The numbers below are the test: the optimized shared thetaK, "
            "the correlation of the normalized feature with Hc, and the energy gaps "
            "versus thetaK=0 and versus equal-RMS controls. A negative thetaK is "
            "consistent with the heuristic only when those controls lose."
        ),
        "Hc_Fk_correlation": {
            str(n): targets[str(n)].get("Hc_Fk_correlation") for n in targets
        },
        "optimized": by_depth,
    }


def build_reading(summary: dict) -> list[str]:
    """Qualified statements that follow the tables. Not a program-level seal."""
    rows = summary["comparisons"]
    lines = []
    both = [r for r in rows if r["both_better_than_qaoa"]]
    lines.append(
        f"Equal depth: family A is better on both energy and fidelity in {len(both)} of {len(rows)} computed cells. "
        "That comparison gives A more parameters (2+3L versus 2L)."
    )
    survived = [
        r for r in rows
        if r["both_better_than_qaoa"] and r.get("both_better_than_qaoa_Lplus1")
    ]
    failed_l1 = [
        f"N={r['n']} L={r['L']}" for r in rows
        if r["both_better_than_qaoa"] and r.get("both_better_than_qaoa_Lplus1") is False
    ]
    lines.append(
        f"QAOA at L+1: the equal-depth both-metric edge also beats QAOA at L+1 in {len(survived)} cells. "
        + ("It does not at " + ", ".join(failed_l1) + "." if failed_l1 else "No equal-depth both-metric cell lost to QAOA at L+1.")
    )
    matched_both = [
        f"N={r['n']} L={r['L']} (QAOA L={r['matched_qaoa_depth']})"
        for r in rows
        if r.get("energy_better_than_matched_qaoa") and r.get("fidelity_better_than_matched_qaoa")
    ]
    matched_loss = [
        f"N={r['n']} L={r['L']} (QAOA L={r['matched_qaoa_depth']})"
        for r in rows
        if r.get("matched_qaoa_E") is not None and not (
            r.get("energy_better_than_matched_qaoa") and r.get("fidelity_better_than_matched_qaoa")
        )
    ]
    missing = [
        f"N={r['n']} L={r['L']} (QAOA L={r['matched_qaoa_depth']})"
        for r in rows
        if r.get("matched_qaoa_E") is None
    ]
    lines.append(
        "Matched parameter count, QAOA depth ceil((2+3L)/2): both-metric edge survives at "
        + (", ".join(matched_both) if matched_both else "no computed cell")
        + ". It does not survive at "
        + (", ".join(matched_loss) if matched_loss else "no computed cell")
        + (
            ". Not run: " + ", ".join(missing) + "."
            if missing
            else ". Every listed cell has its matched QAOA depth."
        )
    )
    lines.append(
        "The N=6 matched-parameter wins are already present in native D, which fixes thetaK=0 and "
        "uses fewer parameters than family A. They are evidence for the nonlinear shell under a shared "
        "gamma, not for the physical K matrix. At N=12 native D loses to equal-depth QAOA, and family A "
        "loses to parameter-matched QAOA at every depth that was checked."
    )
    lines.append(
        "Controls: at N=12 the physical K feature is ahead of native D (thetaK=0), equal-RMS random K, "
        "and equal-RMS reflection-symmetric K on energy at every computed depth, with optimized thetaK negative. "
        "At N=6 that gap is large at L=1 and is gone by L=4, where a reflection-symmetric random K matches or "
        "beats physical K. Native D alone already beats equal-depth QAOA at N=6 for L>=2 and does not beat "
        "equal-depth QAOA at N=12. The N=12 equal-depth edge is the K feature, not the nonlinear shell by itself."
    )
    lines.append(
        "The negative thetaK at N=12 is consistent with the effective-gamma-ramp heuristic and, at that size, "
        "the physical matrix beats both equal-RMS controls. The heuristic is not a derivation: thetaK changes "
        "sign across N=6 depths, and a reflection-symmetric random matrix can do as well once the shell is expressive."
    )
    lines.append(
        "Restart standard deviations are large because they include failed starts. They are not an error bar "
        "on the reported best energy. QAOA rescues with extra fresh starts did not lower the QAOA best energies. "
        "Family A used more evaluations than QAOA because it has more parameters and more starts; QAOA was not cut off by the shared cap."
    )
    lines.append(
        "N=6 to N=12 transfer with frozen angles is a different result from retraining. "
        "The predeclared kappa rescaling carries L=1 family A to an energy close to the retrained L=1 result. "
        "For L>=2 every disclosed gamma scale (raw, kappa, rms) leaves the nonlinear shell far from the ground state, "
        "while transferred QAOA with the same kappa rule stays near its retrained energy. "
        "A shared shell schedule fit at N=6 does not transfer."
    )
    lines.append(
        "Fidelity along the optimized family-A schedules is not monotone at every depth. "
        "Best-fidelity starts were recorded and were not allowed to replace best-energy angles."
    )
    lines.append(
        "This is not a sealed win. Equal-depth gains that die under parameter matching, or that a random "
        "reflection-symmetric K can copy, are not a demonstrated advantage of this feature."
    )
    return lines


def write_markdown(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# Round 1 direction A — shared-gamma F_K versus QAOA")
    lines.append("")
    lines.append(
        "Generator `G = Hc + thetaK * sK * F_K` with `F_K = sum_{j<k} K_jk Z_j Z_k`, "
        "`K = offdiag(J^2)`, `sK` the upper-triangle RMS match to `J`, and the "
        "coefficient of `Hc` fixed at 1. `gamma` and `thetaK` are one scalar each "
        "at every depth. Each layer has shell controls `(c1, c2, c3)`."
    )
    lines.append("")
    lines.append("## Targets")
    lines.append("")
    lines.append("| N | B | E0 | gap | gap in P=R=+1 | S (bits) | P | R | residual | H hash |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for n, t in summary["targets"].items():
        gap_s = t.get("low_lying", {}).get("gap_within_PR_plus")
        gap_txt = "n/a" if gap_s is None else f"{gap_s:.6e}"
        lines.append(
            f"| {n} | {t['B']:.4f} | {t['E0']:.6f} | {t['gap']:.3e} | {gap_txt} | "
            f"{t['S_log2']:.6f} | {t['parity_expectation']:.4f} | "
            f"{t['reflection_expectation']:.4f} | {t['residual']:.2e} | `{t['H_hash'][:12]}` |"
        )
    lines.append("")
    lines.append(
        "B is the half-chain entropy maximum on the contract grid, refined to 1e-4. "
        "It is this run's numerical realization, not a published Pagano field. "
        "The ground state is the full-space vector. `gap` is the full-space gap; "
        "`gap in P=R=+1` is the gap to the next state that shares the vacuum's quantum numbers. "
        "A tiny full-space gap can be a symmetry partner outside that sector."
    )
    for n, t in summary["targets"].items():
        plat = t.get("coarse_plateau_1e-4_bits") or []
        if len(plat) >= 3:
            lines.append(
                f"N={n} entropy is flat: {len(plat)} coarse-grid fields sit within 1e-4 bits of the maximum "
                f"(B from {min(plat):.3f} to {max(plat):.3f}). The frozen B is the argmax under the stated tie rule, "
                "not evidence of a sharp peak."
            )
    lines.append("")
    lines.append("## Depth table")
    lines.append("")
    lines.append(
        "Energy is the objective. Fidelity `F` and `eta = (E-Emax)/(E0-Emax)` are diagnostics. "
        "Negative `dE` means family A has lower energy than independent-angle QAOA at the same depth."
    )
    lines.append("")
    lines.append(
        "| N | L | method | params | E | E-E0 | F | eta | theta | gamma | evals | starts | init |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in sorted(summary["jobs"], key=lambda r: (r["n"], r["L"], r["method"])):
        theta = j.get("theta")
        theta_s = "" if theta is None else f"{theta:.4f}"
        gamma = j["gamma"][0] if j.get("gamma") else float("nan")
        lines.append(
            f"| {j['n']} | {j['L']} | {j['method']} | {j['param_count']} | {j['E']:.6f} | "
            f"{j['energy_error']:.4e} | {j['F']:.4f} | {j['eta']:.4f} | {theta_s} | "
            f"{gamma:.4f} | {j['n_eval']} | {j['n_starts']} | {j.get('init_label')} |"
        )
    lines.append("")
    lines.append("## Family A versus controls")
    lines.append("")
    lines.append(
        "| N | L | dE vs QAOA | dF vs QAOA | both | dE vs D | dE vs random K | dE vs refl K | "
        "QAOA L+1 E | both vs L+1 | matched L | matched E |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["comparisons"]:
        def fmt(v):
            return "" if v is None else f"{v:.4e}"

        lines.append(
            f"| {r['n']} | {r['L']} | {fmt(r['dE_A_minus_qaoa'])} | {r['dF_A_minus_qaoa']:.4f} | "
            f"{r['both_better_than_qaoa']} | {fmt(r['dE_vs_native'])} | {fmt(r['dE_vs_random_K'])} | "
            f"{fmt(r['dE_vs_refl_K'])} | {fmt(r['qaoa_Lplus1_E'])} | {r['both_better_than_qaoa_Lplus1']} | "
            f"{r['matched_qaoa_depth']} | {fmt(r['matched_qaoa_E'])} |"
        )
    lines.append("")
    lines.append("## Transfer N=6 to N=12")
    lines.append("")
    lines.append(
        f"Primary disclosed scale is `{PRIMARY_TRANSFER_SCALE}` "
        "(gamma_12 = gamma_6 * kappa_6 / kappa_12). "
        "Theta is unchanged because each size RMS-matches its own K to its own J. "
        "Shell controls are unchanged. Raw gamma and an RMS rescaling are sensitivity checks, "
        "not a selection on the N=12 ground state. This is not the retrained N=12 column."
    )
    lines.append("")
    if summary.get("transfer"):
        lines.append("| source L | method | scale | E | E-E0 | F | eta | primary |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in summary["transfer"]:
            lines.append(
                f"| {row['L']} | {row['method']} | {row['scale']} | {row['E']:.6f} | "
                f"{row['energy_error']:.4e} | {row['F']:.4f} | {row['eta']:.4f} | {row['primary']} |"
            )
        lines.append("")
    lines.append("## Gamma-ramp heuristic")
    lines.append("")
    lines.append(summary["gamma_ramp_heuristic"]["statement"])
    lines.append("")
    lines.append("## Failures and skips")
    lines.append("")
    if not summary["failures"] and not summary["skipped"]:
        lines.append("None recorded.")
    else:
        for item in summary["failures"]:
            lines.append(f"- {item}")
        for item in summary["skipped"]:
            lines.append(f"- skipped: {item}")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    for line in summary.get("reading") or []:
        lines.append(line)
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    prov = summary["provenance"]
    lines.append(f"- git: `{prov.get('git')}`")
    lines.append(f"- python {prov.get('python')} numpy {prov.get('numpy')} scipy {prov.get('scipy')}")
    lines.append(f"- wall seconds, last process: {prov.get('wall_seconds')}")
    if prov.get("timing_note"):
        lines.append(f"- timing: {prov.get('timing_note')}")
    lines.append(f"- validation passed: {summary['validation']['passed']}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def dump_summary(path: Path, summary: dict) -> None:
    json_dump(path, summary)
