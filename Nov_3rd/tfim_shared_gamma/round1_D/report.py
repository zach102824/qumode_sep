"""Tables, figures, and the acceptance verdict.

The numeric gates below were written before the optimization output was inspected.
They are effect-size gates, not captions. A cell that misses one gate is not a win.
"""
from __future__ import annotations

import json
from pathlib import Path

import thread_env  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Acceptance gates. Do not relax these after seeing the numbers.
ABS_ENERGY = 1e-3
REL_RESIDUAL = 0.05
FIDELITY_ABS = 0.01
ABLATION_ENERGY = 1e-4
RESCUE_ENERGY = 1e-3
REPRO_ABS = 1e-3
REPRO_FRAC = 0.25
MIN_DEPTHS = 2

METHODS = ("qaoa", "shared_beta", "nov3_D", "drop_c2", "drop_c3")
COLORS = {
    "qaoa": "#1f4e79",
    "shared_beta": "#7f8c8d",
    "nov3_D": "#c0392b",
    "drop_c2": "#e67e22",
    "drop_c3": "#8e44ad",
}


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def matched_qaoa_depth(L: int) -> int:
    """Smallest QAOA depth whose 2 L_Q parameters are at least the 1+3L shell count."""
    return int(np.ceil((1 + 3 * L) / 2.0))


def _cell(rows, n, L, method):
    for r in rows:
        if r.get("n") == n and r.get("L") == L and r.get("method") == method and r.get("status") == "complete":
            return r
    return None


def _second_best_gap(row) -> float | None:
    energies = [s["E"] for s in row.get("starts", []) if s.get("E") is not None]
    if len(energies) < 2:
        return None
    energies = sorted(energies)
    return float(energies[1] - energies[0])


def classify_cell(rows, n, L) -> dict:
    D = _cell(rows, n, L, "nov3_D")
    Q = _cell(rows, n, L, "qaoa")
    out = {"n": n, "L": L, "robust": False, "reasons": []}
    if D is None or Q is None:
        out["reasons"].append("missing D or QAOA")
        return out
    E0 = float(D["E0"])
    improve = float(Q["E"]) - float(D["E"])
    dF = float(D["F"]) - float(Q["F"])
    qaoa_resid = float(Q["E"]) - E0
    need = max(ABS_ENERGY, REL_RESIDUAL * max(qaoa_resid, 0.0))
    out.update(
        {
            "E_D": D["E"],
            "E_Q": Q["E"],
            "F_D": D["F"],
            "F_Q": Q["F"],
            "energy_improvement": improve,
            "fidelity_improvement": dF,
            "required_energy_improvement": need,
            "qaoa_residual": qaoa_resid,
        }
    )
    ok = True
    if improve <= need:
        ok = False
        out["reasons"].append(f"energy improvement {improve:.6g} does not exceed {need:.6g}")
    if dF < FIDELITY_ABS:
        ok = False
        out["reasons"].append(f"fidelity improvement {dF:.6g} < {FIDELITY_ABS}")
    for abl in ("drop_c2", "drop_c3", "shared_beta"):
        A = _cell(rows, n, L, abl)
        if A is None:
            ok = False
            out["reasons"].append(f"missing {abl}")
            continue
        margin = ABLATION_ENERGY if abl != "shared_beta" else 1e-5
        if float(D["E"]) > float(A["E"]) - margin:
            ok = False
            out["reasons"].append(
                f"D energy {D['E']:.8g} does not beat {abl} {A['E']:.8g} by {margin}"
            )
    Qnext = _cell(rows, n, L + 1, "qaoa")
    if Qnext is None:
        ok = False
        out["reasons"].append("missing QAOA at L+1")
    elif float(D["E"]) >= float(Qnext["E"]) - RESCUE_ENERGY:
        ok = False
        out["reasons"].append(
            f"D does not beat QAOA L+1 ({Qnext['E']:.8g}) by {RESCUE_ENERGY}"
        )
    Lm = matched_qaoa_depth(L)
    out["matched_qaoa_L"] = Lm
    Qm = _cell(rows, n, Lm, "qaoa")
    if Qm is None:
        ok = False
        out["reasons"].append(f"missing matched-parameter QAOA at L={Lm}")
    elif float(D["E"]) >= float(Qm["E"]) - RESCUE_ENERGY:
        ok = False
        out["reasons"].append(
            f"D does not beat matched QAOA L={Lm} ({Qm['E']:.8g}) by {RESCUE_ENERGY}"
        )
        out["matched_qaoa_E"] = Qm["E"]
    elif Qm is not None:
        out["matched_qaoa_E"] = Qm["E"]
    gap = _second_best_gap(D)
    out["second_start_gap"] = gap
    if gap is None:
        ok = False
        out["reasons"].append("D has fewer than two finished starts")
    else:
        tol = max(REPRO_ABS, REPRO_FRAC * max(improve, 0.0))
        out["repro_tolerance"] = tol
        if gap > tol:
            ok = False
            out["reasons"].append(f"second D start is {gap:.6g} worse than the best; tolerance {tol:.6g}")
    # Equal-depth descriptive flag. This is not acceptance.
    out["equal_depth_ahead"] = bool(improve > 0.0 and dF > 0.0)
    out["robust"] = bool(ok and not out["reasons"])
    if out["robust"]:
        out["reasons"].append("passed every gate")
    return out


def build_verdict(rows: list[dict]) -> dict:
    cells = []
    ns = sorted({r["n"] for r in rows})
    Ls = sorted({r["L"] for r in rows})
    for n in ns:
        for L in Ls:
            if _cell(rows, n, L, "nov3_D") is None and _cell(rows, n, L, "qaoa") is None:
                continue
            cells.append(classify_cell(rows, n, L))
    robust = [c for c in cells if c.get("robust")]
    by_n = {}
    for n in ns:
        by_n[str(n)] = sorted({c["L"] for c in robust if c["n"] == n})
    program = all(len(by_n.get(str(n), [])) >= MIN_DEPTHS for n in (6, 12))
    return {
        "gates": {
            "absolute_energy": ABS_ENERGY,
            "residual_fraction": REL_RESIDUAL,
            "fidelity_absolute": FIDELITY_ABS,
            "ablation_energy": ABLATION_ENERGY,
            "qaoa_rescue_energy": RESCUE_ENERGY,
            "repro_absolute": REPRO_ABS,
            "repro_fraction_of_gain": REPRO_FRAC,
            "min_depths_per_n": MIN_DEPTHS,
            "requires_both_n": [6, 12],
            "requires_qaoa_Lplus1": True,
            "requires_matched_parameter_qaoa": True,
            "requires_drop_c2_and_drop_c3_and_shared_beta": True,
            "objective": "best energy over starts; fidelity is a separate gate, never the training objective",
        },
        "cells": cells,
        "robust_depths_by_n": by_n,
        "program_success": bool(program),
        "headline": (
            "Robust win under the pre-set gates."
            if program
            else "No robust win. Equal-depth differences, where present, did not clear every gate on both sizes."
        ),
    }


def _fmt(v, digits=6):
    if v is None:
        return "—"
    return f"{float(v):.{digits}g}"


def write_results_md(path: Path, rows: list[dict], targets: dict, verdict: dict, transfer: list, notes: list[str]) -> None:
    lines = []
    lines.append("# Round 1 direction D — shared-gamma native shell vs QAOA")
    lines.append("")
    lines.append(verdict["headline"])
    lines.append("")
    lines.append("Training minimized energy only. Fidelity is a post-hoc diagnostic of the best-energy state.")
    lines.append("Family D uses G = Hc, one gamma shared by every layer, and per-layer shell coefficients (c1, c2, c3).")
    lines.append("Split gamma and per-layer theta were not optimized.")
    lines.append("")
    lines.append("## Frozen fields")
    lines.append("")
    lines.append("| N | B | hex | entropy (bits) | E0 | gap | Emax | GS residual | P+ weight | R+ weight | H sha256 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, t in targets.items():
        lines.append(
            f"| {n} | {t['B']:.6f} | `{t['B_hex']}` | {t['entropy_bits']:.6f} | {t['E0']:.8f} | "
            f"{t['gap']:.3e} | {t['Emax']:.8f} | {t['gs_residual']:.2e} | "
            f"{t['parity']['P_plus_weight']:.8f} | {t['reflection']['R_plus_weight']:.8f} | "
            f"`{t['hashes']['H_dense_sha256'][:12]}` |"
        )
    lines.append("")
    lines.append("B maximizes the half-chain entropy of the full-space ground state. It is this run's numerical realization, not a published Pagano field. Alpha = 1.1, J0 = 1, open chain, J_ij = |i-j|^{-1.1}.")
    lines.append("The gap column is the full-space splitting to the P=-1, R=-1 partner. Both circuits stay in P=+1, R=+1, which contains the ground state. The gap to the next state in that sector is larger and is given in the notes.")
    lines.append("")
    lines.append("## Best energy and the fidelity of that state")
    lines.append("")
    lines.append("| N | L | method | params | E | E-E0 | F | eta | median E | starts | nfev | tag |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: (z["n"], z["L"], METHODS.index(z["method"]) if z["method"] in METHODS else 99)):
        if r.get("status") != "complete":
            lines.append(f"| {r.get('n')} | {r.get('L')} | {r.get('method')} | | | | | | | | | {r.get('status')} |")
            continue
        lines.append(
            f"| {r['n']} | {r['L']} | {r['method']} | {r['n_params']} | {r['E']:.8f} | {r['energy_error']:.6e} | "
            f"{r['F']:.6f} | {r['eta']:.4f} | {r['E_median']:.6f} | {r['n_finished']}/{r['n_starts']} | "
            f"{r['nfev_sum']} | {r.get('best_tag','')} |"
        )
    lines.append("")
    lines.append("## Effect sizes versus equal-depth QAOA")
    lines.append("")
    lines.append("Negative energy difference means the method has lower energy than QAOA. Positive fidelity difference means higher fidelity. These columns are measurements. The verdict section applies the gates.")
    lines.append("")
    lines.append("| N | L | method | E - E_QAOA | F - F_QAOA | fraction of QAOA residual closed |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: (z["n"], z["L"], z["method"])):
        if r.get("status") != "complete" or r["method"] == "qaoa":
            continue
        Q = _cell(rows, r["n"], r["L"], "qaoa")
        if Q is None:
            continue
        dE = float(r["E"]) - float(Q["E"])
        dF = float(r["F"]) - float(Q["F"])
        resid = float(Q["E"]) - float(Q["E0"])
        frac = (float(Q["E"]) - float(r["E"])) / resid if resid > 1e-12 else float("nan")
        lines.append(f"| {r['n']} | {r['L']} | {r['method']} | {dE:.6e} | {dF:.6e} | {frac:.4f} |")
    lines.append("")
    lines.append("## Measured reading")
    lines.append("")
    lines.append(
        "Program success requires the pre-set gates at two or more depths on both N=6 and N=12. "
        "That did not happen. N=12 native D is behind equal-depth QAOA from L=2 through L=6 on energy and on fidelity. "
        "Extra adiabatic seeds, differential evolution, and an unbounded random blitz did not reverse that ordering."
    )
    lines.append("")
    lines.append(
        "The cell flag `robust` is only the best-of-restarts gate. It is not a claim that a typical random start wins, "
        "and it is not program success. Compare the median energy with QAOA's best energy:"
    )
    lines.append("")
    lines.append("| N | L | best D | median D | best QAOA | median D minus best QAOA |")
    lines.append("|---|---|---|---|---|---|")
    for n in (6, 12):
        for L in range(1, 7):
            D = _cell(rows, n, L, "nov3_D")
            Q = _cell(rows, n, L, "qaoa")
            if D is None or Q is None or D.get("E_median") is None:
                continue
            lines.append(
                f"| {n} | {L} | {D['E']:.6f} | {D['E_median']:.6f} | {Q['E']:.6f} | "
                f"{float(D['E_median']) - float(Q['E']):.6f} |"
            )
    lines.append("")
    lines.append(
        "A positive last column means the median D start is worse than the best QAOA energy. "
        "At N=6 the best shell energy is ahead of equal-depth QAOA for L>=2, and L=3,4,5 also clear the matched-depth and ablation gates. "
        "Dropping c3 removes most of that N=6 gain. Dropping c2 does not. "
        "Zero-shot transfer of those N=6 shell angles onto N=12 collapses; kappa rescaling of the cost angle does not repair it. "
        "Shared-beta is the same ansatz as QAOA at L=1 and is behind independent-angle QAOA at every L>1."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Program success: **{verdict['program_success']}**.")
    lines.append("")
    lines.append("| N | L | equal-depth ahead | robust | why |")
    lines.append("|---|---|---|---|---|")
    for c in verdict["cells"]:
        why = "; ".join(c.get("reasons", []))
        lines.append(
            f"| {c['n']} | {c['L']} | {c.get('equal_depth_ahead')} | {c.get('robust')} | {why} |"
        )
    lines.append("")
    lines.append("## Zero-shot N=6 to N=12")
    lines.append("")
    lines.append("Angles trained at N=6 are evaluated on the N=12 Hamiltonian with no further training. `kappa` multiplies only cost angles by kappa(6)/kappa(12). Shell coefficients and mixer betas are copied. This rescaling is a J-only hypothesis, not a fitted transfer.")
    lines.append("")
    if transfer:
        lines.append("| L | method | kind | E | E-E0 | F |")
        lines.append("|---|---|---|---|---|---|")
        for t in transfer:
            lines.append(
                f"| {t['L']} | {t['method']} | {t['kind']} | {t['E']:.6f} | {t['energy_error']:.6e} | {t['F']:.6f} |"
            )
    else:
        lines.append("No transfer rows.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines))


def write_figures(fig_dir: Path, rows: list[dict], targets: dict) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # Entropy curves
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for n, t in targets.items():
        curve = t.get("coarse_curve") or []
        if not curve:
            continue
        Bs = [c["B"] for c in curve]
        Ss = [c["entropy_bits"] for c in curve]
        ax.plot(Bs, Ss, label=f"N={n}")
        ax.axvline(t["B"], linestyle="--", linewidth=1)
    ax.set_xlabel("B")
    ax.set_ylabel("half-chain entropy (bits)")
    ax.set_title("Entropy-maximizing field")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "entropy_vs_B.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    for n in sorted({r["n"] for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
        for method in METHODS:
            pts = [r for r in rows if r.get("status") == "complete" and r["n"] == n and r["method"] == method]
            pts = sorted(pts, key=lambda r: r["L"])
            if not pts:
                continue
            Ls = [r["L"] for r in pts]
            axes[0].plot(
                Ls,
                [max(float(r["energy_error"]), 1e-16) for r in pts],
                marker="o",
                color=COLORS[method],
                label=method,
            )
            axes[1].plot(Ls, [r["F"] for r in pts], marker="o", color=COLORS[method], label=method)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("depth L")
        axes[0].set_ylabel("energy error E - E0")
        axes[1].set_xlabel("depth L")
        axes[1].set_ylabel("fidelity of best-energy state")
        axes[0].set_title(f"N={n} energy error")
        axes[1].set_title(f"N={n} fidelity")
        axes[1].set_ylim(-0.02, 1.02)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        p = fig_dir / f"depth_curves_N{n}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    # Schedules at L=4 when present, else the deepest complete D.
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    for col, n in enumerate((6, 12)):
        for method, ax, title in (
            ("qaoa", axes[0, col], f"N={n} QAOA"),
            ("nov3_D", axes[1, col], f"N={n} native D"),
        ):
            candidates = [
                r
                for r in rows
                if r.get("status") == "complete" and r["n"] == n and r["method"] == method
            ]
            if not candidates:
                ax.set_title(title + " (missing)")
                continue
            prefer = [r for r in candidates if r["L"] == 4] or candidates
            r = max(prefer, key=lambda z: z["L"])
            x = np.asarray(r.get("x_canonical") or r["x_raw"], dtype=float)
            L = r["L"]
            s = np.linspace(0.0, 1.0, L) if L > 1 else np.array([0.0])
            if method == "qaoa":
                ax.plot(s, x[:L], marker="o", label="gamma")
                ax.plot(s, x[L:], marker="s", label="beta")
            else:
                shell = x[1:].reshape(L, 3)
                ax.plot(s, shell[:, 0], marker="o", label="c1")
                ax.plot(s, shell[:, 1], marker="s", label="c2")
                ax.plot(s, shell[:, 2], marker="^", label="c3")
                ax.axhline(x[0], color="k", linewidth=1, label=f"gamma={x[0]:.3f}")
            ax.set_title(f"{title} L={L}")
            ax.set_xlabel("normalized layer time")
            ax.legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "schedules.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))
    return written


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    header = [
        "n", "L", "method", "n_params", "E", "E0", "Emax", "energy_error", "F", "eta",
        "E_median", "n_starts", "n_finished", "nfev_sum", "grad_norm", "best_tag", "seconds", "status",
    ]
    lines = [",".join(header)]
    for r in sorted(rows, key=lambda z: (z.get("n", 0), z.get("L", 0), z.get("method", ""))):
        vals = []
        for k in header:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.12g}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")
