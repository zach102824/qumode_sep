"""Tables for family H. The relaxed gamma search is not reported as a ceiling."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_round2 import matched_p


def _rows(rows, method, L=None):
    out = [r for r in rows if r.get("method") == method and r.get("status") == "complete"]
    if L is not None:
        out = [r for r in out if r.get("L") == L]
    return out


def _one(rows, method, L):
    hit = _rows(rows, method, L)
    return hit[-1] if hit else None


def write_report(results: Path, rows: list, notes: list, spec: dict, meta: dict) -> None:
    results = Path(results)
    fig = results / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Round 2 family H — shared-gamma homotopy of full-K A")
    lines.append("")
    lines.append("Legal candidates use one shared gamma and one shared theta_K. The independent-gamma homotopy stage is an initializer only. It is not a certified ceiling and it is not a legal ansatz.")
    lines.append("")
    lines.append("## Frozen target")
    lines.append("")
    lines.append(f"- Manifest SHA256 `{meta.get('manifest_sha256')}`")
    lines.append(f"- Hc diagonal SHA256 (little-endian float64) `{meta.get('hc_sha256_le')}`")
    lines.append(f"- N=12 B={spec.get('B', meta.get('B_hex'))} full-space gap={spec.get('full_space_gap')} accessible P=R=+1 gap={spec.get('accessible_PR_plus_gap')}")
    lines.append(f"- E0={spec.get('E0')} Emax={spec.get('Emax')} entropy_bits={spec.get('entropy_bits')} GS residual={spec.get('gs_residual')}")
    lines.append(f"- FK RMS scale rms(J)/rms(K)={meta.get('fk_scale')} (full upper triangle, including zeros).")
    lines.append("- The full-space gap is the split to the P=-1, R=-1 partner. The accessible gap is larger. Fidelity uses the full-space ground state.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| method | L | P | E | E-E0 | F | median E | starts | nfev | tag |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: (z.get("L", 0), z.get("method", ""))):
        lines.append(
            f"| {r['method']} | {r['L']} | {len(r.get('x_raw') or [])} | {r['E']:.6f} | {r['energy_error']:.6e} | "
            f"{r['F']:.4f} | {r['E_median']:.6f} | {r['n_finished']}/{r['n_starts']} | {r['nfev_sum']} | {r.get('best_tag','')} |"
        )
    lines.append("")
    lines.append("## Family H versus QAOA")
    lines.append("")
    lines.append("Candidate is H_homotopy for L>1 and H_plain for L=1, where the homotopy is vacuous. H_plain is the equally budgeted shared-gamma bootstrap, reported separately. Negative energy difference means the row has lower energy than that QAOA depth.")
    lines.append("")
    lines.append("| L | candidate | E_c - E_qaoa(L) | F_c - F_qaoa(L) | vs L+1 | vs ceil(P/2) | plain - qaoa(L) | theta0 - qaoa(L) | median candidate - best QAOA(L) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for L in range(1, 7):
        cand = _one(rows, "H_homotopy", L) or _one(rows, "H_plain", L)
        q = _one(rows, "qaoa", L)
        qp = _one(rows, "qaoa", L + 1)
        qm = _one(rows, "qaoa", matched_p(L))
        plain = _one(rows, "H_plain", L)
        th = _one(rows, "thetaK0", L)
        if cand is None or q is None:
            lines.append(f"| {L} | missing | | | | | | | |")
            continue
        def dE(a, b):
            if a is None or b is None:
                return "—"
            return f"{a['E'] - b['E']:.4f}"

        def dF(a, b):
            if a is None or b is None:
                return "—"
            return f"{a['F'] - b['F']:.4f}"

        med = cand["E_median"] - q["E"]
        lines.append(
            f"| {L} | {cand['method']} | {dE(cand, q)} | {dF(cand, q)} | {dE(cand, qp)} | {dE(cand, qm)} | "
            f"{dE(plain, q)} | {dE(th, q)} | {med:.4f} |"
        )
    lines.append("")
    lines.append("Equal-L differences are one ledger. Parameter-matched QAOA is p=ceil((3L+2)/2). A lower energy at equal L is not by itself a quantum-advantage claim, and a loss at matched parameter count is not a reason to hide the equal-L numbers.")
    lines.append("")
    lines.append("## Measured reading")
    lines.append("")
    lines.append("Homotopy did not beat the plain shared-gamma bootstrap. At every N=12 depth the legal homotopy energy matches H_plain. The independent-gamma stage reached a lower relaxed energy and the exact projection back to one gamma did not keep that gain. Those relaxed numbers are initializers. They are not legal candidates and they are not a certified ceiling: a failed or successful local relaxed search does not prove what shared gamma can or cannot reach.")
    lines.append("")
    lines.append("Against equal-depth QAOA, the legal full-K ansatz has lower energy and higher fidelity at L=1 through L=6. The margin survives the QAOA rescue top-ups, which added many starts without lowering the best QAOA energy. At L=6 the plain median (-8.160) is still below the best equal-depth QAOA energy (-8.057).")
    lines.append("")
    lines.append("Against p=ceil(P/2) the ordering reverses: that QAOA depth has lower energy at every L from 1 to 6. Several equal-L wins also remain ahead of QAOA at p=L+1 (L=1,3,4,5,6). L=2 does not. Beating p=L, or even p=L+1, is not a seal under the stricter matched-parameter ledger.")
    lines.append("")
    lines.append("theta_K=0 stays well behind full-K and behind equal-depth QAOA for L>=2, in line with round-1 native D on this target. One frozen Gaussian K with the same RMS, and a smaller start budget, stays near theta_K=0 rather than near FK. That is evidence about this draw, not a full random-matrix survey.")
    lines.append("")
    lines.append("Legal angles are one gamma and one theta_K. From H_plain:")
    lines.append("")
    for L in range(1, 7):
        plain = _one(rows, "H_plain", L)
        if plain is None:
            continue
        x = plain["x_raw"]
        lines.append(f"- L={L}: gamma={x[0]:.6f}, theta_K={x[1]:.6f}")
    lines.append("")
    lines.append("| L | P | H energy error | QAOA L | QAOA L+1 | QAOA ceil(P/2) | H F | QAOA L F |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for L in range(1, 7):
        cand = _one(rows, "H_homotopy", L) or _one(rows, "H_plain", L)
        q = _one(rows, "qaoa", L)
        qp = _one(rows, "qaoa", L + 1)
        qm = _one(rows, "qaoa", matched_p(L))
        if cand is None or q is None:
            continue
        lines.append(
            f"| {L} | {3*L+2} | {cand['energy_error']:.4f} | {q['energy_error']:.4f} | "
            f"{qp['energy_error']:.4f} | {qm['energy_error']:.4f} | {cand['F']:.4f} | {q['F']:.4f} |"
            if qp is not None and qm is not None
            else f"| {L} | partial |"
        )
    lines.append("")
    # curves
    fig_p, ax = plt.subplots(figsize=(7.2, 4.2))
    for method, color in (
        ("qaoa", "#1f4e79"),
        ("H_plain", "#7f8c8d"),
        ("H_homotopy", "#c0392b"),
        ("thetaK0", "#27ae60"),
        ("randomK", "#8e44ad"),
    ):
        pts = sorted(_rows(rows, method), key=lambda r: r["L"])
        if not pts:
            continue
        ax.plot([r["L"] for r in pts], [max(r["energy_error"], 1e-8) for r in pts], marker="o", color=color, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("depth")
    ax.set_ylabel("energy error E-E0")
    ax.set_title("N=12 energy error")
    ax.legend(fontsize=8)
    fig_p.tight_layout()
    fig_p.savefig(fig / "energy_error_N12.png", dpi=140)
    plt.close(fig_p)

    lines.append("## Notes")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    (results / "RESULTS.md").write_text("\n".join(lines))
    (results / "summary.json").write_text(json.dumps({"notes": notes, "rows": rows}, indent=2, default=str))
    # csv
    header = ["method", "L", "E", "energy_error", "F", "E_median", "nfev_sum", "n_starts", "best_tag"]
    csv = [",".join(header)]
    for r in rows:
        csv.append(",".join(str(r.get(k, "")) for k in header))
    (results / "summary.csv").write_text("\n".join(csv) + "\n")
