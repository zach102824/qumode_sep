"""Figures and the round-3 F table."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _last(rows, method, L):
    hits = [r for r in rows if r.get("method") == method and r.get("L") == L and r.get("status") == "complete"]
    return hits[-1] if hits else None


def matched_p(L: int) -> int:
    return (3 * L + 2 + 1) // 2


def write_report(results: Path, rows: list, spec: dict) -> None:
    results = Path(results)
    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Round 3 family F — tied polynomial of Hc")
    lines.append("")
    lines.append("Legal generator `G = Hc + theta_K (FK + Q4) / rK = Hc + theta_K / (2 rK) (Hc^2 - c0 I)`. One shared gamma and one shared theta_K. This is a function of Hc. Free theta_Q is a separate diagnostic and is not f(Hc).")
    lines.append("")
    lines.append("## Target and support")
    lines.append("")
    lines.append(f"- Manifest SHA256 `{spec.get('manifest_sha256', '14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9')}`")
    lines.append(f"- Hc diagonal SHA256 `{spec.get('hc_sha256_le')}`")
    lines.append(f"- N=12 B={spec.get('B')} E0={spec.get('E0')} full-space gap={spec.get('full_space_gap')} accessible gap={spec.get('accessible_PR_plus_gap')}")
    lines.append(f"- rK={spec.get('rK')} c0={spec.get('c0')} two-body supports={spec.get('n_pairs')} four-body supports={spec.get('n_fourbody')}")
    lines.append("- The 495 four-body supports are charged. Being a polynomial of Hc is not a promise that the hardware cost equals two-body A.")
    lines.append("- Full-space gap is the P=-1, R=-1 partner. Fidelity uses the full-space ground state.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| method | L | P | E | E-E0 | F | Var(H) | median | IQR | nfev | tag |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda z: (z.get("L", 0), str(z.get("method")))):
        iqr = r.get("E_iqr")
        iqr_s = "—" if iqr is None else f"{iqr:.4f}"
        var = r.get("variance_H")
        var_s = "—" if var is None else f"{var:.4f}"
        med = r.get("E_median")
        med_s = "—" if med is None else f"{med:.4f}"
        lines.append(
            f"| {r['method']} | {r['L']} | {r.get('n_params','')} | {r['E']:.6f} | {r['energy_error']:.4e} | "
            f"{r['F']:.4f} | {var_s} | {med_s} | {iqr_s} | {r.get('nfev_sum')} | {r.get('best_tag','')} |"
        )
    lines.append("")
    lines.append("## Ledgers")
    lines.append("")
    lines.append("Negative `E_F - E_other` means F has lower energy. Equal depth and matched depth are separate.")
    lines.append("")
    lines.append("| L | E_F-E_A | E_F-E_Q(L) | F_F-F_Q(L) | E_F-E_Q(L+1) | E_F-E_Q(match) | F_F-F_Q(match) |")
    lines.append("|---|---|---|---|---|---|---|")
    for L in range(1, 7):
        f = _last(rows, "F", L)
        a = _last(rows, "A", L)
        q = _last(rows, "qaoa", L)
        qp = _last(rows, "qaoa", L + 1)
        qm = _last(rows, "qaoa", matched_p(L))
        if f is None or q is None:
            lines.append(f"| {L} | missing | | | | | |")
            continue

        def de(other):
            return "—" if other is None else f"{f['E'] - other['E']:.4f}"

        def df(other):
            return "—" if other is None else f"{f['F'] - other['F']:.4f}"

        lines.append(f"| {L} | {de(a)} | {de(q)} | {df(q)} | {de(qp)} | {de(qm)} | {df(qm)} |")
    lines.append("")
    lines.append("A is the round-2 full-K replay (FK only, same pair coefficients and same P). Shuffle4 permutes the 495 four-body coefficients with seed 31108 and keeps FK. thetaK0 is G=Hc. sameG and freeQ4 are diagnostics. freeQ4 is not f(Hc).")
    lines.append("")
    lines.append("Polishing F from the stored A angles did not lower F's energy at any depth. The tied optimum is a different point, with a smaller |theta_K| than A.")
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    series = (
        ("F", "#c0392b"),
        ("A", "#1f4e79"),
        ("qaoa", "#7f8c8d"),
        ("thetaK0", "#27ae60"),
        ("shuffle4", "#e67e22"),
        ("freeQ4", "#8e44ad"),
    )
    for method, color in series:
        pts = sorted((r for r in rows if r.get("method") == method), key=lambda r: r["L"])
        if not pts:
            continue
        axes[0].plot([r["L"] for r in pts], [max(r["energy_error"], 1e-8) for r in pts], marker="o", color=color, label=method)
        axes[1].plot([r["L"] for r in pts], [r["F"] for r in pts], marker="o", color=color, label=method)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("depth")
    axes[0].set_ylabel("energy error")
    axes[1].set_xlabel("depth")
    axes[1].set_ylabel("full-GS fidelity")
    axes[0].set_title("N=12 energy error")
    axes[1].set_title("N=12 fidelity")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "f_vs_qaoa_N12.png", dpi=140)
    plt.close(fig)

    (results / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (results / "summary.json").write_text(json.dumps({"rows": rows, "spec": {k: spec[k] for k in spec if k != "gs"}}, indent=2, default=str))
