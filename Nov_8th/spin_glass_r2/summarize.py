"""Builds RESULTS tables from cells/*.json. Reads only energy-selected cells."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "spin_glass_r1"
FAMS = ("KM", "K", "nodata", "qaoa")


def load():
    cells, metas = [], []
    for p in sorted((HERE / "cells").glob("n*_k*.json")):
        d = json.loads(p.read_text())
        cells += d["cells"]
        metas.append(d["meta"])
    return cells, metas


def r1_fsub():
    return {(r["N"], r["k"], r["family"]): r["F_sub_1e-6"]
            for r in json.loads((R1 / "subspace_fidelity.json").read_text())}


def main():
    cells, metas = load()
    C = {(c["N"], c["k"], c["family"], c["L"]): c for c in cells}
    out = ["# Spin glass round 2: results", "",
           "Energy-selected states only. F_sub is the weight in the eigenspace within 1e-6 of E0.",
           "Selection used energy alone, over all round-2 starts plus the round-1 selected state.", ""]
    out.append("## Mean F_sub over k = 0..3 by N and L")
    out.append("")
    for N in (6, 9, 12):
        out.append(f"### N={N}")
        out.append("")
        out.append("| L | " + " | ".join(FAMS) + " |")
        out.append("|---|" + "---|" * len(FAMS))
        for L in range(1, 7):
            vals = []
            for f in FAMS:
                xs = [C[(N, k, f, L)]["F_sub"] for k in range(4) if (N, k, f, L) in C]
                vals.append(f"{np.mean(xs):.3f}" + ("" if len(xs) == 4 else f" (n={len(xs)})") if xs else "missing")
            out.append(f"| {L} | " + " | ".join(vals) + " |")
        out.append("")
    out.append("## Per-instance F_sub (KM / K / nodata / QAOA); * marks a cell not flagged converged")
    out.append("")
    for N in (6, 9, 12):
        out.append(f"### N={N}")
        out.append("")
        out.append("| k | L | KM | K | nodata | QAOA | E_KM - E_QAOA | params KM/QAOA |")
        out.append("|---|---|---|---|---|---|---|---|")
        for k in range(4):
            for L in range(1, 7):
                row = []
                for f in FAMS:
                    c = C.get((N, k, f, L))
                    row.append("missing" if c is None else f"{c['F_sub']:.3f}" + ("" if c["converged"] else "*"))
                km, q = C.get((N, k, "KM", L)), C.get((N, k, "qaoa", L))
                dE = f"{km['E'] - q['E']:+.4f}" if km and q else ""
                out.append(f"| {k} | {L} | " + " | ".join(row) + f" | {dE} | {3 + 3 * L}/{2 * L} |")
        out.append("")
    out.append("## Round 1 versus round 2 at L=6 (F_sub)")
    out.append("")
    out.append("| N | k | KM r1 | KM r2 | K r1 | K r2 | QAOA r1 | QAOA r2 |")
    out.append("|---|---|---|---|---|---|---|---|")
    old = r1_fsub()
    for N in (6, 9, 12):
        for k in range(4):
            vals = []
            for f in ("KM", "K", "qaoa"):
                c = C.get((N, k, f, 6))
                vals += [f"{old.get((N, k, f), float('nan')):.3f}", "missing" if c is None else f"{c['F_sub']:.3f}"]
            out.append(f"| {N} | {k} | " + " | ".join(vals) + " |")
    out.append("")
    out.append("## Frozen claim rule")
    out.append("")
    verdict = {}
    for f in ("KM", "K"):
        ok_all = True
        for N in (6, 9, 12):
            for L in (5, 6):
                wins = 0
                complete = True
                for k in range(4):
                    a, q = C.get((N, k, f, L)), C.get((N, k, "qaoa", L))
                    if a is None or q is None:
                        complete = False
                        continue
                    if (a["F_sub"] >= q["F_sub"] + 0.05 and a["converged"] and q["converged"]
                            and a["E"] <= q["E"] + 1e-9):
                        wins += 1
                good = complete and wins >= 3
                ok_all &= good
                out.append(f"- {f} vs QAOA, N={N}, L={L}: qualifying instances {wins}/4"
                           + ("" if complete else " (incomplete cells)") + (" pass" if good else " fail"))
        verdict[f] = ok_all
        out.append(f"- **{f}: {'reliable equal-layer advantage' if ok_all else 'no reliable advantage under the frozen rule'}**")
        out.append("")
    out.append("## Optimizer accounting")
    out.append("")
    out.append("| N | family | starts | evals | capped starts | converged cells / 24 | energy ties KM>=K violated |")
    out.append("|---|---|---|---|---|---|---|")
    for N in (6, 9, 12):
        for f in FAMS:
            cs = [c for c in cells if c["N"] == N and c["family"] == f]
            viol = ""
            if f == "KM":
                viol = str(sum(1 for c in cs if (N, c["k"], "K", c["L"]) in C and c["E"] > C[(N, c["k"], "K", c["L"])]["E"] + 1e-9))
            out.append(f"| {N} | {f} | {sum(c['n_starts'] for c in cs)} | {sum(c['n_eval'] for c in cs)} | "
                       f"{sum(c['n_capped'] for c in cs)} | {sum(c['converged'] for c in cs)} | {viol} |")
    out.append("")
    mono = [(c["N"], c["k"], c["family"], c["L"]) for c in cells
            if (c["N"], c["k"], c["family"], c["L"] - 1) in C and c["E"] > C[(c["N"], c["k"], c["family"], c["L"] - 1)]["E"] + 1e-9]
    out.append(f"Cells where best energy rose from L-1 to L: {len(mono)} {mono}")
    out.append("")
    out.append(f"Cells whose selected state is the preserved round-1 state: "
               f"{[(c['N'], c['k'], c['family'], c['L']) for c in cells if c['source'] == 'r1_preserved']}")
    out.append("")
    out.append("## Secondary: F_even and single-vector F at L=6")
    out.append("")
    out.append("| N | k | n_sub | even ground in window | KM F_even | QAOA F_even | KM F_single | QAOA F_single | KM E-E0 | QAOA E-E0 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    mm = {(m["N"], m["k"]): m for m in metas}
    for N in (6, 9, 12):
        for k in range(4):
            a, q = C.get((N, k, "KM", 6)), C.get((N, k, "qaoa", 6))
            m = mm.get((N, k), {})
            if a and q:
                out.append(f"| {N} | {k} | {m.get('n_sub')} | {m.get('even_ground_in_window')} | {a['F_even']:.3f} | {q['F_even']:.3f} | "
                           f"{a['F_single']:.3f} | {q['F_single']:.3f} | {a['E'] - a['E0']:.4f} | {q['E'] - q['E0']:.4f} |")
    out.append("")
    out.append("## Jobs")
    out.append("")
    for m in metas:
        out.append(f"- N={m['N']} k={m['k']}: {m['status']}; grad check max {max(v for kk, v in m['grad_check'].items() if kk != 'r1_r2_KM_overlap'):.1e}, "
                   f"r1/r2 KM overlap {m['grad_check']['r1_r2_KM_overlap']:.12f}")
    (HERE / "RESULTS.md").write_text("\n".join(out) + "\n")
    (HERE / "selected.json").write_text(json.dumps(cells) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, N in zip(axes, (6, 9, 12)):
        for f, mk in (("KM", "o"), ("K", "s"), ("nodata", "+"), ("qaoa", "x")):
            xs, ys = [], []
            for L in range(1, 7):
                v = [C[(N, k, f, L)]["F_sub"] for k in range(4) if (N, k, f, L) in C]
                if len(v) == 4:
                    xs.append(L); ys.append(np.mean(v))
            ax.plot(xs, ys, marker=mk, label=f)
        ax.set_title(f"N={N}"); ax.set_xlabel("L")
    axes[0].set_ylabel("mean F_sub (k=0..3)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "fsub_vs_L.png", dpi=120)
    print("\n".join(out[:60]))
    print(verdict)


if __name__ == "__main__":
    main()
