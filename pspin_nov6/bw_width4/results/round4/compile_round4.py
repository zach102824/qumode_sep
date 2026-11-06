"""Compile Gamma=0.8 figures from saved tables. No QAOA."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/agent/repos/qumode_sep")
A = "origin/cursor/sparse-p3-qaoa-round1-13b4"
OUT = Path(__file__).resolve().parent


def show(path: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{A}:{path}"], text=True)


def a_rows():
    raw = show("pspin_nov6/baxter_wu_p3/runs/round3/qaoa_matched.csv")
    return list(csv.DictReader(raw.splitlines()))


def series_A(n, gamma=0.8):
    rows = [r for r in a_rows() if int(r["n"]) == n and abs(float(r["gamma"]) - gamma) < 1e-9]
    rows.sort(key=lambda r: int(r["L"]))
    return rows


def series_4x4():
    p = REPO / "pspin_nov6/bw_width4/results/round3/qaoa.csv"
    rows = [r for r in csv.DictReader(p.open()) if r["lx"] == "4" and r["ly"] == "4" and abs(float(r["Gamma"]) - 0.8) < 1e-9]
    rows.sort(key=lambda r: int(float(r["L"])))
    return rows


# Energy-selected F(L), including L=0 = |<+|GS>|^2 from the ground-state table.
F0 = {12: 0.07000895095119412, 15: 0.12653188850778593, 18: 0.012854966053563241, 16: 0.11528546}
GEOM = {12: "3×4", 15: "3×5", 16: "4×4", 18: "3×6"}
COLOR = {12: "#c0392b", 15: "#2471a3", 16: "#1e8449", 18: "#d68910"}


def energy_curve(n):
    if n == 16:
        rr = series_4x4()
        return [(int(float(r["L"])), float(r["F"])) for r in rr]
    rr = series_A(n)
    pts = [(0, F0[n])] + [(int(r["L"]), float(r["F"])) for r in rr]
    return pts


def main():
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    # Fig 1
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for n in (12, 15, 16, 18):
        pts = energy_curve(n)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=COLOR[n], ms=6, label=f"{GEOM[n]}, N={n}, energy-selected")
    # N=18 extra semantics at L=5
    ax.scatter([5], [0.4709952444238817], s=80, marker="D", color=COLOR[18], zorder=5,
               label="N=18 best-energy polish, F=0.471")
    ax.scatter([5.12], [0.5235870342914763], s=90, marker="o", facecolors="none",
               edgecolors=COLOR[18], linewidths=1.8, zorder=5,
               label="N=18 transferred 3×4 angles, F=0.524 (not best energy)")
    ax.set_xlabel("QAOA depth L  (L=0 is the initial |+⟩ state)")
    ax.set_ylabel("squared overlap with the full-space ground state")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(range(0, 7))
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Staggered Baxter–Wu at Γ = 0.8, unscaled\nSolid curves: energy-selected schedules. No smoothing.")
    fig.text(0.01, 0.01,
             "N=12 is 3×4 (branch A). 4×3 is the same N=12 Hamiltonian and is not drawn as another size.\n"
             "N=18 solid endpoint is the blind 8-restart search (F=0.420). The diamond is a lower energy; the open marker is a higher overlap.",
             fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figdir / "fig1_F_vs_L_gamma08.png", dpi=160)
    fig.savefig(figdir / "fig1_F_vs_L_gamma08.pdf")
    plt.close(fig)

    # Fig 2, no fit
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    Ns = np.array([12, 15, 16, 18], dtype=float)
    # L=1 energy-selected
    f1e = {
        12: 0.23603648873924069,
        15: 0.42626853662306574,
        16: 0.38541553,
        18: 0.07863314381547523,
    }
    # grid-checked L=1 fidelity (before/without treating local polish as the same object)
    f1_grid = {12: 0.26903548707903907, 15: 0.453413851545816, 16: 0.40779571, 18: 0.08988406713299907}
    # local fidelity maximization, best-found
    f1_loc = {12: 0.26949754462061465, 15: 0.4597174428674855, 16: 0.4129654, 18: 0.09522872210943426}
    f_deep_E = {12: 0.8949299000280136, 15: 0.9755065564447986, 16: 0.97336376, 18: 0.4709952444238817}
    x = 1.0 / Ns
    ax.scatter(x, [f1e[int(n)] for n in Ns], marker="o", s=50, color="#1a5276", label="L=1 energy-selected F", zorder=3)
    ax.scatter(x, [f1_grid[int(n)] for n in Ns], marker="s", s=45, color="#b9770e", label="L=1 best fidelity on the period grid", zorder=3)
    ax.scatter(x, [f1_loc[int(n)] for n in Ns], marker="^", s=50, color="#6c3483", label="L=1 fidelity-max after local search", zorder=3)
    ax.scatter(x, [f_deep_E[int(n)] for n in Ns], marker="D", s=50, color="#196f3d", label="deepest energy-selected F", zorder=3)
    ax.scatter([1 / 18], [0.5235870342914763], s=90, facecolors="none", edgecolors="#196f3d",
               linewidths=1.8, zorder=5, label="N=18 transferred F (not best energy)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{GEOM[int(n)]}\nN={int(n)}" for n in Ns])
    ax.set_xlabel("geometry  (horizontal coordinate is 1/N; no fit)")
    ax.set_ylabel("squared ground-state overlap")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Γ = 0.8 only. N=15 above N=12 is shown, not fitted.\n4×3 is not a second N=12 point.")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_F_vs_invN_gamma08.png", dpi=160)
    fig.savefig(figdir / "fig2_F_vs_invN_gamma08.pdf")
    plt.close(fig)

    # Fig 3: energy residual for all four; triangle error only where saved (4×4)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    E0 = {12: -17.08406777766207, 15: -19.86028869115473, 18: -25.559302228060673}
    E0[16] = float(series_4x4()[0]["E0"])
    for n in (12, 15, 18):
        pts = []
        # L=0 energy = -Gamma * N
        pts.append((0, (-0.8 * n - E0[n]) / abs(E0[n])))
        for r in series_A(n):
            pts.append((int(r["L"]), (float(r["energy"]) - E0[n]) / abs(E0[n])))
        xs, ys = zip(*pts)
        axes[0].plot(xs, ys, "-o", color=COLOR[n], label=f"{GEOM[n]} N={n}")
    rr = series_4x4()
    axes[0].plot([int(float(r["L"])) for r in rr],
                 [(float(r["energy"]) - E0[16]) / abs(E0[16]) for r in rr],
                 "-o", color=COLOR[16], label="4×4 N=16")
    axes[0].set_xlabel("L")
    axes[0].set_ylabel(r"$(E-E_0)/|E_0|$")
    axes[0].set_title("Relative energy error (saved E and E0)")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=8)

    # triangle error only 4x4
    axes[1].plot([int(float(r["L"])) for r in rr if int(float(r["L"])) >= 1],
                 [float(r["err_zzz_up"]) for r in rr if int(float(r["L"])) >= 1],
                 "-o", color=COLOR[16], label=r"4×4 $|\Delta\langle ZZZ\rangle_▲|$")
    axes[1].plot([int(float(r["L"])) for r in rr if int(float(r["L"])) >= 1],
                 [float(r["err_zz_connected"]) for r in rr if int(float(r["L"])) >= 1],
                 "--s", color="#1a5276", label="4×4 connected-ZZ error")
    axes[1].set_xlabel("L")
    axes[1].set_title("Triangle and correlator error\n(only 4×4 saved these per depth)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Γ = 0.8. ⟨X⟩ error was not stored and is not plotted.", fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_errors_vs_L_gamma08.png", dpi=160)
    fig.savefig(figdir / "fig3_errors_vs_L_gamma08.pdf")
    plt.close(fig)
    print("wrote", figdir)


if __name__ == "__main__":
    main()
