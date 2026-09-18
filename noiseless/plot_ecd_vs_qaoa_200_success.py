#!/usr/bin/env python3
"""Plot success rate for the matched-parameter ECD/QAOA @ 200-step bake-off."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "noiseless" / "results" / "fleet_ecd_vs_qaoa_20260918T060037Z_summary.json"
OUT_PNG = REPO / "noiseless" / "results" / "figures" / "ecd_vs_qaoa_200_success.png"
OUT_PDF = REPO / "noiseless" / "results" / "figures" / "ecd_vs_qaoa_200_success.pdf"

PARAMS = np.array([16, 24, 32])
DEPTH_LABELS = [r"$L^*=2$ / $p=8$", r"$L^*=3$ / $p=12$", r"$L^*=4$ / $p=16$"]
COLORS = {"ecd": "#0072B2", "qaoa_full": "#E69F00"}
LABELS = {"ecd": r"ECD ($U=\mathrm{bs}_{\pi/4}$, SPSA 200)", "qaoa_full": "QAOA-full (SPSA 200)"}


def load_rates() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read aggregate rates and compute SEM over the 20 Hamiltonians."""
    data = json.loads(SUMMARY.read_text())
    rates: dict[str, np.ndarray] = {}
    sems: dict[str, np.ndarray] = {}
    for arm in ("ecd", "qaoa_full"):
        means = []
        sem = []
        for n_params in PARAMS:
            cells = [
                cell["success_rate"]
                for cell in data["cells"]
                if cell["arm"] == arm and cell["n_params"] == int(n_params)
            ]
            if len(cells) != 20:
                raise ValueError(f"expected 20 Hamiltonians for {arm} @ {n_params}, got {len(cells)}")
            values = np.asarray(cells, dtype=float)
            means.append(values.mean())
            sem.append(values.std(ddof=1) / np.sqrt(values.size))
        rates[arm] = np.asarray(means)
        sems[arm] = np.asarray(sem)
    return rates, sems


def main() -> None:
    rates, sems = load_rates()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=False)
    x = np.arange(PARAMS.size)
    for arm, marker in (("ecd", "o"), ("qaoa_full", "s")):
        ax.errorbar(
            x,
            rates[arm],
            yerr=sems[arm],
            label=LABELS[arm],
            color=COLORS[arm],
            marker=marker,
            markersize=7,
            linewidth=2.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3.5,
            capthick=1.2,
            elinewidth=1.2,
            zorder=3,
        )

    ax.set_title(
        "Fixed SPSA budget (200 steps): ECD landscape improves with depth; QAOA-full degrades",
        loc="left",
        pad=12,
        fontweight="bold",
    )
    ax.set_xlabel("Matched parameter count", labelpad=10)
    ax.set_ylabel("Success rate")
    ax.set_xticks(x, [f"{p}\n{label}" for p, label in zip(PARAMS, DEPTH_LABELS)])
    ax.set_xlim(-0.15, x[-1] + 0.15)
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color="#B0B0B0", linewidth=0.7, alpha=0.32)
    ax.grid(axis="x", color="#B0B0B0", linewidth=0.5, alpha=0.16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015), frameon=True, framealpha=0.94, edgecolor="#D0D0D0")
    ax.text(
        0.99,
        0.98,
        "Error bars: SEM across 20 Hamiltonians",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#666666",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.83)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")
    for arm in ("ecd", "qaoa_full"):
        print(arm, "rates", rates[arm].tolist(), "sem", sems[arm].tolist())


if __name__ == "__main__":
    main()
