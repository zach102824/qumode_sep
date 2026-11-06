"""Plots for the Family C round-1 scan. Reads results/round1/*.csv."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEPTH_STYLE = {
    0: dict(color="black", marker="o", ls="--", label="L=0"),
    1: dict(color="crimson", marker="D", ls="--", label="L=1"),
    2: dict(color="royalblue", marker="x", ls="--", label="L=2"),
    3: dict(color="seagreen", marker="^", ls="--", label="L=3"),
    4: dict(color="darkturquoise", marker="v", ls="--", label="L=4"),
    5: dict(color="magenta", marker="s", ls="--", label="L=5"),
}


def _f(row, key, default=np.nan):
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def featured_gammas(sel_path: Path) -> dict[str, float]:
    import json
    sel = json.loads(sel_path.read_text())
    out = {}
    for name, block in sel.items():
        feat = block.get("featured") or {}
        if "Gamma" in feat:
            out[name] = float(feat["Gamma"])
    return out


def _series(qaoa, sign, gamma, method="qaoa_energy"):
    """dict L -> list of (N, F_reach, eta) for pure model."""
    buckets = defaultdict(list)
    for r in qaoa:
        if r["model"] != "pure" or r["sign_name"] != sign or r["method"] != method:
            continue
        if abs(_f(r, "Gamma") - gamma) > 1e-9:
            continue
        L = int(float(r["L"]))
        buckets[L].append((int(float(r["N"])), _f(r, "F_reach"), _f(r, "eta_full"), _f(r, "F_global")))
    return buckets


def plot_f_vs_invN(qaoa, sign, gamma, path: Path, title_note: str):
    buckets = _series(qaoa, sign, gamma)
    buckets[0] = _series(qaoa, sign, gamma, method="init").get(0, [])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    Ns_all = sorted({n for rows in buckets.values() for n, *_ in rows})
    for L, style in DEPTH_STYLE.items():
        rows = sorted(buckets.get(L, []), key=lambda t: t[0])
        if not rows:
            continue
        Ns = np.array([t[0] for t in rows], dtype=float)
        Fs = np.array([t[1] for t in rows], dtype=float)
        ax.plot(1.0 / Ns, Fs, **style)
    if Ns_all:
        ax.set_xticks([1.0 / n for n in Ns_all])
        ax.set_xticklabels([str(n) for n in Ns_all])
    ax.set_xlabel(r"system size $N$  (axis coordinate is $1/N$)")
    ax.set_ylabel(r"squared overlap $F$ with the reachable GS")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{sign}, $\\Gamma={gamma:g}$\n{title_note}", fontsize=11)
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_f_vs_L_over_N(qaoa, sign, gamma, path: Path):
    buckets = _series(qaoa, sign, gamma)
    buckets[0] = _series(qaoa, sign, gamma, method="init").get(0, [])
    # regroup by N
    by_n = defaultdict(list)
    for L, rows in buckets.items():
        for n, F, eta, Fg in rows:
            by_n[n].append((L, F))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for n, rows in sorted(by_n.items()):
        rows = sorted(rows)
        xs = [L / n for L, _ in rows]
        ys = [F for _, F in rows]
        ax.plot(xs, ys, marker="o", label=f"N={n}")
    ax.set_xlabel(r"depth / size  $L/N$")
    ax.set_ylabel(r"$F$ (energy-optimal QAOA)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{sign}, $\\Gamma={gamma:g}$: $F$ versus $L/N$")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_small_multiples(qaoa, sign, path: Path):
    gammas = sorted({_f(r, "Gamma") for r in qaoa if r["model"] == "pure" and r["sign_name"] == sign and r["method"] == "qaoa_energy"})
    # keep the pre-registered window, drop huge-N-only duplicates
    n = len(gammas)
    ncols = 4
    nrows = int(math_ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, g in zip(axes, gammas):
        buckets = _series(qaoa, sign, g)
        for L, style in DEPTH_STYLE.items():
            rows = sorted(buckets.get(L, []), key=lambda t: t[0])
            # small-multiples focus on the promised N<=12 scan; larger N is on the main panel
            rows = [t for t in rows if t[0] <= 12]
            if not rows:
                continue
            Ns = np.array([t[0] for t in rows], dtype=float)
            Fs = np.array([t[1] for t in rows])
            ax.plot(1.0 / Ns, Fs, color=style["color"], marker=style["marker"], ls="--", ms=4)
        if rows:
            Ns_all = sorted({t[0] for Lrows in buckets.values() for t in Lrows if t[0] <= 12})
            ax.set_xticks([1 / n for n in Ns_all])
            ax.set_xticklabels([str(n) for n in Ns_all], fontsize=8)
        ax.set_title(f"$\\Gamma={g:g}$", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(gammas):]:
        ax.axis("off")
    fig.suptitle(f"{sign}: energy-QAOA $F$ vs $1/N$ at every scanned $\\Gamma$ (N≤12)", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def math_ceil(x):
    return int(np.ceil(x))


def plot_regime(spec, sign, path: Path):
    rows = [r for r in spec if r["model"] == "pure" and r["sign_name"] == sign and int(float(r["N"])) in (6, 8, 10, 12)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharex=True)
    for n in (6, 8, 10, 12):
        sub = sorted([r for r in rows if int(float(r["N"])) == n], key=lambda r: _f(r, "Gamma"))
        g = [_f(r, "Gamma") for r in sub]
        axes[0].plot(g, [_f(r, "F0") for r in sub], marker="o", label=f"N={n}")
        axes[1].plot(g, [_f(r, "F_cat") for r in sub], marker="o", label=f"N={n}")
        axes[2].plot(g, [_f(r, "S_half_bits") for r in sub], marker="o", label=f"N={n}")
    axes[0].set_ylabel(r"$F_0=|\langle+|GS\rangle|^2$")
    axes[1].set_ylabel(r"even-cat overlap")
    axes[2].set_ylabel("half-cut entropy (bits)")
    for ax in axes:
        ax.set_xlabel(r"$\Gamma$")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{sign} ground-state regime (full H, even sector is the GS where the scan says so)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pure_vs_ctrl(overlap_json: Path, path: Path):
    import json
    rows = json.loads(overlap_json.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, sign in zip(axes, ("ferro", "af")):
        for n in (6, 8, 10, 12):
            sub = sorted(
                [r for r in rows if r["sign_name"] == sign and int(r["N"]) == n],
                key=lambda r: r["Gamma"],
            )
            ax.plot([r["Gamma"] for r in sub], [r["F_pure_vs_ctrl_gs"] for r in sub], marker="o", label=f"N={n}")
        ax.set_title(sign)
        ax.set_xlabel(r"$\Gamma$")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(r"$|\langle GS_{pure}|GS_{ctrl}\rangle|^2$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Pure distinct $p{=}4$ vs collective $N m_z^4$ (even GS)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_angles(angle_json: Path, sign_s: int, gamma: float, n: int, path: Path):
    import json
    blob = json.loads(angle_json.read_text())
    rec = None
    for b in blob:
        if b["model"] == "pure" and int(b["s"]) == sign_s and int(b["N"]) == n and abs(float(b["Gamma"]) - gamma) < 1e-9:
            rec = b
            break
    if rec is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, key, title in (
        (axes[0], "raw", "raw optimizer output"),
        (axes[1], "canonical", "folded into one period, sign-fixed"),
    ):
        for depth in rec["depths"]:
            if depth["method"] != "qaoa_energy":
                continue
            L = depth["L"]
            ys = depth["gammas_" + key] if key == "raw" else depth["gammas_canonical"]
            # key raw uses gammas_raw
            ys = depth["gammas_raw"] if key == "raw" else depth["gammas_canonical"]
            xs = np.arange(1, L + 1)
            ax.plot(xs, ys, marker="o", label=f"L={L}")
        ax.set_xlabel("layer k (k=1 acts first)")
        ax.set_ylabel(r"$\gamma_k$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"N={n}, s={sign_s}, $\\Gamma={gamma:g}$: $\\gamma$ schedules (not smoothed)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_baseline(qaoa, sign, gamma, path: Path):
    """Energy-QAOA F against anneal best-T and shared-gamma, N<=12."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method, fmt in (
        ("qaoa_energy", dict(color="magenta", marker="s")),
        ("shared_gamma", dict(color="gray", marker="o")),
        ("anneal_bestT", dict(color="black", marker="x")),
        ("anneal_matched", dict(color="darkorange", marker="+")),
    ):
        buckets = _series(qaoa, sign, gamma, method=method)
        # plot F vs L at N=10 if present else largest N<=12
        Ns = sorted({n for rows in buckets.values() for n, *_ in rows if n <= 12})
        if not Ns:
            continue
        nstar = 10 if 10 in Ns else Ns[-1]
        xs, ys = [], []
        for L, rows in buckets.items():
            for n, F, eta, Fg in rows:
                if n == nstar:
                    xs.append(L)
                    ys.append(F)
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], label=f"{method}", **fmt)
    ax.set_xlabel("depth L")
    ax.set_ylabel("F")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{sign} $\\Gamma={gamma:g}$: QAOA vs shared-$\\gamma$ vs anneal")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent / "results" / "round1"
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)
    qaoa = load_csv(root / "qaoa.csv")
    spec = load_csv(root / "spectrum.csv")
    feats = featured_gammas(root / "gamma_selection.json")
    import json
    sel = json.loads((root / "gamma_selection.json").read_text())
    for sign, g in feats.items():
        feat = sel[sign]["featured"]
        note = f"top GS-only score; positive on {feat['n_pass']}/4 sizes, mean {feat['mean_score']:.2f}"
        plot_f_vs_invN(qaoa, sign, g, figdir / f"F_vs_invN_{sign}.png", note)
        plot_f_vs_L_over_N(qaoa, sign, g, figdir / f"F_vs_L_over_N_{sign}.png")
        plot_small_multiples(qaoa, sign, figdir / f"F_small_multiples_{sign}.png")
        plot_regime(spec, sign, figdir / f"regime_{sign}.png")
        plot_baseline(qaoa, sign, g, figdir / f"baseline_{sign}.png")
        s = -1 if sign == "ferro" else 1
        # angle plot at N=8 (period is moderate and branch folding is visible)
        plot_angles(root / "angles.json", s, g, 8, figdir / f"angles_{sign}_N8.png")
    plot_pure_vs_ctrl(root / "pure_vs_collective_gs.json", figdir / "pure_vs_collective.png")
    print(f"wrote figures in {figdir}")


if __name__ == "__main__":
    main()
