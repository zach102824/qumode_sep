"""n=6 L=4 Γ=0.5: fidelity vs LBFGS iteration for free QAOA vs shared-γ+β (+ shared-γ+φ).

Logs train/hold MedF every round; writes CSV + PNG under runs/compare/.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG.parent))
sys.path.insert(0, str(PKG))

from pspin import make_dataset, hc_torch, gs_torch  # noqa: E402
from circuit_snap import make_ansatz, fidelity, B_SHELL  # noqa: E402
from circuit_fourier import shell_phi_from_betas, init_fourier_angles  # noqa: E402

OUT = PKG / "runs/compare"
CSV_PATH = OUT / "n6_L4_fidelity_vs_iter.csv"
PNG_PATH = OUT / "n6_L4_fidelity_vs_iter.png"

n, L, Gamma = 6, 4, 0.5
B_train, B_hold, rounds, batch = 64, 32, 20, 32
SEEDS = (0, 1)  # seed1 if quick


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fid-vs-iter]", *a, flush=True)


def pack(insts, device):
    return [(hc_torch(it, device), gs_torch(it, device), it.E0) for it in insts]


@torch.no_grad()
def medF(sim, gamma_vec, phi, packed):
    Fs = [float(fidelity(sim, Hc, gs, gamma_vec, phi).cpu()) for Hc, gs, _ in packed]
    arr = np.asarray(Fs, dtype=np.float64)
    return float(np.median(arr)), float(arr.mean())


def expand_g(g_scalar, L):
    return g_scalar.expand(L).contiguous()


def expand_phi_shared(phi4, L):
    return phi4.unsqueeze(0).expand(L, -1).contiguous()


def train_curve(method: str, seed: int, trains, holds, sim, device):
    """Return list of dicts per round including init (round=-1 / r=0 after first step).

    Methods:
      free_qaoa: γ[L] + β[L]
      shared_gb: 1 γ + β[L]
      shared_gphi: 1 γ + φ[4] shared across layers
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    beta0, gamma0 = init_fourier_angles(L, seed=seed)

    if method == "free_qaoa":
        gamma = torch.nn.Parameter(gamma0.clone().to(device))
        beta = torch.nn.Parameter(beta0.clone().to(device))
        params = [gamma, beta]
        n_params = 2 * L

        def angles():
            return gamma, shell_phi_from_betas(beta)

        def angles_det():
            return gamma.detach(), shell_phi_from_betas(beta.detach())

    elif method == "shared_gb":
        beta = torch.nn.Parameter(beta0.clone().to(device))
        g_scalar = torch.nn.Parameter(
            torch.tensor(float(gamma0.mean()), dtype=torch.float32, device=device)
        )
        params = [beta, g_scalar]
        n_params = L + 1

        def angles():
            return expand_g(g_scalar, L), shell_phi_from_betas(beta)

        def angles_det():
            return expand_g(g_scalar.detach(), L), shell_phi_from_betas(beta.detach())

    elif method == "shared_gphi":
        # cold init: mean gamma + random-ish shell from beta0 projected shared
        g_scalar = torch.nn.Parameter(
            torch.tensor(float(gamma0.mean()), dtype=torch.float32, device=device)
        )
        phi_layers0 = shell_phi_from_betas(beta0.to(device))
        phi4 = torch.nn.Parameter(phi_layers0.mean(dim=0).detach().clone())
        params = [g_scalar, phi4]
        n_params = 1 + B_SHELL

        def angles():
            return expand_g(g_scalar, L), expand_phi_shared(phi4, L)

        def angles_det():
            return expand_g(g_scalar.detach(), L), expand_phi_shared(phi4.detach(), L)

    else:
        raise ValueError(method)

    hist = []
    # init (before any step) as iteration 0
    with torch.no_grad():
        gvec, phi = angles_det()
        tr_med, tr_mean = medF(sim, gvec, phi, trains)
        ho_med, ho_mean = medF(sim, gvec, phi, holds)
    hist.append(
        {
            "method": method,
            "seed": seed,
            "iter": 0,
            "train_med": tr_med,
            "train_mean": tr_mean,
            "hold_med": ho_med,
            "hold_mean": ho_mean,
            "n_params": n_params,
            "best_hold": ho_med,
        }
    )
    log(
        f"{method} seed{seed} iter0 (init) tr={tr_med:.4f} hold={ho_med:.4f} np={n_params}"
    )

    best_hold = ho_med
    for r in range(rounds):
        opt = torch.optim.LBFGS(
            params,
            lr=0.25,
            max_iter=8,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        idx = rng.choice(len(trains), size=min(batch, len(trains)), replace=False)

        def closure():
            opt.zero_grad()
            gvec, phi = angles()
            losses = [
                1.0 - fidelity(sim, trains[int(j)][0], trains[int(j)][1], gvec, phi)
                for j in idx
            ]
            lt = torch.stack(losses).mean()
            lt.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return lt

        try:
            opt.step(closure)
        except Exception as e:
            log(f"{method} seed{seed} r{r} skip {e}")
            continue

        with torch.no_grad():
            gvec, phi = angles_det()
            tr_med, tr_mean = medF(sim, gvec, phi, trains)
            ho_med, ho_mean = medF(sim, gvec, phi, holds)
        best_hold = max(best_hold, ho_med)
        it = r + 1  # after round r
        hist.append(
            {
                "method": method,
                "seed": seed,
                "iter": it,
                "train_med": tr_med,
                "train_mean": tr_mean,
                "hold_med": ho_med,
                "hold_mean": ho_mean,
                "n_params": n_params,
                "best_hold": best_hold,
            }
        )
        if it % 5 == 0 or it == rounds:
            log(
                f"{method} seed{seed} iter{it} tr={tr_med:.4f} hold={ho_med:.4f} "
                f"best={best_hold:.4f}"
            )
    return hist


def iters_to_frac(hist, frac=0.95, key="hold_med"):
    """First iter where best-so-far hold reaches frac * final best hold."""
    finals = [h["best_hold"] for h in hist]
    final = finals[-1]
    target = frac * final
    for h in hist:
        if h["best_hold"] >= target - 1e-12:
            return h["iter"], target, final
    return hist[-1]["iter"], target, final


def plateau_iter(hist, window=3, tol=0.005, key="hold_med"):
    """First iter where best_hold improves < tol over next `window` iters (or end)."""
    bests = [h["best_hold"] for h in hist]
    for i in range(len(bests)):
        j = min(i + window, len(bests) - 1)
        if bests[j] - bests[i] < tol and i >= 1:
            # also require we're near final
            if bests[i] >= 0.95 * bests[-1]:
                return hist[i]["iter"]
    # fallback: first time at 95%
    it95, _, _ = iters_to_frac(hist, 0.95)
    return it95


def main():
    device = torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"build data n={n} Γ={Gamma} B_train={B_train} B_hold={B_hold}")
    tr = make_dataset(n, Gamma, B_train, seed0=1000 * n + int(round(10 * Gamma)) + 0)
    ho = make_dataset(n, Gamma, B_hold, seed0=1000 * n + int(round(10 * Gamma)) + 50_000)
    trains = pack(tr, device)
    holds = pack(ho, device)
    sim = make_ansatz(n, L, device)

    methods = ["free_qaoa", "shared_gb", "shared_gphi"]
    all_rows = []
    summaries = []

    for method in methods:
        for seed in SEEDS:
            t0 = time.time()
            hist = train_curve(method, seed, trains, holds, sim, device)
            dt = time.time() - t0
            all_rows.extend(hist)
            it95, target, final = iters_to_frac(hist, 0.95)
            plat = plateau_iter(hist)
            summaries.append(
                {
                    "method": method,
                    "seed": seed,
                    "final_hold": final,
                    "final_hold_last": hist[-1]["hold_med"],
                    "it95": it95,
                    "target95": target,
                    "plateau_iter": plat,
                    "seconds": dt,
                    "n_params": hist[0]["n_params"],
                }
            )
            log(
                f"DONE {method} seed{seed}: final_best_hold={final:.4f} "
                f"it95={it95} plateau≈{plat} ({dt:.1f}s)"
            )

    # CSV
    fields = [
        "method",
        "seed",
        "iter",
        "train_med",
        "train_mean",
        "hold_med",
        "hold_mean",
        "best_hold",
        "n_params",
    ]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in fields})
    log(f"wrote {CSV_PATH}")

    if not HAS_MPL:
        log("matplotlib missing — skip PNG (CSV already written); plot later with system python")
    else:
        # Plot: hold MedF vs iteration (current, not best-so-far), both methods; seed0 solid, seed1 dashed
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        colors = {
            "free_qaoa": "#1f77b4",
            "shared_gb": "#ff7f0e",
            "shared_gphi": "#2ca02c",
        }
        labels = {
            "free_qaoa": r"free QAOA $\gamma_\ell+\beta_\ell$ (8p)",
            "shared_gb": r"shared $\gamma$ + $\beta_\ell$ (5p)",
            "shared_gphi": r"shared $\gamma$ + $\phi_k$ (5p)",
        }
        styles = {0: "-", 1: "--"}

        for method in methods:
            for seed in SEEDS:
                rows = [r for r in all_rows if r["method"] == method and r["seed"] == seed]
                xs = [r["iter"] for r in rows]
                ys = [r["hold_med"] for r in rows]
                ax.plot(
                    xs,
                    ys,
                    styles[seed],
                    color=colors[method],
                    lw=2.0 if seed == 0 else 1.4,
                    alpha=0.95 if seed == 0 else 0.75,
                    label=f"{labels[method]} seed{seed}",
                )

        # mark plateau / 95% for seed0 of each method
        ymin, ymax = ax.get_ylim() if False else (None, None)
        for s in summaries:
            if s["seed"] != 0:
                continue
            c = colors[s["method"]]
            it95 = s["it95"]
            final = s["final_hold"]
            ax.axvline(it95, color=c, ls=":", alpha=0.55, lw=1.2)
            ax.scatter(
                [it95],
                [final * 0.95],
                color=c,
                s=40,
                zorder=5,
                marker="o",
                edgecolors="k",
                linewidths=0.4,
            )
            ax.annotate(
                f"95%@iter{it95}\n→{final:.3f}",
                xy=(it95, final * 0.95),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
                color=c,
            )

        ax.set_xlabel("LBFGS round (iter)", fontsize=11)
        ax.set_ylabel("hold MedF", fontsize=11)
        ax.set_title(
            f"n={n} L={L} Γ={Gamma}  |  B_train={B_train} hold={B_hold}  |  "
            f"LBFGS max_iter=8/round",
            fontsize=11,
        )
        ax.set_xticks(range(0, rounds + 1, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(PNG_PATH, dpi=150)
        plt.close(fig)
        log(f"wrote {PNG_PATH}")

    # summary print
    print("\n=== SUMMARY n6 L4 fidelity vs iter ===")
    print(f"{'method':16s} {'seed':>4} {'np':>3} {'final_hold':>10} {'it95':>5} {'plateau':>7} {'sec':>7}")
    for s in summaries:
        print(
            f"{s['method']:16s} {s['seed']:4d} {s['n_params']:3d} "
            f"{s['final_hold']:10.4f} {s['it95']:5d} {s['plateau_iter']:7d} {s['seconds']:7.1f}"
        )

    # also dump a tiny md note next to plot
    md = OUT / "n6_L4_fidelity_vs_iter.md"
    lines = [
        "# n=6 L=4 Γ=0.5 — fidelity vs LBFGS iteration",
        "",
        f"Protocol: B_train={B_train}, B_hold={B_hold}, 20 LBFGS rounds (max_iter=8 each), seeds {list(SEEDS)}.",
        "",
        "| method | seed | params | final best hold | iters to 95% of final | plateau≈ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['method']} | {s['seed']} | {s['n_params']} | {s['final_hold']:.4f} | "
            f"{s['it95']} | {s['plateau_iter']} |"
        )
    lines += [
        "",
        "Artifacts: `n6_L4_fidelity_vs_iter.csv`, `n6_L4_fidelity_vs_iter.png`.",
        "",
        "Notes: iter=0 is init (before first LBFGS step). Dotted verticals / markers = first iter where best-so-far hold ≥ 95% of that run's final best hold.",
    ]
    md.write_text("\n".join(lines) + "\n")
    log(f"wrote {md}")


if __name__ == "__main__":
    main()
