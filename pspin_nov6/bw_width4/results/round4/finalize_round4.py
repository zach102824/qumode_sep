"""Compile the Gamma=0.8 delivery figures from stored angles.

No optimizer is called. A state is accepted only when its replayed
fidelity matches the published table.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from geometry import build_triangles, coloring_report  # noqa: E402
from model import classical_manifold, cost_diagonal, ground_state  # noqa: E402
from observe import z_one, zzz_means  # noqa: E402
from qaoa import apply_qaoa, plus_state  # noqa: E402

GAMMA = 0.8
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
A_REF = "origin/cursor/sparse-p3-qaoa-round1-13b4"


def git_show(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{A_REF}:{path}"], text=True)


def mean_x(psi, n: int) -> float:
    idx = np.arange(psi.shape[0], dtype=np.int64)
    acc = 0.0
    for q in range(n):
        acc += float(np.vdot(psi, psi[idx ^ (1 << q)]).real)
    return acc / n


def energy(psi, hc, n: int) -> float:
    hc_exp = float(np.dot(np.abs(psi) ** 2, hc))
    return hc_exp - GAMMA * mean_x(psi, n) * n


def flat_and_weight(gs, hc):
    idx = classical_manifold(hc)["indices"]
    cat = np.zeros_like(gs)
    cat[idx] = 1.0 / np.sqrt(idx.size)
    return float(np.dot(gs[idx], gs[idx])), float(np.dot(gs, cat) ** 2), int(idx.size)


def sublattice_error(psi, gs, lx, ly):
    n = lx * ly
    color = np.array([(i % lx + i // lx) % 3 for i in range(n)])
    mz = z_one(psi, n)
    mg = z_one(gs, n)
    err = []
    for c in (0, 1, 2):
        sites = np.flatnonzero(color == c)
        err.append(abs(float(mz[sites].mean()) - float(mg[sites].mean())))
    return float(np.mean(err))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    a_angles = json.loads(git_show("pspin_nov6/baxter_wu_p3/runs/round3/angles.json"))
    n18 = json.loads(git_show("pspin_nov6/baxter_wu_p3/runs/round3/n18_transfer_reopt.json"))
    local = json.loads((ROOT / "results/round3/angles.json").read_text())

    specs = [
        (3, 4, 12, "3x4", a_angles, "gamma"),
        (3, 5, 15, "3x5", a_angles, "gamma"),
        (4, 4, 16, "4x4", local, "Gamma"),
        (3, 6, 18, "3x6", a_angles, "gamma"),
    ]
    rows = []
    for lx, ly, n, name, pool, gkey in specs:
        geom = build_triangles(lx, ly)
        col = coloring_report(lx, ly, geom)
        hc = cost_diagonal(geom, 1.0)
        gs_pack = ground_state(hc, GAMMA, n)
        gs = gs_pack["gs"]
        w, f_flat, deg = flat_and_weight(gs, hc)
        gs_zzz = zzz_means(gs, geom)
        gs_x = mean_x(gs, n)
        print(
            f"{name} E0={gs_pack['E0']:.8f} gap={gs_pack['gap']:.6f} "
            f"resid={gs_pack['residual']:.2e} method={gs_pack['method']} "
            f"coloring={col['face_coloring_exists']} w={w:.6f} Fflat={f_flat:.6f} deg={deg}",
            flush=True,
        )
        ladder = [
            a
            for a in pool
            if int(a["lx"]) == lx
            and int(a["ly"]) == ly
            and abs(float(a[gkey]) - GAMMA) < 1e-9
        ]
        ladder.sort(key=lambda a: int(a["L"]))
        if not any(int(a["L"]) == 0 for a in ladder):
            ladder.insert(0, {"lx": lx, "ly": ly, "L": 0, "gammas": [], "betas": [], "F": None, gkey: GAMMA})
        published_f0 = {
            "3x4": 0.07000895095119412,
            "3x5": 0.12653188850778593,
            "3x6": 0.012854966053563241,
            "4x4": 0.11528546,
        }[name]
        for a in ladder:
            L = int(a["L"])
            if L == 0 or not a.get("gammas"):
                psi = plus_state(n)
                published_F = published_f0 if a.get("F") is None else float(a["F"])
            else:
                psi = apply_qaoa(plus_state(n), hc, n, a["gammas"], a["betas"])
                published_F = float(a["F"])
            F = float(abs(np.vdot(gs, psi)) ** 2)
            E = energy(psi, hc, n)
            zzz = zzz_means(psi, geom)
            dF = abs(F - published_F)
            ok = dF < 1e-6 or (L == 0 and dF < 1e-5)
            rec = {
                "geom": name,
                "lx": lx,
                "ly": ly,
                "n": n,
                "L": L,
                "kind": "energy_selected_ladder",
                "F_replay": F,
                "F_published": published_F,
                "dF": dF,
                "accepted": ok,
                "energy": E,
                "E0": gs_pack["E0"],
                "gap": gs_pack["gap"],
                "rel_energy_err": (E - gs_pack["E0"]) / abs(gs_pack["E0"]),
                "err_zzz_up": abs(zzz["mean_zzz_up"] - gs_zzz["mean_zzz_up"]),
                "err_zzz_down": abs(zzz["mean_zzz_down"] - gs_zzz["mean_zzz_down"]),
                "err_X": abs(mean_x(psi, n) - gs_x),
                "err_sublattice": (
                    sublattice_error(psi, gs, lx, ly) if col["face_coloring_exists"] else None
                ),
                "coloring": bool(col["face_coloring_exists"]),
                "w": w,
                "F_flat": f_flat,
                "gs_residual": gs_pack["residual"],
                "gs_method": gs_pack["method"],
            }
            rows.append(rec)
            print(
                f"  L={L} Freplay={F:.8f} Fpub={published_F:.8f} dF={dF:.2e} "
                f"relE={rec['rel_energy_err']:.4f} dZZZ={rec['err_zzz_up']:.4f} dX={rec['err_X']:.4f} ok={ok}",
                flush=True,
            )
        if name == "3x6":
            for item in n18:
                if abs(float(item["gamma"]) - GAMMA) > 1e-9:
                    continue
                psi = apply_qaoa(plus_state(n), hc, n, item["gammas"], item["betas"])
                F = float(abs(np.vdot(gs, psi)) ** 2)
                E = energy(psi, hc, n)
                zzz = zzz_means(psi, geom)
                # The stored polish fidelity is reopt_F. The same angles are the
                # polished state, so that is the comparison. raw_transfer_F is
                # the seed before this polish and is not this vector.
                published = float(item["reopt_F"])
                dF = abs(F - published)
                rec = {
                    "geom": name,
                    "lx": lx,
                    "ly": ly,
                    "n": n,
                    "L": 5,
                    "kind": item["name"],
                    "F_replay": F,
                    "F_published": published,
                    "dF": dF,
                    "accepted": dF < 1e-6,
                    "energy": E,
                    "E_published": item["reopt_E"],
                    "E0": gs_pack["E0"],
                    "gap": gs_pack["gap"],
                    "rel_energy_err": (E - gs_pack["E0"]) / abs(gs_pack["E0"]),
                    "err_zzz_up": abs(zzz["mean_zzz_up"] - gs_zzz["mean_zzz_up"]),
                    "err_zzz_down": abs(zzz["mean_zzz_down"] - gs_zzz["mean_zzz_down"]),
                    "err_X": abs(mean_x(psi, n) - gs_x),
                    "err_sublattice": sublattice_error(psi, gs, lx, ly),
                    "coloring": True,
                    "w": w,
                    "F_flat": f_flat,
                    "raw_transfer_F_published": item["raw_transfer_F"],
                    "gs_residual": gs_pack["residual"],
                    "gs_method": gs_pack["method"],
                }
                rows.append(rec)
                print(
                    f"  polish Freplay={F:.8f} Fpub={published:.8f} dF={dF:.2e} E={E:.6f} Epub={item['reopt_E']}",
                    flush=True,
                )

    (OUT / "replay_observables.json").write_text(json.dumps(rows, indent=2))
    plot(rows)
    print("wrote", FIG)


def n18_energy_curve(rows):
    """Energy-selected N=18 schedule. Final L=5 and L=6 replace the earlier ladder."""
    base = [r for r in rows if r["geom"] == "3x6" and r["kind"] == "energy_selected_ladder" and r["accepted"]]
    final = {r["L"]: r for r in rows if r["geom"] == "3x6" and r["kind"] == "energy_selected_final" and r["accepted"]}
    out = []
    seen = set()
    for r in base:
        if r["L"] in final or r["L"] in seen:
            continue
        out.append(r)
        seen.add(r["L"])
    for r in final.values():
        out.append(r)
    out.sort(key=lambda r: r["L"])
    return out


def series(rows, geom, kind="energy_selected_ladder"):
    pts = [r for r in rows if r["geom"] == geom and r["kind"] == kind and r["accepted"]]
    pts.sort(key=lambda r: r["L"])
    return pts


def plot(rows):
    order = [("3x4", "3×4", "N=12", "#1a5276"),
             ("3x5", "3×5", "N=15", "#b9770e"),
             ("4x4", "4×4", "N=16", "#196f3d"),
             ("3x6", "3×6", "N=18", "#6c3483")]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

    # Fig 1. Energy-optimal curve. N=18 L=5 uses the lower-energy polish.
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for geom, label, nlab, color in order:
        pts = series(rows, geom)
        Ls = [p["L"] for p in pts]
        Fs = [p["F_replay"] for p in pts]
        if geom == "3x6":
            pts = n18_energy_curve(rows)
            Ls = [p["L"] for p in pts]
            Fs = [p["F_replay"] for p in pts]
            ax.plot(Ls, Fs, "-o", color=color, lw=1.8, ms=6, label=f"{label}, {nlab}", zorder=3)
            ax.scatter([1.18], [0.09522872210943426], s=70, facecolors="none",
                       edgecolors=color, linewidths=1.6, zorder=4,
                       label="N=18 L=1 fidelity-max 0.095 (not a certificate)")
            ax.scatter([4.72], [0.4197711914716309], marker="x", s=45, color=color, zorder=4,
                       label="earlier N=18 L=5: blind 0.420")
            ax.scatter([5.28], [0.5235870342914763], s=55, facecolors="none",
                       edgecolors=color, linewidths=1.4, zorder=4,
                       label="earlier N=18 L=5: transfer 0.524")
            ax.scatter([4.88], [0.4709952444238817], marker="^", s=40, color=color, zorder=4,
                       label="earlier N=18 L=5: polish 0.471")
        else:
            ax.plot(Ls, Fs, "-o", color=color, lw=1.8, ms=6, label=f"{label}, {nlab}", zorder=3)
    ax.set_xlabel("QAOA depth L")
    ax.set_ylabel("squared ground-state overlap")
    ax.set_xlim(-0.2, 6.6)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 7))
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.5, framealpha=0.95)
    ax.set_title("Γ = 0.8, unscaled staggered Baxter–Wu\nsolid: energy-selected state at that depth")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(FIG / "fig1_F_vs_L_gamma08.png", dpi=160)
    fig.savefig(FIG / "fig1_F_vs_L_gamma08.pdf")
    plt.close()

    # Fig 2. Markers only. No line, so the plot is not a fit.
    # Refined L=1 fidelity maxima. Branch B reproduced these; they are not the
    # coarser period-grid values 0.269, 0.453, 0.408, 0.090.
    grid = {"3x4": 0.26949754462061465, "3x5": 0.4597174428674855,
            "4x4": 0.4129654, "3x6": 0.09522872210943426}
    deep = {}
    for geom, *_ in order:
        pts = n18_energy_curve(rows) if geom == "3x6" else series(rows, geom)
        deep[geom] = pts[-1]["F_replay"]
    inv = {"3x4": 1 / 12, "3x5": 1 / 15, "4x4": 1 / 16, "3x6": 1 / 18}
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    xs = [inv[g] for g, *_ in order]
    ax.scatter(xs, [grid[g] for g, *_ in order], marker="s", s=55, color="#b9770e",
               label="L=1 fidelity-max, best-found, not a certificate", zorder=3)
    ax.scatter(xs, [deep[g] for g, *_ in order], marker="D", s=55, color="#196f3d",
               label="deepest energy-selected F (N=18 is L=6)", zorder=3)
    ax.scatter([1 / 18 - 0.0016], [0.4197711914716309], marker="x", s=40, color="#6c3483", zorder=4)
    ax.scatter([1 / 18], [0.4709952444238817], marker="^", s=36, color="#6c3483", zorder=4)
    ax.scatter([1 / 18 + 0.0016], [0.5235870342914763], s=50, facecolors="none",
               edgecolors="#6c3483", linewidths=1.3, zorder=4,
               label="earlier N=18 L=5 states, not the winner")
    tick_pos = [1 / 18, 1 / 16, 1 / 15, 1 / 12]
    tick_lab = ["3×6\nN=18", "4×4\nN=16", "3×5\nN=15", "3×4\nN=12"]
    ax.set_xticks(tick_pos, tick_lab)
    ax.set_xlim(1 / 18 - 0.012, 1 / 12 + 0.012)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("squared ground-state overlap")
    ax.set_xlabel("geometry  (horizontal position is 1/N; no fit)")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="center right", fontsize=8, framealpha=0.95)
    ax.set_title("Γ = 0.8 only. N=15 above N=12 is shown, not fitted.\n4×3 is not a second N=12 point.")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_F_vs_invN_gamma08.png", dpi=160)
    fig.savefig(FIG / "fig2_F_vs_invN_gamma08.pdf")
    plt.close()

    # Fig 3. Three stored-angle errors, plus sublattice only where a coloring exists.
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.4), sharex=True)
    panels = [
        (axes[0], "rel_energy_err", "relative energy error\n(E − E0) / |E0|"),
        (axes[1], "err_zzz_up", "mean up-triangle\n|Δ⟨ZZZ⟩|"),
        (axes[2], "err_X", "site-averaged\n|Δ⟨X⟩|"),
    ]
    for ax, key, title in panels:
        for geom, label, nlab, color in order:
            pts = n18_energy_curve(rows) if geom == "3x6" else series(rows, geom)
            ys = [p.get(key) for p in pts]
            if any(y is None for y in ys):
                continue
            ax.plot([p["L"] for p in pts], ys, "-o", color=color, lw=1.6, ms=5, label=f"{label} {nlab}")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("depth L")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Γ = 0.8, stored-angle replay. 3×6 is the only colored torus here;\n"
        "its sublattice magnetizations are 0 on the ground state and on these circuits, so no m_A curve is drawn.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig3_errors_vs_L_gamma08.png", dpi=160)
    fig.savefig(FIG / "fig3_errors_vs_L_gamma08.pdf")
    plt.close()


if __name__ == "__main__":
    main()
