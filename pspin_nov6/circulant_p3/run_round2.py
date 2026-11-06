"""Bounded round-2 scan of the two-window circulant ring.

The full Hamiltonian's ground state is the target. Reflection is not used.
Depth ladders run only where the L=1 fidelity maximum is still moderate;
rows that are already easy at L=1 are recorded and not given a fake deep search.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

import hamiltonian as ham
import qaoa
from observables import (
    classical_manifold_overlap,
    ground_state,
    half_cut_entropy_bits,
    orbit_overlap,
    plus_overlap,
)
from hamiltonian import cost_diagonal, gamma_period, translate_bits, write_term_list
from verify_algebra import run_checks


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def orbit_upper(gs, diag, n, rng, n_rand: int = 5) -> dict:
    from scipy.optimize import minimize

    e_min = float(np.min(diag))
    mask = np.where(np.abs(diag - e_min) < 1e-8)[0]
    seen = set()
    best_clas = 0.0
    n_orb = 0
    for x in mask:
        x = int(x)
        if x in seen:
            continue
        n_orb += 1
        th = np.array([0.0 if ((x >> i) & 1) == 0 else math.pi for i in range(n)])
        best_clas = max(best_clas, orbit_overlap(gs, th, n))
        y = x
        for _ in range(n):
            seen.add(y)
            y = translate_bits(y, n, 1)
    best = best_clas

    def neg(th):
        return -orbit_overlap(gs, th, n)

    for _ in range(n_rand):
        th0 = rng.uniform(0.0, math.pi, size=n)
        res = minimize(neg, th0, method="L-BFGS-B", bounds=[(0.0, math.pi)] * n, options={"maxiter": 40})
        best = max(best, -float(res.fun))
    clas = classical_manifold_overlap(gs, diag)
    return {"F_orbit_classical": best_clas, "F_orbit_opt": best, "n_classical_orbits": n_orb, **clas}


def light_l1(diag, gamma, n, gs, period):
    grid = qaoa.l1_grid(diag, gamma, n, gs, period, n_g=81, n_b=41)
    best_e = None
    for e0, g0, b0, f0 in grid["basins"][:4]:
        g, b, e, f = qaoa._opt(diag, gamma, n, gs, [g0], [b0], "energy", maxiter=100)
        if best_e is None or e < best_e[0]:
            best_e = (e, f, g, b, "l1_grid_basin")
    gf, bf = grid["best_fidelity_angles"]
    gF, bF, eF, fF = qaoa._opt(diag, gamma, n, gs, [gf], [bf], "fidelity", maxiter=120)
    return {
        "F_E": best_e[1],
        "energy": best_e[0],
        "F1max": fF,
        "E_at_F1max": eF,
        "gamma_raw": best_e[2],
        "beta_raw": best_e[3],
        "source": best_e[4],
        "grid_Fmax": grid["best_fidelity"],
        "f1_gamma": float(gF[0] if np.ndim(gF) else gF),
        "f1_beta": float(bF[0] if np.ndim(bF) else bF),
    }


def reference_evolve(diag, gammas, betas, n):
    """Bit-slice mixer, written separately from qaoa.apply_mixer."""
    psi = qaoa.plus_state(n)
    dim = 1 << n
    for gv, bv in zip(np.atleast_1d(gammas), np.atleast_1d(betas)):
        psi = psi * np.exp(-1j * float(gv) * diag)
        cb = math.cos(float(bv))
        sb = math.sin(float(bv))
        for i in range(n):
            step = 1 << i
            new = np.empty_like(psi)
            for base in range(0, dim, step << 1):
                a = slice(base, base + step)
                b = slice(base + step, base + 2 * step)
                pa, pb = psi[a], psi[b]
                new[a] = cb * pa - 1j * sb * pb
                new[b] = cb * pb - 1j * sb * pa
            psi = new
    return psi


def main():
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = run_checks()
    # Emax = -E0 from global Y, checked with a largest-magnitude eigenpair.
    n, kappa, gamma = 8, 1.0, 1.0
    diag = cost_diagonal(n, kappa)
    dim = 1 << n

    def matvec(v, sign=1.0):
        return sign * (diag * v - gamma * ham.apply_sum_x(v, n))

    op = LinearOperator((dim, dim), matvec=lambda v: matvec(v, 1.0), dtype=np.float64)
    lo = eigsh(op, k=1, which="SA", tol=1e-10, return_eigenvectors=False)[0]
    op_hi = LinearOperator((dim, dim), matvec=lambda v: matvec(v, 1.0), dtype=np.float64)
    hi = eigsh(op_hi, k=1, which="LA", tol=1e-10, return_eigenvectors=False)[0]
    log.append(f"N=8 kappa=1 Gamma=1: E0={lo:.8f} Emax={hi:.8f} E0+Emax={lo+hi:.3e}")
    if abs(lo + hi) > 1e-6:
        raise AssertionError("Emax is not -E0; eta definition needs the computed edge")
    # Replay and sign-flip.
    rng = np.random.default_rng(0)
    g = rng.normal(size=2) * 0.2
    b = rng.normal(size=2) * 0.3
    psi_a = qaoa.evolve(diag, g, b, n)
    psi_b = reference_evolve(diag, g, b, n)
    ov = abs(np.vdot(psi_a, psi_b))
    log.append(f"N=8 replay overlap between two mixer implementations {ov:.12f}")
    if abs(ov - 1) > 1e-8:
        raise AssertionError("replay mismatch")
    # Sign-flipped cost.
    evals, gs, _ = ground_state(n, kappa, gamma, diag=diag)
    evals_f, gs_f, _ = ground_state(n, kappa, gamma, diag=-diag)
    _e1, f1 = qaoa.energy_fidelity(qaoa.evolve(diag, g, b, n), diag, gamma, n, gs)
    _e2, f2 = qaoa.energy_fidelity(qaoa.evolve(-diag, g, b, n), -diag, gamma, n, gs_f)
    log.append(f"sign-flipped couplings: F={f1:.10f} vs {f2:.10f}, abs diff {abs(f1-f2):.3e}")
    if abs(f1 - f2) > 1e-8:
        raise AssertionError("odd-p sign flip fidelity mismatch")

    survey = []
    for n in (8, 9, 10, 11, 12):
        for gamma in (1.0, 1.3):
            survey.append((n, 1.0, gamma))
    survey.append((12, 1.0, 1.2))
    survey.append((12, 0.5, 1.0))

    rows = []
    angles = []
    ladders_done = []
    for n, kappa, gamma in survey:
        if time.perf_counter() - t0 > 40 * 60:
            log.append("time budget reached before finishing the survey")
            break
        diag = cost_diagonal(n, kappa)
        per = gamma_period(diag)
        evals, gs, _ = ground_state(n, kappa, gamma, diag=diag)
        ent, cut = half_cut_entropy_bits(gs, n)
        orb = orbit_upper(gs, diag, n, np.random.default_rng(1000 * n + int(gamma * 100)))
        l1 = light_l1(diag, gamma, n, gs, per["period"])
        e0 = float(evals[0])
        emax = -e0
        def eta(e):
            return (e - emax) / (e0 - emax)
        row = {
            "N": n,
            "kappa": kappa,
            "Gamma": gamma,
            "L": 1,
            "E0": e0,
            "Emax": emax,
            "gap": float(evals[1] - evals[0]),
            "F_plus": plus_overlap(gs, n),
            "entropy_bits": ent,
            "entropy_cut": cut,
            "F_orbit_opt": orb["F_orbit_opt"],
            "F_orbit_classical": orb["F_orbit_classical"],
            "F_classical_uniform": orb["F_classical_uniform"],
            "p_classical_subspace": orb["p_classical_subspace"],
            "max_bitstring_prob": orb["max_bitstring_prob"],
            "classical_degeneracy": orb["classical_degeneracy"],
            "n_classical_orbits": orb["n_classical_orbits"],
            "F_E": l1["F_E"],
            "energy": l1["energy"],
            "eta": eta(l1["energy"]),
            "F1max": l1["F1max"],
            "E_at_F1max": l1["E_at_F1max"],
            "gamma_period": per["period"],
            "sum_abs_gamma": float(np.sum(np.abs(l1["gamma_raw"]))),
            "sum_abs_beta": float(np.sum(np.abs(l1["beta_raw"]))),
            "source": l1["source"],
            "momentum_dev": 0.0,
        }
        rows.append(row)
        angles.append({
            "N": n, "kappa": kappa, "Gamma": gamma, "L": 1, "kind": "energy",
            "gamma": l1["gamma_raw"], "beta": l1["beta_raw"],
            "F": l1["F_E"], "energy": l1["energy"],
        })
        angles.append({
            "N": n, "kappa": kappa, "Gamma": gamma, "L": 1, "kind": "fidelity_max",
            "gamma": l1["f1_gamma"], "beta": l1["f1_beta"],
            "F": l1["F1max"], "energy": l1["E_at_F1max"],
        })
        log.append(
            f"L1 N={n} k={kappa} G={gamma:.2f} F1max={l1['F1max']:.3f} F_E={l1['F_E']:.3f} "
            f"F0={row['F_plus']:.3f} S={ent:.2f} Forbit={orb['F_orbit_opt']:.3f} gap={row['gap']:.3f}"
        )
        print(log[-1], flush=True)
        # Depth only where L=1 cannot already reach a high overlap.
        if l1["F1max"] < 0.55 and time.perf_counter() - t0 < 38 * 60:
            print(f"ladder N={n} G={gamma}", flush=True)
            recs = qaoa.search_ladder(
                diag, gamma, n, gs, per["period"], max_depth=6,
                rng=np.random.default_rng(20261106 + 17 * n + int(gamma * 100)),
                n_random=8,
                maxiter=40,
            )
            for rec in recs:
                r = dict(row)
                r.update({
                    "L": rec["L"],
                    "F_E": rec["F"],
                    "energy": rec["energy"],
                    "eta": eta(rec["energy"]),
                    "F1max": rec["F1max"] if rec["F1max"] is not None else l1["F1max"],
                    "sum_abs_gamma": rec["sum_abs_gamma"],
                    "sum_abs_beta": rec["sum_abs_beta"],
                    "source": rec["source"],
                    "F_min_endpoints": rec["F_min"],
                    "F_max_endpoints": rec["F_max"],
                    "F_std_endpoints": rec["F_std"],
                    "canon_notes": "|".join(rec["canon_notes"]),
                    "n_starts": rec["n_starts"],
                })
                rows.append(r)
                angles.append({
                    "N": n, "kappa": kappa, "Gamma": gamma, "L": rec["L"], "kind": "energy_ladder",
                    "gamma": rec["gamma_raw"], "beta": rec["beta_raw"],
                    "gamma_can": rec["gamma_can"], "beta_can": rec["beta_can"],
                    "canon_notes": rec["canon_notes"],
                    "F": rec["F"], "energy": rec["energy"], "source": rec["source"],
                    "sum_abs_gamma": rec["sum_abs_gamma"], "sum_abs_beta": rec["sum_abs_beta"],
                })
                log.append(
                    f"  L={rec['L']} F_E={rec['F']:.3f} E={rec['energy']:.3f} src={rec['source']} "
                    f"Fend[{rec['F_min']:.2f},{rec['F_max']:.2f}] notes={rec['canon_notes']}"
                )
                print(log[-1], flush=True)
            ladders_done.append((n, kappa, gamma))

    for n in (8, 10, 12):
        write_term_list(str(out / f"hamiltonian_terms_N{n}_k1.csv"), n, 1.0)
    write_term_list(str(out / "hamiltonian_terms_N12_k0.5.csv"), 12, 0.5)

    # CSV
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (out / "qaoa_round2.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "angles.json").write_text(json.dumps(_jsonable(angles), indent=2))
    (out / "verification.log").write_text("\n".join(log) + "\n")

    # Plots: L=1 survey and any ladders.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for n in sorted({r["N"] for r in rows if r["kappa"] == 1.0 and r["L"] == 1}):
        sub = [r for r in rows if r["N"] == n and r["kappa"] == 1.0 and r["L"] == 1]
        sub.sort(key=lambda r: r["Gamma"])
        ax.plot([r["Gamma"] for r in sub], [r["F1max"] for r in sub], marker="o", label=f"N={n} F1max")
        ax.plot([r["Gamma"] for r in sub], [r["F_E"] for r in sub], marker="x", ls="--", label=f"N={n} F_E(1)")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("squared overlap")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("L=1 max fidelity and energy-selected fidelity, kappa=1")
    fig.tight_layout()
    fig.savefig(out / "F1_survey.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for n in sorted({int(r["N"]) for r in rows if r["L"] == 1 and r["kappa"] == 1.0}):
        sub = [r for r in rows if r["N"] == n and r["kappa"] == 1.0 and r["L"] == 1]
        sub.sort(key=lambda r: r["Gamma"])
        ax.plot([r["Gamma"] for r in sub], [r["F_orbit_opt"] for r in sub], marker="o", label=f"N={n} orbit")
        ax.plot([r["Gamma"] for r in sub], [r["entropy_bits"] for r in sub], marker="s", ls="--", label=f"N={n} S")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("orbit overlap or entropy (bits)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("GS orbit-product overlap and half-arc entropy")
    fig.tight_layout()
    fig.savefig(out / "gs_orbit_entropy.png", dpi=140)
    plt.close(fig)

    deep = [r for r in rows if r.get("n_starts")]
    if deep:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        labels = sorted({(r["N"], r["kappa"], r["Gamma"]) for r in deep})
        for n, kappa, gamma in labels:
            sub = sorted((r for r in deep if r["N"] == n and r["kappa"] == kappa and r["Gamma"] == gamma), key=lambda r: r["L"])
            # include the light L=1 if the ladder replaced it; ladder rows have n_starts
            ax.plot([r["L"] for r in sub], [r["F_E"] for r in sub], marker="o", label=f"N={n} k={kappa:g} G={gamma:g}")
        ax.set_xlabel("L")
        ax.set_ylabel("F at best energy")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.set_title("Energy-selected overlap versus depth")
        fig.tight_layout()
        fig.savefig(out / "F_vs_L_deep.png", dpi=140)
        plt.close(fig)

    log.append(f"elapsed_s {time.perf_counter()-t0:.1f} ladders {ladders_done}")
    (out / "verification.log").write_text("\n".join(log) + "\n")
    print(log[-1], flush=True)


if __name__ == "__main__":
    main()
