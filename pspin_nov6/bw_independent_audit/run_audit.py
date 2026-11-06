"""Independent audit of the staggered Baxter-Wu anchors.

Reads published angles and term lists from reference/. Does not import the
other branch's builder or optimizer. Circuit order is cost then mixer,
layer 0 closest to |+>. If that misses the stored fidelity by more than
0.03, one alternate order is tried and reported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

from bw_model import (
    apply_mixer,
    apply_sum_x,
    clauses,
    coloring_conflicts,
    cost_diagonal,
    evolve,
    faces,
    gamma_period_from_diag,
    shift_state,
    site,
    sublattice_flip_commutes,
    term_hash,
)

HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
OUT = HERE / "results"


def published_hash(terms: list[dict]) -> str:
    lines = []
    n_x = 0
    for t in terms:
        if t["pauli"] == "X":
            n_x += 1
            if abs(float(t["coeff"]) + 1.0) > 1e-12 or t.get("face") != "field":
                raise AssertionError(f"unexpected field term {t}")
            continue
        q = sorted(int(x) for x in t["qubits"])
        if len(q) != 3:
            raise AssertionError(t)
        lines.append(f"{t['face']} {float(t['coeff']):.1f} {q[0]} {q[1]} {q[2]}")
    digest = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()
    return digest


def classical_summary(diag: np.ndarray) -> dict:
    e_min = float(np.min(diag))
    e_max = float(np.max(diag))
    deg = int(np.sum(np.abs(diag - e_min) < 1e-8))
    uniq, counts = np.unique(np.rint(diag).astype(np.int64), return_counts=True)
    return {
        "Emin": e_min,
        "Emax_diag": e_max,
        "degeneracy": deg,
        "n_distinct_energies": int(uniq.size),
        "histogram_sha256": hashlib.sha256(
            np.array(list(zip(uniq.tolist(), counts.tolist())), dtype=np.int64).tobytes()
        ).hexdigest(),
    }


def krylov_ground(diag: np.ndarray, gamma: float, n: int, k: int = 3) -> dict:
    def mv(v):
        z = np.asarray(v, dtype=np.float64)
        return diag * z - gamma * apply_sum_x(z, n)

    op = LinearOperator((1 << n, 1 << n), matvec=mv, dtype=np.float64)
    ncv = min(1 << n, max(24, 10 * k))
    evals, vecs = eigsh(op, k=k, which="SA", tol=1e-11, ncv=ncv, maxiter=4000)
    order = np.argsort(evals)
    evals = evals[order]
    vecs = vecs[:, order]
    residuals = []
    for i in range(k):
        r = mv(vecs[:, i]) - float(evals[i]) * vecs[:, i]
        residuals.append(float(np.linalg.norm(r)))
    gs = vecs[:, 0].real.copy()
    if float(np.sum(gs)) < 0.0:
        gs *= -1.0
    n_neg = int(np.sum(gs < -1e-8))
    return {
        "evals": [float(x) for x in evals],
        "gap": float(evals[1] - evals[0]),
        "residuals": residuals,
        "gs": gs,
        "n_negative_amplitudes": n_neg,
        "min_amplitude": float(np.min(gs)),
        "vectors": vecs,
    }


def lambda_max_cut(psi: np.ndarray, n: int, sites_a: list[int]) -> float:
    b_sites = [i for i in range(n) if i not in set(sites_a)]
    order = list(sites_a) + b_sites
    idx = np.arange(1 << n, dtype=np.int64)
    perm = np.zeros_like(idx)
    for new, old in enumerate(order):
        perm |= ((idx >> np.int64(old)) & np.int64(1)) << np.int64(new)
    arranged = np.empty_like(psi, dtype=np.complex128)
    arranged[perm] = np.asarray(psi, dtype=np.complex128)
    na = len(sites_a)
    amp = arranged.reshape(1 << (n - na), 1 << na)
    rho = amp.conj().T @ amp
    ev = np.linalg.eigvalsh(rho).real
    return float(ev[-1])


def product_upper(psi: np.ndarray, lx: int, ly: int, n_random: int = 24) -> dict:
    """Min largest Schmidt eigenvalue over tested cuts.

    Each value upper-bounds the overlap with any state that factors across
    that cut, hence with any fully factorized product. A smaller sample
    still yields a valid, possibly loose, upper bound. It does not bound
    translation-orbit cats.
    """
    n = lx * ly
    cuts = []
    # Lowest N//2 indices.
    cuts.append(("lowest_indices", list(range(n // 2))))
    # Spatial half in y when even, else first ly//2 rows.
    rows = max(ly // 2, 1)
    spatial = [site(x, y, lx, ly) for y in range(rows) for x in range(lx)]
    if len(spatial) not in (0, n):
        cuts.append((f"y_lt_{rows}", spatial))
    rng = np.random.default_rng(0)
    for i in range(n_random):
        choice = rng.choice(n, size=n // 2, replace=False)
        cuts.append((f"random_{i}", sorted(int(x) for x in choice)))
    vals = []
    for name, sites_a in cuts:
        vals.append((name, lambda_max_cut(psi, n, sites_a), sites_a))
    best = min(vals, key=lambda t: t[1])
    spatial_val = next(v for name, v, _s in vals if name.startswith("y_lt_") or name == "lowest_indices")
    return {
        "upper": best[1],
        "tightest_cut": best[0],
        "tightest_sites": best[2],
        "n_cuts": len(vals),
        "spatial_or_lowest": spatial_val,
    }


def half_entropy(psi: np.ndarray, sites_a: list[int], n: int) -> float:
    # reuse lambda machinery via the reduced spectrum
    b_sites = [i for i in range(n) if i not in set(sites_a)]
    order = list(sites_a) + b_sites
    idx = np.arange(1 << n, dtype=np.int64)
    perm = np.zeros_like(idx)
    for new, old in enumerate(order):
        perm |= ((idx >> np.int64(old)) & np.int64(1)) << np.int64(new)
    arranged = np.empty(1 << n, dtype=np.complex128)
    arranged[perm] = np.asarray(psi, dtype=np.complex128)
    na = len(sites_a)
    amp = arranged.reshape(1 << (n - na), 1 << na)
    rho = amp.conj().T @ amp
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-15]
    return float(-np.sum(ev * np.log2(ev)))


def manifold_stats(gs: np.ndarray, diag: np.ndarray, lx: int, ly: int) -> dict:
    e_min = float(np.min(diag))
    mask = np.where(np.abs(diag - e_min) < 1e-8)[0]
    weight = float(np.sum(gs[mask] ** 2))
    uniform = float(np.sum(gs[mask]) ** 2 / max(len(mask), 1))
    # torus orbits of the classical minima
    seen = set()
    orbit_overlaps = []
    for x0 in mask:
        x0 = int(x0)
        if x0 in seen:
            continue
        orb = []
        for dy in range(ly):
            for dx in range(lx):
                y = 0
                # translate bits
                for i in range(lx * ly):
                    if (x0 >> i) & 1:
                        ix, iy = i % lx, i // lx
                        j = ((ix + dx) % lx) + lx * ((iy + dy) % ly)
                        y |= 1 << j
                if y not in seen:
                    seen.add(y)
                    orb.append(y)
        if not orb:
            continue
        ov = float(np.sum(gs[orb]) ** 2 / len(orb))
        orbit_overlaps.append((ov, len(orb)))
    orbit_overlaps.sort(reverse=True)
    return {
        "degeneracy": int(mask.size),
        "manifold_weight": weight,
        "uniform_cat_overlap": uniform,
        "n_torus_orbits": len(orbit_overlaps),
        "best_single_orbit_overlap": orbit_overlaps[0][0] if orbit_overlaps else None,
        "best_single_orbit_size": orbit_overlaps[0][1] if orbit_overlaps else None,
    }


def plus_overlap(gs: np.ndarray, n: int) -> float:
    return float(np.sum(gs) / math.sqrt(1 << n)) ** 2


def fidelity(psi: np.ndarray, gs: np.ndarray) -> float:
    return float(abs(np.vdot(gs, psi)) ** 2)


def energy_of(psi, diag, gamma, n):
    hc = float(np.real(np.vdot(psi, diag * psi)))
    hx = float(np.real(np.vdot(psi, apply_sum_x(psi, n))))
    return hc - gamma * hx


def l1_search(diag, gamma, n, gs, period: float, n_g: int, n_b: int) -> dict:
    """Best-found on a uniform grid of one verified gamma period and one beta period.

    Not a certificate that a narrower peak was missed. Local refinement is Nelder-Mead.
    """
    gammas = np.linspace(0.0, period, n_g, endpoint=False)
    betas = np.linspace(0.0, math.pi, n_b, endpoint=False)
    plus = np.full(1 << n, 2.0 ** (-0.5 * n), dtype=np.complex128)
    best_e, best_e_ang, f_at_e = math.inf, (0.0, 0.0), 0.0
    best_f, best_f_ang, e_at_f = -1.0, (0.0, 0.0), 0.0
    basins_e = []
    basins_f = []
    for g in gammas:
        phased = plus * np.exp(-1j * float(g) * diag)
        for b in betas:
            psi = apply_mixer(phased, float(b), n)
            e = energy_of(psi, diag, gamma, n)
            f = fidelity(psi, gs)
            if e < best_e:
                best_e, best_e_ang, f_at_e = e, (float(g), float(b)), f
            if f > best_f:
                best_f, best_f_ang, e_at_f = f, (float(g), float(b)), e
            basins_e.append((e, float(g), float(b)))
            basins_f.append((f, float(g), float(b)))
    basins_e.sort()
    basins_f.sort(reverse=True)

    def polish(g0, b0, mode):
        def fun(x):
            psi = evolve(diag, [x[0]], [x[1]], n)
            e = energy_of(psi, diag, gamma, n)
            f = fidelity(psi, gs)
            return e if mode == "E" else -f

        res = minimize(fun, np.array([g0, b0]), method="Nelder-Mead", options={"maxiter": 80, "xatol": 1e-4, "fatol": 1e-8})
        psi = evolve(diag, [res.x[0]], [res.x[1]], n)
        return float(res.x[0]), float(res.x[1]), energy_of(psi, diag, gamma, n), fidelity(psi, gs)

    pe = None
    for _e, g0, b0 in basins_e[:4]:
        rec = polish(g0, b0, "E")
        if pe is None or rec[2] < pe[2]:
            pe = rec
    pf = None
    for _f, g0, b0 in basins_f[:3]:
        rec = polish(g0, b0, "F")
        if pf is None or rec[3] > pf[3]:
            pf = rec
    return {
        "grid_best_F": best_f,
        "grid_F_at_best_E": f_at_e,
        "grid_best_E": best_e,
        "refined_F_at_best_E": pe[3],
        "refined_best_E": pe[2],
        "F1max_best_found": pf[3],
        "E_at_F1max": pf[2],
        "n_g": n_g,
        "n_b": n_b,
        "certified_global": False,
    }


def replay(diag, gamma, n, gs, gammas, betas) -> dict:
    orders = ("cost_then_mixer", "mixer_then_cost")
    out = {}
    for order in orders:
        # also try reversed layer index once inside cost_then_mixer
        trials = {
            order: (gammas, betas),
        }
        if order == "cost_then_mixer":
            trials["cost_then_mixer_reversed_layers"] = (list(reversed(gammas)), list(reversed(betas)))
        for name, (g, b) in trials.items():
            psi = evolve(diag, g, b, n, order="cost_then_mixer" if "cost" in name else "mixer_then_cost")
            out[name] = {
                "F": fidelity(psi, gs),
                "energy": energy_of(psi, diag, gamma, n),
            }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log: list[str] = []

    def say(msg: str) -> None:
        log.append(msg)
        print(msg, flush=True)

    published_terms = json.loads((REF / "term_lists_branchA.json").read_text())
    angles = json.loads((REF / "angles_branchA.json").read_text())
    say("independent BW audit start")

    term_report = []
    for rec in published_terms:
        lx, ly, lam = int(rec["lx"]), int(rec["ly"]), float(rec["lam"])
        mine = term_hash(lx, ly, lam)
        theirs = published_hash(rec["terms"])
        up, down = faces(lx, ly)
        term_report.append({
            "lx": lx, "ly": ly, "lam": lam,
            "hash_match": mine == theirs,
            "my_hash": mine,
            "published_hash": theirs,
            "n_up": len(up), "n_down": len(down),
            "n_shared_supports": len(set(up) & set(down)),
        })
        say(f"terms {lx}x{ly}: hash_match={mine == theirs} shared={len(set(up)&set(down))} M={len(up)+len(down)}")
        if mine != theirs:
            raise SystemExit("term list does not match the published geometry; stop before QAOA")

    # Symmetry and coloring, no diagonalization yet.
    sym_rows = []
    for lx, ly in ((3, 3), (3, 4), (4, 3), (3, 5), (3, 6)):
        col = coloring_conflicts(lx, ly)
        flips = sublattice_flip_commutes(lx, ly)
        # translation invariance of the face sets
        up, down = faces(lx, ly)
        def shifted(face_list, dx, dy):
            out = []
            for f in face_list:
                coords = []
                for s in f:
                    x, y = s % lx, s // lx
                    coords.append(site(x + dx, y + dy, lx, ly))
                out.append(tuple(sorted(coords)))
            return set(out)
        trans_ok = shifted(up, 1, 0) == set(up) and shifted(down, 0, 1) == set(down)
        sym_rows.append({
            "geom": f"{lx}x{ly}",
            "both_div_3": lx % 3 == 0 and ly % 3 == 0,
            "formula_monochrome": col["formula_monochrome_edges"],
            "three_coloring_exists": col["three_coloring_exists"],
            "propagation": col["propagation_outcomes"],
            "z2xz2_commutes": flips["commutes"],
            "translation_fixes_faces": trans_ok,
        })
        say(
            f"sym {lx}x{ly}: coloring={col['three_coloring_exists']} mono_edges={col['formula_monochrome_edges']} "
            f"z2={flips['commutes']} translation={trans_ok} outcomes={col['propagation_outcomes']}"
        )

    # Periods and classical spectra.
    spectra = {}
    for lx, ly in ((3, 3), (3, 4), (3, 5), (3, 6)):
        diag = cost_diagonal(lx, ly, 1.0)
        per = gamma_period_from_diag(diag)
        clas = classical_summary(diag)
        spectra[(lx, ly)] = {"diag": diag, "period": per, "classical": clas}
        say(
            f"classical {lx}x{ly}: Emin={clas['Emin']} deg={clas['degeneracy']} Emax={clas['Emax_diag']} "
            f"gamma_period={per['period']:.6f} half_not={per['half_is_not_period']} spread={per['full_spread']:.2e}"
        )

    # Dense cross-check at N=9.
    diag9 = spectra[(3, 3)]["diag"]
    h9 = np.diag(diag9)
    n9 = 9
    # Pair each basis state with the single bit flip. A rectangular block would
    # connect the wrong pairs.
    for i in range(n9):
        step = 1 << i
        for base in range(0, 1 << n9, step << 1):
            for k in range(step):
                a = base + k
                b = base + step + k
                h9[a, b] -= 1.0
                h9[b, a] -= 1.0
    ev9 = np.linalg.eigvalsh(h9)
    kry9 = krylov_ground(diag9, 1.0, 9, k=2)
    say(
        f"N=9 dense vs Krylov: dE0={abs(ev9[0]-kry9['evals'][0]):.3e} "
        f"gap_dense={ev9[1]-ev9[0]:.6f} gap_krylov={kry9['gap']:.6f} resid={kry9['residuals'][0]:.3e}"
    )

    anchors = [(3, 4, 0.7), (3, 5, 0.6), (3, 6, 1.0)]
    extra = [(3, 4, 0.8), (3, 4, 1.0), (3, 5, 0.8), (3, 5, 1.0), (3, 6, 0.8), (3, 6, 1.0)]
    # 3x6 at 1.0 is already an anchor. Dedup later.
    wanted = []
    for item in anchors + extra:
        if item not in wanted:
            wanted.append(item)

    gs_cache = {}
    replay_rows = []
    gs_rows = []
    l1_rows = []

    for lx, ly, gamma in wanted:
        if time.perf_counter() - t0 > 52 * 60:
            say("time budget: stop before further geometries")
            break
        n = lx * ly
        diag = spectra[(lx, ly)]["diag"]
        say(f"Krylov {lx}x{ly} Gamma={gamma}")
        kg = krylov_ground(diag, gamma, n, k=3)
        gs = kg["gs"]
        gs_cache[(lx, ly, gamma)] = kg
        # translation deviations
        tx = shift_state(gs, lx, ly, 1, 0)
        ty = shift_state(gs, lx, ly, 0, 1)
        # spatial cut for entropy: y < ceil(ly/2) rows, matching a contiguous strip
        rows_keep = ly // 2
        spatial_sites = [site(x, y, lx, ly) for y in range(rows_keep) for x in range(lx)]
        if ly % 2 == 1:
            # both periods odd for 3x5: also report lowest indices, and use N//2 lowest as the quoted cut
            spatial_sites = list(range(n // 2))
            cut_name = f"lowest {n // 2} indices"
        else:
            cut_name = f"y < {rows_keep} (sites {spatial_sites})"
        ent = half_entropy(gs, spatial_sites, n)
        man = manifold_stats(gs, diag, lx, ly)
        # product upper: fewer random cuts at N=18
        n_rand = 12 if n >= 18 else 30
        upper = product_upper(gs, lx, ly, n_random=n_rand)
        row = {
            "lx": lx, "ly": ly, "Gamma": gamma,
            "E0": kg["evals"][0], "E1": kg["evals"][1], "E2": kg["evals"][2],
            "gap": kg["gap"], "gap12": kg["evals"][2] - kg["evals"][0],
            "residuals": kg["residuals"],
            "n_negative": kg["n_negative_amplitudes"],
            "min_amplitude": kg["min_amplitude"],
            "translation_dev_x": float(np.linalg.norm(tx - gs)),
            "translation_dev_y": float(np.linalg.norm(ty - gs)),
            "entropy_bits": ent,
            "entropy_cut": cut_name,
            "F_plus": plus_overlap(gs, n),
            "product_upper": upper["upper"],
            "product_upper_cut": upper["tightest_cut"],
            "product_upper_n_cuts": upper["n_cuts"],
            **man,
        }
        gs_rows.append(row)
        say(
            f"  E0={row['E0']:.8f} gap={row['gap']:.6f} resid={kg['residuals'][0]:.2e} "
            f"neg={row['n_negative']} Tx={row['translation_dev_x']:.2e} Ty={row['translation_dev_y']:.2e} "
            f"S={ent:.3f} F+={row['F_plus']:.4f} Pup={upper['upper']:.4f} "
            f"Pman={man['manifold_weight']:.3f} Ucat={man['uniform_cat_overlap']:.3f} "
            f"orb={man['best_single_orbit_overlap']:.3f} n_orb={man['n_torus_orbits']}"
        )
        # replay stored angles for this field
        for ang in angles:
            if int(ang["lx"]) == lx and int(ang["ly"]) == ly and abs(float(ang["gamma_field"]) - gamma) < 1e-9:
                rep = replay(diag, gamma, n, gs, ang["gammas_raw"], ang["betas_raw"])
                # choose the convention with F closest to theirs
                best_name = min(rep, key=lambda k: abs(rep[k]["F"] - float(ang["F"])))
                dF = rep[best_name]["F"] - float(ang["F"])
                dE = rep[best_name]["energy"] - float(ang["energy"])
                rec = {
                    "lx": lx, "ly": ly, "Gamma": gamma, "L": int(ang["L"]),
                    "F_published": float(ang["F"]),
                    "E_published": float(ang["energy"]),
                    "convention": best_name,
                    "F_replay": rep[best_name]["F"],
                    "E_replay": rep[best_name]["energy"],
                    "dF": dF,
                    "dE": dE,
                    "F_cost_mixer": rep["cost_then_mixer"]["F"],
                    "F_reversed_layers": rep["cost_then_mixer_reversed_layers"]["F"],
                    "F_mixer_first": rep["mixer_then_cost"]["F"],
                }
                replay_rows.append(rec)
                say(
                    f"  replay L={ang['L']} pubF={ang['F']:.6f} myF={rec['F_replay']:.6f} dF={dF:.3e} "
                    f"dE={dE:.3e} conv={best_name}"
                )
        # L1 search on anchors and on 0.8/1.0
        if gamma in (0.6, 0.7, 0.8, 1.0) or (lx, ly, gamma) in anchors:
            if n >= 18:
                ng, nb = 36, 18
            elif n >= 15:
                ng, nb = 48, 24
            else:
                ng, nb = 72, 36
            say(f"  L1 grid {ng}x{nb}")
            period = spectra[(lx, ly)]["period"]["period"]
            found = l1_search(diag, gamma, n, gs, period, ng, nb)
            found.update({"lx": lx, "ly": ly, "Gamma": gamma})
            l1_rows.append(found)
            say(
                f"  L1 best-found Fmax={found['F1max_best_found']:.4f} "
                f"F_at_E={found['refined_F_at_best_E']:.4f} gridF={found['grid_best_F']:.4f}"
            )
        # persist as we go
        (OUT / "gs.json").write_text(json.dumps(gs_rows, indent=2))
        (OUT / "replay.json").write_text(json.dumps(replay_rows, indent=2))
        (OUT / "l1.json").write_text(json.dumps(l1_rows, indent=2))

    (OUT / "symmetries.json").write_text(json.dumps(sym_rows, indent=2))
    (OUT / "terms.json").write_text(json.dumps(term_report, indent=2))
    clas_out = {
        f"{lx}x{ly}": {"classical": spectra[(lx, ly)]["classical"], "period": spectra[(lx, ly)]["period"]}
        for lx, ly in spectra
    }
    (OUT / "classical_periods.json").write_text(json.dumps(clas_out, indent=2))
    (OUT / "verification.log").write_text("\n".join(log) + f"\nelapsed_s {time.perf_counter()-t0:.1f}\n")
    say(f"elapsed_s {time.perf_counter()-t0:.1f}")
    (OUT / "verification.log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
