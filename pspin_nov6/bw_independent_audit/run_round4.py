"""Round-4 replay of Gamma=0.8 Baxter-Wu schedules.

Uses this directory's Hamiltonian, not the other branches' builders.
Snapshots:
  A cursor/sparse-p3-qaoa-round1-13b4 @ fbaac837
  C cursor/pure-p4-ferro-af-7dc4 @ b8d2d35e
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from bw_model import (
    apply_mixer,
    apply_sum_x,
    cost_diagonal,
    evolve,
    gamma_period_from_diag,
    site,
)

HERE = Path(__file__).resolve().parent
REF = HERE / "reference_round4"
OUT = HERE / "results_round4"

A_COMMIT = "fbaac8375a9159678c0ece3e54bde60bbf62d5f1"
C_COMMIT = "b8d2d35e96315a9083ac1dc164b3be987c8ed187"


def krylov(diag, gamma, n, k=2):
    from scipy.sparse.linalg import LinearOperator, eigsh

    def mv(v):
        z = np.asarray(v, dtype=np.float64)
        return diag * z - float(gamma) * apply_sum_x(z, n)

    op = LinearOperator((1 << n, 1 << n), matvec=mv, dtype=np.float64)
    ncv = min(1 << n, max(28, 12 * k))
    evals, vecs = eigsh(op, k=k, which="SA", tol=1e-12, ncv=ncv, maxiter=5000)
    order = np.argsort(evals)
    evals = evals[order]
    vecs = vecs[:, order]
    resids = []
    for i in range(k):
        r = mv(vecs[:, i]) - float(evals[i]) * vecs[:, i]
        resids.append(float(np.linalg.norm(r)))
    gs = vecs[:, 0].real.copy()
    if float(np.sum(gs)) < 0:
        gs *= -1
    return {
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "gap": float(evals[1] - evals[0]),
        "residual": resids[0],
        "residual_e1": resids[1],
        "gs": gs,
        "n_negative": int(np.sum(gs < -1e-8)),
        "min_amplitude": float(np.min(gs)),
    }


def energy_fidelity(psi, diag, gamma, n, gs):
    e = float(np.real(np.vdot(psi, diag * psi))) - float(gamma) * float(np.real(np.vdot(psi, apply_sum_x(psi, n))))
    f = float(abs(np.vdot(gs, psi)) ** 2)
    return e, f


def replay(diag, gamma, n, gs, gammas, betas):
    psi = evolve(diag, gammas, betas, n, order="cost_then_mixer")
    e, f = energy_fidelity(psi, diag, gamma, n, gs)
    return e, f


def entropy_bits(psi, sites, n):
    b_sites = [i for i in range(n) if i not in set(sites)]
    order = list(sites) + b_sites
    idx = np.arange(1 << n, dtype=np.int64)
    perm = np.zeros_like(idx)
    for new, old in enumerate(order):
        perm |= ((idx >> np.int64(old)) & 1) << np.int64(new)
    arranged = np.empty(1 << n, dtype=np.complex128)
    arranged[perm] = np.asarray(psi, dtype=np.complex128)
    na = len(sites)
    amp = arranged.reshape(1 << (n - na), 1 << na)
    rho = amp.conj().T @ amp
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-15]
    s = float(-np.sum(ev * np.log2(ev)))
    lam = float(np.max(ev)) if ev.size else 0.0
    return s, lam


def spatial_cut(lx, ly, axis="y"):
    n = lx * ly
    if axis == "y":
        rows = ly // 2
        sites = [site(x, y, lx, ly) for y in range(rows) for x in range(lx)]
        name = f"y < {rows}"
    else:
        cols = lx // 2
        sites = [site(x, y, lx, ly) for y in range(ly) for x in range(cols)]
        name = f"x < {cols}"
    if len(sites) == 0 or len(sites) == n:
        sites = list(range(n // 2))
        name = f"lowest {n // 2} indices"
    return name, sites


def manifold(gs, diag, lx, ly):
    e_min = float(np.min(diag))
    mask = np.where(np.abs(diag - e_min) < 1e-8)[0]
    amps = gs[mask].astype(np.float64)
    w = float(np.sum(amps ** 2))
    m = int(mask.size)
    # flat cat, equal positive phase
    f_flat = float(np.sum(amps) ** 2 / m) if m else 0.0
    # Cauchy-Schwarz: (sum a)^2 <= m * sum a^2 for real a, so F_flat <= w
    diff = w - f_flat
    if w > 0:
        a = amps / math.sqrt(w)  # normalized projection, sum |a|^2 = 1
    else:
        a = amps
    sum_a2 = float(np.sum(a ** 2))
    sum_a4 = float(np.sum(a ** 4))
    pr = (sum_a2 ** 2 / sum_a4) if sum_a4 > 0 else 0.0
    mean_abs = float(np.mean(np.abs(a))) if a.size else 0.0
    var_abs = float(np.mean((np.abs(a) - mean_abs) ** 2)) if a.size else 0.0
    cv = float(math.sqrt(var_abs) / mean_abs) if mean_abs > 0 else float("nan")
    # phases of the raw amplitudes
    ang = np.angle(gs[mask].astype(np.complex128))
    # fold to [-pi, pi], spread of those that have weight
    if amps.size:
        phase_spread = float(np.max(ang) - np.min(ang))
    else:
        phase_spread = 0.0
    # torus orbits
    seen = set()
    orbits = []
    n = lx * ly
    for x0 in mask.tolist():
        if x0 in seen:
            continue
        orb = []
        for dy in range(ly):
            for dx in range(lx):
                y = 0
                xx = x0
                for i in range(n):
                    if (xx >> i) & 1:
                        ix, iy = i % lx, i // lx
                        j = ((ix + dx) % lx) + lx * ((iy + dy) % ly)
                        y |= 1 << j
                if y not in seen:
                    seen.add(y)
                    orb.append(y)
        if orb:
            ov = float(np.sum(gs[orb]) ** 2 / len(orb))
            orbits.append({"size": len(orb), "F_flat_orbit": ov, "weight": float(np.sum(gs[orb] ** 2))})
    orbits.sort(key=lambda r: -r["weight"])
    return {
        "m": m,
        "Emin_diag": e_min,
        "w": w,
        "F_flat": f_flat,
        "w_minus_F_flat": diff,
        "F_flat_over_w": (f_flat / w) if w > 0 else None,
        "PR": pr,
        "PR_over_m": (pr / m) if m else None,
        "cv_abs_amplitude": cv,
        "var_abs_amplitude": var_abs,
        "phase_spread": phase_spread,
        "n_negative_on_M": int(np.sum(amps < -1e-10)),
        "n_orbits": len(orbits),
        "orbit_sizes": [o["size"] for o in orbits],
        "best_orbit_weight": orbits[0]["weight"] if orbits else None,
        "best_orbit_F_flat": orbits[0]["F_flat_orbit"] if orbits else None,
    }


def expectations(gs, diag_up, diag_down, n):
    p = gs ** 2
    up = float(np.dot(p, diag_up))
    down = float(np.dot(p, diag_down))
    # <sum X>
    sx = float(np.real(np.vdot(gs, apply_sum_x(gs, n))))
    return {"ZZZ_up": up, "ZZZ_down": down, "sum_X": sx, "X_per_site": sx / n}


def l1_converge(diag, gamma, n, gs, period, start_g=40, start_b=20, levels=3):
    """Best-found fidelity. Halve the local step until the change is under 1e-4.

    The first grid covers one period. Later grids only refine a window around
    the current best cell. This is not a certificate of the global maximum.
    """
    def scan(g0, g1, b0, b1, ng, nb):
        gammas = np.linspace(g0, g1, ng, endpoint=False)
        betas = np.linspace(b0, b1, nb, endpoint=False)
        # step sizes for the next window
        dg = (g1 - g0) / ng
        db = (b1 - b0) / nb
        plus = np.full(1 << n, 2.0 ** (-0.5 * n), dtype=np.complex128)
        best_f, best_e, ang = -1.0, None, (0.0, 0.0)
        best_fe = -1.0
        e_star, ang_e = math.inf, (0.0, 0.0)
        for g in gammas:
            phased = plus * np.exp(-1j * float(g) * diag)
            for b in betas:
                psi = apply_mixer(phased, float(b), n)
                e = float(np.real(np.vdot(psi, diag * psi))) - float(gamma) * float(np.real(np.vdot(psi, apply_sum_x(psi, n))))
                f = float(abs(np.vdot(gs, psi)) ** 2)
                if f > best_f:
                    best_f, best_e, ang = f, e, (float(g), float(b))
                if e < e_star:
                    e_star, ang_e, best_fe = e, (float(g), float(b)), f
        return best_f, best_e, ang, dg, db, best_fe, e_star

    history = []
    f, e, ang, dg, db, fe, ee = scan(0.0, period, 0.0, math.pi, start_g, start_b)
    history.append({"stage": "full_period", "F": f, "F_at_best_E": fe, "n_g": start_g, "n_b": start_b, "dg": dg, "db": db})
    for level in range(levels):
        g0 = ang[0] - 1.5 * dg
        g1 = ang[0] + 1.5 * dg
        b0 = ang[1] - 1.5 * db
        b1 = ang[1] + 1.5 * db
        ng, nb = start_g, start_b
        f2, e2, ang2, dg, db, fe2, ee2 = scan(g0, g1, b0, b1, ng, nb)
        # also polish
        def fun(x):
            psi = evolve(diag, [x[0]], [x[1]], n)
            return -float(abs(np.vdot(gs, psi)) ** 2)

        res = minimize(fun, np.array(ang2), method="Nelder-Mead", options={"maxiter": 60, "xatol": 1e-5, "fatol": 1e-10})
        psi = evolve(diag, [res.x[0]], [res.x[1]], n)
        f_ref = float(abs(np.vdot(gs, psi)) ** 2)
        delta = abs(f_ref - f)
        history.append({
            "stage": f"refine_{level}",
            "grid_F": f2,
            "refined_F": f_ref,
            "delta_vs_previous_best": delta,
            "dg": dg,
            "db": db,
        })
        f = max(f, f_ref, f2)
        ang = (float(res.x[0]), float(res.x[1]))
        if delta < 1e-4 and level > 0:
            break
    return {"F1max_best_found": f, "certified_global": False, "history": history}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log = []

    def say(msg):
        log.append(msg)
        print(msg, flush=True)

    A = json.loads((REF / "A_angles.json").read_text())
    C = json.loads((REF / "C_angles.json").read_text())
    T = json.loads((REF / "A_n18_transfer.json").read_text())
    say(f"snapshot A {A_COMMIT}")
    say(f"snapshot C {C_COMMIT}")
    say("no round4 directory on A at this fetch")

    geos = [(3, 4), (4, 3), (3, 5), (3, 6), (4, 4)]
    diags = {}
    periods = {}
    for lx, ly in geos:
        d = cost_diagonal(lx, ly, 1.0)
        per = gamma_period_from_diag(d)
        diags[(lx, ly)] = d
        periods[(lx, ly)] = per
        say(
            f"period {lx}x{ly}: T={per['period']:.10f} gcd_half={per['gcd_half_units']} "
            f"full_spread={per['full_spread']:.2e} half_spread={per['half_spread']:.3e} half_not={per['half_is_not_period']} "
            f"Emin={d.min()} deg={int(np.sum(np.abs(d-d.min())<1e-8))}"
        )
    # 3x4 vs 4x3 histogram
    h1 = np.unique(np.rint(diags[(3, 4)]).astype(np.int64), return_counts=True)
    h2 = np.unique(np.rint(diags[(4, 3)]).astype(np.int64), return_counts=True)
    same_hist = h1[0].tolist() == h2[0].tolist() and h1[1].tolist() == h2[1].tolist()
    say(f"3x4 and 4x3 cost histograms identical: {same_hist}")

    # Ground states we need
    needed = [(3, 4, 0.8), (3, 5, 0.8), (3, 6, 0.8), (4, 4, 0.8), (3, 4, 0.6), (4, 3, 0.6), (4, 3, 0.8)]
    gs = {}
    for lx, ly, gamma in needed:
        say(f"GS {lx}x{ly} Gamma={gamma}")
        gs[(lx, ly, gamma)] = krylov(diags[(lx, ly)], gamma, lx * ly)
        g = gs[(lx, ly, gamma)]
        say(f"  E0={g['E0']:.10f} gap={g['gap']:.6e} resid={g['residual']:.3e} neg={g['n_negative']} minA={g['min_amplitude']:.3e}")

    def do_replay(tag, lx, ly, gamma, gammas, betas, f_pub, e_pub, source):
        g = gs[(lx, ly, gamma)]
        e, f = replay(diags[(lx, ly)], gamma, lx * ly, g["gs"], gammas, betas)
        dF = f - float(f_pub)
        dE = e - float(e_pub)
        rec = {
            "tag": tag, "lx": lx, "ly": ly, "Gamma": gamma, "L": len(list(gammas)),
            "F_published": float(f_pub), "E_published": float(e_pub),
            "F_replay": f, "E_replay": e, "dF": dF, "dE": dE,
            "pass_1e-6": abs(dF) <= 1e-6 and abs(dE) <= 1e-6,
            "pass_1e-10": abs(dF) <= 1e-10 and abs(dE) <= 1e-8,
            "source": source,
        }
        say(f"replay {tag} {lx}x{ly} G={gamma} L={rec['L']} dF={dF:.3e} dE={dE:.3e} pass1e-6={rec['pass_1e-6']}")
        return rec

    replays = []
    # A matched Gamma 0.8, all stored depths, for 3x4, 3x5, 3x6
    for r in A:
        if abs(r["gamma"] - 0.8) < 1e-9 and (r["lx"], r["ly"]) in ((3, 4), (3, 5), (3, 6)):
            label = "blind" if (r["n"] == 18) else "matched"
            replays.append(do_replay(
                f"A_{label}", r["lx"], r["ly"], 0.8, r["gammas"], r["betas"], r["F"], r["energy"],
                f"A:{A_COMMIT[:12]} role={r['role']} winner={r['winner']}",
            ))
    # C 4x4 and 4x3 at 0.8, all depths
    for r in C:
        if abs(r["Gamma"] - 0.8) < 1e-9 and (r["lx"], r["ly"]) in ((4, 4), (4, 3)):
            replays.append(do_replay(
                "C", r["lx"], r["ly"], 0.8, r["gammas"], r["betas"], r["F"], r["energy"],
                f"C:{C_COMMIT[:12]} winner={r['winner']}",
            ))
    # N18 transfer: raw 3x4 L=5 angles on 3x6, and polished angles
    a_l5 = next(r for r in A if r["n"] == 12 and abs(r["gamma"] - 0.8) < 1e-9 and r["L"] == 5)
    replays.append(do_replay(
        "N18_transferred_raw_3x4_L5", 3, 6, 0.8, a_l5["gammas"], a_l5["betas"],
        next(t["raw_transfer_F"] for t in T if abs(t["gamma"] - 0.8) < 1e-9),
        # energy of the raw transfer was not stored; compare F only, energy published as nan sentinel
        a_l5["energy"],  # placeholder, overwritten below
        "raw transfer of A 3x4 L=5 angles onto 3x6; published energy is the 3x4 energy, not the 3x6 energy",
    ))
    # fix: don't treat 3x4 energy as the published 3x6 energy
    replays[-1]["E_published"] = None
    replays[-1]["dE"] = None
    replays[-1]["pass_1e-6"] = abs(replays[-1]["dF"]) <= 1e-6
    replays[-1]["pass_1e-10"] = abs(replays[-1]["dF"]) <= 1e-10
    say(f"  transferred raw F_replay={replays[-1]['F_replay']:.6f} published {replays[-1]['F_published']:.6f}")

    pol = next(t for t in T if abs(t["gamma"] - 0.8) < 1e-9)
    replays.append(do_replay(
        "N18_transfer_polish", 3, 6, 0.8, pol["gammas"], pol["betas"], pol["reopt_F"], pol["reopt_E"],
        "A n18_transfer_reopt.json polish",
    ))

    # Both N12 Gamma 0.6 ladders
    for r in A:
        if r["n"] == 12 and abs(r["gamma"] - 0.6) < 1e-9:
            replays.append(do_replay(
                "A_N12_G0.6", 3, 4, 0.6, r["gammas"], r["betas"], r["F"], r["energy"],
                f"A 3x4 winner={r['winner']}",
            ))
    for r in C:
        if r["lx"] == 4 and r["ly"] == 3 and abs(r["Gamma"] - 0.6) < 1e-9:
            replays.append(do_replay(
                "C_N12_G0.6_orientation", 4, 3, 0.6, r["gammas"], r["betas"], r["F"], r["energy"],
                f"C 4x3 winner={r['winner']}",
            ))

    # Diagnostics at Gamma 0.8
    diag_rows = []
    for lx, ly in ((3, 4), (3, 5), (3, 6), (4, 4), (4, 3)):
        gamma = 0.8
        g = gs[(lx, ly, gamma)]
        man = manifold(g["gs"], diags[(lx, ly)], lx, ly)
        # separate up/down diagonals
        # rebuild by zeroing: use cost pieces
        from bw_model import clauses
        n = lx * ly
        xs = np.arange(1 << n, dtype=np.uint32)
        up = np.zeros(xs.shape[0])
        down = np.zeros(xs.shape[0])
        for kind, qubits, coeff in clauses(lx, ly, 1.0):
            parity = np.zeros(xs.shape[0], dtype=np.uint32)
            for q in qubits:
                parity ^= (xs >> np.uint32(q)) & np.uint32(1)
            prod = 1.0 - 2.0 * parity
            if kind == "up":
                up += coeff * prod  # coeff is -1, so this is the up contribution to H
            else:
                down += coeff * prod
        exp = expectations(g["gs"], up, down, n)
        cuts = {}
        for axis in ("y", "x"):
            name, sites = spatial_cut(lx, ly, axis)
            if len(sites) == 0 or len(set(sites)) in (0, n):
                continue
            s, lam = entropy_bits(g["gs"], sites, n)
            cuts[name] = {"entropy_bits": s, "lambda_max": lam, "sites": sites}
        # a few random cuts for a tighter product upper
        rng = np.random.default_rng(1)
        best_lam = min(v["lambda_max"] for v in cuts.values())
        best_name = min(cuts, key=lambda k: cuts[k]["lambda_max"])
        for i in range(16):
            choice = sorted(int(x) for x in rng.choice(n, size=n // 2, replace=False))
            _s, lam = entropy_bits(g["gs"], choice, n)
            if lam < best_lam:
                best_lam, best_name = lam, f"random_{i}"
        row = {
            "lx": lx, "ly": ly, "Gamma": gamma,
            "E0": g["E0"], "gap": g["gap"], "residual": g["residual"],
            "n_negative": g["n_negative"],
            "cuts": {k: {kk: vv for kk, vv in v.items() if kk != "sites"} | {"n_sites": len(v["sites"])} for k, v in cuts.items()},
            "product_upper_schmidt": best_lam,
            "product_upper_cut": best_name,
            "expectations": exp,
            **man,
        }
        diag_rows.append(row)
        say(
            f"diag {lx}x{ly} w={man['w']:.6f} Fflat={man['F_flat']:.6f} diff={man['w_minus_F_flat']:.6e} "
            f"PR={man['PR']:.3f}/m={man['m']} CV={man['cv_abs_amplitude']:.4f} "
            f"orbits={man['n_orbits']} best_orbit_w={man['best_orbit_weight']:.4f} "
            f"Schmidt={best_lam:.4f} gap={g['gap']:.4e} Skeys={list(cuts)}"
        )

    # L1 convergence at Gamma 0.8 for the four commit sizes
    l1_rows = []
    for lx, ly, ng, nb in ((3, 4, 48, 24), (3, 5, 36, 18), (4, 4, 32, 16), (3, 6, 28, 14)):
        say(f"L1 converge {lx}x{ly}")
        per = periods[(lx, ly)]["period"]
        found = l1_converge(diags[(lx, ly)], 0.8, lx * ly, gs[(lx, ly, 0.8)]["gs"], per, start_g=ng, start_b=nb, levels=2)
        found.update({"lx": lx, "ly": ly, "Gamma": 0.8, "period": per})
        l1_rows.append(found)
        say(f"  F1max_best_found={found['F1max_best_found']:.6f} history_deltas={[h.get('delta_vs_previous_best') for h in found['history']]}")

    # ratio check using replayed deep F and this F1max
    ratios = []
    deep_map = {
        (3, 4): next(r for r in replays if r["tag"] == "A_matched" and r["lx"] == 3 and r["ly"] == 4 and r["L"] == 6),
        (3, 5): next(r for r in replays if r["tag"] == "A_matched" and r["lx"] == 3 and r["ly"] == 5 and r["L"] == 6),
        (3, 6): next(r for r in replays if r["tag"] == "A_blind" and r["L"] == 5),
        (4, 4): next(r for r in replays if r["tag"] == "C" and r["lx"] == 4 and r["ly"] == 4 and r["L"] == 5),
    }
    for row in l1_rows:
        key = (row["lx"], row["ly"])
        deep = deep_map[key]["F_replay"]
        f1 = row["F1max_best_found"]
        ratios.append({
            "lx": row["lx"], "ly": row["ly"],
            "F1max_best_found": f1,
            "F_deep_replay": deep,
            "ratio_deep_over_F1": deep / f1 if f1 else None,
            "deep_ge_2_F1": deep >= 2 * f1,
            "deep_ge_0.4": deep >= 0.4,
        })
        say(f"ratio {row['lx']}x{row['ly']} Fdeep={deep:.4f} F1={f1:.4f} two_x={deep >= 2*f1}")

    payload = {
        "A_commit": A_COMMIT,
        "C_commit": C_COMMIT,
        "replays": replays,
        "diagnostics": diag_rows,
        "l1": l1_rows,
        "ratios": ratios,
        "elapsed_s": time.perf_counter() - t0,
    }
    (OUT / "round4.json").write_text(json.dumps(payload, indent=2))
    (OUT / "verification.log").write_text("\n".join(log) + "\n")
    say(f"elapsed {payload['elapsed_s']:.1f}")
    (OUT / "verification.log").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
