"""Close the N=18, Gamma=0.8 energy search. No new Hamiltonian and no N=21.

48 starts at L=5 and at L=6 on the 3x6 staggered Baxter-Wu torus.
Each start is polished locally. Best-energy and best-fidelity are both kept.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import differential_evolution

import geometry as geo
import model as M
import qaoa

THIS = Path(__file__).resolve().parent
OUT = THIS / "runs" / "round4"
GAMMA = 0.8
LX, LY, N = 3, 6, 18
BUDGET = 48
STALL_WINDOW = 16
REL_TOL = 1e-4

HC = None
GS = None
PERIOD = None
PSI0 = None


def say(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as f:
        f.write(line + "\n")


def energy_F(x):
    L = len(x) // 2
    psi = qaoa.apply_qaoa(PSI0, HC, N, x[:L], x[L:])
    E = qaoa.energy_of(psi, HC, N, GAMMA)[0]
    F = float(abs(np.vdot(GS, psi)) ** 2)
    return E, F


def grad_norm(x):
    """Forward-difference Euclidean norm of the energy gradient."""
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-5
    e0, _ = energy_F(x)
    g2 = 0.0
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        e1, _ = energy_F(xp)
        g2 += ((e1 - e0) / eps) ** 2
    return float(np.sqrt(g2))


def polish(x0, maxiter=16):
    x0 = np.asarray(x0, dtype=np.float64)
    L = len(x0) // 2
    bounds = [(-PERIOD, 2 * PERIOD)] * L + [(-np.pi, 2 * np.pi)] * L
    e_seed, f_seed = energy_F(x0)

    def obj(x):
        return energy_F(x)[0]

    x_opt, val, nfev = qaoa._lbfgs(obj, x0, maxiter, bounds)
    e_opt, f_opt = energy_F(x_opt)
    if e_seed <= e_opt:
        return x0, e_seed, f_seed, e_seed, f_seed, True, int(nfev)
    return x_opt, e_opt, f_opt, e_seed, f_seed, False, int(nfev)


def de_vector(L, seed):
    bounds = [(-PERIOD, 2 * PERIOD)] * L + [(-np.pi, 2 * np.pi)] * L

    def obj(x):
        return energy_F(x)[0]

    res = differential_evolution(
        obj,
        bounds,
        seed=int(seed),
        maxiter=4,
        popsize=5,
        mutation=0.5,
        recombination=0.7,
        atol=0.0,
        tol=1e-3,
        workers=1,
        updating="immediate",
        polish=False,
    )
    return np.asarray(res.x, dtype=np.float64)


def worker(job):
    sid, seed_type, L, x0, de_seed = job
    t0 = time.time()
    if x0 is None:
        x0 = de_vector(L, de_seed)
    x_seed = np.asarray(x0, dtype=np.float64)
    x, E, F, E_seed, F_seed, kept, nfev = polish(x_seed)
    g = grad_norm(x)
    return {
        "start_id": int(sid),
        "seed_type": seed_type,
        "L": int(L),
        "E": float(E),
        "F": float(F),
        "E_seed": float(E_seed),
        "F_seed": float(F_seed),
        "grad_norm": g,
        "kept_seed": bool(kept),
        "nfev": nfev,
        "seconds": time.time() - t0,
        "gammas": [float(v) for v in x[:L]],
        "betas": [float(v) for v in x[L:]],
        "gammas_seed": [float(v) for v in x_seed[:L]],
        "betas_seed": [float(v) for v in x_seed[L:]],
    }


def rel_drop(e_old, e_new):
    if not np.isfinite(e_old):
        return 1.0
    return float((e_old - e_new) / max(abs(e_old), 1e-12))


def load_angles():
    cells = [json.loads(line) for line in open(THIS / "runs/round3/cells.jsonl")]
    found = {}
    for c in cells:
        t = c["task"]
        if abs(t["gamma"] - 0.8) < 1e-9 and (t["lx"], t["ly"]) in ((3, 4), (3, 5), (3, 6)):
            found[(t["lx"], t["ly"])] = c
    reopt = json.load(open(THIS / "runs/round3/n18_transfer_reopt.json"))
    reopt08 = next(r for r in reopt if abs(r["gamma"] - 0.8) < 1e-9)
    return found, reopt08


def pack_known(ang):
    return list(ang["gammas"]) + list(ang["betas"])


def interp_pack(ang_short, L):
    g = qaoa.interpolate_schedule(ang_short["gammas"], L)
    b = qaoa.interpolate_schedule(ang_short["betas"], L)
    return list(g) + list(b)


def pad_pack(ang_short, L, where):
    g = list(ang_short["gammas"])
    b = list(ang_short["betas"])
    while len(g) < L:
        if where == "append":
            g.append(0.0)
            b.append(0.0)
        else:
            g.insert(0, 0.0)
            b.insert(0, 0.0)
    return g[:L] + b[:L]


def build_seed_list(L, found, reopt08, rng, bestE_prev, bestF_prev):
    """Structured seeds first, then random, capped at BUDGET."""
    seeds = []
    n12 = found[(3, 4)]["angles_for_transfer"]
    n15 = found[(3, 5)]["angles_for_transfer"]
    n18 = found[(3, 6)]["angles_for_transfer"]
    if str(L) in n12:
        seeds.append(("n12_transfer", pack_known(n12[str(L)])))
    if str(L) in n15:
        seeds.append(("n15_transfer", pack_known(n15[str(L)])))
    if str(L) in n18 and len(n18[str(L)]["gammas"]) == L:
        seeds.append(("n18_round3_bestE", pack_known(n18[str(L)])))
    if L == 5:
        seeds.append(("n18_transfer_polish", list(reopt08["gammas"]) + list(reopt08["betas"])))
    if bestE_prev is not None and len(bestE_prev["gammas"]) == L - 1:
        seeds.append(("pagano_interp_bestE", interp_pack(bestE_prev, L)))
        seeds.append(("zeropad_append_bestE", pad_pack(bestE_prev, L, "append")))
        seeds.append(("zeropad_prepend_bestE", pad_pack(bestE_prev, L, "prepend")))
    if bestF_prev is not None and len(bestF_prev["gammas"]) == L - 1:
        seeds.append(("pagano_interp_bestF", interp_pack(bestF_prev, L)))
        seeds.append(("zeropad_append_bestF", pad_pack(bestF_prev, L, "append")))
    # Two differential-evolution starts, marked by None vector.
    seeds.append(("de", None))
    seeds.append(("de", None))
    while len(seeds) < BUDGET:
        x = np.concatenate([
            rng.uniform(0.0, PERIOD, L),
            rng.uniform(0.0, np.pi, L),
        ])
        seeds.append(("random", x))
    return seeds[:BUDGET]


def run_depth(L, seeds):
    """Run up to 48 starts. Stop if a block of 16 does not improve best energy."""
    import multiprocessing as mp

    jobs = []
    for i, (kind, vec) in enumerate(seeds):
        x0 = None if vec is None else [float(v) for v in vec]
        jobs.append((i, kind, L, x0, 4000 + 100 * L + i))
    rows = []
    best_E = np.inf
    best_at_block_start = np.inf
    stop_reason = "budget_exhausted"
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=4) as pool:
        pending = list(jobs)
        while pending:
            block = pending[:STALL_WINDOW]
            pending = pending[STALL_WINDOW:]
            best_at_block_start = best_E
            for row in pool.imap_unordered(worker, block):
                rows.append(row)
                if row["E"] < best_E:
                    best_E = row["E"]
                say(
                    f"  L={L} id={row['start_id']} {row['seed_type']} "
                    f"E={row['E']:.6f} F={row['F']:.4f} Fseed={row['F_seed']:.4f} "
                    f"|g|={row['grad_norm']:.3e} {row['seconds']:.0f}s"
                )
            gain = rel_drop(best_at_block_start, best_E)
            say(f"  L={L} block done n={len(rows)} bestE={best_E:.6f} rel_gain={gain:.3e}")
            if len(rows) >= STALL_WINDOW and gain < REL_TOL:
                stop_reason = "stall_rel_1e-4_over_16_starts"
                break
    rows.sort(key=lambda r: r["start_id"])
    return rows, stop_reason, float(best_E)


def pick(rows):
    """Best polished energy, and best fidelity among seed and polished values."""
    best_e = min(rows, key=lambda r: r["E"])
    # Fidelity candidate may be the unpolished seed.
    best_f = None
    best_f_val = -1.0
    for r in rows:
        if r["F"] > best_f_val:
            best_f_val = r["F"]
            best_f = {
                "source": "polished",
                "start_id": r["start_id"],
                "seed_type": r["seed_type"],
                "E": r["E"],
                "F": r["F"],
                "grad_norm": r["grad_norm"],
                "gammas": r["gammas"],
                "betas": r["betas"],
            }
        if r["F_seed"] > best_f_val:
            best_f_val = r["F_seed"]
            best_f = {
                "source": "seed_before_polish",
                "start_id": r["start_id"],
                "seed_type": r["seed_type"],
                "E": r["E_seed"],
                "F": r["F_seed"],
                "grad_norm": None,
                "gammas": r["gammas_seed"],
                "betas": r["betas_seed"],
                "note": "These angles are the unpolished seed. Polishing lowered F while changing E; both values are in all_runs.csv.",
            }
    return best_e, best_f


def prior_ladder():
    cells = [json.loads(line) for line in open(THIS / "runs/round3/cells.jsonl")]
    cell = next(
        c for c in cells
        if c["task"]["n"] == 18 and abs(c["task"]["gamma"] - 0.8) < 1e-9
    )
    energy_rows = []
    fidelity_rows = []
    for r in cell["ladder"]:
        if r["L"] > 4:
            continue
        energy_rows.append({
            "L": r["L"],
            "E": r["energy"],
            "F": r["F"],
            "source": "round3_energy_search",
            "gammas": r.get("gammas"),
            "betas": r.get("betas"),
        })
        f_best = r["F"]
        e_at = r["energy"]
        if r["L"] == 1:
            f_best = max(f_best, r.get("l1_grid_max_fidelity", f_best), r.get("l1_fidelity_max_F", f_best))
            e_at = r.get("l1_fidelity_max_energy", e_at)
        for b in r.get("top_basins") or []:
            if b["F"] > f_best:
                f_best = b["F"]
                e_at = b["energy"]
        fidelity_rows.append({
            "L": r["L"],
            "E": e_at,
            "F": f_best,
            "source": "round3_best_F_among_logged_basins" if r["L"] > 1 else "round3_L1_fidelity_search",
        })
    return cell, energy_rows, fidelity_rows


def shared_diag(angles_g, angles_b):
    L = len(angles_g)
    starts = [("mean", np.concatenate([[float(np.mean(angles_g))], np.asarray(angles_b, dtype=np.float64)]))]
    rng = np.random.default_rng(80 + L)
    for i in range(4):
        starts.append((
            f"r{i}",
            np.concatenate([
                rng.uniform(0.0, PERIOD, 1),
                rng.uniform(-np.pi, np.pi, L),
            ]),
        ))
    bounds = [(-PERIOD, 2 * PERIOD)] + [(-np.pi, 2 * np.pi)] * L

    def obj(x):
        psi = qaoa.apply_qaoa(PSI0, HC, N, [float(x[0])] * L, x[1:])
        return qaoa.energy_of(psi, HC, N, GAMMA)[0]

    best_x, best_E = None, np.inf
    for _, x0 in starts:
        e_seed = float(obj(x0))
        x_opt, val, _ = qaoa._lbfgs(obj, x0, 20, bounds)
        if e_seed < val:
            x_opt, val = np.asarray(x0, dtype=np.float64), e_seed
        if val < best_E:
            best_x, best_E = x_opt, val
    psi = qaoa.apply_qaoa(PSI0, HC, N, [float(best_x[0])] * L, best_x[1:])
    F = float(abs(np.vdot(GS, psi)) ** 2)
    return {"L": L, "E": float(best_E), "F": F, "gamma": float(best_x[0]), "role": "shared_gamma_diagnostic"}


def write_csv(path, rows):
    import csv
    cols = [
        "start_id", "seed_type", "L", "E", "F", "grad_norm",
        "E_seed", "F_seed", "kept_seed", "nfev", "seconds", "gammas", "betas",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["gammas"] = json.dumps(r["gammas"])
            out["betas"] = json.dumps(r["betas"])
            w.writerow(out)


def write_report(energy_ladder, fidelity_ladder, rows5, rows6, stop5, stop6, shared, best5, best6):
    def fmt_row(r):
        return f"| {r['L']} | {r['E']:.6f} | {r['F']:.4f} | {r.get('source', '')} |"

    # Cross: at L=5 and L=6, is the max-F point the same as the min-E point?
    def same_point(be, bf):
        if bf.get("gammas") is None:
            return False
        return abs(be["E"] - bf["E"]) < 1e-6 and abs(be["F"] - bf["F"]) < 1e-6

    cross_lines = []
    for L, rows, be, bf in (
        (5, rows5, best5[0], best5[1]),
        (6, rows6, best6[0], best6[1]),
    ):
        distinct = not same_point(be, bf)
        cross_lines.append(
            f"L={L}: best-energy F={be['F']:.4f} at E={be['E']:.6f}; "
            f"best-fidelity F={bf['F']:.4f} at E={bf['E']:.6f} "
            f"({bf.get('source', 'polished')}, seed {bf['seed_type']}). "
            + ("These are different states." if distinct else "These are the same state.")
        )
    # Do the ladders cross? Compare which one is ahead in F.
    eF = {r["L"]: r["F"] for r in energy_ladder}
    fF = {r["L"]: r["F"] for r in fidelity_ladder}
    ahead = []
    for L in range(1, 7):
        if L not in eF or L not in fF:
            continue
        if fF[L] > eF[L] + 1e-4:
            ahead.append(f"L={L} best-F is higher by {fF[L] - eF[L]:.4f}")
        else:
            ahead.append(f"L={L} best-E and best-F agree within 0.0001")
    lines = []
    lines.append("# Round 4 report — N=18, Gamma=0.8")
    lines.append("")
    lines.append("Hamiltonian, unscaled, 3×6 torus only:")
    lines.append("")
    lines.append("`H = -sum_▲ ZZZ + sum_▼ ZZZ - 0.8 * sum X`")
    lines.append("")
    lines.append("Full-space ground state. Independent (γ_ℓ, β_ℓ), energy objective.")
    lines.append("Shared-γ is a side diagnostic and is not the primary schedule.")
    lines.append("")
    lines.append("## Stop")
    lines.append("")
    lines.append(f"- L=5: {stop5}, starts logged {len(rows5)} of {BUDGET}.")
    lines.append(f"- L=6: {stop6}, starts logged {len(rows6)} of {BUDGET}.")
    lines.append("")
    lines.append("Stop rule: budget 48, or the best energy improves by less than 1e-4 relative over a block of 16 starts.")
    lines.append("Relative gain is (E_before - E_after) / |E_before|.")
    lines.append("")
    lines.append("## One-line cross statement")
    lines.append("")
    for s in cross_lines:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("The ladders differ wherever best-fidelity is strictly above best-energy fidelity: " + "; ".join(ahead) + ".")
    lines.append("")
    lines.append("The round-3 transfer comparison is kept: raw 3×4 angles at L=5 have F≈0.524, and an energy polish of that seed previously reached F≈0.471 at a lower energy. Both the seed fidelity and the polished fidelity are in `all_runs.csv` (`F_seed` and `F`).")
    lines.append("")
    lines.append("## Best-energy ladder")
    lines.append("")
    lines.append("| L | E | F | source |")
    lines.append("|---:|---:|---:|---|")
    for r in energy_ladder:
        lines.append(fmt_row(r))
    lines.append("")
    lines.append("## Best-fidelity ladder")
    lines.append("")
    lines.append("| L | E | F | source |")
    lines.append("|---:|---:|---:|---|")
    for r in fidelity_ladder:
        lines.append(fmt_row(r))
    lines.append("")
    g5 = best5[0]["grad_norm"]
    g6 = best6[0]["grad_norm"]
    lines.append("## Convergence")
    lines.append("")
    lines.append(f"Best-energy gradient norm at L=5 is {g5:.3e}, at L=6 is {g6:.3e}.")
    lines.append("Deep fidelities are the fidelities of the best energies located in this budget.")
    if stop5 != "stall_rel_1e-4_over_16_starts" or stop6 != "stall_rel_1e-4_over_16_starts" or g5 > 1e-2 or g6 > 1e-2:
        lines.append("Where the stall rule did not fire or the gradient norm is still above 1e-2, treat that deep F as a lower bound on the fidelity of a fully settled energy minimum: a longer search can still move the energy, and the fidelity can rise or fall when it does.")
    else:
        lines.append("Both depths hit the stall rule and the best-energy gradients are at most 1e-2. The located points are locally stationary to that tolerance; the fidelities are still not a certificate of the global minimum.")
    lines.append("")
    lines.append("## Shared-gamma diagnostic")
    lines.append("")
    for s in shared:
        lines.append(f"- L={s['L']}: shared-γ E={s['E']:.6f}, F={s['F']:.4f}. Not the primary ansatz.")
    lines.append("")
    lines.append("L=1..4 rows are the round-3 search on this same state, not a new optimization. L=2..4 best-fidelity uses the best F among the basins that round 3 logged, which was an energy search. L=5 and L=6 best-fidelity is the maximum of F_seed and polished F in this budget.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "REPORT.md").write_text(text)
    return text


def main():
    global HC, GS, PERIOD, PSI0
    OUT.mkdir(parents=True, exist_ok=True)
    say("round4 N=18 Gamma=0.8")
    geom = geo.build_triangles(LX, LY)
    tris, coeffs = M.faces_and_coeffs(geom, 1.0)
    HC = M.cost_diagonal(N, tris, coeffs)
    PERIOD = float(M.energy_step_and_period(HC)["gamma_period"])
    spec = M.ground_state(HC, GAMMA, N)
    GS = spec["gs"]
    PSI0 = M.plus_state(N)
    say(f"GS E0={spec['E0']:.8f} gap={spec['gap']:.6f} resid={spec['residual']:.2e} period={PERIOD:.6f}")
    found, reopt08 = load_angles()
    # Sanity: raw N12 L=5 transfer fidelity should be near 0.524.
    x_check = pack_known_safe(found[(3, 4)]["angles_for_transfer"]["5"])
    e_c, f_c = energy_F(x_check)
    say(f"sanity N12 L5 transfer on N18: E={e_c:.6f} F={f_c:.4f}")

    cell, e_prior, f_prior = prior_ladder()
    rng = np.random.default_rng(np.random.PCG64(20261106))
    def _ang(row):
        if row and row.get("gammas") and row.get("betas"):
            return row
        return None

    seeds5 = build_seed_list(5, found, reopt08, rng, _ang(e_prior[-1] if e_prior else None), _ang(f_prior[-1] if f_prior else None))
    # e_prior last is L=4 only if we filtered L<=4 and ladder includes L=0..4. prior returns L=1..4. Last is L=4.
    say(f"L=5 seeds structured={sum(1 for k,_ in seeds5 if k!='random')} random={sum(1 for k,_ in seeds5 if k=='random')}")
    rows5, stop5, _ = run_depth(5, seeds5)
    bestE5, bestF5 = pick(rows5)
    say(f"L=5 stop={stop5} bestE={bestE5['E']:.6f} F={bestE5['F']:.4f} bestF={bestF5['F']:.4f}")

    bestE_prev = {"gammas": bestE5["gammas"], "betas": bestE5["betas"]}
    bestF_prev = {"gammas": bestF5["gammas"], "betas": bestF5["betas"]}
    seeds6 = build_seed_list(6, found, reopt08, rng, bestE_prev, bestF_prev)
    say(f"L=6 seeds structured={sum(1 for k,_ in seeds6 if k!='random')}")
    rows6, stop6, _ = run_depth(6, seeds6)
    bestE6, bestF6 = pick(rows6)
    say(f"L=6 stop={stop6} bestE={bestE6['E']:.6f} F={bestE6['F']:.4f} bestF={bestF6['F']:.4f}")

    shared = [
        shared_diag(bestE5["gammas"], bestE5["betas"]),
        shared_diag(bestE6["gammas"], bestE6["betas"]),
    ]
    say(f"shared L5 F={shared[0]['F']:.4f} L6 F={shared[1]['F']:.4f}")

    all_rows = rows5 + rows6
    write_csv(OUT / "all_runs.csv", all_rows)
    (OUT / "angles_best_energy.json").write_text(json.dumps({
        "L5": {k: bestE5[k] for k in ("E", "F", "grad_norm", "gammas", "betas", "seed_type", "start_id")},
        "L6": {k: bestE6[k] for k in ("E", "F", "grad_norm", "gammas", "betas", "seed_type", "start_id")},
        "gamma": GAMMA, "lx": LX, "ly": LY,
        "ansatz": "independent gamma_l beta_l",
        "objective": "energy",
    }, indent=2) + "\n")
    (OUT / "angles_best_fidelity.json").write_text(json.dumps({
        "L5": bestF5, "L6": bestF6, "gamma": GAMMA, "lx": LX, "ly": LY,
        "note": "Maximum of seed fidelity and polished fidelity in this budget. Not the energy objective.",
    }, indent=2) + "\n")

    energy_ladder = list(e_prior) + [
        {"L": 5, "E": bestE5["E"], "F": bestE5["F"], "source": "round4_best_energy", "gammas": bestE5["gammas"], "betas": bestE5["betas"]},
        {"L": 6, "E": bestE6["E"], "F": bestE6["F"], "source": "round4_best_energy", "gammas": bestE6["gammas"], "betas": bestE6["betas"]},
    ]
    fidelity_ladder = list(f_prior) + [
        {"L": 5, "E": bestF5["E"], "F": bestF5["F"], "source": "round4_" + bestF5.get("source", "polished")},
        {"L": 6, "E": bestF6["E"], "F": bestF6["F"], "source": "round4_" + bestF6.get("source", "polished")},
    ]
    (OUT / "ladders.json").write_text(json.dumps({
        "best_energy": energy_ladder, "best_fidelity": fidelity_ladder,
        "stop_L5": stop5, "stop_L6": stop6, "shared_gamma": shared,
        "gs": {"E0": spec["E0"], "gap": spec["gap"], "residual": spec["residual"]},
    }, indent=2) + "\n")
    text = write_report(energy_ladder, fidelity_ladder, rows5, rows6, stop5, stop6, shared, (bestE5, bestF5), (bestE6, bestF6))
    say("wrote " + str(OUT / "REPORT.md"))
    print(text)


def pack_known_safe(ang):
    return np.asarray(list(ang["gammas"]) + list(ang["betas"]), dtype=np.float64)


if __name__ == "__main__":
    main()
