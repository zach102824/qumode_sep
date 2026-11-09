"""Historical B=0.1 selection diagnostic.

The fit is the interleaved shared-gamma KM ladder and the same QAOA ladder.
No repulsion, no popcount shells, no pi/4 or pi/2 shell shift, no packing union,
no extra QAOA starts. Ground states are not loaded. Fidelity is not computed here.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_cri"))
import layout_gate  # noqa: E402

sys.path.insert(0, str(HERE))
import window_rule  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "f5dc9032705c77387f816bc98a02a3c154265000e221c4e0feff13193050c29a"


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    return digest


def bind():
    names = PREREG["family_order"]
    if names != ["nodata", "K", "KM", "qaoa"]:
        raise SystemExit("family order is not the standard four")
    m4.HERE = HERE
    m4.PREREG = PREREG
    m4.FAMS = tuple(names)
    m4.CAP = int(PREREG["maxfun"])


def interleaved(N):
    m = int(N) // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def standard_seeds(inst, fam, L, r, rows_by, best):
    seeds = []
    grid_add = 0
    if L == 1:
        grid = m4.grid_starts(inst, fam, r)
        grid_add = 64
        seeds += [(lab, x, None) for lab, x in grid]
        seeds += [(f"crn_{s}", m4.random_l1(inst, fam, r, s), None) for s in range(PREREG["L1_random"][fam])]
    else:
        for pid, par in enumerate(m4.m2.distinct_parents(rows_by[(fam, L - 1)], PREREG["beam"])):
            seeds += [(lab, x, par) for lab, x in m4.continuation(inst, fam, par["x_final"], L, pid, r)]
        seeds += [("fresh_ramp", m4.ramp(inst, fam, L, r, False), None),
                  ("fresh_ramp_jitter", m4.ramp(inst, fam, L, r, True), None)]
        if fam == "qaoa":
            par = best[("qaoa", L - 1)]
            a, _, _, cs = m4.m2.split("qaoa", par["x_final"], L - 1)
            x = m4.m2.join("qaoa", np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))
            z = m4.seed(20261134, inst.N, inst.k, L, r)
            seeds.append(("best_append_zero_perturb_0.05", x + 0.05 * z.normal(size=x.size), par))
    sub = PREREG["nesting"].get(fam)
    if sub and (sub, L) in best:
        seeds.append((f"nested_{sub}", m4.nested_start(fam, best[(sub, L)]["x_final"], L),
                      {**best[(sub, L)], "nested_anc": True}))
    return seeds, grid_add


def run_one(inst, fam, label, x0, par, N, k, r, L, fh, rows, best):
    row = m4.one_start(inst, fam, x0, L)
    row.update({"N": N, "k": k, "r": r, "family": fam, "L": L, "label": label,
                "root": label if par is None else par["root"],
                "nested_anc": bool(par is not None and par.get("nested_anc", False)),
                "n_param": m4.nparam(fam, L),
                "x0": [float(v) for v in m4.clip_of(inst, fam, x0, L)]})
    rows.append(row)
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    if (fam, L) not in best or row["E"] < best[(fam, L)]["E"]:
        best[(fam, L)] = row
    return row


def _rule_checks():
    tol = float(PREREG["diagnosis"]["energy_legal_tol"])
    if abs(tol - 1e-6) > 1e-15:
        raise SystemExit("energy-legal tolerance is not 1e-6")
    rows = [
        {"E": -1.0, "F": 0.10, "r": 0, "label": "a"},
        {"E": -0.90, "F": 0.40, "r": 1, "label": "b"},
        {"E": -0.50, "F": 0.90, "r": 0, "label": "c"},
    ]
    chosen = window_rule.pick(rows, -0.90, tol)
    if chosen["label"] != "b":
        raise SystemExit("picker kept a state above the QAOA energy")
    chosen = window_rule.pick(rows, -0.20, tol)
    if chosen["label"] != "c":
        raise SystemExit("picker did not take the highest overlap inside the window")
    tied = [
        {"E": -1.0, "F": 0.2, "r": 1, "label": "z"},
        {"E": -1.0, "F": 0.2, "r": 0, "label": "m"},
        {"E": -0.9, "F": 0.2, "r": 0, "label": "a"},
    ]
    chosen = window_rule.pick(tied, -0.5, tol)
    if (chosen["r"], chosen["label"]) != (0, "m"):
        raise SystemExit("tie break is not energy, then repeat, then label")
    if window_rule.pick(rows, -1.1, tol) is not None:
        raise SystemExit("an empty pool returned a state")
    best = window_rule.energy_best(rows)
    if best["label"] != "a":
        raise SystemExit("energy-only tie break drifted")
    top = window_rule.unrestricted(rows)
    if top["label"] != "c":
        raise SystemExit("unrestricted max is not the highest overlap")


def _outcome_checks():
    import summarize  # noqa: E402

    d = {"even_ground_in_window": True}
    joint, dF, dE = summarize.outcome(d, {"F_sub": 0.50, "E": -1.0}, {"F_sub": 0.20, "E": -0.9})
    if joint != "win_joint" or dF <= 0 or dE >= 0:
        raise SystemExit(f"joint win drifted: {joint}")
    only, _, _ = summarize.outcome(d, {"F_sub": 0.50, "E": -0.8}, {"F_sub": 0.20, "E": -1.0})
    if only != "win_fidelity_only":
        raise SystemExit(f"fidelity-only win drifted: {only}")
    both, _, _ = summarize.outcome(d, {"F_sub": 0.01, "E": -1.0}, {"F_sub": 0.01, "E": -0.9})
    if both != "tie_both_failed":
        raise SystemExit(f"floor tie drifted: {both}")
    bad = {"even_ground_in_window": False}
    unreachable, _, _ = summarize.outcome(bad, {"F_sub": 0.5, "E": -1.0}, {"F_sub": 0.2, "E": -0.9})
    if unreachable != "tie_unreachable":
        raise SystemExit("unreachable tie drifted")


def self_check():
    layout_gate.main()
    _rule_checks()
    _outcome_checks()
    if list(PREREG["k_confirm"]) != list(range(130, 142)):
        raise SystemExit("confirmation window is not k=130..141")
    if 100 in PREREG["k_confirm"] or 101 in PREREG["k_confirm"]:
        raise SystemExit("held-out index is in the confirmation window")
    if m4.FAMS.index("KM") != 2 or m4.FAMS.index("qaoa") != 3:
        raise SystemExit("family index drifted")
    out = {}
    rng = np.random.default_rng(2)
    for N in (6, 9, 12):
        inst = m4.m3.instance(N, 4)
        if inst.B != 0.1 or inst.k != 4:
            raise SystemExit("phase check left the development instance")
        if m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12:
            raise SystemExit("parameter count is not the shared-gamma count")
        parent = m4.clip_of(inst, "KM", rng.normal(scale=0.2, size=9), 2)
        _, _, psi = m4.energy_of(inst, "KM", parent, 2)
        shifted = np.array(parent, copy=True)
        shifted[3::3] = shifted[3::3] + 0.5 * np.pi
        _, _, psi_n = m4.energy_of(inst, "KM", shifted, 2)
        if abs(abs(np.vdot(psi, psi_n)) - 1.0) > 1e-8:
            raise SystemExit(f"a pi/2 shell shift was expected to be a global phase at N={N}")
        _, snaps = m4.r1.forward(inst, "KM", parent, 2, store=True)
        gammas = [float(s[-1]) for s in snaps]
        if max(gammas) - min(gammas) > 1e-12:
            raise SystemExit("KM circuit does not share gamma")
    inst = m4.m3.instance(6, 4)
    for fam, depth in (("KM", 2), ("qaoa", 3), ("K", 2), ("nodata", 2)):
        z = m4.clip_of(inst, fam, rng.normal(scale=0.15, size=m4.nparam(fam, depth)), depth)
        _, g, _ = m4.energy_of(inst, fam, z, depth)
        err = 0.0
        for i in range(z.size):
            e = np.zeros_like(z)
            e[i] = 1e-6
            num = (m4.energy_of(inst, fam, z + e, depth)[0] - m4.energy_of(inst, fam, z - e, depth)[0]) / 2e-6
            err = max(err, abs(num - g[i]))
        out[fam] = float(err)
        if err > 1e-6:
            raise SystemExit(f"gradient check failed {fam} {err}")
    return out


def run_repeat(inst, N, k, r, deadline, grid_evals):
    log = HERE / "logs" / f"n{N}_k{k}_r{r}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    best, rows_by, status = {}, {}, "complete"
    with log.open("w") as fh:
        for L in range(1, 7):
            for fam in m4.FAMS:
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_{fam}"
                    break
                seeds, grid_add = standard_seeds(inst, fam, L, r, rows_by, best)
                grid_evals[0] += grid_add
                rows = []
                for label, x0, par in seeds:
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{fam}"
                        break
                    run_one(inst, fam, label, x0, par, N, k, r, L, fh, rows, best)
                if status != "complete":
                    break
                rows_by[(fam, L)] = rows
            if status != "complete":
                break
    return status, best


def run_job(job):
    N, k, deadline = job
    if k not in PREREG["k_confirm"] or k in (100, 101):
        raise SystemExit(f"k={k} is outside the frozen confirmation window")
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}")
    t0 = time.time()
    grid_evals = [0]
    repeats = {}
    status = "complete"
    for r in range(PREREG["repeats_primary"]):
        st, best = run_repeat(inst, N, k, r, deadline, grid_evals)
        repeats[str(r)] = {"status": st, "selected": list(best.values())}
        if st != "complete":
            status = st
            break
    out = {"N": N, "k": k, "status": status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_interleaved": interleaved(N),
           "grid_energy_evals": grid_evals[0],
           "repeats": repeats}
    (HERE / "jobs" / f"n{N}_k{k}.json").write_text(json.dumps(out) + "\n")
    print(f"job N{N} k{k} {status} {out['seconds']:.1f}s", flush=True)
    return {"N": N, "k": k, "status": status, "seconds": out["seconds"]}


def main():
    digest = check_hash()
    bind()
    checks = self_check()
    for d in ("logs", "jobs"):
        (HERE / d).mkdir(exist_ok=True)
    t0 = time.time()
    deadline = t0 + float(PREREG["optimize_wall_seconds"])
    jobs = [(N, k, deadline) for k in PREREG["k_confirm"] for N in (12, 9, 6)]
    mp.set_start_method("fork", force=True)
    with mp.Pool(int(PREREG["workers"])) as pool:
        metas = list(pool.imap(run_job, jobs, chunksize=1))
    (HERE / "run_meta.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "rule": "standard interleaved KM; energy-legal fidelity column is not computed in the fit",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)
    print("jobs", len(metas), flush=True)


if __name__ == "__main__":
    main()
