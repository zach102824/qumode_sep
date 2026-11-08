"""Historical B=0.1: twelve uniform L=6 starts for shared-gamma KM.

The continuation search is the round-4 list, two repeats, interleaved packing.
The twelve draws are extra L=6 energy starts. The continuation vector stays
a candidate. No ground state is loaded.
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

PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "8131d186053868438881ea1b5b5fbce06f7975e5c9104c178c5d8b5a88c14fb1"
BOX_N = int(PREREG["box_count"])
BOX_SEED = 20261302


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


def box_starts(inst, N, k):
    """Twelve uniform draws in the KM box. Independent of any fitted vector."""
    L = 6
    A = 2.0 * np.pi / inst.sJ
    shell = 2.0 * np.pi
    out = []
    for s in range(BOX_N):
        rng = m4.seed(BOX_SEED, N, k, s)
        head = rng.uniform(-A, A, size=3)
        shells = rng.uniform(-shell, shell, size=3 * L)
        out.append((f"box_{s}", np.concatenate([head, shells])))
    return out


def self_check():
    layout_gate.main()
    out = {}
    if BOX_N != 12 or BOX_SEED != 20261302:
        raise SystemExit("box budget is not the frozen budget")
    if abs(float(PREREG["diagnosis"]["drop_min"]) - 0.01) > 1e-15 or int(PREREG["diagnosis"]["drop_count"]) != 4:
        raise SystemExit("diagnosis threshold moved")
    if list(PREREG["k_confirm"]) != list(range(142, 154)):
        raise SystemExit("confirmation window is not k=142..153")
    rng = np.random.default_rng(2)
    for N in (6, 9, 12):
        inst = m4.m3.instance(N, 4)
        if inst.B != 0.1 or inst.k != 4:
            raise SystemExit("phase check left the development instance")
        if m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12:
            raise SystemExit("parameter count is not the shared-gamma count")
        starts = box_starts(inst, N, 4)
        if len(starts) != 12:
            raise SystemExit("box start count drifted")
        vecs = [v for _, v in starts]
        if len({np.round(v, 8).tobytes() for v in vecs}) != 12:
            raise SystemExit("box starts collided")
        for name, vec in starts:
            if not name.startswith("box_") or vec.size != 21:
                raise SystemExit("box vector is not the KM layout")
            got = m4.clip_of(inst, "KM", vec, 6)
            if np.max(np.abs(got - vec)) > 1e-12:
                raise SystemExit("box vector left the KM box")
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
    kept = "historical"
    energy = -0.1
    for label, value in (("box_0", 0.2), ("box_1", -0.1)):
        if value < energy:
            energy = value
            kept = label
    if kept != "historical" or energy != -0.1:
        raise SystemExit("a tie or a worse start replaced the continuation vector")
    if -0.3 < energy:
        energy = -0.3
        kept = "box_2"
    if kept != "box_2" or energy != -0.3:
        raise SystemExit("a strictly lower start was discarded")
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


def run_box(inst, N, k, base_row, deadline):
    log = HERE / "logs" / f"n{N}_k{k}_box.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    L = 6
    winner = {"E": base_row["E"], "x_final": base_row["x_final"], "label": "historical",
              "n_eval": 0, "capped": False, "stop_reason": "historical", "pg_inf": base_row["pg_inf"],
              "touch_bound": base_row["touch_bound"], "seconds": 0.0, "n_param": 21,
              "root": base_row["root"], "nested_anc": False, "family": "KM_box", "L": L,
              "r": -1, "N": N, "k": k}
    status = "complete"
    with log.open("w") as fh:
        for label, x0 in box_starts(inst, N, k):
            if time.time() >= deadline:
                status = "wall_in_box"
                break
            row = m4.one_start(inst, "KM", x0, L)
            row.update({"N": N, "k": k, "r": -1, "family": "KM_box", "L": L, "label": label,
                        "root": label, "nested_anc": False, "n_param": 21,
                        "x0": [float(v) for v in m4.clip_of(inst, "KM", x0, L)]})
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row["E"] < winner["E"]:
                winner = row
    return status, winner


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
    box_status = "not_run"
    base_l6 = None
    for r in range(PREREG["repeats_primary"]):
        st, best = run_repeat(inst, N, k, r, deadline, grid_evals)
        repeats[str(r)] = {"status": st, "selected": list(best.values())}
        if st != "complete":
            status = st
            break
        row = best[("KM", 6)]
        if base_l6 is None or row["E"] < base_l6["E"]:
            base_l6 = row
    box_winner = None
    if status == "complete":
        box_status, box_winner = run_box(inst, N, k, base_l6, deadline)
        status = box_status
        if box_winner["E"] > base_l6["E"] + 1e-9:
            raise SystemExit("box selection discarded the continuation vector")
    out = {"N": N, "k": k, "status": status, "box_status": box_status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_interleaved": interleaved(N),
           "grid_energy_evals": grid_evals[0],
           "repeats": repeats,
           "km_base_L6": None if base_l6 is None else {"E": base_l6["E"], "label": base_l6["label"], "r": base_l6["r"]},
           "km_box": box_winner}
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
        "rule": "interleaved shared-gamma KM; continuation vector plus twelve uniform L=6 starts",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)
    print("jobs", len(metas), flush=True)


if __name__ == "__main__":
    main()
