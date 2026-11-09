"""Historical B=0.1 confirmation of frozen interleaved shared-gamma KM.

No second packing. No per-layer gamma. Ground states are not loaded.
k=202..225 only, after the protocol commit.
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
PREREG_SHA = "1946d1a2463f4d5af9e39973ea2d3dc51414010004e2cdc0baa549cd936d1ced"
ROUND4_FAMS = ("nodata", "K", "KM", "nodata_pack", "K_pack", "KM_pack", "KM_gl", "qaoa")
FORBIDDEN = ("KM_gl", "KM_pack", "KM_fid", "KM_fe", "KM_mm")


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    return digest


def bind():
    order = list(PREREG["family_order"])
    if order != ["nodata", "K", "KM", "qaoa"]:
        raise SystemExit(f"family order is not the frozen four: {order}")
    if any(name in order for name in FORBIDDEN):
        raise SystemExit("a forbidden family is in the protocol")
    blocks = list(PREREG["block_A"]) + list(PREREG["block_B"])
    if blocks != list(PREREG["k_confirm"]):
        raise SystemExit("blocks are not the frozen window in order")
    if len(PREREG["block_A"]) != 12 or len(PREREG["block_B"]) != 12:
        raise SystemExit("blocks are not a 12+12 split")
    if set(PREREG["block_A"]) & set(PREREG["block_B"]):
        raise SystemExit("blocks overlap")
    if min(PREREG["k_confirm"]) != 202 or max(PREREG["k_confirm"]) != 225 or len(PREREG["k_confirm"]) != 24:
        raise SystemExit("window is not k=202..225")
    if 100 in PREREG["k_confirm"] or 101 in PREREG["k_confirm"]:
        raise SystemExit("a held-out k is in the window")
    idx = PREREG["family_index_round4"]
    if [idx[f] for f in order] != [0, 1, 2, 7]:
        raise SystemExit("perturbation indices are not the round-4 indices")
    m4.HERE = HERE
    m4.PREREG = PREREG
    m4.FAMS = ROUND4_FAMS
    m4.CAP = int(PREREG["maxfun"])
    if [m4.FAMS.index(f) for f in order] != [0, 1, 2, 7]:
        raise SystemExit("runner family index drifted from round 4")


def grad_and_layout():
    layout_gate.main()
    inst = m4.m3.instance(6, 4)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}, not the historical 0.1")
    if m4.interleaved(6) != [[0, 2, 4], [1, 3, 5]]:
        raise SystemExit("interleaved formula drifted")
    same = m4.with_packing(inst, m4.interleaved(inst.N))
    if float(np.max(np.abs(same.M_hat - inst.M_hat))) > 1e-12:
        raise SystemExit("instance M is not the interleaved block mean")
    import importlib.util
    spec = importlib.util.spec_from_file_location("win_replay", HERE / "replay_independent.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    part = replay.interleaved(inst.N)
    Kh, Mh = replay.features(inst.J, part)
    if float(np.max(np.abs(Kh - inst.K_hat))) > 1e-9 or float(np.max(np.abs(Mh - inst.M_hat))) > 1e-9:
        raise SystemExit("independent features disagree with the model")
    if float(np.max(np.abs(inst.w1 - inst.wX))) > 1e-12:
        raise SystemExit("interleaved w1 is not sum X")
    if m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12 or m4.nparam("KM", 3) != 12:
        raise SystemExit("parameter count is not the shared-gamma count")
    L = 4
    shells = np.arange(3 * L, dtype=float).reshape(L, 3) * 0.01
    x = m4.r1.pack("KM", 0.2, {"K": -0.1, "M": 0.05}, shells)
    _, snaps = m4.r1.forward(inst, "KM", x, L, store=True)
    gammas = [float(s[-1]) for s in snaps]
    if max(gammas) - min(gammas) > 1e-12:
        raise SystemExit("circuit does not share gamma")
    g0 = snaps[0][4]
    if any(float(np.max(np.abs(s[4] - g0))) > 1e-12 for s in snaps):
        raise SystemExit("D changes across layers")
    rng = np.random.default_rng(2)
    out = {}
    for fam, depth in (("KM", 2), ("qaoa", 3)):
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


def run_job(job):
    N, k, r, deadline = job
    if k not in PREREG["k_confirm"] or k in (100, 101) or k < 202:
        raise SystemExit(f"k={k} is outside the frozen window")
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}")
    base = m4.interleaved(N)
    pscore = m4.packing_score(inst.J, base)
    log = HERE / "logs" / f"n{N}_k{k}_r{r}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    best, rows_by, status, grid_evals = {}, {}, "complete", 0
    t0 = time.time()
    with log.open("w") as fh:
        for L in range(1, 7):
            for fam in PREREG["family_order"]:
                if fam in FORBIDDEN:
                    raise SystemExit(f"forbidden family {fam}")
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_{fam}"
                    break
                seeds = []
                if L == 1:
                    grid = m4.grid_starts(inst, fam, r)
                    grid_evals += 64
                    seeds += [(lab, x, None) for lab, x in grid]
                    seeds += [(f"crn_{s}", m4.random_l1(inst, fam, r, s), None) for s in range(PREREG["L1_random"][fam])]
                else:
                    for pid, par in enumerate(m4.m2.distinct_parents(rows_by[(fam, L - 1)], PREREG["beam"])):
                        seeds += [(lab, x, par) for lab, x in m4.continuation(inst, fam, par["x_final"], L, pid, r)]
                    seeds += [("fresh_ramp", m4.ramp(inst, fam, L, r, False), None),
                              ("fresh_ramp_jitter", m4.ramp(inst, fam, L, r, True), None)]
                    if fam == "qaoa":
                        par = best[(fam, L - 1)]
                        a, _, _, cs = m4.m2.split("qaoa", par["x_final"], L - 1)
                        x = m4.m2.join("qaoa", np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))
                        z = m4.seed(20261134, N, k, L, r)
                        seeds.append(("best_append_zero_perturb_0.05", x + 0.05 * z.normal(size=x.size), par))
                sub = PREREG["nesting"].get(fam)
                if sub and (sub, L) in best:
                    seeds.append((f"nested_{sub}", m4.nested_start(fam, best[(sub, L)]["x_final"], L),
                                  {**best[(sub, L)], "nested_anc": True}))
                rows = []
                for label, x0, par in seeds:
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{fam}"
                        break
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
                if status != "complete":
                    break
                rows_by[(fam, L)] = rows
            if status != "complete":
                break
    out = {"N": N, "k": k, "r": r, "status": status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_abs": base, "packing_score_abs": pscore,
           "packing_interleaved": base, "packing_score_interleaved": pscore,
           "grid_energy_evals": grid_evals, "selected": list(best.values()),
           "cell_counts": {f"{f}|{L}": len(v) for (f, L), v in rows_by.items()}}
    (HERE / "jobs" / f"n{N}_k{k}_r{r}.json").write_text(json.dumps(out) + "\n")
    print(f"job N{N} k{k} r{r} {status} {out['seconds']:.1f}s", flush=True)
    return {"N": N, "k": k, "r": r, "status": status, "seconds": out["seconds"]}


def main():
    digest = check_hash()
    bind()
    checks = grad_and_layout()
    for d in ("logs", "jobs"):
        (HERE / d).mkdir(exist_ok=True)
    t0 = time.time()
    deadline = t0 + float(PREREG["optimize_wall_seconds"])
    jobs = [(N, k, r, deadline) for k in PREREG["k_confirm"] for N in (12, 9, 6) for r in range(PREREG["repeats"])]
    mp.set_start_method("fork", force=True)
    with mp.Pool(int(PREREG["workers"])) as pool:
        metas = list(pool.imap(run_job, jobs, chunksize=1))
    (HERE / "run_meta.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "circuit": "interleaved shared-gamma KM",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
