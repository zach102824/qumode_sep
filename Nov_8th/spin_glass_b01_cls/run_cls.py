"""Historical B=0.1 fits from classical popcount shells.

Interleaved shared-gamma KM only. No per-layer gamma. No packing union.
Ground states are not loaded.
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
sys.path.insert(0, str(HERE))
import classical_init  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_cri"))
import layout_gate  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "45957ee313312515005432d069a670a11ad30ee1bc5e075fe01ebcdb53822d2c"
SCALES = (0.25, 0.5, 0.75)
GAMMA_SIGNS = (1.0, -1.0)
SHELL_SIGNS = (1.0, -1.0)


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    return digest


def bind():
    if "KM_gl" in PREREG["family_order"] or any(str(f).endswith("_pack") or str(f).endswith("_fid") for f in PREREG["family_order"]):
        raise SystemExit("a forbidden family is in the protocol")
    m4.HERE = HERE
    m4.PREREG = PREREG
    m4.FAMS = tuple(PREREG["family_order"])
    m4.CAP = int(PREREG["maxfun"])


def jitter_shells(x, N, k, L, r, index):
    y = np.array(x, dtype=np.float64, copy=True)
    if int(r) != 1:
        return y
    z = m4.seed(20261292, N, k, L, r, index)
    y[3:] = y[3:] + 0.05 * z.normal(size=y.size - 3)
    return y


def classical_starts(inst, L, r, gamma_parent):
    i_star, pops, coef = classical_init.classical_shell(inst.hc, inst.N)
    out = []
    index = 0
    if L == 1:
        for sign_g in GAMMA_SIGNS:
            for scale in SCALES:
                gamma = float(sign_g * scale * np.pi / inst.sJ)
                for sign_s in SHELL_SIGNS:
                    row = np.tile(sign_s * coef, (1, 1))
                    x = m4.m2.join("KM", [gamma], 0.0, 0.0, row)
                    gtag = "p" if sign_g > 0 else "m"
                    stag = "p" if sign_s > 0 else "m"
                    label = f"cls_g{gtag}_a{scale:.2f}_s{stag}"
                    out.append((label, jitter_shells(x, inst.N, inst.k, L, r, index)))
                    index += 1
        return i_star, pops, coef, out
    q = (L - 1) / float(L)
    gamma = float(gamma_parent) * q
    for sign_s in SHELL_SIGNS:
        row = np.tile(sign_s * coef, (L, 1))
        x = m4.m2.join("KM", np.full(L, gamma), 0.0, 0.0, row)
        stag = "p" if sign_s > 0 else "m"
        out.append((f"cls_reseed_{stag}", jitter_shells(x, inst.N, inst.k, L, r, index)))
        index += 1
    return i_star, pops, coef, out


def grad_and_layout():
    layout_gate.main()
    out = {}
    for N in (6, 9, 12):
        inst = m4.m3.instance(N, 4)
        if inst.k != 4 or inst.B != 0.1:
            raise SystemExit("phase check left the development instance")
        if m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12:
            raise SystemExit("parameter count is not the shared-gamma count")
        i_star, pops, coef = classical_init.classical_shell(inst.hc, N)
        if i_star < 0 or len(pops) != N // 3 or any(p not in (0, 1, 2, 3) for p in pops):
            raise SystemExit("classical shell metadata is not a popcount")
        L = 2
        gamma = 0.5 * np.pi / inst.sJ
        bare = m4.m2.join("KM", [gamma, gamma], 0.0, 0.0, np.zeros((L, 3)))
        full = m4.m2.join("KM", [gamma, gamma], 0.0, 0.0, np.tile(coef, (L, 1)))
        _, _, psi_b = m4.energy_of(inst, "KM", bare, L)
        _, _, psi_f = m4.energy_of(inst, "KM", full, L)
        ov = abs(np.vdot(psi_b, psi_f))
        out[f"overlap_N{N}"] = float(ov)
        if ov > 0.95:
            raise SystemExit(f"classical shells are nearly trivial at N={N}")
        y = np.array(full, copy=True)
        y[3::3] = y[3::3] + 0.5 * np.pi
        _, _, psi_n = m4.energy_of(inst, "KM", y, L)
        if abs(abs(np.vdot(psi_f, psi_n)) - 1.0) > 1e-8:
            raise SystemExit(f"a pi/2 shell shift was expected to be a global phase at N={N}")
        _, snaps = m4.r1.forward(inst, "KM", full, L, store=True)
        gammas = [float(s[-1]) for s in snaps]
        if max(gammas) - min(gammas) > 1e-12:
            raise SystemExit("classical circuit does not share gamma")
    rng = np.random.default_rng(2)
    inst = m4.m3.instance(6, 4)
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


def qaoa_seeds(inst, L, r, rows_by, best):
    seeds = []
    if L == 1:
        grid = m4.grid_starts(inst, "qaoa", r)
        seeds += [(lab, x, None) for lab, x in grid]
        seeds += [(f"crn_{s}", m4.random_l1(inst, "qaoa", r, s), None) for s in range(PREREG["L1_random"]["qaoa"])]
        return seeds, 64
    for pid, par in enumerate(m4.m2.distinct_parents(rows_by[("qaoa", L - 1)], PREREG["beam"])):
        seeds += [(lab, x, par) for lab, x in m4.continuation(inst, "qaoa", par["x_final"], L, pid, r)]
    seeds += [("fresh_ramp", m4.ramp(inst, "qaoa", L, r, False), None),
              ("fresh_ramp_jitter", m4.ramp(inst, "qaoa", L, r, True), None)]
    par = best[("qaoa", L - 1)]
    a, _, _, cs = m4.m2.split("qaoa", par["x_final"], L - 1)
    x = m4.m2.join("qaoa", np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))
    z = m4.seed(20261134, inst.N, inst.k, L, r)
    seeds.append(("best_append_zero_perturb_0.05", x + 0.05 * z.normal(size=x.size), par))
    return seeds, 0


def run_job(job):
    N, k, r, deadline = job
    if k not in PREREG["k_confirm"] or k in (100, 101):
        raise SystemExit(f"k={k} is outside the frozen confirmation window")
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}")
    i_star, pops, coef, _ = classical_starts(inst, 1, 0, 0.0)
    log = HERE / "logs" / f"n{N}_k{k}_r{r}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    best, rows_by, status, grid_evals = {}, {}, "complete", 0
    t0 = time.time()
    with log.open("w") as fh:
        for L in range(1, 7):
            for fam in m4.FAMS:
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_{fam}"
                    break
                if fam == "KM":
                    parent_gamma = 0.0 if L == 1 else float(best[("KM", L - 1)]["x_final"][0])
                    _, _, _, cls = classical_starts(inst, L, r, parent_gamma)
                    seeds = [(lab, x, None) for lab, x in cls]
                    if L > 1:
                        for pid, par in enumerate(m4.m2.distinct_parents(rows_by[("KM", L - 1)], PREREG["beam"])):
                            seeds = [(lab, x, par) for lab, x in m4.continuation(inst, "KM", par["x_final"], L, pid, r)] + seeds
                    grid_add = 0
                else:
                    seeds, grid_add = qaoa_seeds(inst, L, r, rows_by, best)
                    grid_evals += grid_add
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
    out = {"N": N, "k": k, "r": r, "status": status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_interleaved": classical_init.interleaved(N),
           "classical_index": int(i_star), "classical_pops": [int(p) for p in pops],
           "classical_shell": [float(v) for v in coef],
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
    jobs = [(N, k, r, deadline) for k in PREREG["k_confirm"] for N in (12, 9, 6) for r in range(PREREG["repeats_primary"])]
    mp.set_start_method("fork", force=True)
    with mp.Pool(int(PREREG["workers"])) as pool:
        metas = list(pool.imap(run_job, jobs, chunksize=1))
    (HERE / "run_meta.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "rule": "classical popcount shells, interleaved shared-gamma KM",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)
    print("jobs", len(metas), flush=True)


if __name__ == "__main__":
    main()
