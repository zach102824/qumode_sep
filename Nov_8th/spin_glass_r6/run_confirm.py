"""Confirmation of the original KM family plus the four extra starts. Runs only if the probe gate says so."""
from __future__ import annotations

import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from km_original import PREREG, assert_original, m2, m3, one_start, qproj_starts, seed

HERE = Path(__file__).resolve().parent
PREREG_SHA = "933cf6a85e88e64685ba2440b96d3ab8b90fe65353f400137e669c91b4b53352"
CONF = PREREG["confirmation_if_launched"]
FAMS = tuple(CONF["family_order_within_L"])
INDEX = CONF["fam_index"]


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    verdict = json.loads((HERE / "probe_verdict.json").read_text())
    if not verdict.get("launch_confirmation"):
        raise SystemExit("probe gate did not launch confirmation")
    return digest


def grid_starts(inst, fam, r):
    z = seed(20261130, inst.N, inst.k, r).uniform(size=1 + len(CONF["grid_c0"]))
    delta, c0s = float(z[0]), [g + 0.05 * (2 * z[1 + i] - 1) for i, g in enumerate(CONF["grid_c0"])]
    pts = []
    for j in range(16):
        a = np.pi * (j + delta) / 16.0 / inst.sJ
        for c0 in c0s:
            cs = np.zeros((1, 3)); cs[0, 0] = c0
            pts.append(m2.join(fam, [a], 0.0, 0.0, cs))
    energies = [m2.energy_grad(inst, fam, p, 1)[0] for p in pts]
    order = np.argsort(energies, kind="stable")[: CONF["L1_grid_top"]]
    return [(f"grid_{int(i)}", pts[int(i)]) for i in order]


def random_l1(inst, fam, r, s):
    u = seed(20261131, inst.N, inst.k, 1, r, s).uniform(size=6)
    a = np.pi * u[0] / inst.sJ
    cs = np.zeros((1, 3))
    cs[0] = [2.0 * u[1] - 1.0, 1.2 * u[2] - 0.6, 1.2 * u[3] - 0.6]
    b, c = a * (2.0 * u[4] - 1.0), a * (2.0 * u[5] - 1.0)
    return m2.join(fam, [a], b, c, cs)


def ramp(inst, fam, L, r, jitter):
    t = (np.arange(L) + 0.5) / L
    d = CONF["ramp_delta"]
    u, beta = d * t, -d * (1.0 - t)
    if jitter:
        z = seed(20261133, inst.N, inst.k, L, r).uniform(size=2 * L)
        u = u + 0.1 * (2 * z[:L] - 1)
        beta = beta + 0.1 * (2 * z[L:] - 1)
    cs = np.zeros((L, 3)); cs[:, 0] = beta
    a = u / inst.sJ
    if fam == "qaoa":
        return m2.join("qaoa", a, 0, 0, cs)
    return m2.join(fam, np.full(L, float(np.mean(a))), 0.0, 0.0, cs)


def continuation(inst, fam, parent, L, pid, r):
    a, b, c, cs = m2.split(fam, parent, L - 1)
    csi = np.column_stack([m2.zhou_interp(cs[:, j]) for j in range(3)])
    if fam == "qaoa":
        interp = m2.join("qaoa", m2.zhou_interp(a), 0, 0, csi)
        out = [(f"p{pid}_interp", interp),
               (f"p{pid}_append_zero", m2.join("qaoa", np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))])))]
    else:
        interp = m2.join(fam, a, b, c, csi)
        q = (L - 1) / L
        out = [(f"p{pid}_interp", interp),
               (f"p{pid}_time_rescaled", m2.join(fam, a * q, b * q, c * q, csi))]
    for j, sig in enumerate((0.1, 0.3)):
        z = seed(20261132, inst.N, inst.k, L, r, INDEX[fam], pid, j)
        out.append((f"p{pid}_perturb_{sig}", interp + sig * z.normal(size=interp.size)))
    return out


def run_job(job):
    N, k, r, deadline = job
    inst = m3.instance(N, k)
    assert_original(inst)
    log = HERE / "logs" / f"n{N}_k{k}_r{r}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    best, rows_by, status = {}, {}, "complete"
    t0 = time.time()
    with log.open("w") as fh:
        for L in range(1, 7):
            for fam in FAMS:
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_{fam}"
                    break
                seeds = []
                if L == 1:
                    seeds += [(lab, x, None) for lab, x in grid_starts(inst, fam, r)]
                    seeds += [(f"crn_{s}", random_l1(inst, fam, r, s), None) for s in range(CONF["L1_random"][fam])]
                else:
                    for pid, par in enumerate(m2.distinct_parents(rows_by[(fam, L - 1)], CONF["beam"])):
                        seeds += [(lab, x, par) for lab, x in continuation(inst, fam, par["x_final"], L, pid, r)]
                    seeds += [("fresh_ramp", ramp(inst, fam, L, r, False), None),
                              ("fresh_ramp_jitter", ramp(inst, fam, L, r, True), None)]
                    if fam == "qaoa":
                        par = best[(fam, L - 1)]
                        aa, _, _, cs = m2.split("qaoa", par["x_final"], L - 1)
                        x = m2.join("qaoa", np.append(aa, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))
                        z = seed(20261134, N, k, L, r)
                        seeds.append(("best_append_zero_perturb_0.05", x + 0.05 * z.normal(size=x.size), par))
                sub = CONF["nesting"].get(fam)
                if sub and (sub, L) in best:
                    seeds.append((f"nested_{sub}", m2.pad_nested(fam, best[(sub, L)]["x_final"], sub),
                                  {**best[(sub, L)], "nested_anc": True}))
                if fam == "KM":
                    for lab, x in qproj_starts(inst, best[("qaoa", L)]["x_final"], L, r):
                        seeds.append((lab, x, None))
                rows = []
                for label, x0, par in seeds:
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{fam}"
                        break
                    row = one_start(inst, fam, x0, L)
                    row.update({"N": N, "k": k, "r": r, "family": fam, "L": L, "label": label,
                                "root": label if par is None else par["root"],
                                "nested_anc": bool(par is not None and par.get("nested_anc", False)),
                                "n_param": m2.nparam(fam, L),
                                "extra_start": label.startswith("qproj") or label == "couple"})
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
           "selected": list(best.values()),
           "cell_counts": {f"{f}|{L}": len(v) for (f, L), v in rows_by.items()}}
    (HERE / "jobs" / f"n{N}_k{k}_r{r}.json").write_text(json.dumps(out) + "\n")
    print(f"job N{N} k{k} r{r} {status} {out['seconds']:.1f}s", flush=True)
    return {"N": N, "k": k, "r": r, "status": status, "seconds": out["seconds"]}


def main():
    digest = check_hash()
    for d in ("logs", "jobs"):
        (HERE / d).mkdir(exist_ok=True)
    t0 = time.time()
    deadline = t0 + float(CONF["optimize_wall_seconds"])
    jobs = [(N, k, r, deadline) for k in CONF["k"] for N in (12, 9, 6) for r in range(CONF["repeats"])]
    with Pool(int(CONF["workers"])) as pool:
        metas = list(pool.imap(run_job, jobs, chunksize=1))
    (HERE / "run_meta.json").write_text(json.dumps({"prereg_sha256": digest, "wall_s": time.time() - t0, "jobs": metas}, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
