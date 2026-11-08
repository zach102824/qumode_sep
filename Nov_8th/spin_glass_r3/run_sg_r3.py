"""Spin-glass round 3: robustness. R2 parametrization, box, beam and nesting; fresh instances and independent repeats."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r2"))
import run_sg_r2 as m2  # noqa: E402

r1 = m2.r1
PREREG = json.loads((HERE / "PREREG.json").read_text())
FAMS = ("nodata", "K", "KM", "qaoa", "qaoa_heavy")
CAP = int(PREREG["maxfun"])


def circ(fam):
    return "qaoa" if fam.startswith("qaoa") else fam


def make_J(N, k):
    rng = np.random.default_rng(np.random.SeedSequence([20261115, N, k]))
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = J[j, i] = float(rng.normal())
    return J


def j_sha(J):
    return hashlib.sha256(np.ascontiguousarray(J, dtype="<f8").tobytes()).hexdigest()


def instance(N, k):
    J = make_J(N, k)
    doc = {"N": N, "k": k, "seed": [20261115, N, k], "J_sha256_f64le": j_sha(J), "role": "r3"}
    inst = r1.Inst(N, k, J, doc)
    inst.sJ = m2.s_J(J)
    return inst


def seed(*parts):
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def one_start(inst, fam, x0, L):
    c = circ(fam)
    x0 = m2.clip(inst, c, x0, L)
    bnds = m2.bounds(inst, c, L)
    best = {"E": np.inf, "x": x0.copy()}
    cache = {}
    used = [0]

    def ev(x):
        if used[0] >= CAP:
            raise StopIteration
        used[0] += 1
        E, g, _ = m2.energy_grad(inst, c, x, L)
        cache["x"] = np.array(x, copy=True); cache["g"] = g
        if E < best["E"]:
            best.update(E=float(E), x=np.array(x, copy=True))
        return E, g

    def fun(x):
        return ev(x)[0]

    def jac(x):
        if "x" in cache and np.array_equal(cache["x"], x):
            return cache["g"]
        return ev(x)[1]

    t0 = time.perf_counter()
    reason = "returned"
    try:
        res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": CAP, "maxfun": CAP, "ftol": PREREG["ftol"], "gtol": PREREG["gtol"]})
        reason = str(res.message)
    except StopIteration:
        reason = "start_ceiling"
    x = best["x"]
    _, g, _ = m2.energy_grad(inst, c, x, L)
    b = np.array(bnds)
    pg = np.where((x <= b[:, 0] + 1e-12) & (g > 0), 0.0, g)
    pg = np.where((x >= b[:, 1] - 1e-12) & (pg < 0), 0.0, pg)
    touch = bool(np.any((x - b[:, 0] < 1e-6) | (b[:, 1] - x < 1e-6)))
    return {"E": best["E"], "x_final": [float(v) for v in x], "n_eval": used[0],
            "capped": used[0] >= CAP, "stop_reason": reason, "pg_inf": float(np.max(np.abs(pg))),
            "touch_bound": touch, "seconds": time.perf_counter() - t0}


def grid_starts(inst, fam, r):
    c = circ(fam)
    z = seed(20261121, inst.N, inst.k, r).uniform(size=1 + len(PREREG["grid_c0"]))
    delta = float(z[0])
    c0s = [g + 0.05 * (2 * z[1 + i] - 1) for i, g in enumerate(PREREG["grid_c0"])]
    pts = []
    for j in range(16):
        u = np.pi * (j + delta) / 16.0
        for c0 in c0s:
            cs = np.zeros((1, 3)); cs[0, 0] = c0
            pts.append(m2.join(c, [u / inst.sJ], 0.0, 0.0, cs))
    Es = [m2.energy_grad(inst, c, p, 1)[0] for p in pts]
    order = np.argsort(Es, kind="stable")[: PREREG["L1_grid_top"]]
    return [(f"grid_{int(i)}", pts[int(i)]) for i in order], len(pts)


def random_l1(inst, fam, r, s):
    u = seed(20261120, inst.N, inst.k, 1, r, s).uniform(size=6)
    a = np.pi * u[0] / inst.sJ
    cs = np.zeros((1, 3))
    cs[0, 0] = 2.0 * u[1] - 1.0
    cs[0, 1] = 1.2 * u[2] - 0.6
    cs[0, 2] = 1.2 * u[3] - 0.6
    return m2.join(circ(fam), [a], a * (2.0 * u[4] - 1.0), a * (2.0 * u[5] - 1.0), cs)


def ramp(inst, fam, L, r, jitter):
    t = (np.arange(L) + 0.5) / L
    d = PREREG["ramp_delta"]
    u, beta = d * t, -d * (1.0 - t)
    if jitter:
        z = seed(20261123, inst.N, inst.k, L, r).uniform(size=2 * L)
        u = u + 0.1 * (2 * z[:L] - 1)
        beta = beta + 0.1 * (2 * z[L:] - 1)
    cs = np.zeros((L, 3)); cs[:, 0] = beta
    a = u / inst.sJ
    if circ(fam) != "qaoa":
        a = np.full(L, float(np.mean(a)))
    return m2.join(circ(fam), a, 0.0, 0.0, cs)


def continuation(inst, fam, parent, L, pid, r):
    c = circ(fam)
    a, b, cc, cs = m2.split(c, parent, L - 1)
    csi = np.column_stack([m2.zhou_interp(cs[:, j]) for j in range(3)])
    out = []
    if c == "qaoa":
        interp = m2.join(c, m2.zhou_interp(a), 0, 0, csi)
        out.append((f"p{pid}_interp", interp))
        out.append((f"p{pid}_append_zero", m2.join(c, np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))))
    else:
        interp = m2.join(c, a, b, cc, csi)
        q = (L - 1) / L
        out.append((f"p{pid}_interp", interp))
        out.append((f"p{pid}_time_rescaled", m2.join(c, a * q, b * q, cc * q, csi)))
    draws = 2 if fam == "qaoa_heavy" else 1
    for j, sig in enumerate((0.1, 0.3)):
        for d in range(draws):
            z = seed(20261122, inst.N, inst.k, L, r, pid, j, d)
            out.append((f"p{pid}_perturb_{sig}_{d}", interp + sig * z.normal(size=interp.size)))
    return out


def run_job(job):
    N, k, r, deadline = job
    inst = instance(N, k)
    log = HERE / "logs" / f"n{N}_k{k}_r{r}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    best, rows_by = {}, {}
    status = "complete"
    t0 = time.time()
    with log.open("w") as fh:
        for L in range(1, 7):
            for fam in FAMS:
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_{fam}"
                    break
                c = circ(fam)
                seeds = []
                if L == 1:
                    g, _ = grid_starts(inst, fam, r)
                    seeds += [(lab, x, None) for lab, x in g]
                    nr = PREREG["L1_random"][fam]
                    seeds += [(f"crn_{s}", random_l1(inst, fam, r, s), None) for s in range(nr)]
                else:
                    for pid, par in enumerate(m2.distinct_parents(rows_by[(fam, L - 1)], PREREG["beam"])):
                        seeds += [(lab, x, par) for lab, x in continuation(inst, fam, par["x_final"], L, pid, r)]
                    seeds.append(("fresh_ramp", ramp(inst, fam, L, r, False), None))
                    seeds.append(("fresh_ramp_jitter", ramp(inst, fam, L, r, True), None))
                    if c == "qaoa":
                        par = best[(fam, L - 1)]
                        a, _, _, cs = m2.split(c, par["x_final"], L - 1)
                        x = m2.join(c, np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))
                        z = seed(20261124, N, k, L, r, 1 if fam == "qaoa_heavy" else 0)
                        seeds.append(("best_append_zero_perturb_0.05", x + 0.05 * z.normal(size=x.size), par))
                sub = {"K": "nodata", "KM": "K"}.get(fam)
                if sub and (sub, L) in best:
                    sb = best[(sub, L)]
                    seeds.append((f"nested_{sub}", m2.pad_nested(fam, sb["x_final"], sub), {**sb, "nested_anc": True}))
                rows = []
                for label, x0, par in seeds:
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{fam}"
                        break
                    row = one_start(inst, fam, x0, L)
                    row.update({"N": N, "k": k, "r": r, "family": fam, "L": L, "label": label,
                                "root": label if par is None else par["root"],
                                "nested_anc": bool(par is not None and par.get("nested_anc", False)),
                                "n_param": m2.nparam(c, L)})
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    if (fam, L) not in best or row["E"] < best[(fam, L)]["E"]:
                        best[(fam, L)] = row
                if status != "complete":
                    break
                rows_by[(fam, L)] = rows
            if status != "complete":
                break
    sel = [{k2: v for k2, v in b.items()} for b in best.values()]
    out = {"N": N, "k": k, "r": r, "status": status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ, "selected": sel,
           "cell_counts": {f"{f}|{L}": len(v) for (f, L), v in rows_by.items()}}
    (HERE / "jobs" / f"n{N}_k{k}_r{r}.json").write_text(json.dumps(out) + "\n")
    print(f"job N{N} k{k} r{r} {status} {out['seconds']:.1f}s", flush=True)
    return {kk: out[kk] for kk in ("N", "k", "r", "status", "seconds")}


def check_manifests():
    rows = []
    for N in PREREG["Ns"]:
        for k in PREREG["k_secondary"]:
            doc = json.loads((r1.HERE / "manifests" / f"n{N}_k{k}.json").read_text())
            ok = j_sha(make_J(N, k)) == doc["J_sha256_f64le"]
            rows.append({"N": N, "k": k, "match": ok})
            if not ok:
                raise SystemExit(f"seed formula does not reproduce manifest N={N} k={k}")
    return rows


def main():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    for d in ("logs", "jobs"):
        (HERE / d).mkdir(exist_ok=True)
    man = check_manifests()
    t0 = time.time()
    deadline = t0 + float(PREREG["optimize_wall_seconds"])
    ks = PREREG["k_primary"] + PREREG["k_secondary"]
    jobs = [(N, k, r, deadline) for k in ks for N in (12, 9, 6) for r in range(PREREG["repeats"])]
    with Pool(int(PREREG["workers"])) as pool:
        metas = list(pool.imap(run_job, jobs, chunksize=1))
    (HERE / "run_meta.json").write_text(json.dumps({"prereg_sha256": digest, "manifest_check": man,
                                                     "wall_s": time.time() - t0, "jobs": metas}, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)


PREREG_SHA = "c46c578fa34f73a8b8bbd29fb211189cb18ed6c0462d5660bed93c275491fcec"

if __name__ == "__main__":
    main()
