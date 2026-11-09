"""Round 4 development run. Shared-gamma KM, |J|-max packing variant, per-layer KM_gl, and QAOA."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r3"))
import run_sg_r3 as m3  # noqa: E402

m2 = m3.m2
r1 = m3.r1
PREREG = json.loads((HERE / "PREREG.json").read_text())
FAMS = ("nodata", "K", "KM", "KM_gl", "qaoa")
BASE = {"nodata_pack": "nodata", "K_pack": "K", "KM_pack": "KM"}
CAP = int(PREREG["maxfun"])
SHELL = (-2.0 * np.pi, 2.0 * np.pi)


def kind(fam):
    return "gl" if fam == "KM_gl" else ("qaoa" if fam == "qaoa" else BASE.get(fam, fam))


def packed(fam):
    return fam.endswith("_pack")


def seed(*parts):
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def best_packing(J):
    """Partition into triples maximizing the sum of |J| inside triples. Lexicographic tie break."""
    sites = list(range(J.shape[0]))
    best, best_score = None, -1.0

    def rec(rem, acc):
        nonlocal best, best_score
        if not rem:
            part = tuple(sorted(tuple(sorted(t)) for t in acc))
            score = 0.0
            for t in part:
                score += abs(J[t[0], t[1]]) + abs(J[t[0], t[2]]) + abs(J[t[1], t[2]])
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (best is None or part < best)):
                best, best_score = part, score
            return
        s0, rest = rem[0], rem[1:]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                rec([rest[k] for k in range(len(rest)) if k != i and k != j], acc + [(s0, rest[i], rest[j])])

    rec(sites, [])
    return [list(t) for t in best], float(best_score)


def interleaved(N):
    m = N // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def with_packing(inst, packing):
    out = copy.copy(inst)
    out.M_hat, out.scale_M = r1.hat(r1.block_coeff(inst.J, packing), inst.rms_J)
    out.dM = r1.pair_diag(out.M_hat, inst.z)
    out.w1, out.w2, out.w3 = r1.shell_weights(inst.N, packing)
    out.packing = packing
    return out


def inst_for(inst, inst_pack, fam):
    return inst_pack if packed(fam) else inst


def nparam(fam, L):
    return 2 + 4 * L if fam == "KM_gl" else m2.nparam(kind(fam), L)


def bounds_of(inst, fam, L):
    if fam == "KM_gl":
        A = 2.0 * np.pi / inst.sJ
        return [(-A, A)] * (2 + L) + [SHELL] * (3 * L)
    return m2.bounds(inst, kind(fam), L)


def clip_of(inst, fam, x, L):
    b = np.array(bounds_of(inst, fam, L), float)
    return np.minimum(np.maximum(np.asarray(x, float), b[:, 0]), b[:, 1])


def gl_from_km(x, L):
    x = np.asarray(x, float)
    return np.concatenate([[x[1], x[2]], np.full(L, x[0]), x[3:]])


def energy_grad_gl(inst, x, L):
    x = np.asarray(x, float)
    b, c = float(x[0]), float(x[1])
    a = x[2:2 + L]
    cs = x[2 + L:].reshape(L, 3)
    psi = inst.plus.astype(np.complex128).copy()
    snaps = []
    for ell in range(L):
        phase = a[ell] * inst.hc + b * inst.dK + c * inst.dM
        mid = psi * np.exp(-1j * phase)
        phi = cs[ell, 0] * inst.w1 + cs[ell, 1] * inst.w2 + cs[ell, 2] * inst.w3
        x2 = r1.fwht(mid) * np.exp(-1j * phi)
        psi = r1.fwht(x2)
        snaps.append((phase, mid, phi, x2))
    Hpsi = inst.hc * psi + inst.B * r1.apply_X_sum(psi, inst.N)
    E = float(np.vdot(psi, Hpsi).real)
    lam = Hpsi
    ga, gb, gc = np.zeros(L), 0.0, 0.0
    gcs = np.zeros((L, 3))
    for ell in range(L - 1, -1, -1):
        phase, mid, phi, x2 = snaps[ell]
        lam_x = r1.fwht(lam)
        ov = np.conj(lam_x) * x2
        for t, w in enumerate((inst.w1, inst.w2, inst.w3)):
            gcs[ell, t] = 2.0 * float(np.imag(np.sum(ov * w)))
        lam_mid = r1.fwht(lam_x * np.exp(1j * phi))
        ovD = np.conj(lam_mid) * mid
        ga[ell] = 2.0 * float(np.imag(np.sum(ovD * inst.hc)))
        gb += 2.0 * float(np.imag(np.sum(ovD * inst.dK)))
        gc += 2.0 * float(np.imag(np.sum(ovD * inst.dM)))
        lam = lam_mid * np.exp(1j * phase)
    return E, np.concatenate([[gb, gc], ga, gcs.reshape(-1)]), psi


def energy_of(inst, fam, x, L):
    if fam == "KM_gl":
        return energy_grad_gl(inst, x, L)
    return m2.energy_grad(inst, kind(fam), x, L)


def one_start(inst, fam, x0, L):
    x0 = clip_of(inst, fam, x0, L)
    bnds = bounds_of(inst, fam, L)
    best = {"E": np.inf, "x": x0.copy()}
    cache, used = {}, [0]

    def ev(x):
        if used[0] >= CAP:
            raise StopIteration
        used[0] += 1
        E, g, _ = energy_of(inst, fam, x, L)
        cache["x"], cache["g"] = np.array(x, copy=True), g
        if E < best["E"]:
            best.update(E=float(E), x=np.array(x, copy=True))
        return E, g

    t0 = time.perf_counter()
    reason = "returned"
    try:
        res = minimize(lambda x: ev(x)[0], x0, jac=lambda x: cache["g"] if "x" in cache and np.array_equal(cache["x"], x) else ev(x)[1],
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": CAP, "maxfun": CAP, "ftol": PREREG["ftol"], "gtol": PREREG["gtol"]})
        reason = str(res.message)
    except StopIteration:
        reason = "start_ceiling"
    x = best["x"]
    _, g, _ = energy_of(inst, fam, x, L)
    b = np.array(bnds, float)
    pg = np.where((x <= b[:, 0] + 1e-12) & (g > 0), 0.0, g)
    pg = np.where((x >= b[:, 1] - 1e-12) & (pg < 0), 0.0, pg)
    touch = bool(np.any((x - b[:, 0] < 1e-6) | (b[:, 1] - x < 1e-6)))
    return {"E": best["E"], "x_final": [float(v) for v in x], "n_eval": used[0], "capped": used[0] >= CAP,
            "stop_reason": reason, "pg_inf": float(np.max(np.abs(pg))), "touch_bound": touch,
            "seconds": time.perf_counter() - t0}


def grid_starts(inst, fam, r):
    z = seed(20261130, inst.N, inst.k, r).uniform(size=1 + len(PREREG["grid_c0"]))
    delta, c0s = float(z[0]), [g + 0.05 * (2 * z[1 + i] - 1) for i, g in enumerate(PREREG["grid_c0"])]
    pts = []
    for j in range(16):
        a = np.pi * (j + delta) / 16.0 / inst.sJ
        for c0 in c0s:
            cs = np.zeros((1, 3)); cs[0, 0] = c0
            if fam == "KM_gl":
                pts.append(np.concatenate([[0.0, 0.0, a], cs.reshape(-1)]))
            else:
                pts.append(m2.join(kind(fam), [a], 0.0, 0.0, cs))
    Es = [energy_of(inst, fam, p, 1)[0] for p in pts]
    order = np.argsort(Es, kind="stable")[: PREREG["L1_grid_top"]]
    return [(f"grid_{int(i)}", pts[int(i)]) for i in order]


def random_l1(inst, fam, r, s):
    u = seed(20261131, inst.N, inst.k, 1, r, s).uniform(size=6)
    a = np.pi * u[0] / inst.sJ
    cs = np.zeros((1, 3))
    cs[0] = [2.0 * u[1] - 1.0, 1.2 * u[2] - 0.6, 1.2 * u[3] - 0.6]
    b, c = a * (2.0 * u[4] - 1.0), a * (2.0 * u[5] - 1.0)
    if fam == "KM_gl":
        return np.concatenate([[b, c, a], cs.reshape(-1)])
    return m2.join(kind(fam), [a], b, c, cs)


def ramp(inst, fam, L, r, jitter):
    t = (np.arange(L) + 0.5) / L
    d = PREREG["ramp_delta"]
    u, beta = d * t, -d * (1.0 - t)
    if jitter:
        z = seed(20261133, inst.N, inst.k, L, r).uniform(size=2 * L)
        u = u + 0.1 * (2 * z[:L] - 1)
        beta = beta + 0.1 * (2 * z[L:] - 1)
    cs = np.zeros((L, 3)); cs[:, 0] = beta
    a = u / inst.sJ
    if fam == "qaoa":
        return m2.join("qaoa", a, 0, 0, cs)
    if fam == "KM_gl":
        return np.concatenate([[0.0, 0.0], a, cs.reshape(-1)])
    return m2.join(kind(fam), np.full(L, float(np.mean(a))), 0.0, 0.0, cs)


def continuation(inst, fam, parent, L, pid, r):
    fid = FAMS.index(fam)
    out = []
    if fam == "KM_gl":
        x = np.asarray(parent, float)
        b, c, a = float(x[0]), float(x[1]), x[2:2 + (L - 1)]
        cs = x[2 + (L - 1):].reshape(L - 1, 3)
        csi = np.column_stack([m2.zhou_interp(cs[:, j]) for j in range(3)])
        interp = np.concatenate([[b, c], m2.zhou_interp(a), csi.reshape(-1)])
        out.append((f"p{pid}_interp", interp))
        out.append((f"p{pid}_append_zero", np.concatenate([[b, c], a, [0.0], cs.reshape(-1), np.zeros(3)])))
    elif fam == "qaoa":
        a, _, _, cs = m2.split("qaoa", parent, L - 1)
        csi = np.column_stack([m2.zhou_interp(cs[:, j]) for j in range(3)])
        interp = m2.join("qaoa", m2.zhou_interp(a), 0, 0, csi)
        out.append((f"p{pid}_interp", interp))
        out.append((f"p{pid}_append_zero", m2.join("qaoa", np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))))
    else:
        a, b, c, cs = m2.split(kind(fam), parent, L - 1)
        csi = np.column_stack([m2.zhou_interp(cs[:, j]) for j in range(3)])
        interp = m2.join(kind(fam), a, b, c, csi)
        q = (L - 1) / L
        out.append((f"p{pid}_interp", interp))
        out.append((f"p{pid}_time_rescaled", m2.join(kind(fam), a * q, b * q, c * q, csi)))
    for j, sig in enumerate((0.1, 0.3)):
        z = seed(20261132, inst.N, inst.k, L, r, fid, pid, j)
        out.append((f"p{pid}_perturb_{sig}", interp + sig * z.normal(size=interp.size)))
    return out


def nested_start(fam, parent_x, L):
    sub = PREREG["nesting"][fam]
    if fam == "KM_gl":
        return gl_from_km(parent_x, L)
    return m2.pad_nested(kind(fam), parent_x, kind(sub))


def run_job(job):
    N, k, r, deadline = job
    inst = m3.instance(N, k)
    part, pscore = best_packing(inst.J)
    base = interleaved(N)
    inst_pack = with_packing(inst, part)
    log = HERE / "logs" / f"n{N}_k{k}_r{r}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    best, rows_by, status, grid_evals = {}, {}, "complete", 0
    t0 = time.time()
    with log.open("w") as fh:
        for L in range(1, 7):
            for fam in FAMS:
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_{fam}"
                    break
                use = inst_for(inst, inst_pack, fam)
                seeds = []
                if L == 1:
                    grid = grid_starts(use, fam, r)
                    grid_evals += 64
                    seeds += [(lab, x, None) for lab, x in grid]
                    seeds += [(f"crn_{s}", random_l1(use, fam, r, s), None) for s in range(PREREG["L1_random"][fam])]
                else:
                    for pid, par in enumerate(m2.distinct_parents(rows_by[(fam, L - 1)], PREREG["beam"])):
                        seeds += [(lab, x, par) for lab, x in continuation(use, fam, par["x_final"], L, pid, r)]
                    seeds += [("fresh_ramp", ramp(use, fam, L, r, False), None),
                              ("fresh_ramp_jitter", ramp(use, fam, L, r, True), None)]
                    if fam == "qaoa":
                        par = best[(fam, L - 1)]
                        a, _, _, cs = m2.split("qaoa", par["x_final"], L - 1)
                        x = m2.join("qaoa", np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))
                        z = seed(20261134, N, k, L, r)
                        seeds.append(("best_append_zero_perturb_0.05", x + 0.05 * z.normal(size=x.size), par))
                sub = PREREG["nesting"].get(fam)
                if sub and (sub, L) in best:
                    seeds.append((f"nested_{sub}", nested_start(fam, best[(sub, L)]["x_final"], L),
                                  {**best[(sub, L)], "nested_anc": True}))
                rows = []
                for label, x0, par in seeds:
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{fam}"
                        break
                    row = one_start(use, fam, x0, L)
                    row.update({"N": N, "k": k, "r": r, "family": fam, "L": L, "label": label,
                                "root": label if par is None else par["root"],
                                "nested_anc": bool(par is not None and par.get("nested_anc", False)),
                                "n_param": nparam(fam, L), "x0": [float(v) for v in clip_of(use, fam, x0, L)]})
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    if (fam, L) not in best or row["E"] < best[(fam, L)]["E"]:
                        best[(fam, L)] = row
                if status != "complete":
                    break
                rows_by[(fam, L)] = rows
            if status != "complete":
                break
    out = {"N": N, "k": k, "r": r, "status": status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_abs": part, "packing_score_abs": pscore,
           "packing_interleaved": base, "packing_score_interleaved": packing_score(inst.J, base),
           "grid_energy_evals": grid_evals, "selected": list(best.values()),
           "cell_counts": {f"{f}|{L}": len(v) for (f, L), v in rows_by.items()}}
    (HERE / "jobs" / f"n{N}_k{k}_r{r}.json").write_text(json.dumps(out) + "\n")
    print(f"job N{N} k{k} r{r} {status} {out['seconds']:.1f}s", flush=True)
    return {"N": N, "k": k, "r": r, "status": status, "seconds": out["seconds"]}


def packing_score(J, part):
    s = 0.0
    for t in part:
        s += abs(J[t[0], t[1]]) + abs(J[t[0], t[2]]) + abs(J[t[1], t[2]])
    return float(s)


def grad_check():
    inst = m3.instance(6, 4)
    part, _ = best_packing(inst.J)
    ip = with_packing(inst, part)
    w1_err = float(np.max(np.abs(ip.w1 - ip.wX)))
    out = {"pack_w1_matches_sumX": w1_err}
    rng = np.random.default_rng(2)
    for fam, use, L in (("KM_pack", ip, 2), ("KM_gl", inst, 2), ("KM", inst, 2), ("qaoa", inst, 3)):
        x = clip_of(use, fam, rng.normal(scale=0.15, size=nparam(fam, L)), L)
        _, g, _ = energy_of(use, fam, x, L)
        err = 0.0
        for i in range(x.size):
            e = np.zeros_like(x); e[i] = 1e-6
            err = max(err, abs((energy_of(use, fam, x + e, L)[0] - energy_of(use, fam, x - e, L)[0]) / 2e-6 - g[i]))
        out[fam] = float(err)
    x = clip_of(inst, "KM", rng.normal(scale=0.2, size=nparam("KM", 3)), 3)
    _, _, p1 = energy_of(inst, "KM", x, 3)
    _, _, p2 = energy_of(inst, "KM_gl", gl_from_km(x, 3), 3)
    out["gl_nests_KM_overlap"] = float(abs(np.vdot(p1, p2)))
    if w1_err > 1e-12 or out["gl_nests_KM_overlap"] < 1 - 1e-9 or max(out[f] for f in ("KM_pack", "KM_gl", "KM", "qaoa")) > 1e-6:
        raise SystemExit(f"grad check failed {out}")
    return out


def main():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    for d in ("logs", "jobs"):
        (HERE / d).mkdir(exist_ok=True)
    checks = grad_check()
    t0 = time.time()
    deadline = t0 + float(PREREG["optimize_wall_seconds"])
    jobs = [(N, k, r, deadline) for k in PREREG["k_confirm"] for N in (12, 9, 6) for r in range(PREREG["repeats"])]
    with Pool(int(PREREG["workers"])) as pool:
        metas = list(pool.imap(run_job, jobs, chunksize=1))
    (HERE / "run_meta.json").write_text(json.dumps({"prereg_sha256": digest, "grad_check": checks,
                                                     "wall_s": time.time() - t0, "jobs": metas}, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)


PREREG_SHA = "3f89677e809345e06692b4219a0d3ebcb0c3e3157d3f5b64376b37105be375af"

if __name__ == "__main__":
    main()
