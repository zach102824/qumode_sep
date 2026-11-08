"""Spin-glass round 2. Same manifests and circuits as round 1; optimizer protocol from PREREG.json."""
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
R1 = HERE.parent / "spin_glass_r1"
sys.path.insert(0, str(R1))
import run_sg as r1  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
FAMS = ("nodata", "K", "KM", "qaoa")
NHEAD = {"nodata": 1, "K": 2, "KM": 3}
SHELL = (-2.0 * np.pi, 2.0 * np.pi)
CAP = int(PREREG["maxfun"])


def s_J(J):
    return float(np.sqrt(np.mean(np.sum(J * J, axis=1))))


def nparam(fam, L):
    return 2 * L if fam == "qaoa" else NHEAD[fam] + 3 * L


def bounds(inst, fam, L):
    A = 2.0 * np.pi / inst.sJ
    if fam == "qaoa":
        return [(-A, A)] * L + [SHELL] * L
    return [(-A, A)] * NHEAD[fam] + [SHELL] * (3 * L)


def clip(inst, fam, x, L):
    b = np.array(bounds(inst, fam, L))
    return np.minimum(np.maximum(np.asarray(x, float), b[:, 0]), b[:, 1])


def split(fam, x, L):
    """Returns per-layer diagonal coefficients (a_l, b, c) and shell controls (L x 3)."""
    x = np.asarray(x, float)
    if fam == "qaoa":
        cs = np.zeros((L, 3)); cs[:, 0] = x[L:]
        return x[:L].copy(), 0.0, 0.0, cs
    h = NHEAD[fam]
    a = np.full(L, x[0])
    b = x[1] if h > 1 else 0.0
    c = x[2] if h > 2 else 0.0
    return a, float(b), float(c), x[h:].reshape(L, 3).copy()


def join(fam, a, b, c, cs):
    cs = np.asarray(cs, float).reshape(-1, 3)
    if fam == "qaoa":
        return np.concatenate([np.asarray(a, float).reshape(-1), cs[:, 0]])
    head = [float(np.mean(a))] + ([b] if NHEAD[fam] > 1 else []) + ([c] if NHEAD[fam] > 2 else [])
    return np.concatenate([head, cs.reshape(-1)])


def energy_grad(inst, fam, x, L):
    a, b, c, cs = split(fam, x, L)
    psi = inst.plus.astype(np.complex128).copy()
    snaps = []
    for ell in range(L):
        phase = a[ell] * inst.hc if fam == "qaoa" else a[ell] * inst.hc + b * inst.dK + c * inst.dM
        mid = psi * np.exp(-1j * phase)
        phi = cs[ell, 0] * inst.w1 + cs[ell, 1] * inst.w2 + cs[ell, 2] * inst.w3
        x2 = r1.fwht(mid) * np.exp(-1j * phi)
        psi = r1.fwht(x2)
        snaps.append((phase, mid, phi, x2))
    Hpsi = inst.hc * psi + inst.B * r1.apply_X_sum(psi, inst.N)
    E = float(np.vdot(psi, Hpsi).real)
    lam = Hpsi
    ga = np.zeros(L); gb = gc = 0.0
    gcs = np.zeros((L, 3))
    for ell in range(L - 1, -1, -1):
        phase, mid, phi, x2 = snaps[ell]
        lam_x = r1.fwht(lam)
        ov = np.conj(lam_x) * x2
        gcs[ell, 0] = 2.0 * float(np.imag(np.sum(ov * inst.w1)))
        gcs[ell, 1] = 2.0 * float(np.imag(np.sum(ov * inst.w2)))
        gcs[ell, 2] = 2.0 * float(np.imag(np.sum(ov * inst.w3)))
        lam_mid = r1.fwht(lam_x * np.exp(1j * phi))
        ovD = np.conj(lam_mid) * mid
        ga[ell] = 2.0 * float(np.imag(np.sum(ovD * inst.hc)))
        gb += 2.0 * float(np.imag(np.sum(ovD * inst.dK)))
        gc += 2.0 * float(np.imag(np.sum(ovD * inst.dM)))
        lam = lam_mid * np.exp(1j * phase)
    if fam == "qaoa":
        grad = np.concatenate([ga, gcs[:, 0]])
    else:
        head = [ga.sum()] + ([gb] if NHEAD[fam] > 1 else []) + ([gc] if NHEAD[fam] > 2 else [])
        grad = np.concatenate([head, gcs.reshape(-1)])
    return E, grad, psi


def one_start(inst, fam, x0, L):
    x0 = clip(inst, fam, x0, L)
    best = {"E": np.inf, "x": x0.copy(), "psi": None}
    cache = {}
    used = [0]

    def ev(x):
        if used[0] >= CAP:
            raise StopIteration
        used[0] += 1
        E, g, psi = energy_grad(inst, fam, x, L)
        cache["x"] = np.array(x, copy=True); cache["g"] = g
        if E < best["E"]:
            best.update(E=float(E), x=np.array(x, copy=True), psi=psi)
        return E, g

    def fun(x):
        return ev(x)[0]

    def jac(x):
        if "x" in cache and np.array_equal(cache["x"], x):
            return cache["g"]
        return ev(x)[1]

    t0 = time.perf_counter()
    reason, success = "converged_or_returned", False
    try:
        res = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=bounds(inst, fam, L),
                       options={"maxiter": CAP, "maxfun": CAP, "ftol": PREREG["ftol"], "gtol": PREREG["gtol"]})
        success = bool(res.success)
        reason = str(res.message)
    except StopIteration:
        reason = "start_ceiling"
    return {"E": best["E"], "x_final": [float(v) for v in best["x"]], "psi": best["psi"],
            "n_eval": used[0], "capped": reason == "start_ceiling" or used[0] >= CAP,
            "optimizer_success": success, "stop_reason": reason,
            "seconds": time.perf_counter() - t0}


def crn(N, k, L, s, n):
    rng = np.random.default_rng(np.random.SeedSequence([20261117, N, k, L, s]))
    return rng.uniform(size=n)


def grid_starts(inst, fam):
    """Energy-only L1 grid shared by all families: shared-gamma point with only c0 set equals QAOA."""
    pts = []
    for j in range(1, 17):
        u = np.pi * j / 16.0
        for c0 in PREREG["grid_c0"]:
            a = u / inst.sJ
            cs = np.zeros((1, 3)); cs[0, 0] = c0
            pts.append(join(fam, [a], 0.0, 0.0, cs))
    Es = [energy_grad(inst, fam, p, 1)[0] for p in pts]
    order = np.argsort(Es, kind="stable")[: PREREG["L1_grid_top"]]
    return [(f"grid_{int(i)}", pts[int(i)]) for i in order], len(pts)


def random_l1(inst, fam, s):
    u = crn(inst.N, inst.k, 1, s, 6)
    a = np.pi * u[0] / inst.sJ
    cs = np.zeros((1, 3))
    cs[0, 0] = 2.0 * u[1] - 1.0
    cs[0, 1] = 1.2 * u[2] - 0.6
    cs[0, 2] = 1.2 * u[3] - 0.6
    b = a * (2.0 * u[4] - 1.0)
    c = a * (2.0 * u[5] - 1.0)
    return join(fam, [a], b, c, cs)


def ramp(inst, fam, L, delta, s=None):
    t = (np.arange(L) + 0.5) / L
    u = delta * t
    beta = -delta * (1.0 - t)
    if s is not None:
        z = crn(inst.N, inst.k, L, 1000 + s, 2 * L)
        u = u + 0.1 * (2 * z[:L] - 1)
        beta = beta + 0.1 * (2 * z[L:] - 1)
    cs = np.zeros((L, 3)); cs[:, 0] = beta
    a = u / inst.sJ
    if fam != "qaoa":
        a = np.full(L, float(np.mean(a)))
    return join(fam, a, 0.0, 0.0, cs)


def zhou_interp(v):
    v = np.asarray(v, float); p = v.size
    out = np.zeros(p + 1)
    for i in range(1, p + 2):
        lo = v[i - 2] if i - 2 >= 0 else 0.0
        hi = v[i - 1] if i - 1 < p else 0.0
        out[i - 1] = (i - 1) / p * lo + (p - i + 1) / p * hi
    return out


def continuation(inst, fam, parent, L, pid):
    a, b, c, cs = split(fam, parent, L - 1)
    out = []
    csi = np.column_stack([zhou_interp(cs[:, j]) for j in range(3)])
    if fam == "qaoa":
        interp = join(fam, zhou_interp(a), 0, 0, csi)
        cz = np.vstack([cs, np.zeros((1, 3))])
        out.append((f"p{pid}_interp", interp))
        out.append((f"p{pid}_append_zero", join(fam, np.append(a, 0.0), 0, 0, cz)))
    else:
        interp = join(fam, a, b, c, csi)
        r = (L - 1) / L
        out.append((f"p{pid}_interp", interp))
        out.append((f"p{pid}_time_rescaled", join(fam, a * r, b * r, c * r, csi)))
    for j, sig in enumerate((0.1, 0.3)):
        z = np.random.default_rng(np.random.SeedSequence([20261118, inst.N, inst.k, L, pid, j]))
        noise = z.normal(size=interp.size)
        out.append((f"p{pid}_perturb_{sig}", interp + sig * noise))
    return out


def distinct_parents(rows, n):
    chosen = []
    for r in sorted(rows, key=lambda r: r["E"]):
        x = np.array(r["x_final"])
        if all(abs(r["E"] - q["E"]) > 1e-6 or np.linalg.norm(x - np.array(q["x_final"])) > 0.1 for q in chosen):
            chosen.append(r)
        if len(chosen) == n:
            break
    return chosen


def r1_to_r2(fam, x, L):
    x = np.asarray(x, float)
    if fam == "qaoa":
        return x.copy()
    g = x[0]
    head = [g]
    if NHEAD[fam] > 1:
        head.append(g * x[1])
    if NHEAD[fam] > 2:
        head.append(g * x[2])
    return np.concatenate([head, x[NHEAD[fam]:]])


def pad_nested(fam, x_sub, sub):
    x_sub = np.asarray(x_sub, float)
    h_sub = NHEAD[sub]
    head = list(x_sub[:h_sub]) + [0.0] * (NHEAD[fam] - h_sub)
    return np.concatenate([head, x_sub[h_sub:]])


def ground_data(inst):
    dim = 1 << inst.N
    idx = np.arange(dim)
    H = np.diag(inst.hc).astype(float)
    for q in range(inst.N):
        H[idx, idx ^ (1 << q)] += inst.B
    w, V = np.linalg.eigh(H)
    E0 = float(w[0])
    sub = V[:, w - E0 <= PREREG["F_sub_tol"]]
    flip = idx ^ (dim - 1)
    reps = idx[idx < flip]
    Bm = np.zeros((dim, reps.size))
    Bm[reps, np.arange(reps.size)] = 2 ** -0.5
    Bm[flip[reps], np.arange(reps.size)] = 2 ** -0.5
    we, Ve = np.linalg.eigh(Bm.T @ H @ Bm)
    g_even = Bm @ Ve[:, 0]
    v0 = V[:, 0].copy()
    if v0.sum() < 0:
        v0 = -v0
    return {"E0": E0, "Emax": float(w[-1]), "sub": sub, "n_sub": int(sub.shape[1]),
            "E0_even": float(we[0]), "gap_even": float(we[1] - we[0]), "g_even": g_even, "v0": v0,
            "even_ground_in_window": bool(we[0] - E0 <= PREREG["F_sub_tol"])}


def metrics(inst, gd, psi):
    Hpsi = inst.hc * psi + inst.B * r1.apply_X_sum(psi, inst.N)
    E = float(np.vdot(psi, Hpsi).real)
    return {"E_check": E, "var_H": float(np.vdot(Hpsi, Hpsi).real - E * E),
            "F_sub": float(np.sum(np.abs(gd["sub"].T.conj() @ psi) ** 2)),
            "F_even": float(abs(np.vdot(gd["g_even"], psi)) ** 2),
            "F_single": float(abs(np.vdot(gd["v0"], psi)) ** 2),
            "eta": (E - gd["Emax"]) / (gd["E0"] - gd["Emax"]),
            "state_sha256": r1.sha256_state(psi)}


def run_job(job):
    N, k, deadline = job
    doc, J = r1.load_instance(N, k)
    inst = r1.Inst(N, k, J, doc)
    inst.sJ = s_J(J)
    log = HERE / "logs" / f"starts_n{N}_k{k}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists; refuse to mix runs")
    r1_sel = {(r["family"], r["L"]): r for r in json.loads((R1 / "selected.json").read_text())
              if r["N"] == N and r["k"] == k}
    best = {}
    rows_by = {}
    status = "complete"
    states = {}
    with log.open("w") as fh:
        for L in range(1, 7):
            for fam in FAMS:
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_{fam}"
                    break
                seeds = []
                n_grid = 0
                if L == 1:
                    g, n_grid = grid_starts(inst, fam)
                    seeds += g
                    nrand = PREREG["L1_random"] + (1 if fam == "qaoa" else 0)
                    seeds += [(f"crn_{s}", random_l1(inst, fam, s)) for s in range(nrand)]
                else:
                    for pid, par in enumerate(distinct_parents(rows_by[(fam, L - 1)], PREREG["beam"])):
                        seeds += continuation(inst, fam, par["x_final"], L, pid)
                    seeds.append(("fresh_ramp", ramp(inst, fam, L, PREREG["ramp_delta"])))
                    seeds.append(("fresh_ramp_jitter", ramp(inst, fam, L, PREREG["ramp_delta"], s=0)))
                    if fam == "qaoa":
                        a, _, _, cs = split(fam, best[(fam, L - 1)]["x_final"], L - 1)
                        z = np.random.default_rng(np.random.SeedSequence([20261119, N, k, L]))
                        x = join(fam, np.append(a, 0.0), 0, 0, np.vstack([cs, np.zeros((1, 3))]))
                        seeds.append(("best_append_zero_perturb_0.05", x + 0.05 * z.normal(size=x.size)))
                if (fam, L) in r1_sel:
                    seeds.append(("r1_best", r1_to_r2(fam, r1_sel[(fam, L)]["x_final"], L)))
                sub = {"K": "nodata", "KM": "K"}.get(fam)
                if sub and (sub, L) in best:
                    seeds.append((f"nested_{sub}", pad_nested(fam, best[(sub, L)]["x_final"], sub)))
                rows = []
                for label, x0 in seeds:
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{fam}"
                        break
                    row = one_start(inst, fam, x0, L)
                    psi = row.pop("psi")
                    row.update({"N": N, "k": k, "family": fam, "L": L, "label": label,
                                "n_param": nparam(fam, L), "x0": [float(v) for v in clip(inst, fam, x0, L)]})
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    if (fam, L) not in best or row["E"] < best[(fam, L)]["E"]:
                        best[(fam, L)] = row
                        states[(fam, L)] = psi
                    print(f"N{N} k{k} {fam} L{L} {label} E={row['E']:.8f} n={row['n_eval']}", flush=True)
                if rows:
                    rows_by[(fam, L)] = rows
                    fh.write(json.dumps({"cell_summary": True, "N": N, "k": k, "family": fam, "L": L,
                                         "n_grid_energy_evals": n_grid}) + "\n"); fh.flush()
            if status != "complete":
                break
    gd = ground_data(inst)
    cells = []
    for (fam, L), b in best.items():
        rows = rows_by.get((fam, L), [])
        r1r = r1_sel.get((fam, L))
        src, psi, x = "r2", states[(fam, L)], b["x_final"]
        if r1r is not None and r1r["E"] < b["E"] - 1e-12:
            psi, _ = r1.forward(inst, fam, np.array(r1r["x_final"]), L)
            src, x = "r1_preserved", r1r["x_final"]
        m = metrics(inst, gd, psi)
        n_hit = sum(1 for r in rows if r["E"] - b["E"] <= 1e-6)
        n_cap = sum(1 for r in rows if r["capped"])
        cells.append({"N": N, "k": k, "family": fam, "L": L, "E": m["E_check"], "source": src,
                      "label": b["label"] if src == "r2" else "r1_selected", "x_final": x,
                      "coords": "r2_linear" if src == "r2" else "r1_product",
                      "n_param": nparam(fam, L), "n_starts": len(rows),
                      "n_eval": int(sum(r["n_eval"] for r in rows)), "n_capped": n_cap,
                      "n_hit_best_1e-6": n_hit,
                      "converged": bool(n_hit >= 2 and n_cap < 0.5 * len(rows)),
                      "r1_E": None if r1r is None else r1r["E"],
                      "E0": gd["E0"], "E0_even": gd["E0_even"], "n_sub": gd["n_sub"], **m})
        np.save(HERE / "states" / f"n{N}_k{k}_{fam}_L{L}.npy", psi)
    meta = {"N": N, "k": k, "status": status, "s_J": inst.sJ, "E0": gd["E0"], "E0_even": gd["E0_even"],
            "gap_even": gd["gap_even"], "n_sub": gd["n_sub"],
            "even_ground_in_window": gd["even_ground_in_window"], "grad_check": grad_check(inst)}
    (HERE / "cells" / f"n{N}_k{k}.json").write_text(json.dumps({"meta": meta, "cells": cells}) + "\n")
    return meta


def grad_check(inst):
    out = {}
    for fam in FAMS:
        L = 3
        rng = np.random.default_rng(1)
        x = clip(inst, fam, rng.normal(scale=0.2, size=nparam(fam, L)), L)
        _, g, psi = energy_grad(inst, fam, x, L)
        err = 0.0
        for i in range(x.size):
            e = np.zeros_like(x); e[i] = 1e-6
            fd = (energy_grad(inst, fam, x + e, L)[0] - energy_grad(inst, fam, x - e, L)[0]) / 2e-6
            err = max(err, abs(fd - g[i]))
        out[fam] = err
    # linear-coefficient KM equals the round-1 product form
    x1 = np.array([0.17, -0.5, 0.3, 0.1, 0.2, -0.1, -0.4, 0.0, 0.3])
    p1, _ = r1.forward(inst, "KM", x1, 2)
    _, _, p2 = energy_grad(inst, "KM", r1_to_r2("KM", x1, 2), 2)
    out["r1_r2_KM_overlap"] = float(abs(np.vdot(p1, p2)))
    return out


def main():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    for d in ("logs", "cells", "states"):
        (HERE / d).mkdir(exist_ok=True)
    deadline = time.time() + float(PREREG["wall_seconds"])
    jobs = [(N, k, deadline) for N in (12, 9, 6) for k in (0, 1, 2, 3)]
    t0 = time.time()
    with Pool(int(PREREG["workers"])) as pool:
        metas = pool.map(run_job, jobs, chunksize=1)
    (HERE / "run_meta.json").write_text(json.dumps({"jobs": metas, "wall_s": time.time() - t0,
                                                     "prereg_sha256": digest}, indent=2) + "\n")
    print("done", time.time() - t0, flush=True)


PREREG_SHA = "41f48303604e1efa445b8a5fa2075b18707d8ddba89e4698d08470368cf13df4"

if __name__ == "__main__":
    main()
