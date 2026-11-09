"""Historical B=0.1 fits with a repulsion escape from the KM energy trap.

Interleaved shared-gamma families only. Shell angles are not shifted by pi/4 or pi/2.
The penalty is not the selection value. Ground states are not loaded.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_cri"))
import layout_gate  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "93ed4230099ce65d77a3f14804aabe5fd0bcf0b036f38f098e3c821bb2e14907"
LAM = float(PREREG["repulsion_lambda"])


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    return digest


def bind():
    names = PREREG["family_order"]
    if "KM_gl" in names or any(str(f).endswith("_pack") or str(f).endswith("_fid") for f in names):
        raise SystemExit("a forbidden family is in the protocol")
    if names[:3] != ["nodata", "K", "KM"] or "qaoa" not in names:
        raise SystemExit("family order does not keep the standard nest ahead of KM")
    m4.HERE = HERE
    m4.PREREG = PREREG
    m4.FAMS = tuple(names)
    m4.CAP = int(PREREG["maxfun"])


def interleaved(N):
    m = int(N) // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def km_repulsion(inst, x, L, phi, coeff):
    """True energy, penalty, joint gradient, and state. coeff=0 is the energy gradient."""
    a, b, c, cs = m4.m2.split("KM", x, L)
    psi = inst.plus.astype(np.complex128).copy()
    snaps = []
    for ell in range(L):
        phase = a[ell] * inst.hc + b * inst.dK + c * inst.dM
        mid = psi * np.exp(-1j * phase)
        shell = cs[ell, 0] * inst.w1 + cs[ell, 1] * inst.w2 + cs[ell, 2] * inst.w3
        x2 = m4.r1.fwht(mid) * np.exp(-1j * shell)
        psi = m4.r1.fwht(x2)
        snaps.append((phase, mid, shell, x2))
    Hpsi = inst.hc * psi + inst.B * m4.r1.apply_X_sum(psi, inst.N)
    E = float(np.vdot(psi, Hpsi).real)
    ov = np.vdot(phi, psi)
    pen = float(coeff * (ov.real * ov.real + ov.imag * ov.imag))
    adj = Hpsi + (coeff * ov) * phi
    ga = np.zeros(L)
    gb = gc = 0.0
    gcs = np.zeros((L, 3))
    for ell in range(L - 1, -1, -1):
        phase, mid, shell, x2 = snaps[ell]
        lam_x = m4.r1.fwht(adj)
        prod = np.conj(lam_x) * x2
        gcs[ell, 0] = 2.0 * float(np.imag(np.sum(prod * inst.w1)))
        gcs[ell, 1] = 2.0 * float(np.imag(np.sum(prod * inst.w2)))
        gcs[ell, 2] = 2.0 * float(np.imag(np.sum(prod * inst.w3)))
        lam_mid = m4.r1.fwht(lam_x * np.exp(1j * shell))
        ovD = np.conj(lam_mid) * mid
        ga[ell] = 2.0 * float(np.imag(np.sum(ovD * inst.hc)))
        gb += 2.0 * float(np.imag(np.sum(ovD * inst.dK)))
        gc += 2.0 * float(np.imag(np.sum(ovD * inst.dM)))
        adj = lam_mid * np.exp(1j * phase)
    grad = np.concatenate([[ga.sum(), gb, gc], gcs.reshape(-1)])
    return E, pen, grad, psi


def kicks(parent, N, k, L, r):
    parent = np.asarray(parent, float)
    sign = parent.copy()
    sign[1] *= -1.0
    sign[2] *= -1.0
    swap = parent.copy()
    swap[1], swap[2] = parent[2], parent[1]
    head = parent.copy()
    z = m4.seed(20261294, N, k, L, r)
    head[:3] = head[:3] + z.normal(size=3)
    out = (("esc_signflip", sign), ("esc_swap", swap), ("esc_headkick", head))
    for _name, vec in out:
        if np.max(np.abs(vec[3:] - parent[3:])) > 1e-15:
            raise SystemExit("an escape kick changed a shell angle")
    return out


def margin_row(inst, fam, x, L):
    x = np.asarray(x, float)
    _, g, _ = m4.energy_of(inst, fam, x, L)
    b = np.array(m4.bounds_of(inst, fam, L), float)
    pg = np.where((x <= b[:, 0] + 1e-12) & (g > 0), 0.0, g)
    pg = np.where((x >= b[:, 1] - 1e-12) & (pg < 0), 0.0, pg)
    touch = bool(np.any((x - b[:, 0] < 1e-6) | (b[:, 1] - x < 1e-6)))
    return float(np.max(np.abs(pg))), touch


def repulsion_run(inst, x0, L, phi):
    x0 = m4.clip_of(inst, "KM", x0, L)
    bnds = m4.bounds_of(inst, "KM", L)
    best_obj = {"value": np.inf, "x": x0.copy()}
    best_true = {"E": np.inf, "x": x0.copy()}
    cache, used = {}, [0]

    def ev(x):
        if used[0] >= m4.CAP:
            raise StopIteration
        used[0] += 1
        E, pen, g, _psi = km_repulsion(inst, x, L, phi, LAM)
        cache["x"], cache["g"] = np.array(x, copy=True), g
        if E + pen < best_obj["value"]:
            best_obj.update(value=float(E + pen), x=np.array(x, copy=True))
        if E < best_true["E"]:
            best_true.update(E=float(E), x=np.array(x, copy=True))
        return E + pen, g

    t0 = time.perf_counter()
    reason = "returned"
    try:
        res = minimize(lambda x: ev(x)[0], x0, jac=lambda x: cache["g"] if "x" in cache and np.array_equal(cache["x"], x) else ev(x)[1],
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": m4.CAP, "maxfun": m4.CAP, "ftol": PREREG["ftol"], "gtol": PREREG["gtol"]})
        reason = str(res.message)
    except StopIteration:
        reason = "start_ceiling"
    return {"x_pen": best_obj["x"], "x_true": best_true["x"], "E_true": best_true["E"],
            "n_eval": used[0], "capped": used[0] >= m4.CAP, "stop_reason": reason,
            "seconds": time.perf_counter() - t0}


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


def run_one(inst, fam, label, x0, par, N, k, r, L, fh, rows, best, stage):
    row = m4.one_start(inst, fam, x0, L)
    row.update({"N": N, "k": k, "r": r, "family": fam, "L": L, "label": label, "stage": stage,
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


def self_check():
    layout_gate.main()
    out = {}
    if abs(LAM - 1.0) > 1e-15:
        raise SystemExit("repulsion lambda is not the frozen value")
    rng = np.random.default_rng(2)
    for N in (6, 9, 12):
        inst = m4.m3.instance(N, 4)
        if inst.B != 0.1 or inst.k != 4:
            raise SystemExit("phase check left the development instance")
        if m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12:
            raise SystemExit("parameter count is not the shared-gamma count")
        L = 2
        x = m4.clip_of(inst, "KM", rng.normal(scale=0.2, size=m4.nparam("KM", L)), L)
        x[1], x[2] = 0.4, -0.7
        y = m4.clip_of(inst, "KM", rng.normal(scale=0.2, size=x.size), L)
        _, _, phi = m4.energy_of(inst, "KM", y, L)
        phi = np.array(phi, copy=True)
        E0, pen0, g0, psi0 = km_repulsion(inst, x, L, phi, 0.0)
        E1, g1, psi1 = m4.energy_of(inst, "KM", x, L)
        if abs(E0 - E1) > 1e-10 or np.max(np.abs(g0 - g1)) > 1e-8:
            raise SystemExit(f"zero penalty does not match the energy gradient at N={N}")
        if abs(np.vdot(psi0, psi1)) < 1 - 1e-8:
            raise SystemExit(f"repulsion state drifted at N={N}")
        _E, pen, g, _psi = km_repulsion(inst, x, L, phi, LAM)
        err = 0.0
        for i in range(x.size):
            e = np.zeros_like(x)
            e[i] = 1e-6
            vp = km_repulsion(inst, x + e, L, phi, LAM)
            vm = km_repulsion(inst, x - e, L, phi, LAM)
            num = ((vp[0] + vp[1]) - (vm[0] + vm[1])) / 2e-6
            err = max(err, abs(num - g[i]))
        out[f"pen_N{N}"] = float(err)
        if err > 1e-6:
            raise SystemExit(f"penalty gradient failed at N={N} {err}")
        _, _, psi = m4.energy_of(inst, "KM", x, L)
        shifted = np.array(x, copy=True)
        shifted[3::3] = shifted[3::3] + 0.5 * np.pi
        _, _, psi_n = m4.energy_of(inst, "KM", shifted, L)
        if abs(abs(np.vdot(psi, psi_n)) - 1.0) > 1e-8:
            raise SystemExit(f"a pi/2 shell shift was expected to be a global phase at N={N}")
        for name, vec in kicks(x, N, 4, L, 0):
            vec = m4.clip_of(inst, "KM", vec, L)
            _, _, psik = m4.energy_of(inst, "KM", vec, L)
            ov = float(abs(np.vdot(psi, psik)))
            out[f"{name}_N{N}"] = ov
            if ov > 0.95:
                raise SystemExit(f"{name} is nearly trivial at N={N}")
        _, snaps = m4.r1.forward(inst, "KM", x, L, store=True)
        gammas = [float(s[-1]) for s in snaps]
        if max(gammas) - min(gammas) > 1e-12:
            raise SystemExit("escape circuit does not share gamma")
    inst = m4.m3.instance(6, 4)
    for fam, depth in (("KM", 2), ("qaoa", 3), ("K", 2), ("nodata", 2)):
        z = m4.clip_of(inst, fam, rng.normal(scale=0.15, size=m4.nparam(fam, depth)), depth)
        _, g, _ = m4.energy_of(inst, fam, z, depth)
        err = 0.0
        step = 1e-6
        base = m4.energy_of(inst, fam, z, depth)[0]
        for i in range(z.size):
            e = np.zeros_like(z)
            e[i] = step
            num = (m4.energy_of(inst, fam, z + e, depth)[0] - m4.energy_of(inst, fam, z - e, depth)[0]) / (2 * step)
            err = max(err, abs(num - g[i]))
        out[fam] = float(err)
        if err > 1e-6 or not np.isfinite(base):
            raise SystemExit(f"gradient check failed {fam} {err}")
    return out


def run_job(job):
    N, k, r, deadline = job
    if k not in PREREG["k_confirm"] or k in (100, 101):
        raise SystemExit(f"k={k} is outside the frozen confirmation window")
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}")
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
                seeds, grid_add = standard_seeds(inst, fam, L, r, rows_by, best)
                grid_evals += grid_add
                rows = []
                for label, x0, par in seeds:
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{fam}"
                        break
                    run_one(inst, fam, label, x0, par, N, k, r, L, fh, rows, best, "standard")
                if status != "complete":
                    break
                if fam == "KM":
                    trap = best[("KM", L)]
                    _, _, phi = m4.energy_of(inst, "KM", np.array(trap["x_final"]), L)
                    phi = np.array(phi, copy=True)
                    for label, x0 in kicks(trap["x_final"], N, k, L, r):
                        if time.time() >= deadline:
                            status = f"wall_in_L{L}_escape"
                            break
                        rep = repulsion_run(inst, x0, L, phi)
                        pg, touch = margin_row(inst, "KM", rep["x_true"], L)
                        _, _, psi_e = m4.energy_of(inst, "KM", rep["x_true"], L)
                        row = {"E": rep["E_true"], "x_final": [float(v) for v in rep["x_true"]],
                               "n_eval": rep["n_eval"], "capped": rep["capped"], "stop_reason": rep["stop_reason"],
                               "pg_inf": pg, "touch_bound": touch, "seconds": rep["seconds"],
                               "N": N, "k": k, "r": r, "family": "KM", "L": L, "label": label,
                               "stage": "repulsion", "root": trap["root"], "nested_anc": False,
                               "n_param": m4.nparam("KM", L),
                               "x0": [float(v) for v in m4.clip_of(inst, "KM", x0, L)],
                               "trap_overlap": float(abs(np.vdot(phi, psi_e)))}
                        rows.append(row)
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        if row["E"] < best[("KM", L)]["E"]:
                            best[("KM", L)] = row
                        if time.time() >= deadline:
                            status = f"wall_in_L{L}_polish"
                            break
                        prow = run_one(inst, "KM", "polish_" + label[4:], rep["x_pen"], trap, N, k, r, L, fh, rows, best, "polish")
                        _, _, psi_p = m4.energy_of(inst, "KM", np.array(prow["x_final"]), L)
                        prow["trap_overlap"] = float(abs(np.vdot(phi, psi_p)))
                if status != "complete":
                    break
                rows_by[(fam, L)] = rows
            if status != "complete":
                break
    out = {"N": N, "k": k, "r": r, "status": status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_interleaved": interleaved(N),
           "grid_energy_evals": grid_evals, "selected": list(best.values()),
           "cell_counts": {f"{f}|{L}": len(v) for (f, L), v in rows_by.items()}}
    (HERE / "jobs" / f"n{N}_k{k}_r{r}.json").write_text(json.dumps(out) + "\n")
    print(f"job N{N} k{k} r{r} {status} {out['seconds']:.1f}s", flush=True)
    return {"N": N, "k": k, "r": r, "status": status, "seconds": out["seconds"]}


def main():
    digest = check_hash()
    bind()
    checks = self_check()
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
        "rule": "repulsion escape from the energy-best KM state, interleaved shared-gamma",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)
    print("jobs", len(metas), flush=True)


if __name__ == "__main__":
    main()
