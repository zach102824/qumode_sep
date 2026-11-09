"""Historical B=0.1 fits with half-pi shell restarts. The union is applied after the fits.

No per-layer gamma. Ground states are not loaded.
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
sys.path.insert(0, str(HERE.parent / "spin_glass_b01_fid"))
import packing_fiedler  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_cri"))
import layout_gate  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
PREREG_SHA = "824671dd6b7dbad18991d692315b730e6393e8e40937170f818b8c38bdbef5ac"
HALFPI = 0.5 * np.pi
KM_RESTART = ("KM_pack", "KM_fid")


def check_hash():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    return digest


def bind():
    if "KM_gl" in PREREG["family_order"]:
        raise SystemExit("per-layer family is in the protocol")
    m4.HERE = HERE
    m4.PREREG = PREREG
    m4.FAMS = tuple(PREREG["family_order"])
    m4.CAP = int(PREREG["maxfun"])
    m4.BASE.update({
        "nodata_pack": "nodata", "K_pack": "K", "KM_pack": "KM",
        "nodata_fid": "nodata", "K_fid": "K", "KM_fid": "KM",
    })
    m4.packed = lambda fam: str(fam).endswith("_pack") or str(fam).endswith("_fid")


def use_for(inst, inst_pack, inst_fid, fam):
    if str(fam).endswith("_fid"):
        return inst_fid
    if str(fam).endswith("_pack"):
        return inst_pack
    return inst


def halfpi_shifts(parent_x):
    """Four deterministic displacements. A shift of pi is not among them."""
    x = np.asarray(parent_x, float)
    out = []
    for j, name in enumerate(("halfpi_shell_0", "halfpi_shell_1", "halfpi_shell_2")):
        y = x.copy()
        y[3 + j::3] = y[3 + j::3] + HALFPI
        out.append((name, y))
    y = x.copy()
    y[0] = -y[0]
    y[3::3] = y[3::3] + HALFPI
    out.append(("halfpi_gamma_shell0", y))
    return out


def grad_and_layout():
    layout_gate.main()
    inst = m4.m3.instance(6, 4)
    if inst.k != 4:
        raise SystemExit("phase check used a confirmation index")
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}, not the historical 0.1")
    part, _ = m4.best_packing(inst.J)
    fid, _ = packing_fiedler.fiedler_packing(inst.J)
    ip = m4.with_packing(inst, part)
    iff = m4.with_packing(inst, fid)
    for packed, name in ((ip, "abs"), (iff, "fiedler"), (inst, "interleaved")):
        if float(np.max(np.abs(packed.w1 - packed.wX))) > 1e-12:
            raise SystemExit(f"{name} w1 is not sum X")
    if any(m4.nparam(f, 6) != 21 for f in ("KM_pack", "KM_fid")) or m4.nparam("qaoa", 6) != 12:
        raise SystemExit("parameter count is not the shared-gamma count")
    L = 4
    shells = np.arange(3 * L, dtype=float).reshape(L, 3) * 0.01
    x = m4.r1.pack("KM", 0.2, {"K": -0.1, "M": 0.05}, shells)
    for packed in (ip, iff):
        _, snaps = m4.r1.forward(packed, "KM", x, L, store=True)
        gammas = [float(s[-1]) for s in snaps]
        if max(gammas) - min(gammas) > 1e-12:
            raise SystemExit("a packed circuit does not share gamma")
        g0 = snaps[0][4]
        if any(float(np.max(np.abs(s[4] - g0))) > 1e-12 for s in snaps):
            raise SystemExit("D changes across layers")
    shifts = halfpi_shifts(x)
    if [name for name, _ in shifts] != ["halfpi_shell_0", "halfpi_shell_1", "halfpi_shell_2", "halfpi_gamma_shell0"]:
        raise SystemExit("half-pi labels drifted")
    if abs(float(shifts[0][1][3] - x[3]) - HALFPI) > 1e-12:
        raise SystemExit("half-pi channel 0 was not displaced by pi/2")
    if abs(float(shifts[3][1][0] + x[0])) > 1e-12:
        raise SystemExit("gamma was not negated on the fourth restart")
    _, _, psi = m4.energy_of(ip, "KM_pack", x, L)
    y_pi = np.array(x, float, copy=True)
    y_pi[3::3] = y_pi[3::3] + np.pi
    _, _, psi_pi = m4.energy_of(ip, "KM_pack", y_pi, L)
    if abs(abs(np.vdot(psi, psi_pi)) - 1.0) > 1e-8:
        raise SystemExit("a pi shell shift was expected to be a global phase")
    _, _, psi_h = m4.energy_of(ip, "KM_pack", shifts[0][1], L)
    if abs(abs(np.vdot(psi, psi_h)) - 1.0) < 1e-6:
        raise SystemExit("a half-pi shell shift is trivial")
    rng = np.random.default_rng(2)
    out = {}
    for fam, use, depth in (("KM_pack", ip, 2), ("KM_fid", iff, 2), ("qaoa", inst, 3)):
        z = m4.clip_of(use, fam, rng.normal(scale=0.15, size=m4.nparam(fam, depth)), depth)
        _, g, _ = m4.energy_of(use, fam, z, depth)
        err = 0.0
        for i in range(z.size):
            e = np.zeros_like(z)
            e[i] = 1e-6
            num = (m4.energy_of(use, fam, z + e, depth)[0] - m4.energy_of(use, fam, z - e, depth)[0]) / 2e-6
            err = max(err, abs(num - g[i]))
        out[fam] = float(err)
        if err > 1e-6:
            raise SystemExit(f"gradient check failed {fam} {err}")
    out["halfpi_nontrivial"] = True
    out["pi_is_global_phase"] = True
    return out


def run_one(use, fam, label, x0, par, N, k, r, L, fh, rows, best):
    row = m4.one_start(use, fam, x0, L)
    row.update({"N": N, "k": k, "r": r, "family": fam, "L": L, "label": label,
                "root": label if par is None else par["root"],
                "nested_anc": bool(par is not None and par.get("nested_anc", False)),
                "n_param": m4.nparam(fam, L),
                "x0": [float(v) for v in m4.clip_of(use, fam, x0, L)]})
    rows.append(row)
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    if (fam, L) not in best or row["E"] < best[(fam, L)]["E"]:
        best[(fam, L)] = row
    return row


def run_job(job):
    N, k, r, deadline = job
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit(f"B is {inst.B}")
    if k < 80 or k > 91:
        raise SystemExit(f"k={k} is outside the frozen confirmation window")
    part, pscore = m4.best_packing(inst.J)
    fid, fscore = packing_fiedler.fiedler_packing(inst.J)
    base = m4.interleaved(N)
    inst_pack = m4.with_packing(inst, part)
    inst_fid = m4.with_packing(inst, fid)
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
                use = use_for(inst, inst_pack, inst_fid, fam)
                seeds = []
                if L == 1:
                    grid = m4.grid_starts(use, fam, r)
                    grid_evals += 64
                    seeds += [(lab, x, None) for lab, x in grid]
                    seeds += [(f"crn_{s}", m4.random_l1(use, fam, r, s), None) for s in range(PREREG["L1_random"][fam])]
                else:
                    for pid, par in enumerate(m4.m2.distinct_parents(rows_by[(fam, L - 1)], PREREG["beam"])):
                        seeds += [(lab, x, par) for lab, x in m4.continuation(use, fam, par["x_final"], L, pid, r)]
                    seeds += [("fresh_ramp", m4.ramp(use, fam, L, r, False), None),
                              ("fresh_ramp_jitter", m4.ramp(use, fam, L, r, True), None)]
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
                    run_one(use, fam, label, x0, par, N, k, r, L, fh, rows, best)
                if status != "complete":
                    break
                if fam in KM_RESTART:
                    parent = best[(fam, L)]
                    for label, x0 in halfpi_shifts(parent["x_final"]):
                        if time.time() >= deadline:
                            status = f"wall_in_L{L}_{fam}_halfpi"
                            break
                        run_one(use, fam, label, x0, {"root": f"halfpi:{label}", "nested_anc": False},
                                N, k, r, L, fh, rows, best)
                if status != "complete":
                    break
                rows_by[(fam, L)] = rows
            if status != "complete":
                break
    out = {"N": N, "k": k, "r": r, "status": status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_abs": part, "packing_score_abs": pscore,
           "packing_fiedler": fid, "packing_score_fiedler": fscore,
           "packing_interleaved": base,
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
        "rule": "half-pi shell restarts, then energy-only union",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)
    print("jobs", len(metas), flush=True)


if __name__ == "__main__":
    main()
