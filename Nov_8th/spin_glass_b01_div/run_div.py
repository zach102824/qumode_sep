"""Historical B=0.1: continue L=1 KM states the energy beam drops.

The continuation search is the round-4 list, two repeats, interleaved packing.
Up to three discarded L=1 states, each at least 0.5 half-width units from that
beam, are continued on their own through L=6. The continuation vector stays a
candidate. No ground state is loaded.
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
PREREG_SHA = "82ac2ac410aea9512bc5e5e865f0e25f49432d142616bdf85bf012faab80cbe6"
MIN_SEP = float(PREREG["min_sep"])
DIVERSE_COUNT = int(PREREG["diverse_count"])
LINE_SEED = 20261306


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


def half_distance(inst, x, y, L):
    b = np.array(m4.bounds_of(inst, "KM", L), float)
    half = 0.5 * (b[:, 1] - b[:, 0])
    if np.any(half <= 0):
        raise SystemExit("a KM box width is not positive")
    return float(np.linalg.norm((np.asarray(x, float) - np.asarray(y, float)) / half))


def diverse_parents(inst, rows):
    """Up to three optimized L=1 states outside the energy beam and far from it."""
    beam = m4.m2.distinct_parents(rows, int(PREREG["beam"]))
    if len(beam) == 0:
        raise SystemExit("the L=1 beam is empty")
    scored = []
    for row in rows:
        if row in beam:
            continue
        dist = min(half_distance(inst, row["x_final"], kept["x_final"], 1) for kept in beam)
        scored.append((dist, float(row["E"]), str(row["label"]), row))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    kept = []
    for dist, _energy, _label, row in scored:
        if dist < MIN_SEP:
            break
        kept.append((row, dist))
        if len(kept) == DIVERSE_COUNT:
            break
    return kept


def lineage_children(inst, parent_x, L, r, lineage):
    a, b, c, cs = m4.m2.split("KM", parent_x, L - 1)
    csi = np.column_stack([m4.m2.zhou_interp(cs[:, j]) for j in range(3)])
    interp = m4.m2.join("KM", a, b, c, csi)
    scale = (L - 1) / L
    out = [
        (f"div{lineage}_interp", interp),
        (f"div{lineage}_time_rescaled", m4.m2.join("KM", a * scale, b * scale, c * scale, csi)),
    ]
    for j, sig in enumerate((0.1, 0.3)):
        z = m4.seed(LINE_SEED, inst.N, inst.k, L, r, lineage, j)
        out.append((f"div{lineage}_perturb_{sig}", interp + sig * z.normal(size=interp.size)))
    return out


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


def run_one(inst, fam, label, x0, par, N, k, r, L, fh, rows, best, circuit=None):
    circuit = fam if circuit is None else circuit
    row = m4.one_start(inst, circuit, x0, L)
    row.update({"N": N, "k": k, "r": r, "family": fam, "L": L, "label": label,
                "root": label if par is None else par["root"],
                "nested_anc": bool(par is not None and par.get("nested_anc", False)),
                "n_param": m4.nparam(circuit, L),
                "x0": [float(v) for v in m4.clip_of(inst, circuit, x0, L)]})
    rows.append(row)
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    if best is not None and ((fam, L) not in best or row["E"] < best[(fam, L)]["E"]):
        best[(fam, L)] = row
    return row


def self_check():
    layout_gate.main()
    out = {}
    if DIVERSE_COUNT != 3 or abs(MIN_SEP - 0.5) > 1e-15 or LINE_SEED != 20261306:
        raise SystemExit("diverse budget is not the frozen budget")
    if int(PREREG["beam"]) != 3:
        raise SystemExit("the screen is not defined against a beam of three")
    if abs(float(PREREG["diagnosis"]["drop_min"]) - 0.01) > 1e-15 or int(PREREG["diagnosis"]["drop_count"]) != 4:
        raise SystemExit("diagnosis threshold moved")
    if list(PREREG["k_confirm"]) != list(range(166, 178)):
        raise SystemExit("confirmation window is not k=166..177")
    rng = np.random.default_rng(2)
    for N in (6, 9, 12):
        inst = m4.m3.instance(N, 4)
        if inst.B != 0.1 or inst.k != 4:
            raise SystemExit("phase check left the development instance")
        if m4.nparam("KM", 6) != 21 or m4.nparam("qaoa", 6) != 12:
            raise SystemExit("parameter count is not the shared-gamma count")
        rows = []
        for lab, x0 in m4.grid_starts(inst, "KM", 0):
            row = m4.one_start(inst, "KM", x0, 1)
            row.update({"label": lab, "root": lab})
            rows.append(row)
        for s in range(PREREG["L1_random"]["KM"]):
            row = m4.one_start(inst, "KM", m4.random_l1(inst, "KM", 0, s), 1)
            row.update({"label": f"crn_{s}", "root": f"crn_{s}"})
            rows.append(row)
        kept = diverse_parents(inst, rows)
        if len(kept) < 1 or len(kept) > DIVERSE_COUNT:
            raise SystemExit(f"development screen is empty or oversized at N={N}")
        if any(dist < MIN_SEP for _row, dist in kept):
            raise SystemExit(f"a kept lineage is inside the separation at N={N}")
        parent = np.asarray(kept[0][0]["x_final"], float)
        kids = lineage_children(inst, parent, 2, 0, 0)
        names = [name for name, _vec in kids]
        if names != ["div0_interp", "div0_time_rescaled", "div0_perturb_0.1", "div0_perturb_0.3"]:
            raise SystemExit("lineage maps are not the recorded continuation maps")
        a, b, c, cs = m4.m2.split("KM", parent, 1)
        csi = np.column_stack([m4.m2.zhou_interp(cs[:, j]) for j in range(3)])
        expect_interp = m4.m2.join("KM", a, b, c, csi)
        if kids[0][1].size != 9 or np.max(np.abs(kids[0][1] - expect_interp)) > 1e-12:
            raise SystemExit("interpolation is not the Zhou map on the KM layout")
        for name, vec in kids:
            if vec.size != 9:
                raise SystemExit("a lineage child is not the L=2 KM layout")
        z = m4.seed(LINE_SEED, N, 4, 2, 0, 0, 1)
        expect = kids[0][1] + 0.3 * z.normal(size=kids[0][1].size)
        if np.max(np.abs(kids[-1][1] - expect)) > 1e-12:
            raise SystemExit("the lineage perturbation is not the recorded N(0, 0.3) kick")
        probe = m4.clip_of(inst, "KM", rng.normal(scale=0.2, size=9), 2)
        _, _, psi = m4.energy_of(inst, "KM", probe, 2)
        shifted = np.array(probe, copy=True)
        shifted[3::3] = shifted[3::3] + 0.5 * np.pi
        _, _, psi_n = m4.energy_of(inst, "KM", shifted, 2)
        if abs(abs(np.vdot(psi, psi_n)) - 1.0) > 1e-8:
            raise SystemExit(f"a pi/2 shell shift was expected to be a global phase at N={N}")
        _, snaps = m4.r1.forward(inst, "KM", probe, 2, store=True)
        gammas = [float(s[-1]) for s in snaps]
        if max(gammas) - min(gammas) > 1e-12:
            raise SystemExit("KM circuit does not share gamma")
        out[f"n{N}_kept"] = len(kept)
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
    near = {"E": -3.0, "label": "near", "x_final": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    mid = {"E": -2.0, "label": "mid", "x_final": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]}
    side = {"E": -1.0, "label": "side", "x_final": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]}
    far = {"E": -0.1, "label": "far", "x_final": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]}
    copy = {"E": -3.0, "label": "copy", "x_final": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    kept = diverse_parents(inst, [near, mid, side, far, copy])
    if len(kept) != 1 or kept[0][0]["label"] != "far" or kept[0][1] < MIN_SEP:
        raise SystemExit("synthetic screen did not keep only the separated state")
    energy, kept_label = -0.1, "historical"
    for label, value in (("div0_interp", 0.2), ("div0_perturb_0.1", -0.1)):
        if value < energy:
            energy, kept_label = value, label
    if kept_label != "historical" or energy != -0.1:
        raise SystemExit("a tie or a worse start replaced the continuation vector")
    if -0.3 < energy:
        energy, kept_label = -0.3, "div0_perturb_0.3"
    if kept_label != "div0_perturb_0.3" or energy != -0.3:
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
    return status, best, rows_by


def run_diverse(inst, N, k, r, parents, deadline):
    log = HERE / "logs" / f"n{N}_k{k}_div_r{r}.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists")
    state = [{"lineage": i, "x": np.asarray(row["x_final"], float), "root": f"div{i}"}
             for i, (row, _dist) in enumerate(parents)]
    l6 = []
    status = "complete"
    with log.open("w") as fh:
        for L in range(2, 7):
            nxt = []
            for lin in state:
                if time.time() >= deadline:
                    status = f"wall_before_L{L}_div{lin['lineage']}"
                    break
                best = None
                for label, x0 in lineage_children(inst, lin["x"], L, r, lin["lineage"]):
                    if time.time() >= deadline:
                        status = f"wall_in_L{L}_{label}"
                        break
                    row = run_one(inst, "KM_div", label, x0, {"root": lin["root"], "nested_anc": False},
                                  N, k, r, L, fh, l6 if L == 6 else [], None, circuit="KM")
                    if best is None or row["E"] < best["E"]:
                        best = row
                if status != "complete":
                    break
                nxt.append({"lineage": lin["lineage"], "x": np.asarray(best["x_final"], float), "root": lin["root"]})
            if status != "complete":
                break
            state = nxt
    return status, l6


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
    screens = {}
    status = "complete"
    div_status = "not_run"
    base_l6 = None
    for r in range(PREREG["repeats_primary"]):
        st, best, rows_by = run_repeat(inst, N, k, r, deadline, grid_evals)
        repeats[str(r)] = {"status": st, "selected": list(best.values())}
        if st != "complete":
            status = st
            break
        parents = diverse_parents(inst, rows_by[("KM", 1)])
        screens[r] = parents
        repeats[str(r)]["n_diverse"] = len(parents)
        repeats[str(r)]["diverse"] = [
            {"label": row["label"], "E": row["E"], "distance": dist, "root": f"div{i}"}
            for i, (row, dist) in enumerate(parents)]
        row = best[("KM", 6)]
        if base_l6 is None or row["E"] < base_l6["E"]:
            base_l6 = row
    div_winner = None
    if status == "complete":
        div_status = "complete"
        winner = {"E": base_l6["E"], "x_final": base_l6["x_final"], "label": "historical",
                  "n_eval": 0, "capped": False, "stop_reason": "historical", "pg_inf": base_l6["pg_inf"],
                  "touch_bound": base_l6["touch_bound"], "seconds": 0.0, "n_param": 21,
                  "root": base_l6["root"], "nested_anc": False, "family": "KM_div", "L": 6,
                  "r": -1, "N": N, "k": k}
        for r, parents in screens.items():
            st, rows = run_diverse(inst, N, k, r, parents, deadline)
            if st != "complete":
                div_status = st
                break
            for row in rows:
                if row["E"] < winner["E"]:
                    winner = row
        status = div_status
        if div_status == "complete":
            if winner["E"] > base_l6["E"] + 1e-9:
                raise SystemExit("diverse selection discarded the continuation vector")
            div_winner = winner
    out = {"N": N, "k": k, "status": status, "div_status": div_status, "seconds": time.time() - t0,
           "J_sha256_f64le": inst.doc["J_sha256_f64le"], "s_J": inst.sJ,
           "packing_interleaved": interleaved(N),
           "grid_energy_evals": grid_evals[0],
           "repeats": repeats,
           "km_base_L6": None if base_l6 is None else {"E": base_l6["E"], "label": base_l6["label"], "r": base_l6["r"]},
           "km_div": div_winner}
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
        "rule": "interleaved shared-gamma KM; continuation column plus up to three diverse L=1 lineages",
        "prereg_sha256": digest,
        "grad_check": checks,
        "wall_s": time.time() - t0,
        "jobs": metas,
    }, indent=2) + "\n")
    print("optimize done", time.time() - t0, flush=True)
    print("jobs", len(metas), flush=True)


if __name__ == "__main__":
    main()
