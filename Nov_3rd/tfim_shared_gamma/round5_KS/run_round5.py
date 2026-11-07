"""Round-5 KS. Energy only. 75 minute bound.

Order is the frozen priority: replay, fresh KS, equal-budget fresh Rfull,
nested source, neighbor transfer, then QAOA equalization. A partial cell stays partial.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.common import canonical_state, log, sha256_state

from .model import (
    MANIFEST_SHA,
    PACKING_SHA,
    Runner,
    Target,
    coefficient_distance,
    embed_rfull,
    fwht,
    logical_perm,
    shell_weights,
    unpack,
)
from .optimize import ascending_seeds, generator_of, optimize, perturb_seeds, resample_shell

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
BASELINE = json.loads((ROOT / "baseline.json").read_text())
PROTOCOL_SHA = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()


def spectrum(target: Target):
    n = target.n
    dim = 1 << n
    H = np.diag(target.hc.copy())
    idx = np.arange(dim)
    for bit in range(n):
        partner = idx ^ (1 << bit)
        mask = partner > idx
        a, b = idx[mask], partner[mask]
        H[a, b] = target.B
        H[b, a] = target.B
    evals, vecs = np.linalg.eigh(H)
    gs = np.real(vecs[:, 0])
    gs /= np.linalg.norm(gs)
    if float(gs.sum()) < 0:
        gs = -gs
    x = np.arange(dim)
    flip = x ^ (dim - 1)
    y = np.zeros_like(x)
    for i in range(n):
        y |= ((x >> i) & 1) << (n - 1 - i)

    def pr(v):
        r = np.real(v)
        return float(np.dot(r, r[flip])), float(np.dot(r, r[y]))

    gap_acc = None
    for i in range(1, min(16, len(evals))):
        pi, ri = pr(vecs[:, i])
        if pi > 0.5 and ri > 0.5:
            gap_acc = float(evals[i] - evals[0])
            break
    return gs, float(evals[0]), float(evals[-1]), float(evals[1] - evals[0]), gap_acc, pr(gs)


def _preflight(runner: Runner) -> dict:
    t = runner.t
    scales = PROTOCOL["scales_from_manifest_before_fits"]
    for key, attr in (("rK", "rK"), ("rIn", "rIn"), ("rOut", "rOut"), ("rM", "rM"), ("rms_J", "rms_J")):
        if abs(float(scales[key]) - float(getattr(t, attr))) > 1e-12:
            raise RuntimeError(f"scale {key} drifted")
    if hashlib.sha256((ROOT / "baseline.json").read_bytes()).hexdigest() != PROTOCOL["baseline_sha256"]:
        raise RuntimeError("baseline hash drifted")
    if t.hc_sha != PROTOCOL["hc_diag_sha256"]:
        raise RuntimeError("Hc hash drifted")
    out = {"model": t.checks}
    # Physical packing versus logical sublattice.
    pi = [q for cav in t.packings["sublattice"] for q in cav]
    perm = logical_perm(pi, t.n)
    rng = np.random.default_rng(20261112)
    x = rng.normal(scale=0.25, size=1 + 3 + 6)
    psi_l, _, _ = runner.forward("KS", x, 2)
    gammas, theta, cs = unpack("KS", x, 2)
    g_phys = np.empty(1 << t.n)
    g_phys[perm] = t.g_diag("KS", theta)
    w1, w2, w3 = shell_weights(t.n, [[3 * a + r for r in range(3)] for a in range(t.n // 3)])
    psi = runner.plus.copy()
    for ell in range(2):
        psi = psi * np.exp(-1j * float(gammas[0]) * g_phys)
        phi = cs[ell, 0] * w1 + cs[ell, 1] * w2 + cs[ell, 2] * w3
        psi = fwht(fwht(psi) * np.exp(-1j * phi))
    out["physical_maxabs"] = float(np.max(np.abs(psi[perm] - psi_l)))
    x1 = rng.normal(scale=0.3, size=7)
    base, _, _ = runner.forward("KS", x1, 1)
    phase_ok = True
    for j in range(3):
        y = x1.copy()
        y[4 + j] += np.pi
        other, _, _ = runner.forward("KS", y, 1)
        if abs(np.vdot(base, other)) ** 2 < 1 - 1e-10:
            phase_ok = False
    y = x1.copy()
    y[0] *= -1
    y[4:] *= -1
    conj, _, _ = runner.forward("KS", y, 1)
    out["shell_pi_global"] = phase_ok
    out["conjugation_with_conjugate"] = float(abs(np.vdot(conj, np.conj(base))) ** 2)
    out["conjugation_with_same"] = float(abs(np.vdot(conj, base)) ** 2)
    # One-layer resample is a constant schedule.
    row = np.array([[0.2, -0.1, 0.3]])
    held = resample_shell(row, 5)
    out["clamp_is_constant"] = bool(np.max(np.abs(held - row)) < 1e-12)
    replay = {}
    tolE = PROTOCOL["tolerances"]["replay_E"]
    tolF = PROTOCOL["tolerances"]["replay_F"]
    tolO = PROTOCOL["tolerances"]["replay_overlap_deficit"]
    for L in (5, 6):
        blob = json.loads((ROOT / "sources" / f"N12_L{L}_A_sub_M.json").read_text())
        x = np.array(blob["x"], dtype=np.float64)
        Er, _, psir = runner.energy_grad("Rfull", x, L)
        xk = embed_rfull(x, L, t)
        Ek, _, psik = runner.energy_grad("KS", xk, L)
        src = np.load(ROOT / "sources" / f"N12_L{L}_A_sub_M.npz")["psi"]
        row = {
            "source_E": blob["E"], "source_F": blob["F"],
            "replay_E": float(Er), "replay_F": runner.fidelity(psir),
            "nested_E": float(Ek), "nested_F": runner.fidelity(psik),
            "dE_replay": float(Er - blob["E"]), "dF_replay": float(runner.fidelity(psir) - blob["F"]),
            "dE_nested": float(Ek - Er),
            "overlap_source": float(abs(np.vdot(src, psir)) ** 2),
            "overlap_nested": float(abs(np.vdot(psir, psik)) ** 2),
        }
        replay[str(L)] = row
        if abs(row["dE_replay"]) > tolE or abs(row["dF_replay"]) > tolF or row["overlap_source"] < 1 - tolO:
            raise RuntimeError(f"source replay failed L{L} {row}")
        if abs(row["dE_nested"]) > tolE or row["overlap_nested"] < 1 - tolO:
            raise RuntimeError(f"nested embedding failed L{L} {row}")
    out["replay"] = replay
    out["passed"] = bool(
        t.checks["passed"] and out["physical_maxabs"] < 1e-9 and phase_ok
        and out["conjugation_with_conjugate"] > 1 - 1e-10 and out["clamp_is_constant"]
    )
    if not out["passed"]:
        raise RuntimeError(f"preflight failed {out}")
    return out


def _public(runner: Runner, raw: dict) -> dict:
    gen, L, x = raw["arm"], raw["L"], raw["x"]
    E, _, psi = runner.energy_grad(gen, x, L)
    energies = np.array([r["E"] for r in raw["starts"] if "E" in r], dtype=np.float64)
    fs = np.array([r["F"] for r in raw["starts"] if "F" in r], dtype=np.float64)
    order = np.argsort(energies)
    rec = {
        "role": raw["role"], "arm": gen, "n": raw["n"], "L": L, "P": int(x.size),
        "E": float(E), "energy_error": float(E - runner.E0), "F": runner.fidelity(psi),
        "variance_H": runner.variance(psi), "parity": runner.parity_weight(psi),
        "seed_median": raw["seed_median"], "seed_iqr": raw["seed_iqr"], "seed_best": raw["seed_best"],
        "n_eval": raw["n_eval"], "n_starts": raw["n_starts"], "n_success": raw["n_success"],
        "n_failed": raw["n_failed"], "n_skipped": raw["n_skipped"], "seconds": raw["seconds"],
        "init": raw["init"], "gamma": raw["gamma"], "theta": raw["theta"], "shell": raw["shell"],
        "x_exact": [format(float(v), ".17g") for v in x],
        "state_sha256": sha256_state(psi),
        "manifest_sha256": MANIFEST_SHA, "packing_sha256": PACKING_SHA,
        "hc_sha256": runner.t.hc_sha, "protocol_sha256": PROTOCOL_SHA,
        "starts": raw["starts"], "best_so_far": raw["best_so_far"],
        "cdf_by_energy": [{"k": int(i + 1), "E": float(energies[j]), "F": float(fs[j])} for i, j in enumerate(order)],
        "best_F_not_selected": float(fs.max()) if fs.size else None,
        "objective": "energy", "fidelity_used_for_selection": False,
    }
    if gen == "KS":
        th = raw["theta"]
        rec["coefficient_distance"] = coefficient_distance(th["In"], th["Out"], runner.t.rIn, runner.t.rOut, runner.t.rK)
    rec["x"] = np.array(x, copy=True)
    rec["psi"] = psi
    return rec


def _save(rec: dict):
    folder = RESULTS / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    (RESULTS / "states").mkdir(exist_ok=True)
    clean = {k: v for k, v in rec.items() if k not in ("psi", "x")}
    (folder / f"n{rec['n']}_L{rec['L']}_{rec['role']}.json").write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")
    if "psi" in rec:
        np.save(RESULTS / "states" / f"n{rec['n']}_L{rec['L']}_{rec['role']}.npy", canonical_state(rec["psi"]))


def run(budget_min: float = 75.0):
    t0 = time.perf_counter()
    deadline = t0 + budget_min * 60.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    failures = []
    log("spectrum and preflight")
    target = Target(12)
    gs, E0, Emax, gap, gap_acc, pr = spectrum(target)
    runner = Runner(target)
    runner.set_spectrum(gs, E0, Emax, gap, gap_acc)
    log(f"E0={E0:.6f} full_gap={gap:.6e} acc={gap_acc} P={pr[0]:.3f} R={pr[1]:.3f}")
    pre = _preflight(runner)
    log(f"preflight passed nested L6 dE={pre['replay']['6']['dE_nested']:.3e}")
    smoke_target = Target(6)
    if not smoke_target.checks["passed"]:
        raise RuntimeError("N6 split checks failed")
    mem = {}

    def left():
        return deadline - time.perf_counter()

    def fit(role, L, seeds, cap, screen=False):
        key = (role, L)
        path = RESULTS / "jobs" / f"n12_L{L}_{role}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            if rec.get("protocol_sha256") == PROTOCOL_SHA and rec.get("manifest_sha256") == MANIFEST_SHA:
                rec["x"] = np.array([float(v) for v in rec["x_exact"]])
                mem[key] = rec
                log(f"resume {role} L{L} E={rec['E']:.6f}")
                return rec
        if left() < 40:
            failures.append(f"skipped {role} L={L}")
            return None
        gen = generator_of(role)
        log(f"fit {role} L={L} cap={cap} left={left()/60:.1f}min")
        raw = optimize(runner, gen, L, seeds, role, cap, screen=screen)
        rec = _public(runner, raw)
        _save(rec)
        mem[key] = rec
        dist = rec.get("coefficient_distance", {}).get("normalized_distance")
        log(f"  E={rec['E']:.6f} err={rec['energy_error']:.4e} F={rec['F']:.4f} med={rec['seed_median']:.4f} evals={rec['n_eval']} dist={dist}")
        return rec

    def stored_x(L):
        blob = json.loads((ROOT / "sources" / f"N12_L{L}_A_sub_M.json").read_text())
        got = hashlib.sha256((ROOT / "sources" / f"N12_L{L}_A_sub_M.json").read_bytes()).hexdigest()
        if got != BASELINE["stored_rfull"][str(L)]["sha256"]:
            raise RuntimeError("stored source hash mismatch")
        return np.array(blob["x"], dtype=np.float64)

    # Fresh ladders. Equal caps and equal fresh-start counts.
    for role in ("KS_fresh", "Rfull_fresh"):
        gen = generator_of(role)
        for L in range(1, 7):
            prev = None if (role, L - 1) not in mem else mem[(role, L - 1)].get("x")
            if prev is None and (role, L - 1) in mem:
                prev = mem[(role, L - 1)]["x"]
            seeds = ascending_seeds(gen, L, prev if L > 1 else None, role, PROTOCOL["arms"][role]["fresh_starts"])
            fit(role, L, seeds, PROTOCOL["arms"][role]["eval_cap"], screen=(L == 1))

    spec_n = PROTOCOL["arms"]["KS_nested"]
    for L in spec_n["depths"]:
        if left() < 45:
            failures.append(f"skipped KS_nested L={L}")
            continue
        x0 = embed_rfull(stored_x(L), L, target)
        seeds = perturb_seeds(x0, "KS_nested", L, tuple(spec_n["sigmas"]), spec_n["starts_per_sigma"], "nested")
        fit("KS_nested", L, seeds, spec_n["eval_cap"], screen=False)

    spec_t = PROTOCOL["arms"]["KS_transfer"]
    for L in spec_t["depths"]:
        if left() < 45:
            failures.append(f"skipped KS_transfer L={L}")
            continue
        seeds = []
        for name in spec_t["neighbor_order"]:
            src_vec, Lsrc = _neighbor_seed(name, L, target, mem, stored_x)
            if src_vec is None:
                continue
            part = perturb_seeds(
                src_vec, "KS_transfer", L, tuple(spec_t["sigmas"]),
                spec_t["starts_per_sigma_per_neighbor"], f"{name}_L{Lsrc}",
            )
            seeds.extend(part)
        if not seeds:
            failures.append(f"no transfer seeds L={L}")
            continue
        fit("KS_transfer", L, seeds, spec_t["eval_cap"], screen=False)

    qaoa = _qaoa(runner, mem, left, failures)
    public = [{k: v for k, v in rec.items() if k not in ("psi", "x")} for rec in mem.values()]
    summary = {
        "manifest_sha256": MANIFEST_SHA,
        "packing_sha256": PACKING_SHA,
        "hc_sha256": target.hc_sha,
        "protocol_sha256": PROTOCOL_SHA,
        "baseline_sha256": PROTOCOL["baseline_sha256"],
        "E0": E0, "gap_full": gap, "gap_accessible": gap_acc, "gs_parity_reflection": list(pr),
        "preflight": pre,
        "n6_checks_passed": True,
        "jobs": public,
        "qaoa": qaoa,
        "failures": failures,
        "screens": _screens(public),
        "comparisons": _compare(public, qaoa),
        "ledgers": _ledgers(public, qaoa),
        "optimizer_seconds": float(sum(j["seconds"] for j in public)),
        "wall_seconds": time.perf_counter() - t0,
    }
    summary["reading"] = _reading(summary)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _md(RESULTS / "RESULTS.md", summary)
    _plots(summary, RESULTS / "figures")
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    for png in (RESULTS / "figures").glob("*.png"):
        (art / f"round5_KS_{png.name}").write_bytes(png.read_bytes())
    log(f"done {summary['wall_seconds']:.1f}s failures={len(failures)}")
    return summary


def _neighbor_seed(name, L, target, mem, stored_x):
    from .model import pack
    if name == "stored_Rfull_Lm1" and str(L - 1) in BASELINE["stored_rfull"]:
        Lsrc = L - 1
        x_emb = embed_rfull(stored_x(Lsrc), Lsrc, target)
    elif name == "stored_Rfull_Lp1" and str(L + 1) in BASELINE["stored_rfull"]:
        Lsrc = L + 1
        x_emb = embed_rfull(stored_x(Lsrc), Lsrc, target)
    elif name == "fresh_KS_Lp1" and ("KS_fresh", L + 1) in mem:
        Lsrc = L + 1
        x_emb = mem[("KS_fresh", Lsrc)]["x"]
    elif name == "fresh_KS_Lm1" and ("KS_fresh", L - 1) in mem:
        Lsrc = L - 1
        x_emb = mem[("KS_fresh", Lsrc)]["x"]
    else:
        return None, None
    gammas, theta, cs = unpack("KS", x_emb, Lsrc)
    return pack("KS", float(gammas[0]), theta, resample_shell(cs, L)), Lsrc


def _qaoa(runner, mem, left, failures):
    """Replay the same-hash floor. Top up a depth only if its floor is short of the candidate inclusive cost."""
    out = {}
    jobs = Path(__file__).resolve().parents[1] / "round4_N" / "results" / "jobs"
    for p in range(4, 13):
        path = jobs / f"n12_L{p}_qaoa.json"
        blob = json.loads(path.read_text())
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != BASELINE["qaoa_round4_samehash"][str(p)]["sha256"]:
            raise RuntimeError(f"round4 qaoa hash mismatch p={p}")
        x = np.array([float(v) for v in blob["x_exact"]], dtype=np.float64)
        E, _, psi = runner.energy_grad("qaoa", x, p)
        if abs(E - blob["E"]) > 1e-8:
            raise RuntimeError(f"qaoa floor replay p={p} dE={E-blob['E']}")
        out[p] = {
            "p": p, "E": float(E), "F": runner.fidelity(psi), "energy_error": float(E - runner.E0),
            "variance_H": runner.variance(psi), "n_eval_floor": int(blob["n_eval"]),
            "n_eval_new": 0, "seed_median": blob["seed_median"], "seed_iqr": blob["seed_iqr"],
            "n_starts_floor": blob["n_starts"], "provenance": "round4_N same-hash replay",
            "state_sha256": sha256_state(psi),
        }
    # Candidate inclusive costs are filled after we know them. Top-up uses the largest fresh ladder.
    need = 0
    for role in ("KS_fresh", "Rfull_fresh"):
        total = 0
        for L in range(1, 7):
            if (role, L) in mem:
                total += mem[(role, L)]["n_eval"]
        need = max(need, total)
    for L in (4, 5, 6):
        for role in ("KS_nested", "KS_transfer"):
            if (role, L) in mem:
                extra = BASELINE["stored_rfull"].get(str(L), {}).get("nfev_sum", 0) if role == "KS_nested" else 0
                need = max(need, mem[(role, L)]["n_eval"] + int(extra))
    out["_need"] = int(need)
    priority = PROTOCOL["qaoa"]["priority_p"]
    for p in priority:
        floor = out[p]["n_eval_floor"]
        short = need - floor
        if short <= 0:
            out[p]["coverage"] = "floor_covers_candidate_inclusive"
            continue
        if left() < 40:
            failures.append(f"qaoa p={p} short {short}")
            out[p]["coverage"] = f"short_{short}"
            continue
        log(f"qaoa p={p} top-up short={short} left={left()/60:.1f}min")
        prev = None
        # Use this run's lower new starts if any; otherwise no ascending seed, only fresh plus the incumbent.
        seeds = [("floor_incumbent", np.array([float(v) for v in json.loads((jobs / f"n12_L{p}_qaoa.json").read_text())["x_exact"]]), 0.0)]
        seeds.extend(ascending_seeds("qaoa", p, None, "qaoa", 8))
        raw = optimize(runner, "qaoa", p, seeds, "qaoa", max(400, short), screen=False)
        if raw["E"] < out[p]["E"]:
            out[p]["E"] = raw["E"]
            out[p]["F"] = raw["F"]
            out[p]["energy_error"] = float(raw["E"] - runner.E0)
            out[p]["variance_H"] = raw["variance_H"]
            out[p]["state_sha256"] = sha256_state(raw["psi"])
            out[p]["provenance"] = "round4 floor plus this-run top-up"
        out[p]["n_eval_new"] = raw["n_eval"]
        out[p]["topup_median"] = raw["seed_median"]
        out[p]["coverage"] = "topped_up" if raw["n_eval"] + floor >= need else f"still_short_{need - floor - raw['n_eval']}"
        _save_qaoa_topup(raw, runner)
    return out


def _save_qaoa_topup(raw, runner):
    rec = _public(runner, raw)
    rec["role"] = f"qaoa_topup"
    _save(rec)


def _screens(jobs):
    ix = {(j["role"], j["L"]): j for j in jobs}
    stored = BASELINE["stored_rfull"]
    branch = []
    for L in (5, 6):
        for role in ("KS_fresh", "KS_nested", "KS_transfer"):
            cell = ix.get((role, L))
            if cell is None or "coefficient_distance" not in cell:
                continue
            srcE = stored[str(L)]["E"]
            improvement = srcE - cell["E"]  # positive when KS is lower
            dist = cell["coefficient_distance"]["normalized_distance"]
            branch.append({
                "role": role, "L": L, "energy_improvement_vs_stored": improvement,
                "normalized_distance": dist,
                "negligible_energy": improvement < 0.005,
                "small_distance": dist <= 0.001,
            })
    robust = []
    for L in (5, 6):
        a, b = ix.get(("KS_fresh", L)), ix.get(("Rfull_fresh", L))
        if a and b:
            gap = a["seed_median"] - b["seed_median"]
            robust.append({"L": L, "fresh_median_gap": gap, "worse_by_more_than_0.03": gap > 0.03})
    def both_depths(rows, pred):
        got = {r["L"] for r in rows if pred(r)}
        return 5 in got and 6 in got
    # The branch screen is about the best KS energy at each depth, across arms, and that arm's distance.
    best = {}
    for L in (5, 6):
        cands = [r for r in branch if r["L"] == L]
        if cands:
            best[L] = min(cands, key=lambda r: -r["energy_improvement_vs_stored"])
    retire = False
    if 5 in best and 6 in best:
        retire = all(best[L]["negligible_energy"] and best[L]["small_distance"] for L in (5, 6))
    return {
        "per_arm": branch,
        "best_improvement": best,
        "branch_screen_retire_new_candidate": retire,
        "robust_fresh": robust,
        "robust_screen_flag": both_depths(robust, lambda r: r["worse_by_more_than_0.03"]),
    }


def _compare(jobs, qaoa):
    ix = {(j["role"], j["L"]): j for j in jobs}
    rows = []
    tolE = PROTOCOL["tolerances"]["dominance_E"]
    tolF = PROTOCOL["tolerances"]["dominance_F"]
    matched = {int(k): v for k, v in PROTOCOL["matched_raw_P"].items()}
    for role in ("KS_fresh", "KS_nested", "KS_transfer", "Rfull_fresh"):
        for L in range(1, 7):
            cell = ix.get((role, L))
            if cell is None:
                continue
            spec = matched.get(L)
            def q(p):
                return qaoa.get(p) if isinstance(qaoa.get(p), dict) and "E" in qaoa.get(p, {}) else None
            targets = {}
            if spec:
                targets = {"matched": q(spec["matched_p"]), "next": q(spec["next_p"]), "p12": q(12), "equal": q(L), "plus": q(L + 1)}
            def both(other):
                if other is None:
                    return None
                return bool(cell["E"] < other["E"] - tolE and cell["F"] > other["F"] + tolF)
            row = {
                "role": role, "L": L, "P": cell["P"], "E": cell["E"], "F": cell["F"],
                "err": cell["energy_error"], "median": cell["seed_median"],
            }
            if spec:
                row["matched_p"] = spec["matched_p"]
                row["next_p"] = spec["next_p"]
                for name, other in targets.items():
                    row[f"both_{name}"] = both(other)
                    if other is not None:
                        row[f"dE_{name}"] = cell["E"] - other["E"]
                        row[f"dF_{name}"] = cell["F"] - other["F"]
            stored = BASELINE["stored_rfull"].get(str(L))
            if stored and role.startswith("KS"):
                row["dE_stored_rfull"] = cell["E"] - stored["E"]
                row["dF_stored_rfull"] = cell["F"] - stored["F"]
            rows.append(row)
    return rows


def _ledgers(jobs, qaoa):
    marginal = sum(j["n_eval"] for j in jobs)
    external = 0
    used = set()
    for j in jobs:
        if j["role"] == "KS_nested":
            used.add(j["L"])
        if j["role"] == "KS_transfer":
            if j["L"] - 1 >= 3:
                used.add(j["L"] - 1)
            if j["L"] + 1 <= 6:
                used.add(j["L"] + 1)
    for L in used:
        external += int(BASELINE["stored_rfull"][str(L)]["nfev_sum"])
    q_new = sum(v.get("n_eval_new", 0) for v in qaoa.values() if isinstance(v, dict))
    q_floor = sum(BASELINE["qaoa_round4_samehash"][str(p)]["n_eval"] for p in range(4, 13))
    return {
        "marginal_this_run": marginal + q_new,
        "external_stored_rfull_nfev_consumed": external,
        "round4_qaoa_floor_nfev_p4_to_p12": q_floor,
        "campaign_total_with_external_and_floor": marginal + q_new + external + q_floor,
        "qaoa_inclusive_need": qaoa.get("_need"),
        "note": PROTOCOL["amortization"],
    }


def _reading(summary):
    lines = [
        "KS splits full K into cavity-intra Kin and cavity-inter Kout on the sublattice packing. "
        "G = Hc + thetaIn Kin/rIn + thetaOut Kout/rOut + thetaM Mhat, raw P = 3L+4. "
        "Kin + Kout = FK, the two supports are disjoint, and rIn^2 + rOut^2 = rK^2. "
        "The nested embedding thetaIn = thetaK rIn/rK, thetaOut = thetaK rOut/rK reproduces the stored Rfull state.",
        "A within-window or a both-metric cell is not a seal. Dominance needs both energy and fidelity of the same energy-selected state, past the frozen tolerances. "
        "The branch screen retires a new-candidate claim only when the energy gain versus stored Rfull is below 0.005 at both L5 and L6 and the coefficient distance is at most 0.001. "
        "A fresh-median gap above 0.03 is a robustness flag. Neither screen is an impossibility proof.",
    ]
    sc = summary["screens"]
    lines.append(
        f"Branch screen retire={sc['branch_screen_retire_new_candidate']}. Robust fresh flag={sc['robust_screen_flag']}. "
        + json.dumps(sc["best_improvement"])
    )
    return lines


def _md(path, summary):
    lines = ["# Round 5 KS — packing-resolved full K", ""]
    for line in summary["reading"]:
        lines += [line, ""]
    lines.append(f"Manifest `{summary['manifest_sha256']}`. Packing `{summary['packing_sha256']}`. Hc `{summary['hc_sha256']}`.")
    lines.append(f"E0 = {summary['E0']:.10f}. Full gap {summary['gap_full']:.6e}. Accessible gap {summary['gap_accessible']}.")
    lines.append(f"Optimizer time {summary['optimizer_seconds']:.0f} s. Wall {summary['wall_seconds']:.0f} s.")
    lines.append("")
    lines.append("| role | L | P | E | error | F | median | IQR | success | failed | evals | distance |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in sorted(summary["jobs"], key=lambda r: (r["role"], r["L"])):
        dist = j.get("coefficient_distance", {}).get("normalized_distance")
        dist_s = f"{dist:.3e}" if dist is not None else ""
        lines.append(
            f"| {j['role']} | {j['L']} | {j['P']} | {j['E']:.5f} | {j['energy_error']:.4e} | {j['F']:.4f} | "
            f"{j['seed_median']:.4f} | {j['seed_iqr']:.3f} | {j['n_success']} | {j['n_failed']} | {j['n_eval']} | {dist_s} |"
        )
    lines.append("")
    lines.append("## Both-metric cells")
    lines.append("")
    for row in summary["comparisons"]:
        if "both_matched" not in row:
            continue
        lines.append(
            f"- {row['role']} L{row['L']} P={row['P']}: matched p={row.get('matched_p')} both {row.get('both_matched')} "
            f"dE {row.get('dE_matched')}, next both {row.get('both_next')} dE {row.get('dE_next')}, "
            f"p12 both {row.get('both_p12')} dE {row.get('dE_p12')}, stored dE {row.get('dE_stored_rfull')}"
        )
    lines.append("")
    if summary["failures"]:
        lines.append("## Incomplete")
        for item in summary["failures"]:
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n")


def _plots(summary, folder: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jobs = {(j["role"], j["L"]): j for j in summary["jobs"]}
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for role in ("KS_fresh", "KS_nested", "KS_transfer", "Rfull_fresh"):
        xs, ys = [], []
        for L in range(1, 7):
            hit = jobs.get((role, L))
            if hit:
                xs.append(L)
                ys.append(hit["energy_error"])
        if xs:
            ax.plot(xs, ys, marker="o", label=role)
    for p, row in summary["qaoa"].items():
        if isinstance(row, dict) and "energy_error" in row:
            ax.scatter([p], [row["energy_error"]], color="purple", s=18)
    ax.scatter([], [], color="purple", s=18, label="qaoa")
    ax.set_xlabel("depth")
    ax.set_ylabel("energy error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "error_vs_depth.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for role, L in (("KS_fresh", 6), ("KS_nested", 6), ("KS_transfer", 6), ("Rfull_fresh", 6)):
        hit = jobs.get((role, L))
        if hit and hit.get("best_so_far"):
            e0 = hit["E"] - hit["energy_error"]
            ys = [v - e0 for v in hit["best_so_far"]]
            ax.plot(np.linspace(1, hit["n_eval"], len(ys)), ys, label=f"{role} L{L}")
    ax.set_ylim(0.2, 1.6)
    ax.set_xlabel("inclusive evaluations in that cell")
    ax.set_ylabel("best energy error so far")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "best_so_far.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for role in ("KS_fresh", "KS_nested", "KS_transfer", "Rfull_fresh"):
        xs, ys = [], []
        for L in range(1, 7):
            hit = jobs.get((role, L))
            if hit:
                xs.append(hit["energy_error"])
                ys.append(hit["F"])
        if xs:
            ax.scatter(xs, ys, label=role)
    for p, row in summary["qaoa"].items():
        if isinstance(row, dict) and "energy_error" in row:
            ax.scatter([row["energy_error"]], [row["F"]], color="purple", s=18)
    ax.scatter([], [], color="purple", s=18, label="qaoa")
    ax.set_xlabel("energy error")
    ax.set_ylabel("fidelity of the energy-best state")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "pareto_error_fidelity.png", dpi=140)
    plt.close(fig)
    hit = jobs.get(("KS_fresh", 6)) or jobs.get(("KS_nested", 6))
    if hit:
        e0 = hit["E"] - hit["energy_error"]
        err = sorted(r["E"] - e0 for r in hit["starts"] if "E" in r)
        fig, ax = plt.subplots(figsize=(6.2, 4))
        ys = np.arange(1, len(err) + 1) / len(err)
        ax.step(err, ys, where="post")
        ax.set_xlabel("start energy error")
        ax.set_ylabel("CDF")
        ax.set_title(f"{hit['role']} L6, every finished start")
        fig.tight_layout()
        fig.savefig(folder / "ks_L6_cdf.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    run(75.0)
