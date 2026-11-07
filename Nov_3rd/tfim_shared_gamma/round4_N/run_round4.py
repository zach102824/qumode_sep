"""Round-4 N reductions. Energy only. 75 minute bound.

Priority is R_M, then N_red, then R_blockK. R_full is the fresh baseline
those reductions are compared with. A passing reduction is not an advantage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.common import canonical_state, log, sha256_state

from .model import (
    MANIFEST_SHA,
    PACKING_SHA,
    Runner,
    System,
    fwht,
    logical_perm,
    pack,
    shell_weights,
    unpack,
)
from .optimize import _resample, _unwrap, optimize

RESULTS = Path(__file__).resolve().parent / "results"
SRC = "2d81d6e6"
REPO = Path(__file__).resolve().parents[3]
OURS_N = Path(__file__).resolve().parents[1] / "round3_N" / "results" / "jobs"


def spectrum(system: System):
    n = system.n
    dim = 1 << n
    H = np.diag(system.hc.copy())
    idx = np.arange(dim)
    for bit in range(n):
        partner = idx ^ (1 << bit)
        mask = partner > idx
        a, b = idx[mask], partner[mask]
        H[a, b] = system.B
        H[b, a] = system.B
    evals, vecs = np.linalg.eigh(H)
    gs = np.real(vecs[:, 0])
    gs /= np.linalg.norm(gs)
    if gs.sum() < 0:
        gs = -gs
    return gs, float(evals[0]), float(evals[-1]), float(evals[1] - evals[0]), evals, vecs


def parity_reflection(system, vec):
    n = system.n
    x = np.arange(1 << n)
    flip = x ^ ((1 << n) - 1)
    y = np.zeros_like(x)
    for i in range(n):
        y |= ((x >> i) & 1) << (n - 1 - i)
    v = np.real(vec)
    return float(np.dot(v, v[flip])), float(np.dot(v, v[y]))


def _git_bytes(rel: str) -> bytes:
    import subprocess
    return subprocess.check_output(["git", "show", f"{SRC}:{rel}"], cwd=str(REPO))


def _replay_source(runner: Runner) -> dict:
    """Replay stored angles with this simulator. The source code is not imported."""
    out = {}
    arms = {"A_sub_M": "R_full", "A_sub": "A_sub", "A_cont": "A_cont", "qaoa": "qaoa"}
    for L in range(1, 7):
        for name, arm in arms.items():
            rel = f"Nov_3rd/tfim_shared_gamma/round3_R/results/cells/N12_L{L}_{name}.json"
            blob = json.loads(_git_bytes(rel))
            if arm == "qaoa":
                x = np.concatenate([blob["angles"]["gammas"], blob["angles"]["betas"]])
            else:
                x = np.array(blob["x"], dtype=np.float64)
            E, _, psi = runner.energy_grad(arm, x, L)
            F = runner.fidelity(psi)
            row = {
                "source_E": blob["E"], "source_F": blob["F"], "source_P": blob["n_params"],
                "replay_E": float(E), "replay_F": F,
                "dE": float(E - blob["E"]), "dF": float(F - blob["F"]),
                "hc_match": blob["hc_sha256"] == runner.s.hc_sha,
                "packing_match": blob.get("packing_sha256") == PACKING_SHA,
            }
            if name == "A_sub_M" and L in (5, 6):
                state_rel = f"Nov_3rd/tfim_shared_gamma/round3_R/results/states/N12_L{L}_{name}.npz"
                tmp = Path(f"/tmp/src_{name}_L{L}.npz")
                tmp.write_bytes(_git_bytes(state_rel))
                src = np.load(tmp)["psi"]
                row["overlap"] = float(abs(np.vdot(src, psi)) ** 2)
            out[f"{name}_L{L}"] = row
    bad = [k for k, row in out.items() if abs(row["dE"]) > 1e-9 or row.get("overlap", 1.0) < 1 - 1e-12]
    if bad:
        raise RuntimeError(f"source replay failed { {k: out[k] for k in bad} }")
    return out


def checks(runner: Runner) -> dict:
    s = runner.s
    out = {"model": s.checks}
    # Physical formulation versus logical formulation.
    pi = [q for cav in s.packings["sublattice"] for q in cav]
    perm = logical_perm(pi, s.n)
    rng = np.random.default_rng(0)
    theta = {"K": 0.3, "M": -0.4}
    cs = rng.normal(scale=0.3, size=(2, 3))
    x = np.concatenate([[0.35, theta["K"], theta["M"]], cs.reshape(-1)])
    psi_l, _, _ = runner.forward("R_full", x, 2)
    # Physical diagonal is the logical diagonal pulled back by perm.
    g_log = s.g_diag("R_full", theta)
    g_phys = np.empty_like(g_log)
    g_phys[perm] = g_log
    w1, w2, w3 = shell_weights(s.n, [[3 * a + r for r in range(3)] for a in range(s.n // 3)])
    psi = runner.plus.copy()
    for ell in range(2):
        psi = psi * np.exp(-1j * 0.35 * g_phys)
        phi = cs[ell, 0] * w1 + cs[ell, 1] * w2 + cs[ell, 2] * w3
        psi = fwht(fwht(psi) * np.exp(-1j * phi))
    psi_back = psi[perm]
    err = float(np.max(np.abs(psi_back - psi_l)))
    out["physical_match_maxabs"] = err
    # pi shifts are global phases.
    x1 = np.array([0.2, 0.3, 0.4, -0.2, 0.15], dtype=np.float64)
    base, _, _ = runner.forward("R_M", x1, 1)
    phase_ok = True
    for j in range(3):
        y = x1.copy()
        y[2 + j] += np.pi
        other, _, _ = runner.forward("R_M", y, 1)
        if abs(np.vdot(base, other)) ** 2 < 1 - 1e-10:
            phase_ok = False
    out["shell_pi_is_global_phase"] = phase_ok
    # Conjugation. Theta stays put. State becomes the conjugate, not the same vector.
    y = x1.copy()
    y[0] *= -1
    y[2:] *= -1
    conj, _, _ = runner.forward("R_M", y, 1)
    ov_conj = float(abs(np.vdot(conj, np.conj(base))) ** 2)
    ov_same = float(abs(np.vdot(conj, base)) ** 2)
    out["conjugation_overlap_with_conjugate"] = ov_conj
    out["conjugation_overlap_with_same_state"] = ov_same
    out["passed"] = bool(s.checks["passed"] and err < 1e-9 and phase_ok and ov_conj > 1 - 1e-10)
    return out


def _save(rec, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"n{rec['n']}_L{rec['L']}_{rec['arm']}.json"
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    if "psi" in rec:
        np.save(folder.parent / "states" / f"n{rec['n']}_L{rec['L']}_{rec['arm']}.npy", canonical_state(rec["psi"]))


def _public(runner, raw):
    x = raw["x"]
    arm, L = raw["arm"], raw["L"]
    E, _, psi = runner.energy_grad(arm, x, L)
    energies = np.array([r["E"] for r in raw["starts"] if "E" in r], dtype=np.float64)
    fs = np.array([r["F"] for r in raw["starts"] if "F" in r], dtype=np.float64)
    order = np.argsort(energies)
    cdf = [{"eval_order": i + 1, "E": float(energies[k]), "F": float(fs[k])} for i, k in enumerate(order)]
    rec = {
        "arm": arm, "n": raw["n"], "L": L, "P": int(x.size),
        "E": float(E), "energy_error": float(E - runner.E0), "F": runner.fidelity(psi),
        "variance_H": runner.variance(psi),
        "seed_median": raw["seed_median"], "seed_iqr": raw["seed_iqr"], "seed_best": raw["seed_best"],
        "n_eval": raw["n_eval"], "n_starts": raw["n_starts"], "n_success": raw["n_success"], "n_failed": raw["n_failed"],
        "seconds": raw["seconds"], "init": raw["init"], "best_so_far": raw["best_so_far"],
        "gamma": raw["gamma"], "theta": raw["theta"], "shell": raw["shell"],
        "x_exact": [format(float(v), ".17g") for v in x],
        "state_sha256": sha256_state(psi),
        "manifest_sha256": MANIFEST_SHA, "packing_sha256": PACKING_SHA,
        "packing": {"R_M": "sublattice", "R_full": "sublattice", "R_blockK": "sublattice", "N_red": "contiguous", "A_cont": "contiguous", "A_sub": "sublattice", "qaoa": "none"}[arm],
        "best_F_not_selected": float(fs.max()) if fs.size else None,
        "starts": raw["starts"], "trace": raw["trace"], "cdf_by_energy": cdf,
        "objective": "energy", "fidelity_used_for_selection": False,
    }
    rec["psi"] = psi
    return rec


def _as_prev(arm, rec, L):
    """A vector of length L-1. A skipped rung is resampled, not dropped."""
    if rec["L"] == L - 1:
        return rec["x"]
    gammas, theta, cs = unpack(arm, rec["x"], rec["L"])
    cs = _unwrap(cs)
    mid = L - 1
    kind = "linear" if mid <= 3 else "cubic"
    c_new = np.column_stack([_resample(cs[:, j], mid, kind) for j in range(3)])
    if arm == "qaoa":
        return pack(arm, 0.0, {}, c_new, _resample(gammas, mid, kind), _resample(cs[:, 0], mid, kind))
    return pack(arm, float(gammas[0]), theta, c_new)


def _absorb(arm, L, raw, mem, runner):
    fresh = _public(runner, raw)
    old = mem[(arm, L)]
    winner = fresh if fresh["E"] <= old["E"] else old
    starts = list(old["starts"]) + list(fresh["starts"])
    energies = np.array([r["E"] for r in starts if "E" in r], dtype=np.float64)
    fs = np.array([r["F"] for r in starts if "F" in r], dtype=np.float64)
    q25, q75 = np.percentile(energies, [25, 75])
    merged = {k: v for k, v in winner.items()}
    order = np.argsort(energies)
    merged.update({
        "starts": starts,
        "n_eval": int(old["n_eval"] + fresh["n_eval"]),
        "n_starts": int(energies.size),
        "n_success": int(old["n_success"] + fresh["n_success"]),
        "n_failed": int(old["n_failed"] + fresh["n_failed"]),
        "seconds": float(old["seconds"] + fresh["seconds"]),
        "seed_median": float(np.median(energies)),
        "seed_iqr": float(q75 - q25),
        "seed_best": float(energies.min()),
        "best_so_far": list(old.get("best_so_far") or []) + list(fresh.get("best_so_far") or []),
        "cdf_by_energy": [{"eval_order": i + 1, "E": float(energies[k]), "F": float(fs[k])} for i, k in enumerate(order)],
        "best_F_not_selected": float(fs.max()) if fs.size else None,
    })
    _save({k: v for k, v in merged.items() if k != "psi"}, RESULTS / "jobs")
    mem[(arm, L)] = merged


def run(budget_min: float = 75.0):
    t0 = time.perf_counter()
    deadline = t0 + budget_min * 60.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "states").mkdir(exist_ok=True)
    (RESULTS / "jobs").mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    failures = []
    log("spectrum and checks")
    systems = {n: System(n) for n in (6, 12)}
    runners = {}
    for n, sys in systems.items():
        gs, E0, Emax, gap, evals, vecs = spectrum(sys)
        p, r = parity_reflection(sys, gs)
        # accessible gap: next state with the same parity and reflection signs
        p1, r1 = parity_reflection(sys, vecs[:, 1])
        gap_acc = None
        for i in range(1, min(12, len(evals))):
            pi, ri = parity_reflection(sys, vecs[:, i])
            if pi > 0.5 and ri > 0.5:
                gap_acc = float(evals[i] - evals[0])
                break
        runr = Runner(sys)
        runr.set_spectrum(gs, E0, Emax, gap, gap_acc if gap_acc is not None else gap)
        runners[n] = runr
        log(f"N={n} E0={E0:.6f} full_gap={gap:.6e} acc={gap_acc} P={p:.3f} R={r:.3f} partnerP={p1:.3f}")
    report_checks = checks(runners[12])
    log(f"checks passed={report_checks['passed']} phys={report_checks['physical_match_maxabs']:.2e}")
    if not report_checks["passed"]:
        raise RuntimeError(f"physics checks failed {report_checks}")

    # Replay the stored sublattice+M angles with this simulator. Data only.
    replay = _replay_source(runners[12])
    for key, row in replay.items():
        if row.get("overlap") is not None:
            log(f"replay {key} dE={row['dE']:.3e} overlap={row['overlap']:.12f}")

    def left():
        return deadline - time.perf_counter()

    mem = {}

    def fit(arm, L, n_fresh, cap, tag):
        key = (arm, L)
        path = RESULTS / "jobs" / f"n12_L{L}_{arm}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            if rec.get("manifest_sha256") == MANIFEST_SHA:
                rec["x"] = np.array([float(v) for v in rec["x_exact"]])
                mem[key] = rec
                log(f"resume {arm} L{L} E={rec['E']:.6f}")
                return rec
        if left() < 40:
            failures.append(f"skipped {arm} L={L}")
            return None
        prev = None
        for ell in range(L - 1, 0, -1):
            if (arm, ell) in mem:
                prev = mem[(arm, ell)]
                break
        prev_x = None if prev is None else _as_prev(arm, prev, L)
        log(f"fit {arm} L={L} fresh={n_fresh} cap={cap} left={left()/60:.1f}min")
        raw = optimize(runners[12], arm, L, prev_x, n_fresh, tag, cap)
        rec = _public(runners[12], raw)
        _save({k: v for k, v in rec.items() if k != "psi"}, RESULTS / "jobs")
        mem[key] = rec
        log(f"  E={rec['E']:.6f} err={rec['energy_error']:.4e} F={rec['F']:.4f} med={rec['seed_median']:.4f} iqr={rec['seed_iqr']:.3f} evals={rec['n_eval']}")
        return rec

    # Priority: R_M, then N_red, then block-K, then fresh R_full.
    for L in range(1, 7):
        fit("R_M", L, 12, 7000, 1)
    for L, fresh, cap in ((1, 4, 1400), (2, 4, 1600), (3, 4, 1800), (4, 12, 7000), (5, 12, 7000), (6, 12, 7000)):
        if left() < 50 and L >= 4:
            failures.append(f"skipped N_red L={L}")
            continue
        if left() < 30:
            break
        fit("N_red", L, fresh, cap, 2)
    for L, fresh, cap in ((1, 4, 1200), (2, 8, 5000), (3, 4, 1800), (4, 8, 5000), (5, 4, 1800), (6, 8, 5000)):
        if left() < 40:
            failures.append(f"skipped R_blockK L={L}")
            continue
        fit("R_blockK", L, fresh, cap, 3)
    for L in range(1, 7):
        if left() < 40:
            failures.append(f"skipped R_full L={L}")
            continue
        fit("R_full", L, 8, 5000, 4)

    # QAOA through p=12. Inclusive budget at least the deepest finished candidate at that depth.
    def need_at(L):
        vals = [mem[(arm, min(L, 6))]["n_eval"] for arm in ("R_M", "N_red", "R_blockK", "R_full") if (arm, min(L, 6)) in mem]
        return max(vals) if vals else 2000

    for L in range(1, 13):
        cap = max(2000, need_at(L))
        if left() < 35:
            failures.append(f"skipped qaoa L={L}")
            continue
        fit("qaoa", L, 12, cap, 5)
        guard = 0
        while ("qaoa", L) in mem and mem[("qaoa", L)]["n_eval"] < need_at(L) and left() > 40 and guard < 6:
            guard += 1
            short = need_at(L) - mem[("qaoa", L)]["n_eval"]
            n_more = max(4, short // 80)
            log(f"qaoa L={L} top-up {n_more} starts, short {short}")
            raw = optimize(runners[12], "qaoa", L, None, n_more, 50 + L + guard, max(200, short + 30), screen=False,
                           extra=[("incumbent", mem[("qaoa", L)]["x"])])
            _absorb("qaoa", L, raw, mem, runners[12])

    smoke = None
    if left() > 90 and ("R_M", 1) in mem:
        log("N6 smoke L1 R_M")
        raw = optimize(runners[6], "R_M", 1, None, 4, 9, 800, screen=True)
        smoke = {k: v for k, v in _public(runners[6], raw).items() if k != "psi"}
        _save(smoke, RESULTS / "jobs")

    public_jobs = []
    for rec in mem.values():
        public_jobs.append({k: v for k, v in rec.items() if k != "psi"})
    summary = {
        "manifest_sha256": MANIFEST_SHA,
        "packing_sha256": PACKING_SHA,
        "checks": report_checks,
        "source_replay": replay,
        "jobs": public_jobs,
        "failures": failures,
        "comparisons": {arm: _compare(public_jobs, arm) for arm in ("R_M", "N_red", "R_blockK", "R_full")},
        "reduction": _reduction(public_jobs, replay),
        "smoke_n6": smoke,
        "wall_seconds": time.perf_counter() - t0,
        "protocol": json.loads((Path(__file__).resolve().parent / "protocol.json").read_text()),
    }
    summary["reading"] = _reading(summary)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _md(RESULTS / "RESULTS.md", summary)
    figdir = RESULTS / "figures"
    _plots(summary, figdir)
    art = Path("/opt/cursor/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    for png in figdir.glob("*.png"):
        (art / f"round4_N_{png.name}").write_bytes(png.read_bytes())
    log(f"done {summary['wall_seconds']:.1f}s failures={len(failures)}")
    return summary


def _by(jobs):
    return {(j["arm"], j["L"]): j for j in jobs}


def _compare(jobs, arm):
    ix = _by(jobs)
    rows = []
    depths = sorted(L for (a, L) in ix if a == arm)
    for L in depths:
        m = ix.get((arm, L))
        if m is None:
            continue
        P = m["P"]
        matched = int(np.ceil(P / 2))
        def pack(other, name):
            if other is None:
                return {}
            return {f"E_{name}": other["E"], f"F_{name}": other["F"], f"err_{name}": other["energy_error"],
                    f"med_{name}": other["seed_median"], f"dE_{name}": m["E"] - other["E"], f"dF_{name}": m["F"] - other["F"]}
        q, q1, qm, qn = ix.get(("qaoa", L)), ix.get(("qaoa", L + 1)), ix.get(("qaoa", matched)), ix.get(("qaoa", matched + 1))
        q12 = ix.get(("qaoa", 12))
        rows.append({
            "L": L, "P": P, "matched_p": matched,
            "E": m["E"], "F": m["F"], "err": m["energy_error"], "median": m["seed_median"], "iqr": m["seed_iqr"],
            "both_L": _both(m, q), "both_L1": _both(m, q1), "both_matched": _both(m, qm), "both_next": _both(m, qn), "both_p12": _both(m, q12),
            **pack(q, "q"), **pack(q1, "q1"), **pack(qm, "qm"), **pack(qn, "qn"), **pack(q12, "q12"),
        })
    return rows


def _both(a, b):
    if a is None or b is None:
        return None
    return bool(a["E"] < b["E"] - 1e-8 and a["F"] > b["F"] + 1e-6)


def _reduction(jobs, replay):
    ix = _by(jobs)
    # Original contiguous N from round 3, hash-checked.
    original = {}
    for L in (4, 5, 6):
        path = OURS_N / f"n12_L{L}_family_N.json"
        if path.exists():
            d = json.loads(path.read_text())
            if d.get("manifest_sha256") == MANIFEST_SHA:
                original[L] = {"E": d["E"], "F": d["F"], "energy_error": d["energy_error"], "P": d["param_count"]}
    out = {"original_N": original, "cells": []}
    for L in (5, 6):
        a, b = ix.get(("R_M", L)), ix.get(("R_full", L))
        if a and b:
            out["cells"].append({
                "pair": "R_M_vs_R_full", "L": L,
                "dE_best": a["E"] - b["E"], "d_median": a["seed_median"] - b["seed_median"],
                "dF": a["F"] - b["F"],
                "within_0.02_best": abs(a["E"] - b["E"]) <= 0.02,
                "median_within_0.05": abs(a["seed_median"] - b["seed_median"]) <= 0.05,
                "worse_by_more_than_0.03": a["E"] > b["E"] + 0.03,
                "pass_fail_baseline": "fresh R_full",
            })
        src = replay.get(f"A_sub_M_L{L}")
        if a and src:
            out["cells"].append({
                "pair": "R_M_vs_source_R_full_replay", "L": L,
                "dE_best": a["E"] - src["replay_E"], "dF": a["F"] - src["replay_F"],
                "note": "Labeled comparison to the replayed source state. Not the fresh-fit pass/fail cell.",
            })
    for L in (4, 6):
        a, b = ix.get(("R_blockK", L)), ix.get(("R_full", L))
        if a and b:
            out["cells"].append({
                "pair": "blockK_vs_R_full", "L": L,
                "dE_best": a["E"] - b["E"], "d_median": a["seed_median"] - b["seed_median"],
                "dF": a["F"] - b["F"],
                "within_0.01_best": abs(a["E"] - b["E"]) <= 0.01,
                "median_no_worse": a["seed_median"] <= b["seed_median"] + 1e-9,
                "pass_fail_baseline": "fresh R_full",
            })
    for L in (4, 5, 6):
        a = ix.get(("N_red", L))
        b = original.get(L)
        if a and b:
            out["cells"].append({
                "pair": "N_red_vs_original_N", "L": L,
                "dE_best": a["E"] - b["E"], "dF": a["F"] - b["F"],
                "within_0.01_best": abs(a["E"] - b["E"]) <= 0.01,
                "note": "supports a parameter reduction only if within 0.01; not an advantage",
            })
    return out


def _reading(summary):
    lines = [
        "R_M is the main new fit: sublattice shells, G = Hc + thetaM Mhat, counted P = 3L+2. "
        "R_full, block-K, and contiguous N_red are the pre-registered comparisons. "
        "Passing a within-threshold is a reduction statement, not an advantage and not a seal.",
        "M is exactly the block-mean projection of Hinter on the sublattice packing. "
        "Full K is not in the span of J and M, and the block-mean of raw K is not raw K. "
        "The Hc coefficient stays 1, so those pieces are not a scale gauge.",
        "A shift of any one shell coefficient by π is a global phase. "
        "Sending gamma and the shell to their negatives conjugates the state. "
        "That keeps E and F and does not make the state the same vector. Gamma and theta are not wrapped.",
    ]
    cells = summary["reduction"]["cells"]
    if not cells:
        lines.append("Reduction cells were not all finished. See failures.")
    else:
        lines.append("Reduction cells: " + json.dumps(cells))
    return lines


def _md(path, summary):
    lines = ["# Round 4 N — pre-registered reductions", ""]
    for line in summary["reading"]:
        lines += [line, ""]
    lines.append(f"Manifest `{summary['manifest_sha256']}`. Packing `{summary['packing_sha256']}`.")
    lines.append("")
    lines.append("## Replay of stored source angles")
    lines.append("")
    lines.append("These rows reuse the stored angles in this simulator. They are a baseline check, not a new fit.")
    lines.append("")
    for key, row in summary["source_replay"].items():
        extra = f", overlap {row['overlap']:.12f}" if "overlap" in row else ""
        lines.append(f"- {key}: dE {row['dE']:.3e}, dF {row['dF']:.3e}{extra}")
    lines.append("")
    lines.append("| arm | L | P | E | error | F | best F not used | var | median | IQR | success | failed | evals |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in sorted(summary["jobs"], key=lambda r: (r["arm"], r["L"])):
        lines.append(
            f"| {j['arm']} | {j['L']} | {j['P']} | {j['E']:.5f} | {j['energy_error']:.4e} | {j['F']:.4f} | "
            f"{j['best_F_not_selected']:.4f} | {j['variance_H']:.3e} | {j['seed_median']:.4f} | {j['seed_iqr']:.3f} | "
            f"{j['n_success']} | {j['n_failed']} | {j['n_eval']} |"
        )
    lines.append("")
    lines.append("## Equal depth, next depth, matched depth, and p=12")
    lines.append("")
    lines.append("Matched p is ceil(P/2). That is a rounded parameter count, not an exact match and not an exhaustive optimum. A both-metric cell is energy and fidelity together on the energy-selected states.")
    lines.append("")
    for arm, rows in summary["comparisons"].items():
        for row in rows:
            lines.append(
                f"- {arm} L{row['L']} P={row['P']} matched p={row['matched_p']}: "
                f"both vs L {row['both_L']}, L+1 {row['both_L1']}, matched {row['both_matched']}, "
                f"next {row['both_next']}, p12 {row['both_p12']}"
            )
    lines.append("")
    if summary.get("smoke_n6"):
        s = summary["smoke_n6"]
        lines.append(f"N=6 smoke only: R_M L1 E={s['E']:.5f} F={s['F']:.4f}. Not a claim.")
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
    jobs = {(j["arm"], j["L"]): j for j in summary["jobs"]}
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ("R_M", "R_full", "N_red", "R_blockK", "qaoa"):
        xs, ys = [], []
        for L in range(1, 13):
            hit = jobs.get((arm, L))
            if hit:
                xs.append(L)
                ys.append(hit["energy_error"])
        if xs:
            ax.plot(xs, ys, marker="o", label=arm)
    ax.set_xlabel("depth")
    ax.set_ylabel("energy error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "error_vs_depth.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, L in (("R_M", 6), ("N_red", 6), ("R_blockK", 6), ("R_full", 6), ("qaoa", 12)):
        hit = jobs.get((arm, L))
        if hit and hit.get("best_so_far"):
            e0 = hit["E"] - hit["energy_error"]
            ys = [v - e0 for v in hit["best_so_far"]]
            ax.plot(np.linspace(1, hit["n_eval"], len(ys)), ys, label=f"{arm} L{L}")
    ax.set_xlabel("inclusive evaluations")
    ax.set_ylabel("best energy error so far")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "best_so_far.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ("R_M", "R_full", "N_red", "R_blockK", "qaoa"):
        xs, ys = [], []
        for L in range(1, 13):
            hit = jobs.get((arm, L))
            if hit:
                xs.append(hit["energy_error"])
                ys.append(hit["F"])
        if xs:
            ax.scatter(xs, ys, label=arm)
    ax.set_xlabel("energy error")
    ax.set_ylabel("fidelity of the energy-best state")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "pareto_error_fidelity.png", dpi=140)
    plt.close(fig)
    hit = jobs.get(("R_M", 6))
    if hit:
        es = sorted(r["energy_error"] if "energy_error" in r else (r["E"] - (hit["E"] - hit["energy_error"])) for r in hit["starts"] if "E" in r)
        # starts store E not error
        e0 = hit["E"] - hit["energy_error"]
        err = sorted(r["E"] - e0 for r in hit["starts"] if "E" in r)
        fig, ax = plt.subplots(figsize=(6, 4))
        ys = np.arange(1, len(err) + 1) / len(err)
        ax.step(err, ys, where="post")
        ax.set_xlabel("start energy error")
        ax.set_ylabel("CDF")
        ax.set_title("R_M L6 starts, including failures")
        fig.tight_layout()
        fig.savefig(folder / "rm_L6_cdf.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    run(75.0)
