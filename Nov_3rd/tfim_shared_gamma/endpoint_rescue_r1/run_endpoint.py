"""Paired endpoint-versus-center QAOA seeds on the frozen N=12 target.

The preregistration is PREREG.json. Optimizer evaluations stop at 300000 or
at the frozen wall. Diagnostics are not part of that count.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from Nov_3rd.tfim_shared_gamma.round1_A.common import canonical_state, sha256_state
from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum, reflection_perm
from Nov_3rd.tfim_shared_gamma.round5_KS.model import Runner, Target

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREREG_SHA = "52cdefec636d7c47fae7e047c69231d222debb64a0bb0e39fa28895c0177bc16"
PREREG = json.loads((HERE / "PREREG.json").read_text())
GAMMA = tuple(PREREG["optimizer"]["gamma_bounds"])
BETA = tuple(PREREG["optimizer"]["beta_bounds"])
SIGMA = float(PREREG["seeds"]["noise_sigma"])
START_CAP = int(PREREG["caps"]["evals_per_start"])
GLOBAL_CAP = int(PREREG["caps"]["global_optimizer_evaluations"])
WALL = float(PREREG["caps"]["optimizer_wall_seconds"])
TAU = float(PREREG["tau"])


class StopSearch(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _check_prereg():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise RuntimeError(f"PREREG hash {digest} != {PREREG_SHA}")


def _resample(values, p_new: int, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    p_old = int(values.size)
    if mode == "center":
        xs = (np.arange(p_old) + 0.5) / p_old
        xt = (np.arange(p_new) + 0.5) / p_new
    elif mode == "endpoint":
        xs = np.arange(p_old, dtype=np.float64) / (p_old - 1)
        xt = np.arange(p_new, dtype=np.float64) / (p_new - 1)
    else:
        raise KeyError(mode)
    if p_old < 3:
        return np.interp(xt, xs, values).astype(np.float64)
    spline = CubicSpline(xs, values, bc_type="natural")
    return np.asarray(spline(np.clip(xt, float(xs[0]), float(xs[-1]))), dtype=np.float64)


def _clip(x, p):
    y = np.asarray(x, dtype=np.float64).copy()
    for i in range(p):
        y[i] = min(max(float(y[i]), GAMMA[0]), GAMMA[1])
    for i in range(p, 2 * p):
        y[i] = min(max(float(y[i]), BETA[0]), BETA[1])
    return y


def _noise(pair, depth, k, n):
    rng = np.random.default_rng(np.random.SeedSequence([20261114, int(pair), int(depth), 9, int(k)]))
    return rng.normal(scale=SIGMA, size=n)


def _spectrum(target: Target):
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
    if float(gs.sum()) < 0.0:
        gs = -gs
    flip = idx ^ (dim - 1)
    refl = reflection_perm(n)
    plus = []
    for k in range(dim):
        v = vecs[:, k]
        par = float(np.vdot(v, v[flip]).real)
        ref = float(np.vdot(v, v[refl]).real)
        if par > 0.5 and ref > 0.5:
            plus.append(float(evals[k]))
        if len(plus) >= 2:
            break
    return gs, float(evals[0]), float(evals[-1]), plus


def _observables(runner, psi):
    E, Hpsi = runner.energy(psi)
    var = float(np.vdot(Hpsi, Hpsi).real - E * E)
    F = float(runner.fidelity(psi))
    eta = (E - runner.Emax) / (runner.E0 - runner.Emax)
    return float(E), F, var, float(eta)


def _replay(runner):
    blob = json.loads((ROOT / "round4_N" / "results" / "jobs" / "n12_L12_qaoa.json").read_text())
    x = np.array([float(v) for v in blob["x_exact"]], dtype=np.float64)
    _E, _grad, psi = runner.energy_grad("qaoa", x, 12)
    E, F, var, eta = _observables(runner, psi)
    want = PREREG["replay_must_match"]
    tol = float(want["abs_tol"])
    residual = {
        "E": E - float(want["E"]),
        "F": F - float(want["F"]),
        "var_H": var - float(want["var_H"]),
        "eta": eta - float(want["eta"]),
    }
    ok = all(abs(v) <= tol for v in residual.values())
    report = {
        "ok": ok,
        "E": E,
        "F": F,
        "var_H": var,
        "eta": eta,
        "norm": float(np.linalg.norm(psi)),
        "state_sha256": sha256_state(psi),
        "residual_vs_audit": residual,
        "abs_tol": tol,
        "source": "round4_N/results/jobs/n12_L12_qaoa.json x_exact",
        "diagnostic_qaoa_forward": 1,
        "counted_in_optimizer_budget": False,
    }
    (HERE / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    return ok, report


class Budget:
    def __init__(self, n, deadline):
        self.left = int(n)
        self.used = 0
        self.deadline = deadline
        self.shared = 0
        self.endpoint = 0
        self.center = 0

    def charge(self, bucket):
        if time.perf_counter() >= self.deadline:
            raise StopSearch("wall")
        if self.left <= 0:
            raise StopSearch("global_cap")
        self.left -= 1
        self.used += 1
        if bucket == "shared":
            self.shared += 1
        elif bucket == "endpoint":
            self.endpoint += 1
        elif bucket == "center":
            self.center += 1
        else:
            raise KeyError(bucket)


def _one_start(runner, p, x0, budget, bucket, start_cap):
    x0 = _clip(x0, p)
    local_E, local_x, local_psi = float("inf"), x0.copy(), None
    cache = {}
    used = 0
    reason = "optimizer_return"
    fun_calls = 0
    jac_calls = 0

    def take():
        nonlocal used
        if used >= start_cap:
            raise StopSearch("start_ceiling")
        budget.charge(bucket)
        used += 1

    def record(x, E, grad, psi):
        nonlocal local_E, local_x, local_psi
        cache["x"] = np.array(x, copy=True)
        cache["g"] = grad
        if E < local_E:
            local_E = float(E)
            local_x = np.array(x, copy=True)
            local_psi = psi

    def fun(x):
        nonlocal fun_calls
        take()
        fun_calls += 1
        E, grad, psi = runner.energy_grad("qaoa", x, p)
        record(x, E, grad, psi)
        return float(E)

    def jac(x):
        nonlocal jac_calls
        if cache.get("x") is not None and np.allclose(cache["x"], x, atol=1e-12, rtol=0.0):
            return cache["g"]
        take()
        jac_calls += 1
        E, grad, psi = runner.energy_grad("qaoa", x, p)
        record(x, E, grad, psi)
        return grad

    t0 = time.perf_counter()
    try:
        res = minimize(
            fun, x0, method="L-BFGS-B", jac=jac,
            bounds=[GAMMA] * p + [BETA] * p,
            options={"maxiter": 1500, "maxfun": start_cap, "ftol": 1e-14, "gtol": 1e-8},
        )
        success = bool(res.success)
    except StopSearch as exc:
        reason = exc.reason
        success = False
    partial = reason in ("wall", "global_cap")
    obs = None
    if local_psi is not None:
        E, F, var, eta = _observables(runner, local_psi)
        obs = {
            "E": E, "F": F, "var_H": var, "eta": eta,
            "state_sha256": sha256_state(local_psi),
            "norm": float(np.linalg.norm(local_psi)),
        }
    return {
        "E": None if local_psi is None else obs["E"],
        "F": None if obs is None else obs["F"],
        "var_H": None if obs is None else obs["var_H"],
        "eta": None if obs is None else obs["eta"],
        "state_sha256": None if obs is None else obs["state_sha256"],
        "norm": None if obs is None else obs["norm"],
        "x0": [float(v) for v in x0],
        "x_final": None if local_psi is None else [float(v) for v in local_x],
        "n_eval": int(used),
        "fun_calls": int(fun_calls),
        "jac_calls_uncached": int(jac_calls),
        "optimizer_success": success,
        "stop_reason": reason,
        "partial": bool(partial),
        "hit_start_ceiling": bool(reason == "start_ceiling"),
        "wall_seconds": float(time.perf_counter() - t0),
        "bucket": bucket,
    }


def _append(row):
    with (HERE / "chains.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _best_of(rows):
    finite = [r for r in rows if r.get("E") is not None]
    if not finite:
        return None
    return min(finite, key=lambda r: r["E"])


def _seeds_from(g, b, p_new, mode, pair):
    base = np.concatenate([_resample(g, p_new, mode), _resample(b, p_new, mode)])
    out = [(0, base)]
    for k in (1, 2):
        out.append((k, base + _noise(pair, p_new, k, base.size)))
    return out


def _run_depth_shared(runner, budget, pair, depth, seeds, label):
    rows = []
    for k, x0 in seeds:
        if budget.left <= 0 or time.perf_counter() >= budget.deadline:
            break
        row = _one_start(runner, depth, x0, budget, "shared", START_CAP)
        row.update({"pair": pair, "depth": depth, "arm": "shared", "k": k, "label": label, "seed_noise": [20261114, pair, depth, 9, k]})
        rows.append(row)
        _append(row)
        print(f"pair {pair} p{depth} shared k{k} E={row['E']} n={row['n_eval']} {row['stop_reason']}", flush=True)
        if row["partial"]:
            break
    return rows


def _compare(E, F, spec):
    dE = float(E - spec["E"])
    dF = float(F - spec["F"])
    retains = bool(spec["E"] < E - TAU and spec["F"] > F + TAU)
    overturns = bool(E < spec["E"] - TAU and F > spec["F"] + TAU)
    return {
        "dE_qaoa_minus_candidate": dE,
        "dF_qaoa_minus_candidate": dF,
        "candidate_retains_both": retains,
        "qaoa_overturns_both": overturns,
        "tradeoff": bool(not retains and not overturns),
    }


def _summarize(runner, pairs, budget, diag0):
    lines = (HERE / "chains.jsonl").read_text().splitlines() if (HERE / "chains.jsonl").exists() else []
    starts = [json.loads(line) for line in lines if line.strip()]
    by = {}
    for row in starts:
        by.setdefault((row["pair"], row["depth"], row["arm"]), []).append(row)
    selected = []
    for pair in pairs:
        for depth in range(1, 13):
            for arm in ("shared", "endpoint", "center"):
                rows = by.get((pair["pair"], depth, arm), [])
                best = _best_of(rows)
                if best is None or best.get("x_final") is None:
                    continue
                x = np.array(best["x_final"], dtype=np.float64)
                _E, _g, psi = runner.energy_grad("qaoa", x, depth)
                E, F, var, eta = _observables(runner, psi)
                selected.append({
                    "pair": pair["pair"],
                    "depth": depth,
                    "arm": arm,
                    "E": E,
                    "F": F,
                    "var_H": var,
                    "eta": eta,
                    "state_sha256": sha256_state(psi),
                    "logged_E": best["E"],
                    "logged_minus_diagnostic_E": float(best["E"] - E),
                    "from_partial_start": bool(best["partial"]),
                    "k": best["k"],
                    "n_eval_in_selected_start": best["n_eval"],
                    "diagnostic_qaoa_forward": 1,
                })
    hist = {int(k): v for k, v in PREREG["historical_qaoa_recorded_best"].items()}
    candidates = PREREG["candidates"]
    depths = {}
    for depth in (9, 10, 11, 12):
        pool = [s for s in selected if s["depth"] == depth and s["arm"] in ("endpoint", "center", "shared")]
        # shared p=3 is copied onto both arms below; p>=9 has no shared arm
        if not pool:
            depths[str(depth)] = {"status": "not_reached"}
            continue
        winner = min(pool, key=lambda s: s["E"])
        rec = hist[depth]
        comps = {name: _compare(winner["E"], winner["F"], spec) for name, spec in candidates.items()}
        arm_best = {}
        for arm in ("endpoint", "center"):
            arm_rows = [s for s in pool if s["arm"] == arm]
            if arm_rows:
                arm_best[arm] = min(arm_rows, key=lambda s: s["E"])
        depths[str(depth)] = {
            "strongest": winner,
            "improved_vs_recorded": bool(winner["E"] < float(rec["E"]) - TAU),
            "recorded_E": rec["E"],
            "recorded_F": rec["F"],
            "arm_best": arm_best,
            "comparisons": comps,
            "n_selected_records": len(pool),
        }
    complete = []
    partial_pairs = []
    for pair in pairs:
        pid = pair["pair"]
        ok = pair.get("complete_through_p12", False)
        (complete if ok else partial_pairs).append(pid)
    paired = {}
    for depth in (9, 10, 11, 12):
        deltas = []
        for pid in complete:
            e = next((s for s in selected if s["pair"] == pid and s["depth"] == depth and s["arm"] == "endpoint"), None)
            c = next((s for s in selected if s["pair"] == pid and s["depth"] == depth and s["arm"] == "center"), None)
            if e is None or c is None:
                continue
            deltas.append({
                "pair": pid,
                "E_endpoint": e["E"], "E_center": c["E"], "dE_endpoint_minus_center": e["E"] - c["E"],
                "F_endpoint": e["F"], "F_center": c["F"], "dF_endpoint_minus_center": e["F"] - c["F"],
                "eta_endpoint": e["eta"], "eta_center": c["eta"],
                "var_endpoint": e["var_H"], "var_center": c["var_H"],
                "state_endpoint": e["state_sha256"], "state_center": c["state_sha256"],
            })
        dE = np.array([d["dE_endpoint_minus_center"] for d in deltas], dtype=np.float64)
        paired[str(depth)] = {
            "n_complete_pairs": len(deltas),
            "pairs": deltas,
            "median_dE_endpoint_minus_center": None if dE.size == 0 else float(np.median(dE)),
            "n_endpoint_lower_E": int(np.sum(dE < -TAU)) if dE.size else 0,
            "n_center_lower_E": int(np.sum(dE > TAU)) if dE.size else 0,
            "n_E_within_tau": int(np.sum(np.abs(dE) <= TAU)) if dE.size else 0,
        }
    ledger = {
        "optimizer_evaluations_total": budget.used,
        "global_cap": GLOBAL_CAP,
        "shared_prefix_and_coincident_evaluations": budget.shared,
        "endpoint_suffix_evaluations": budget.endpoint,
        "center_suffix_evaluations": budget.center,
        "sum_of_buckets": budget.shared + budget.endpoint + budget.center,
        "per_arm_attribution_if_shared_added_to_each": {
            "endpoint": budget.shared + budget.endpoint,
            "center": budget.shared + budget.center,
            "note": "Adding these two numbers counts the shared work twice. The total above counts it once.",
        },
        "diagnostic_qaoa_forwards": diag0 + len(selected),
        "logged_n_eval_sum": int(sum(int(s["n_eval"]) for s in starts)),
        "n_starts": len(starts),
        "n_partial_starts": int(sum(1 for s in starts if s["partial"])),
        "n_failed_optimizer_success_false": int(sum(1 for s in starts if not s["optimizer_success"])),
        "complete_pairs": complete,
        "incomplete_pairs": partial_pairs,
        "wall_hit": bool(time.perf_counter() >= budget.deadline),
        "cap_hit": bool(budget.left <= 0),
    }
    ledger["logged_matches_budget"] = bool(ledger["logged_n_eval_sum"] == budget.used)
    summary = {
        "parent_audit_commit": PREREG["parent_audit_commit"],
        "prereg_sha256": PREREG_SHA,
        "E0": runner.E0,
        "Emax": runner.Emax,
        "tau": TAU,
        "ledger": ledger,
        "depths": depths,
        "paired": paired,
        "selected": selected,
        "pairs": pairs,
        "historical_budgets_unchanged": True,
        "paper_reproduction": False,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    _plots(starts, selected, complete)
    _report(summary)
    return summary


def _plots(starts, selected, complete):
    fig, ax = plt.subplots(figsize=(7, 4))
    curve_e, curve_c, curve_g = [], [], []
    best_e = best_c = best_g = np.inf
    used = 0
    for row in starts:
        used += int(row["n_eval"])
        if row["E"] is None:
            continue
        best_g = min(best_g, row["E"])
        if row["arm"] in ("endpoint", "shared"):
            best_e = min(best_e, row["E"])
        if row["arm"] in ("center", "shared"):
            best_c = min(best_c, row["E"])
        curve_g.append((used, best_g))
        curve_e.append((used, best_e if np.isfinite(best_e) else np.nan))
        curve_c.append((used, best_c if np.isfinite(best_c) else np.nan))
    if curve_g:
        ax.plot([c[0] for c in curve_g], [c[1] for c in curve_g], color="black", label="best so far")
        ax.plot([c[0] for c in curve_e], [c[1] for c in curve_e], label="endpoint, shared included")
        ax.plot([c[0] for c in curve_c], [c[1] for c in curve_c], label="center, shared included")
    ax.set_xlabel("optimizer evaluations, shared work counted once")
    ax.set_ylabel("best energy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "cumulative_best_energy.png", dpi=120)
    plt.close(fig)

    for key, ylabel, name in (
        ("E", "energy of the energy-selected state", "paired_energy.png"),
        ("F", "post-hoc squared fidelity", "paired_fidelity.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for depth in (9, 10, 11, 12):
            xs, ye, yc = [], [], []
            for pid in complete:
                e = next((s for s in selected if s["pair"] == pid and s["depth"] == depth and s["arm"] == "endpoint"), None)
                c = next((s for s in selected if s["pair"] == pid and s["depth"] == depth and s["arm"] == "center"), None)
                if e is None or c is None:
                    continue
                xs.append(pid)
                ye.append(e[key])
                yc.append(c[key])
            if xs:
                ax.plot(xs, ye, marker="o", label=f"endpoint p{depth}")
                ax.plot(xs, yc, marker="x", label=f"center p{depth}")
        ax.set_xlabel("complete pair")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(HERE / name, dpi=120)
        plt.close(fig)


def _report(summary):
    led = summary["ledger"]
    lines = [
        "# Endpoint rescue r1",
        "",
        f"Parent audit `{summary['parent_audit_commit']}`. Preregistration SHA-256 `{summary['prereg_sha256']}`.",
        "The frozen N=12 target was not changed. This is not a reproduction of Pagano's ion Hamiltonian.",
        "η and the squared fidelity are both reported for the energy-selected vector. They are different quantities.",
        "The accessible P=R=+1 gap is not used to explain fidelity.",
        "",
        "## Cost",
        "",
        f"Optimizer evaluations {led['optimizer_evaluations_total']} of {led['global_cap']}.",
        f"Shared, counted once: {led['shared_prefix_and_coincident_evaluations']}.",
        f"Endpoint suffix: {led['endpoint_suffix_evaluations']}. Center suffix: {led['center_suffix_evaluations']}.",
        f"Partial starts: {led['n_partial_starts']}. Starts whose optimizer did not report success: {led['n_failed_optimizer_success_false']}.",
        f"Diagnostic QAOA forwards, not in the optimizer total: {led['diagnostic_qaoa_forwards']}.",
        f"Complete pairs: {led['complete_pairs']}. Incomplete pairs: {led['incomplete_pairs']}.",
        "",
        "## Depths 9–12",
        "",
    ]
    for depth, block in summary["depths"].items():
        if block.get("status") == "not_reached":
            lines.append(f"p={depth} was not reached.")
            continue
        w = block["strongest"]
        lines.append(
            f"p={depth}: strongest E={w['E']:.15f}, F={w['F']:.15f}, η={w['eta']:.15f}, "
            f"Var={w['var_H']:.15f}, arm={w['arm']}, pair={w['pair']}, "
            f"state `{w['state_sha256']}`."
        )
        lines.append(
            f"Recorded E={block['recorded_E']}. Improved over that record: {block['improved_vs_recorded']}."
        )
        for name, comp in block["comparisons"].items():
            lines.append(
                f"Against {name}: dE={comp['dE_qaoa_minus_candidate']:.6e}, "
                f"dF={comp['dF_qaoa_minus_candidate']:.6e}, "
                f"candidate_retains_both={comp['candidate_retains_both']}, "
                f"qaoa_overturns_both={comp['qaoa_overturns_both']}, tradeoff={comp['tradeoff']}."
            )
        pair = summary["paired"][depth]
        lines.append(
            f"Complete-pair median (endpoint minus center) energy {pair['median_dE_endpoint_minus_center']}; "
            f"endpoint lower on {pair['n_endpoint_lower_E']} pairs, center lower on {pair['n_center_lower_E']}, "
            f"within tau on {pair['n_E_within_tau']}, n={pair['n_complete_pairs']}."
        )
        lines.append("")
    lines.append("A local result on this budget is not a global optimum. The finished historical QAOA allowances stay closed.")
    lines.append("")
    (HERE / "REPORT.md").write_text("\n".join(lines))


def main():
    _check_prereg()
    if (HERE / "chains.jsonl").exists():
        raise RuntimeError("chains.jsonl already exists; this directory is append-only and the run is single-shot")
    t0 = time.perf_counter()
    deadline = t0 + WALL
    target = Target(12)
    if target.hc_sha != "eef806253f93036b128bc3aa2a7c06d35bbe60e7b9bcaf01cf920974b1da6a00":
        raise RuntimeError("Hc hash drifted")
    if abs(target.B - float("-0.3444")) > 0.0:
        raise RuntimeError(f"B drifted {target.B}")
    gs, E0, Emax, plus = _spectrum(target)
    if abs(E0 + 8.6132024589353) > 1e-12:
        raise RuntimeError(f"E0 drifted {E0}")
    gap = {
        "E0": E0,
        "Emax": Emax,
        "lowest_PR_plus": plus,
        "accessible_gap": None if len(plus) < 2 else plus[1] - plus[0],
        "pinned_accessible_gap": PREREG["accessible_gap_not_an_explanation"],
        "used_to_explain_fidelity": False,
    }
    (HERE / "spectrum.json").write_text(json.dumps(gap, indent=2) + "\n")
    runner = Runner(target)
    runner.set_spectrum(gs, E0, Emax, 0.0, gap["accessible_gap"] or 0.0)
    ok, replay = _replay(runner)
    if not ok:
        print("replay failed", replay["residual_vs_audit"], flush=True)
        raise SystemExit(2)
    budget = Budget(GLOBAL_CAP, deadline)
    pair_rows = []
    for spec in PREREG["p1_starts"]:
        if budget.left <= 0 or time.perf_counter() >= deadline:
            break
        pid = int(spec["pair"])
        info = {"pair": pid, "arm_first": spec["arm_first"], "complete_through_p12": False, "stopped": None}
        p1_seeds = [(int(s["k"]), np.array([s["gamma"], s["beta"]], dtype=np.float64)) for s in spec["starts"]]
        p1_rows = _run_depth_shared(runner, budget, pid, 1, p1_seeds, "p1_box")
        best1 = _best_of(p1_rows)
        if best1 is None or any(r["partial"] for r in p1_rows) or len(p1_rows) < 3:
            info["stopped"] = "p1"
            pair_rows.append(info)
            break
        g1, b1 = best1["x_final"][0], best1["x_final"][1]
        base2 = np.array([g1, g1 + float(PREREG["p2_ours_delta_gamma"]), b1, b1 + float(PREREG["p2_ours_delta_beta"])], dtype=np.float64)
        p2_seeds = [(0, base2)] + [(k, base2 + _noise(pid, 2, k, 4)) for k in (1, 2)]
        p2_rows = _run_depth_shared(runner, budget, pid, 2, p2_seeds, "p2_mapped_offset")
        best2 = _best_of(p2_rows)
        if best2 is None or any(r["partial"] for r in p2_rows) or len(p2_rows) < 3:
            info["stopped"] = "p2"
            pair_rows.append(info)
            break
        schedules = {}
        g2 = np.array(best2["x_final"][:2], dtype=np.float64)
        b2 = np.array(best2["x_final"][2:], dtype=np.float64)
        schedules["endpoint"] = (g2.copy(), b2.copy())
        schedules["center"] = (g2.copy(), b2.copy())
        order = ["endpoint", "center"] if spec["arm_first"] == "endpoint" else ["center", "endpoint"]
        stopped = False
        for depth in range(3, 13):
            arm_seeds = {arm: _seeds_from(schedules[arm][0], schedules[arm][1], depth, arm, pid) for arm in order}
            bases = [arm_seeds[arm][0][1] for arm in order]
            coincide = bool(np.allclose(bases[0], bases[1], atol=0.0, rtol=0.0))
            if coincide:
                rows = _run_depth_shared(runner, budget, pid, depth, arm_seeds[order[0]], "coincident_seed")
                if any(r["partial"] for r in rows) or len(rows) < 3:
                    info["stopped"] = f"p{depth}_shared"
                    stopped = True
                    break
                best = _best_of(rows)
                x = np.array(best["x_final"], dtype=np.float64)
                for arm in order:
                    schedules[arm] = (x[:depth].copy(), x[depth:].copy())
                    _append({
                        "pair": pid, "depth": depth, "arm": arm, "k": best["k"], "label": "copied_from_shared",
                        "E": best["E"], "F": best["F"], "var_H": best["var_H"], "eta": best["eta"],
                        "state_sha256": best["state_sha256"], "x_final": best["x_final"], "x0": best["x0"],
                        "n_eval": 0, "partial": False, "stop_reason": "copied",
                        "optimizer_success": True, "copied_from_shared": True,
                        "fun_calls": 0, "jac_calls_uncached": 0, "wall_seconds": 0.0,
                        "hit_start_ceiling": False, "norm": best["norm"], "bucket": "shared",
                    })
                continue
            arm_rows = {arm: [] for arm in order}
            for k in range(3):
                for arm in order:
                    if budget.left <= 0 or time.perf_counter() >= deadline:
                        stopped = True
                        break
                    row = _one_start(runner, depth, arm_seeds[arm][k][1], budget, arm, START_CAP)
                    row.update({
                        "pair": pid, "depth": depth, "arm": arm, "k": k, "label": "suffix",
                        "seed_noise": [20261114, pid, depth, 9, k],
                        "seed_maxabs_vs_other": float(np.max(np.abs(arm_seeds["endpoint"][k][1] - arm_seeds["center"][k][1]))),
                    })
                    arm_rows[arm].append(row)
                    _append(row)
                    print(f"pair {pid} p{depth} {arm} k{k} E={row['E']} n={row['n_eval']} {row['stop_reason']}", flush=True)
                    if row["partial"]:
                        stopped = True
                        break
                if stopped:
                    break
            if stopped or any(len(arm_rows[arm]) < 3 or any(r["partial"] for r in arm_rows[arm]) for arm in order):
                info["stopped"] = f"p{depth}"
                stopped = True
                break
            for arm in order:
                best = _best_of(arm_rows[arm])
                x = np.array(best["x_final"], dtype=np.float64)
                schedules[arm] = (x[:depth].copy(), x[depth:].copy())
        if not stopped:
            info["complete_through_p12"] = True
        pair_rows.append(info)
        (HERE / "pairs_checkpoint.json").write_text(json.dumps(pair_rows, indent=2) + "\n")
        if not info["complete_through_p12"]:
            break
    summary = _summarize(runner, pair_rows, budget, diag0=1)
    print("done", summary["ledger"], flush=True)


if __name__ == "__main__":
    main()
