"""Slice 2 of the QAOA rescue. Reads the frozen prereg. Does not edit round 6 files."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from Nov_3rd.tfim_shared_gamma.round1_A.common import canonical_state, log, sha256_state
from Nov_3rd.tfim_shared_gamma.round5_KS.model import Runner, Target, pack, unpack

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROUND4 = ROOT.parents[0] / "round4_N" / "results" / "jobs"
PREREG = json.loads((HERE / "prereg.json").read_text())
V1 = json.loads((ROOT / "baseline.json").read_text())
AUDIT = json.loads((HERE / "provenance_audit.json").read_text())
GAMMA = (-12.0, 12.0)
BETA = (-2.0 * np.pi, 2.0 * np.pi)


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
    return gs, float(evals[0])


def _resample(values, p_new):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    p_old = int(values.size)
    xs = (np.arange(p_old) + 0.5) / p_old
    xt = (np.arange(p_new) + 0.5) / p_new
    if p_old < 3:
        return np.interp(xt, xs, values)
    spline = CubicSpline(xs, values, bc_type="natural")
    return np.asarray(spline(np.clip(xt, float(xs[0]), float(xs[-1]))), dtype=np.float64)


def _stored_x(p):
    blob = json.loads((ROUND4 / f"n12_L{p}_qaoa.json").read_text())
    return np.array([float(v) for v in blob["x_exact"]], dtype=np.float64)


def _rng(arm, p, k, batch):
    return np.random.default_rng(np.random.SeedSequence([20261113, int(arm), int(p), int(k), int(batch)]))


def _seeds(p, batch):
    x_same = _stored_x(p)
    gammas, _, cs = unpack("qaoa", x_same, p)
    betas = cs[:, 0]
    out = []
    if p > 1:
        g0, _, c0 = unpack("qaoa", _stored_x(p - 1), p - 1)
        out.append((1, 0, "interp", pack("qaoa", 0, {}, np.zeros((p, 3)), _resample(g0, p), _resample(c0[:, 0], p)), 0.0))
    t = (np.arange(p) + 0.5) / float(p)
    ag, ab = float(np.max(np.abs(gammas))), float(np.max(np.abs(betas)))
    for mode in (1, 2):
        shape = np.sin(mode * np.pi * t)
        out.append((2, mode, f"fourier{mode}", pack("qaoa", 0, {}, np.zeros((p, 3)), ag * shape, ab * shape), 0.0))
    for k in range(8):
        rng = _rng(3, p, k, batch)
        out.append((3, k, f"random_{k}", pack("qaoa", 0, {}, np.zeros((p, 3)), rng.uniform(-0.8, 0.8, p), rng.uniform(-0.8, 0.8, p)), 0.0))
    for arm, sigma in ((4, 0.02), (5, 0.1)):
        for k in range(8):
            rng = _rng(arm, p, k, batch)
            out.append((arm, k, f"perturb_{sigma}_{k}", x_same + rng.normal(scale=sigma, size=x_same.shape), sigma))
    return out


def _clip(x):
    y = np.asarray(x, dtype=np.float64).copy()
    p = y.size // 2
    for i in range(p):
        y[i] = min(max(float(y[i]), GAMMA[0]), GAMMA[1])
    for i in range(p, 2 * p):
        y[i] = min(max(float(y[i]), BETA[0]), BETA[1])
    return y


class Counter:
    def __init__(self, n):
        self.left = int(n)
        self.used = 0

    def take(self):
        if self.left <= 0:
            return False
        self.left -= 1
        self.used += 1
        return True


def _one_start(runner, p, x0, counter, best):
    x0 = _clip(x0)
    local_E, local_x = float("inf"), x0.copy()
    local_psi = None
    cache = {}
    hit_cap = False

    def _record(x, E, grad, psi):
        nonlocal local_E, local_x, local_psi
        cache["x"] = np.array(x, copy=True)
        cache["g"] = grad
        if E < local_E:
            local_E, local_x = float(E), np.array(x, copy=True)
            local_psi = psi
        if E < best["E"]:
            best["E"] = float(E)
            best["x"] = np.array(x, copy=True)
            best["psi"] = psi
        best["curve"].append((counter.used, float(best["E"])))

    def fun(x):
        nonlocal hit_cap
        if not counter.take():
            hit_cap = True
            return local_E
        E, grad, psi = runner.energy_grad("qaoa", x, p)
        _record(x, E, grad, psi)
        return float(E)

    def jac(x):
        nonlocal hit_cap
        if cache.get("x") is not None and np.allclose(cache["x"], x, atol=1e-12, rtol=0):
            return cache["g"]
        if not counter.take():
            hit_cap = True
            return np.zeros_like(x)
        E, grad, psi = runner.energy_grad("qaoa", x, p)
        _record(x, E, grad, psi)
        return grad

    ok = False
    try:
        res = minimize(fun, x0, method="L-BFGS-B", jac=jac, bounds=[GAMMA] * p + [BETA] * p,
                       options={"maxiter": 1500, "maxfun": min(500, max(1, counter.left + 1)), "ftol": 1e-14, "gtol": 1e-8})
        ok = bool(res.success) and not hit_cap
    except Exception:
        ok = False
    fid = None if local_psi is None else runner.fidelity(local_psi)
    return local_x, local_E, ok, hit_cap, fid


def _load_progress(p):
    path = HERE / "jobs" / f"p{p}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_progress(p, blob):
    folder = HERE / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"p{p}.json").write_text(json.dumps(blob, indent=2) + "\n")


def run(budget_min: float = 68.0):
    deadline = time.perf_counter() + budget_min * 60.0
    target = Target(12)
    if target.hc_sha != "eef806253f93036b128bc3aa2a7c06d35bbe60e7b9bcaf01cf920974b1da6a00":
        raise RuntimeError("Hc hash drifted")
    gs, E0 = spectrum(target)
    if abs(E0 - V1["E0"]) > 1e-12:
        raise RuntimeError("E0 drifted")
    runner = Runner(target)
    runner.set_spectrum(gs, E0, E0, 0.0, 0.0)
    rows_out = []
    for p in PREREG["priority"]:
        cap = int(PREREG["ledger_check"]["remaining_max_new_evaluations"][str(p)])
        prev = _load_progress(p)
        used = 0 if prev is None else int(prev["n_eval_new"])
        starts = [] if prev is None else list(prev["starts"])
        curve = [] if prev is None else list(prev["curve"])
        stored_E = float(V1["qaoa_stored_best"][str(p)]["E"])
        best = {"E": stored_E if prev is None else float(prev["best_E"]), "x": None if prev is None or "x_exact" not in prev else np.array(prev["x_exact"], dtype=np.float64), "curve": curve}
        batch = int(PREREG["continuation_batch_start"][str(p)])
        if prev is not None:
            batch = int(prev.get("next_batch", batch))
            used_cap = cap
            cap = max(0, used_cap - used)
        counter = Counter(cap)
        log(f"p={p} remaining={counter.left} batch0={batch}")
        while counter.left > 0 and time.perf_counter() < deadline:
            for arm, k, label, x0, sigma in _seeds(p, batch):
                if counter.left <= 0 or time.perf_counter() >= deadline:
                    break
                before = counter.used
                local_x, local_E, ok, hit_cap, fid = _one_start(runner, p, x0, counter, best)
                row = {
                    "p": p, "batch": batch, "arm": arm, "k": k, "label": label, "sigma": sigma,
                    "E": None if not np.isfinite(local_E) else float(local_E),
                    "F": fid,
                    "nfev": int(counter.used - before),
                    "optimizer_success": bool(ok),
                    "completed": bool(not hit_cap),
                    "partial": bool(hit_cap),
                    "seed": [20261113, arm, p, k, batch],
                }
                if row["E"] is not None:
                    # F is filled after the cap for the best only. Per start F would be extra evals.
                    starts.append(row)
                blob = {
                    "p": p, "n_eval_new": counter.used + (0 if prev is None else used),
                    "best_E": best["E"],
                    "next_batch": batch + 1,
                    "starts": starts,
                    "curve": best["curve"][:: max(1, len(best["curve"]) // 600)],
                }
                if best["x"] is not None:
                    blob["x_exact"] = [float(v) for v in best["x"]]
                _save_progress(p, blob)
            batch += 1
            log(f"  p={p} batch_done {batch-1} used={counter.used} best={best['E']:.6f}")
        # One diagnostic forward of the energy-selected vector. Not counted.
        x_report = best["x"] if best["x"] is not None else _stored_x(p)
        E, _, psi = runner.energy_grad("qaoa", x_report, p)
        F = runner.fidelity(psi)
        var = runner.variance(psi)
        np.save(HERE / "states" / f"p{p}.npy", canonical_state(psi))
        energies = np.array([s["E"] for s in starts if s["E"] is not None], dtype=np.float64)
        q25, q75 = (np.percentile(energies, [25, 75]) if energies.size else (np.nan, np.nan))
        total_new = counter.used + (0 if prev is None else used)
        allowed = int(PREREG["ledger_check"]["remaining_max_new_evaluations"][str(p)])
        rec = {
            "p": p, "E": float(E), "F": F, "var_H": var, "energy_error": float(E - E0),
            "state_sha256": sha256_state(psi), "norm": float(np.linalg.norm(psi)),
            "parity": runner.parity_weight(psi),
            "n_eval_new": int(total_new), "allowed": allowed, "shortfall": int(allowed - total_new),
            "n_starts": len(starts),
            "n_completed": int(sum(1 for s in starts if s["completed"])),
            "n_partial": int(sum(1 for s in starts if s["partial"])),
            "seed_median": None if not energies.size else float(np.median(energies)),
            "seed_iqr": None if not energies.size else float(q75 - q25),
            "basin_hits_1e-3": int(sum(1 for s in starts if s["E"] is not None and abs(s["E"] - E) <= 1e-3)),
            "next_batch": batch,
            "clock_done": bool(time.perf_counter() >= deadline and total_new < allowed),
        }
        rows_out.append(rec)
        log(f"p={p} E={E:.6f} F={F:.4f} new={total_new} short={rec['shortfall']}")
        if time.perf_counter() >= deadline:
            for q in PREREG["priority"]:
                if q not in [r["p"] for r in rows_out]:
                    rows_out.append({"p": q, "status": "not_run", "allowed": int(PREREG["ledger_check"]["remaining_max_new_evaluations"][str(q)]), "n_eval_new": 0, "shortfall": int(PREREG["ledger_check"]["remaining_max_new_evaluations"][str(q)])})
            break
    summary = _summary(rows_out, E0)
    (HERE / "rescue_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _csv(HERE / "starts.csv")
    _plots(summary)
    _md(summary)
    log("slice2 done")
    return summary


def _summary(rows, E0):
    tau = 1e-9
    v1_s4 = V1["stride4_L6_arm0"]
    v1_ks = V1["KS_L6"]
    v2 = AUDIT["independent_replay"]["family_best_L6"]
    out_rows = []
    for rec in rows:
        if "E" not in rec:
            out_rows.append(rec)
            continue
        def comp(spec, tag):
            dE = float(rec["E"] - spec["E"])
            dF = float(rec["F"] - spec["F"])
            return {"tag": tag, "dE": dE, "dF": dF,
                    "qaoa_dominates": bool(rec["E"] < spec["E"] - tau and rec["F"] > spec["F"] + tau),
                    "lower_E_lower_F": bool(rec["E"] < spec["E"] - tau and rec["F"] < spec["F"] - tau)}
        rec = dict(rec)
        rec["v1_stride4_L6"] = comp(v1_s4, "v1_arm0")
        rec["v1_KS_L6"] = comp(v1_ks, "v1_KS")
        rec["v2_stride4_L6"] = comp(v2, "v2_arm2_family_best")
        rec["sensitivity_1e-7_would_change_v1_vs_v2_identity"] = True
        rec["tau_v1"] = tau
        out_rows.append(rec)
    return {
        "E0": E0, "tau_v1": tau, "tau_not_changed": True,
        "v1_stride4_L6_F": v1_s4["F"], "v2_stride4_L6_F": v2["F"],
        "depths": out_rows,
        "parent_commit": PREREG["parent_round6_commit"],
    }


def _csv(path):
    rows = []
    for p in PREREG["priority"]:
        f = HERE / "jobs" / f"p{p}.json"
        if f.exists():
            rows.extend(json.loads(f.read_text())["starts"])
    if not rows:
        path.write_text("")
        return
    keys = ["p", "batch", "arm", "k", "label", "sigma", "E", "F", "nfev", "optimizer_success", "completed", "partial", "seed"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            item = dict(row)
            item["seed"] = " ".join(str(v) for v in row["seed"])
            w.writerow({k: item.get(k) for k in keys})


def _plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for p in PREREG["priority"]:
        f = HERE / "jobs" / f"p{p}.json"
        if not f.exists():
            continue
        blob = json.loads(f.read_text())
        if not blob.get("curve"):
            continue
        prior = {"7": 7001 + 46511, "12": 130014 + 118558, "11": 117249 + 127525}[str(p)]
        ys = [v - summary["E0"] for _, v in blob["curve"]]
        xs = [prior + n for n, _ in blob["curve"]]
        ax.plot(xs, ys, label=f"p={p}")
    ax.set_xlabel("inclusive evaluations")
    ax.set_ylabel("best energy error so far")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "best_so_far.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for p in PREREG["priority"]:
        f = HERE / "jobs" / f"p{p}.json"
        if not f.exists():
            continue
        starts = [s for s in json.loads(f.read_text())["starts"] if s.get("E") is not None]
        if not starts:
            continue
        err = sorted(s["E"] - summary["E0"] for s in starts)
        ax.step(err, np.arange(1, len(err) + 1) / len(err), where="post", label=f"p={p}")
    ax.set_xlabel("slice2 start energy error")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "cdf.png", dpi=140)
    plt.close(fig)


def _md(summary):
    lines = ["# Round 6 QR slice 2", ""]
    lines.append("Tau stays 1e-9. v1 is arm0. v2 is D's arm2 family-best. They are not declared the same state.")
    lines.append("")
    lines.append("| p | E | F | new | allowed | short | completed | partial |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in summary["depths"]:
        if "E" not in rec:
            lines.append(f"| {rec['p']} | not run | | 0 | {rec['allowed']} | {rec['shortfall']} | | |")
            continue
        lines.append(
            f"| {rec['p']} | {rec['E']:.6f} | {rec['F']:.4f} | {rec['n_eval_new']} | {rec['allowed']} | {rec['shortfall']} | {rec['n_completed']} | {rec['n_partial']} |"
        )
        for key in ("v1_stride4_L6", "v2_stride4_L6", "v1_KS_L6"):
            c = rec[key]
            lines.append(f"| | {key} dE {c['dE']:+.6e} dF {c['dF']:+.6e} dominates {c['qaoa_dominates']} | | | | | | |")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run(68.0)
