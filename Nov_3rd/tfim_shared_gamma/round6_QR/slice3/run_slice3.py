"""Slice 3: finish the frozen p9 then p10 QAOA allowances. Append-only."""
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


def _clip(x, p):
    y = np.asarray(x, dtype=np.float64).copy()
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
    x0 = _clip(x0, p)
    local_E, local_x, local_psi = float("inf"), x0.copy(), None
    cache = {}
    hit_cap = False

    def _record(x, E, grad, psi):
        nonlocal local_E, local_x, local_psi
        cache["x"] = np.array(x, copy=True)
        cache["g"] = grad
        if E < local_E:
            local_E, local_x, local_psi = float(E), np.array(x, copy=True), psi
        if E < best["E"]:
            best["E"] = float(E)
            best["x"] = np.array(x, copy=True)
        best["curve"].append([counter.used, float(best["E"])])

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
        res = minimize(
            fun, x0, method="L-BFGS-B", jac=jac, bounds=[GAMMA] * p + [BETA] * p,
            options={"maxiter": 1500, "maxfun": min(500, max(1, counter.left)), "ftol": 1e-14, "gtol": 1e-8},
        )
        ok = bool(res.success) and not hit_cap
    except Exception:
        ok = False
    fid = None if local_psi is None else float(runner.fidelity(local_psi))
    return local_E, fid, ok, hit_cap


def run(budget_min: float = 62.0):
    deadline = time.perf_counter() + budget_min * 60.0
    target = Target(12)
    if target.hc_sha != "eef806253f93036b128bc3aa2a7c06d35bbe60e7b9bcaf01cf920974b1da6a00":
        raise RuntimeError("Hc hash drifted")
    gs, E0 = spectrum(target)
    if abs(E0 + 8.6132024589353) > 1e-12:
        raise RuntimeError(f"E0 drifted {E0}")
    runner = Runner(target)
    runner.set_spectrum(gs, E0, E0, 0.0, 0.0)
    reports = []
    for p in PREREG["priority"]:
        cap = int(PREREG["remaining_max_new_evaluations"][str(p)])
        counter = Counter(cap)
        stored_E = float(V1["qaoa_stored_best"][str(p)]["E"])
        best = {"E": stored_E, "x": None, "curve": []}
        starts = []
        batch = int(PREREG["continuation_batch_start"][str(p)])
        log(f"p={p} cap={cap} batch0={batch}")
        while counter.left > 0 and time.perf_counter() < deadline:
            for arm, k, label, x0, sigma in _seeds(p, batch):
                if counter.left <= 0 or time.perf_counter() >= deadline:
                    break
                before = counter.used
                local_E, fid, ok, hit_cap = _one_start(runner, p, x0, counter, best)
                if not np.isfinite(local_E):
                    continue
                starts.append({
                    "p": p, "batch": batch, "arm": arm, "k": int(k), "label": label, "sigma": sigma,
                    "E": float(local_E), "F": fid, "nfev": int(counter.used - before),
                    "optimizer_success": bool(ok), "completed": bool(not hit_cap), "partial": bool(hit_cap),
                    "seed": [20261113, int(arm), int(p), int(k), int(batch)],
                })
                _checkpoint(p, counter.used, best, starts, batch + 1)
            batch += 1
            log(f"  p={p} through batch {batch-1} used={counter.used} best={best['E']:.6f}")
        x_report = best["x"] if best["x"] is not None else _stored_x(p)
        E, _, psi = runner.energy_grad("qaoa", x_report, p)
        reports.append(_record(p, runner, E, psi, E0, counter.used, cap, starts, best, time.perf_counter() >= deadline))
        log(f"p={p} E={E:.6f} new={counter.used} short={cap-counter.used}")
        if time.perf_counter() >= deadline:
            for q in PREREG["priority"]:
                if q not in [r["p"] for r in reports]:
                    reports.append({"p": q, "status": "not_run", "allowed": int(PREREG["remaining_max_new_evaluations"][str(q)]), "n_eval_new": 0, "shortfall": int(PREREG["remaining_max_new_evaluations"][str(q)])})
            break
    summary = {"E0": E0, "tau": 1e-9, "depths": reports, "parent_slice2_commit": PREREG["parent_slice2_commit"]}
    (HERE / "rescue_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _csv()
    _plots(summary)
    _md(summary)
    log("slice3 done")


def _checkpoint(p, used, best, starts, next_batch):
    folder = HERE / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    blob = {"p": p, "n_eval_new": used, "best_E": best["E"], "next_batch": next_batch, "starts": starts,
            "curve": best["curve"][:: max(1, len(best["curve"]) // 700)]}
    if best["x"] is not None:
        blob["x_exact"] = [float(v) for v in best["x"]]
    (folder / f"p{p}.json").write_text(json.dumps(blob) + "\n")


def _record(p, runner, E, psi, E0, used, cap, starts, best, clock):
    tau = 1e-9
    F = runner.fidelity(psi)
    var = runner.variance(psi)
    np.save(HERE / "states" / f"p{p}.npy", canonical_state(psi))
    energies = np.array([s["E"] for s in starts], dtype=np.float64)
    q25, q75 = np.percentile(energies, [25, 75]) if energies.size else (np.nan, np.nan)
    pins = {
        "stride4_L6_v1": V1["stride4_L6_arm0"],
        "stride4_L5_D_arm0": {"E": -8.385155838742358, "F": 0.8898142677626111},
        "stride4_L6_v2_arm2": {"E": -8.445682501760135, "F": 0.9023701795284864},
        "KS_L6": V1["KS_L6"],
    }
    comps = {}
    for name, spec in pins.items():
        dE = float(E - spec["E"])
        dF = float(F - spec["F"])
        comps[name] = {
            "dE_qaoa_minus_candidate": dE,
            "dF_qaoa_minus_candidate": dF,
            "qaoa_dominates": bool(E < spec["E"] - tau and F > spec["F"] + tau),
            "candidate_retains_both": bool(spec["E"] < E - tau and spec["F"] > F + tau),
            "lower_E_lower_F_tradeoff": bool(E < spec["E"] - tau and F < spec["F"] - tau),
        }
    return {
        "p": p, "E": float(E), "F": F, "var_H": var, "energy_error": float(E - E0),
        "state_sha256": sha256_state(psi), "norm": float(np.linalg.norm(psi)),
        "parity": runner.parity_weight(psi),
        "n_eval_new": int(used), "allowed": int(cap), "shortfall": int(cap - used),
        "n_starts": len(starts),
        "n_completed": int(sum(1 for s in starts if s["completed"])),
        "n_partial": int(sum(1 for s in starts if s["partial"])),
        "seed_median": None if not energies.size else float(np.median(energies)),
        "seed_iqr": None if not energies.size else float(q75 - q25),
        "basin_hits_1e-3": int(sum(1 for s in starts if abs(s["E"] - E) <= 1e-3)),
        "clock_done": bool(clock and used < cap),
        "comparisons": comps,
        "diagnostic_forward_not_in_n_eval": 1,
    }


def _csv():
    rows = []
    for p in PREREG["priority"]:
        f = HERE / "jobs" / f"p{p}.json"
        if f.exists():
            rows.extend(json.loads(f.read_text()).get("starts", []))
    keys = ["p", "batch", "arm", "k", "label", "sigma", "E", "F", "nfev", "optimizer_success", "completed", "partial", "seed"]
    with (HERE / "starts.csv").open("w", newline="") as fh:
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
    prior = {"9": 102522 + 71598, "10": 115638 + 64704}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for rec in summary["depths"]:
        if "E" not in rec:
            continue
        blob = json.loads((HERE / "jobs" / f"p{rec['p']}.json").read_text())
        if not blob.get("curve"):
            continue
        ys = [v - summary["E0"] for _, v in blob["curve"]]
        xs = [prior[str(rec["p"])] + n for n, _ in blob["curve"]]
        ax.plot(xs, ys, label=f"p={rec['p']}")
    ax.set_xlabel("inclusive evaluations")
    ax.set_ylabel("best energy error so far")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "best_so_far.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for rec in summary["depths"]:
        if "E" not in rec:
            continue
        starts = json.loads((HERE / "jobs" / f"p{rec['p']}.json").read_text())["starts"]
        err = sorted(s["E"] - summary["E0"] for s in starts)
        if not err:
            continue
        ax.step(err, np.arange(1, len(err) + 1) / len(err), where="post", label=f"p={rec['p']}")
    ax.set_xlabel("slice3 start energy error")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "cdf.png", dpi=140)
    plt.close(fig)


def _md(summary):
    lines = ["# Round 6 QR slice 3", "", "p9 then p10. Tau stays 1e-9. Allowances were not reduced.", ""]
    lines.append("| p | E | F | new | allowed | short | completed | partial |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in summary["depths"]:
        if "E" not in rec:
            lines.append(f"| {rec['p']} | not run | | 0 | {rec['allowed']} | {rec['shortfall']} | | |")
            continue
        lines.append(f"| {rec['p']} | {rec['E']:.6f} | {rec['F']:.4f} | {rec['n_eval_new']} | {rec['allowed']} | {rec['shortfall']} | {rec['n_completed']} | {rec['n_partial']} |")
        for name, c in rec["comparisons"].items():
            lines.append(f"| | {name} dE {c['dE_qaoa_minus_candidate']:+.6e} dF {c['dF_qaoa_minus_candidate']:+.6e} candidate_both {c['candidate_retains_both']} | | | | | | |")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run(62.0)
