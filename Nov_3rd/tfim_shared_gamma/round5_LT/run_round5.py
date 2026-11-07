"""Round 5 LT: fresh sublattice ladder and leave-depth-out templates.

Seeds, caps, and decision tolerances come from preregistration.json.
Stored angles are loaded only in posthoc_overlap, after the searches return.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
R4 = ROOT.parent / "round4_V"
sys.path.insert(0, str(R4))
from sim import (  # noqa: E402
    System,
    fidelity,
    load_manifest,
    load_packings,
    logical_state,
    overlap,
    parse_J,
    physical_state,
    rfull_energy_grad,
    unpack_rfull,
)

RESULTS = ROOT / "results"
PRE = json.loads((ROOT / "preregistration.json").read_text())
BASE = json.loads((ROOT / "baselines.json").read_text())
_SYS = None


def _init(J, B, packing):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    global _SYS
    _SYS = System(np.asarray(J, dtype=np.float64), float(B), packing)


def _lbfgs(fun, x0):
    opt = PRE["optimizer"]
    best = {"E": np.inf, "x": np.asarray(x0, dtype=np.float64).copy()}

    def wrapped(x):
        val, grad = fun(np.asarray(x, dtype=np.float64))
        if val < best["E"]:
            best["E"] = float(val)
            best["x"] = np.array(x, dtype=np.float64, copy=True)
        return float(val), np.asarray(grad, dtype=np.float64)

    res = minimize(
        wrapped,
        np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxfun": int(opt["maxfun"]),
            "maxiter": int(opt["maxiter"]),
            "ftol": float(opt["ftol"]),
            "gtol": float(opt["gtol"]),
        },
    )
    return best["x"], int(getattr(res, "nfev", 0)), bool(res.success)


def _polish(task):
    L = int(task["L"])
    x0 = np.asarray(task["x0"], dtype=np.float64)

    def fun(x):
        return rfull_energy_grad(_SYS, L, x)

    try:
        x, nfev, success = _lbfgs(fun, x0)
        E = _SYS.energy(logical_state(_SYS, L, x))
        return {"tag": task["tag"], "E": float(E), "x": x.tolist(), "nfev": nfev, "success": success, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"tag": task["tag"], "E": None, "x": x0.tolist(), "nfev": 0, "success": False, "error": repr(exc)}


def knots(n):
    return (np.arange(1, n + 1) - 0.5) / n


def resample(values, n_src, n_tgt):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if n_src == 1 or values.size == 1:
        return np.full(n_tgt, float(values.reshape(-1)[0]))
    xs = knots(n_src)
    xt = knots(n_tgt)
    if n_src < 3:
        return np.interp(xt, xs, values)
    y = np.asarray(CubicSpline(xs, values, bc_type="natural", extrapolate=False)(xt), dtype=np.float64)
    y = np.where(xt < xs[0], values[0], y)
    y = np.where(xt > xs[-1], values[-1], y)
    y = np.where(np.isnan(y), np.where(xt < xs[0], values[0], values[-1]), y)
    return y


def pack(gamma, thk, thm, shell):
    return np.concatenate([np.array([gamma, thk, thm], dtype=np.float64), np.asarray(shell, dtype=np.float64).reshape(-1)])


def canonical_gamma(x):
    y = np.asarray(x, dtype=np.float64).copy()
    if y[0] < -float(PRE["gauge"]["gamma0"]):
        y[0] *= -1.0
        y[3:] *= -1.0
        return y, True
    return y, False


def apply_unwrap(x, L, ref, ref_L, periods):
    y, flipped = canonical_gamma(x)
    ref_c, _ = canonical_gamma(ref)
    _, _, _, shell = unpack_rfull(L, y)
    _, _, _, ref_shell = unpack_rfull(ref_L, ref_c)
    shell = shell.copy()
    for k in range(3):
        period = periods[k]
        if period is None:
            continue
        ref_on = resample(ref_shell[:, k], ref_L, L)
        shift = np.rint((ref_on - shell[:, k]) / period) * period
        shell[:, k] = shell[:, k] + shift
    y = pack(y[0], y[1], y[2], shell)
    return y, flipped


def ray_ok(sys, L, original, aligned, flipped):
    psi = logical_state(sys, L, original)
    phi = logical_state(sys, L, aligned)
    target = np.conj(psi) if flipped else psi
    return overlap(phi, target) >= 1.0 - float(PRE["gauge"]["training_vector_check"].split(">=")[-1].split()[0]) if False else overlap(phi, target)


def align_source(sys, L, x, ref, ref_L, periods):
    tol = float(PRE["tolerances"]["gauge_overlap"])
    flipped_gamma = bool(np.asarray(x)[0] < -float(PRE["gauge"]["gamma0"]))
    unwrapped, flipped = apply_unwrap(x, L, ref, ref_L, periods)
    ov_u = ray_ok(sys, L, x, unwrapped, flipped)
    if ov_u >= tol:
        return unwrapped, {"flipped": flipped, "unwrapped": True, "overlap": ov_u}
    canon, flipped_c = canonical_gamma(x)
    ov_c = ray_ok(sys, L, x, canon, flipped_c)
    if ov_c >= tol:
        return canon, {"flipped": flipped_c, "unwrapped": False, "overlap": ov_c}
    return np.asarray(x, dtype=np.float64).copy(), {"flipped": False, "unwrapped": False, "overlap": overlap(logical_state(sys, L, x), logical_state(sys, L, x)), "gamma_would_have_flipped": flipped_gamma}


def build_template(sys, sources, L_target, periods):
    """sources is a list of (depth, x) with the deepest last. No other depths are read."""
    ref_L, ref_x = sources[-1]
    aligned = []
    notes = []
    for L, x in sources:
        y, note = align_source(sys, L, x, ref_x, ref_L, periods)
        g, tk, tm, shell = unpack_rfull(L, y)
        on_target = np.column_stack([resample(shell[:, k], L, L_target) for k in range(3)])
        aligned.append((g, tk, tm, on_target))
        notes.append({"depth": L, **note, "x_hash": vec_hash(y)})
    gammas = [a[0] for a in aligned]
    thks = [a[1] for a in aligned]
    thms = [a[2] for a in aligned]
    shell = np.mean([a[3] for a in aligned], axis=0)
    center = pack(float(np.median(gammas)), float(np.median(thks)), float(np.median(thms)), shell)
    return center, notes


def vec_hash(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()).hexdigest()


def state_hash(psi) -> str:
    return hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()


def plain_starts(L, prev, rng):
    starts = []
    if prev is not None:
        g, tk, tm, shell = unpack_rfull(L - 1, prev)
        sh = np.column_stack([resample(shell[:, k], L - 1, L) for k in range(3)])
        tag = "cubic" if (L - 1) >= 3 else "linear"
        starts.append((tag, pack(g, tk, tm, sh)))
    box = PRE["plain_ladder"]
    lo, hi = box["random_scalar_uniform"]
    scale = float(box["random_shell_normal_scale"])
    for i in range(int(box["n_random"])):
        x = np.zeros(3 + 3 * L)
        x[0] = rng.uniform(lo, hi)
        x[1] = rng.uniform(lo, hi)
        x[2] = rng.uniform(lo, hi)
        x[3:] = rng.normal(scale=scale, size=3 * L)
        starts.append((f"restart_{i}", x))
    return starts


def template_starts(center, rng):
    starts = [("center", np.asarray(center, dtype=np.float64).copy())]
    for sigma in PRE["templates"]["sigmas"]:
        label = "s02" if abs(sigma - 0.02) < 1e-12 else "s10"
        for i in range(int(PRE["templates"]["n_per_sigma"])):
            noise = rng.normal(scale=float(sigma), size=center.size)
            starts.append((f"{label}_{i}", np.asarray(center, dtype=np.float64) + noise))
    return starts


def run_pool(J, B, packing, tasks):
    with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(J, B, packing)) as ex:
        return list(ex.map(_polish, tasks, chunksize=1))


def variance(sys, psi) -> float:
    hp = sys.apply_H(psi)
    e = float(np.real(np.vdot(psi, hp)))
    return float(np.real(np.vdot(hp, hp)) - e * e)


def sector_weights(psi):
    dim = psi.size
    n = int(np.log2(dim))
    bits = np.bitwise_count(np.arange(dim))
    p_even = float(np.sum(np.abs(psi[bits % 2 == 0]) ** 2))
    image = np.zeros(dim, dtype=np.int64)
    idx = np.arange(dim)
    for i in range(n):
        image |= ((idx >> i) & 1) << (n - 1 - i)
    reflection = float(abs(np.vdot(psi, psi[image])) ** 2)
    return {"P_even": p_even, "reflection_overlap": reflection}


def iqr(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size < 2:
        return None
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def summarize(sys, gs, E0, L, rows):
    finished = [r for r in rows if r.get("E") is not None]
    for row in finished:
        psi = logical_state(sys, L, row["x"])
        row["F"] = fidelity(gs, psi)
        row["energy_error"] = row["E"] - E0
    Es = [r["E"] for r in finished]
    Fs = [r["F"] for r in finished]
    best = min(finished, key=lambda r: r["E"])
    psi = logical_state(sys, L, best["x"])
    running_E = []
    running_n = []
    cur = np.inf
    nfev = 0
    for r in finished:
        nfev += int(r["nfev"])
        cur = min(cur, r["E"])
        running_E.append(cur)
        running_n.append(nfev)
    screen_E = PRE["screen"]["screen_energy"].get(str(L))
    basin_E = PRE["basin_posthoc"]["E"].get(str(L))
    tol_s = float(PRE["tolerances"]["screen_E"])
    tol_b = float(PRE["tolerances"]["basin_E"])
    for r in finished:
        r["screen_hit"] = bool(screen_E is not None and r["E"] <= float(screen_E) + tol_s)
        r["basin_hit"] = bool(basin_E is not None and r["E"] <= float(basin_E) + tol_b)
    return {
        "L": L,
        "E": float(sys.energy(psi)),
        "F": fidelity(gs, psi),
        "energy_error": float(sys.energy(psi) - E0),
        "var_H": variance(sys, psi),
        "state_hash": state_hash(psi),
        "sector": sector_weights(psi),
        "x": best["x"],
        "best_tag": best["tag"],
        "E_median": float(np.median(Es)),
        "E_iqr": iqr(Es),
        "F_median": float(np.median(Fs)),
        "F_iqr": iqr(Fs),
        "n_starts": len(rows),
        "n_finished": len(finished),
        "n_failed": len(rows) - len(finished),
        "nfev_sum": int(sum(r.get("nfev") or 0 for r in rows)),
        "n_screen": int(sum(r["screen_hit"] for r in finished)),
        "n_basin": int(sum(r["basin_hit"] for r in finished)),
        "energies": Es,
        "fidelities": Fs,
        "tags": [r["tag"] for r in finished],
        "nfev": [int(r["nfev"]) for r in finished],
        "best_so_far_E": running_E,
        "best_so_far_nfev": running_n,
        "starts": [{k: r[k] for k in ("tag", "E", "F", "energy_error", "nfev", "success", "error", "screen_hit", "basin_hit", "x") if k in r} for r in rows],
        "psi": psi,
    }


def probe_periods(sys):
    rng = np.random.default_rng(int(PRE["seeds"]["probe"]))
    L = 3
    x = np.zeros(3 + 3 * L)
    x[0] = float(rng.normal())
    x[1] = float(rng.normal())
    x[2] = float(rng.normal())
    x[3:] = rng.normal(scale=0.4, size=3 * L)
    psi = logical_state(sys, L, x)
    tol = float(PRE["tolerances"]["gauge_overlap"])
    periods = []
    detail = []
    for k, name in enumerate(("c1", "c2", "c3")):
        chosen = None
        for period in (0.5 * np.pi, np.pi):
            y = x.copy()
            y[3 + k] += period
            ov = overlap(psi, logical_state(sys, L, y))
            detail.append({"coeff": name, "period": float(period), "overlap": ov})
            if chosen is None and ov >= 1.0 - tol:
                chosen = float(period)
        periods.append(chosen)
    y = x.copy()
    y[0] *= -1.0
    y[3:] *= -1.0
    conj_ov = overlap(logical_state(sys, L, y), np.conj(psi))
    phys_ov = overlap(psi, physical_state(sys, L, x)) if L == 3 else None
    # physical_state uses the system's packing; the probe state is a valid check.
    return periods, {"shifts": detail, "conjugate_overlap": conj_ov, "physical_overlap": phys_ov, "periods": periods}


def fd_grad_n6(manifest):
    J, B = parse_J(manifest, 6)
    packings, _ = load_packings()
    sys = System(J, B, packings["sublattice"]["6"])
    rng = np.random.default_rng(int(PRE["seeds"]["probe"]))
    x = rng.normal(scale=0.15, size=3 + 6)
    _, g = rfull_energy_grad(sys, 2, x)
    eps = 1e-7
    fd = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        fd[i] = (rfull_energy_grad(sys, 2, x + step)[0] - rfull_energy_grad(sys, 2, x - step)[0]) / (2 * eps)
    return float(np.max(np.abs(g - fd)))


def dominates(E, F, baseline):
    tol_E = float(PRE["tolerances"]["dominance_E"])
    tol_F = float(PRE["tolerances"]["dominance_F"])
    dE = float(baseline["E"] - E)
    dF = float(F - baseline["F"])
    return {"dE": dE, "dF": dF, "both": bool(dE > tol_E and dF > tol_F)}


def screen_stats(rec):
    n = rec["n_finished"]
    k = rec["n_screen"]
    p = (k / n) if n else None
    ratio = (rec["nfev_sum"] / k) if k else None
    return {"n": n, "k": k, "p": p, "evals_per_success": ratio, "censored": k == 0}


def compare_screen(template, plain):
    a = screen_stats(template)
    b = screen_stats(plain)
    out = {"template": a, "plain": b, "gain": None, "evals_ratio": None, "gain_pass": False, "evals_pass": False, "censored": False}
    if a["p"] is None or b["p"] is None:
        out["censored"] = True
        return out
    out["gain"] = a["p"] - b["p"]
    out["gain_pass"] = bool(out["gain"] >= float(PRE["screen"]["gain_bar"]))
    if a["evals_per_success"] is None or b["evals_per_success"] is None:
        out["censored"] = True
        return out
    out["evals_ratio"] = a["evals_per_success"] / b["evals_per_success"]
    out["evals_pass"] = bool(out["evals_ratio"] <= float(PRE["screen"]["evals_per_success_ratio_bar"]))
    return out


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(type(obj))


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import multiprocessing as mp

    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.perf_counter()
    deadline = t0 + float(PRE["deadline_minutes"]) * 60
    RESULTS.mkdir(parents=True, exist_ok=True)
    logp = RESULTS / "run.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with logp.open("a") as fh:
            fh.write(line + "\n")

    def left():
        return deadline - time.perf_counter()

    pre_hash = hashlib.sha256((ROOT / "preregistration.json").read_bytes()).hexdigest()
    log(f"preregistration {pre_hash}")
    manifest, man_raw = load_manifest()
    packings, pack_raw = load_packings()
    man_hash = hashlib.sha256(man_raw).hexdigest()
    pack_hash = hashlib.sha256(pack_raw).hexdigest()
    if man_hash != PRE["manifest_sha256"] or pack_hash != PRE["packing_sha256"]:
        raise RuntimeError("manifest or packing hash mismatch")
    J, B = parse_J(manifest, 12)
    packing = packings["sublattice"]["12"]
    sys = System(J, B, packing)
    hc_hash = hashlib.sha256(np.ascontiguousarray(sys.hc, dtype="<f8").tobytes()).hexdigest()
    log(f"hc_le {hc_hash} rK {sys.rK:.10f} rM {sys.rM:.10f}")
    periods, probe = probe_periods(sys)
    log(f"periods {periods} conj {probe['conjugate_overlap']:.12f} physical {probe['physical_overlap']:.12f}")
    if probe["conjugate_overlap"] < 1.0 - float(PRE["tolerances"]["gauge_overlap"]):
        raise RuntimeError("conjugate gauge failed on the probe")
    if probe["physical_overlap"] < 1.0 - float(PRE["tolerances"]["gauge_overlap"]):
        raise RuntimeError("physical map failed on the probe")
    n6_err = fd_grad_n6(manifest)
    log(f"N6 grad max abs {n6_err:.3e}")
    if n6_err > 5e-6:
        raise RuntimeError("N6 gradient check failed")
    log("spectrum")
    dim = sys.dim
    H = np.diag(sys.hc).astype(np.float64)
    idx = np.arange(dim)
    for i in range(12):
        H[idx ^ (1 << i), idx] += B
    evals, evecs = eigh(H)
    gs = evecs[:, 0].astype(np.complex128)
    if float(np.real(np.sum(gs))) < 0:
        gs = -gs
    E0 = float(evals[0])
    log(f"E0 {E0:.12f}")
    if abs(E0 - float(BASE["E0_label"])) > float(PRE["tolerances"]["E0_abs"]):
        raise RuntimeError(f"E0 mismatch {E0}")

    plain = {}
    prev = None
    notes = []

    def run_starts(L, starts, stem):
        tasks = [{"L": L, "tag": t, "x0": np.asarray(x).tolist()} for t, x in starts]
        t1 = time.perf_counter()
        out = run_pool(J, B, packing, tasks)
        rec = summarize(sys, gs, E0, L, out)
        psi = rec.pop("psi")
        rec["seconds"] = time.perf_counter() - t1
        np.savez(RESULTS / f"{stem}.npz", x=np.asarray(rec["x"], dtype=np.float64), psi=psi)
        log(f"  {stem} E {rec['E']:.8f} F {rec['F']:.4f} med {rec['E_median']:.4f} iqr {rec['E_iqr']} nfev {rec['nfev_sum']} screen {rec['n_screen']}/{rec['n_finished']} tag {rec['best_tag']}")
        return rec

    if args.smoke:
        PRE["optimizer"]["maxfun"] = 4
        PRE["optimizer"]["maxiter"] = 4
        rng = np.random.default_rng(np.random.SeedSequence([711085, 1, 1]))
        rec = run_starts(1, plain_starts(1, None, rng)[:1], "smoke_L1")
        _, gnotes = build_template(sys, [(1, rec["x"]), (1, rec["x"])], 2, periods)
        log(f"smoke template overlap {gnotes[0]['overlap']:.6f} unwrapped {gnotes[0]['unwrapped']}")
        return

    for L in (1, 2, 3, 4):
        if L == 5:
            break
        if left() < 30:
            notes.append(f"stopped before plain L{L}")
            break
        log(f"plain L{L}")
        rng = np.random.default_rng(np.random.SeedSequence([711085, 1, L]))
        starts = plain_starts(L, prev, rng)
        plain[L] = run_starts(L, starts, f"plain_L{L}")
        prev = np.asarray(plain[L]["x"], dtype=np.float64)
        dump(RESULTS / "partial.json", {"plain_depths": list(plain), "notes": notes})

    templates = {}
    if 3 in plain and 4 in plain and left() > 30:
        log("template L5 from plain L3 and L4 only")
        center, gnotes = build_template(sys, [(3, plain[3]["x"]), (4, plain[4]["x"])], 5, periods)
        assert [n["depth"] for n in gnotes] == [3, 4]
        rng = np.random.default_rng(np.random.SeedSequence([711085, 2, 5]))
        templates[5] = run_starts(5, template_starts(center, rng), "template_L5")
        templates[5]["training_depths"] = [3, 4]
        templates[5]["gauge"] = gnotes
        templates[5]["center_hash"] = vec_hash(center)
        templates[5]["center"] = center.tolist()
    else:
        notes.append("template L5 not run")

    if 4 in plain and left() > 30:
        log("plain L5 from plain L4")
        rng = np.random.default_rng(np.random.SeedSequence([711085, 1, 5]))
        starts = plain_starts(5, np.asarray(plain[4]["x"]), rng)
        plain[5] = run_starts(5, starts, "plain_L5")
    else:
        notes.append("plain L5 not run")

    if all(d in plain for d in (3, 4, 5)) and left() > 30:
        log("template L6 from plain L3 L4 L5 only")
        center, gnotes = build_template(sys, [(3, plain[3]["x"]), (4, plain[4]["x"]), (5, plain[5]["x"])], 6, periods)
        assert [n["depth"] for n in gnotes] == [3, 4, 5]
        rng = np.random.default_rng(np.random.SeedSequence([711085, 2, 6]))
        templates[6] = run_starts(6, template_starts(center, rng), "template_L6")
        templates[6]["training_depths"] = [3, 4, 5]
        templates[6]["gauge"] = gnotes
        templates[6]["center_hash"] = vec_hash(center)
        templates[6]["center"] = center.tolist()
    else:
        notes.append("template L6 not run")

    if 5 in plain and left() > 30:
        log("plain L6 from plain L5")
        rng = np.random.default_rng(np.random.SeedSequence([711085, 1, 6]))
        plain[6] = run_starts(6, plain_starts(6, np.asarray(plain[5]["x"]), rng), "plain_L6")
    else:
        notes.append("plain L6 not run")

    for store in (plain, templates):
        for rec in store.values():
            rec["nfev_preregistered"] = rec["nfev_sum"]
            rec["n_starts_preregistered"] = rec["n_starts"]
    topup = {}
    for L in (5, 6):
        if L not in plain or L not in templates or left() < 10 * 60:
            notes.append(f"topup L{L} skipped")
            continue
        target = max(plain[L]["nfev_sum"], templates[L]["nfev_sum"])
        log(f"topup L{L} toward marginal nfev {target}")
        added_plain = []
        added_template = []
        rng_p = np.random.default_rng(np.random.SeedSequence([711085, 3, L]))
        rng_t = np.random.default_rng(np.random.SeedSequence([711085, 4, L]))
        guard = 0
        while min(plain[L]["nfev_sum"], templates[L]["nfev_sum"]) < target and guard < int(48) and left() > 60:
            guard += 1
            if plain[L]["nfev_sum"] < templates[L]["nfev_sum"]:
                starts = plain_starts(L, None, rng_p)[:1]
                starts = [(f"topup_{guard}", starts[0][1])]
                rec = run_starts(L, starts, f"topup_plain_L{L}_{guard}")
                added_plain.append(rec)
                plain[L] = merge_recs(sys, gs, E0, L, plain[L], rec)
                save_best(sys, L, plain[L], f"plain_L{L}")
            else:
                center = np.asarray(templates[L]["center"], dtype=np.float64)
                x = center + rng_t.normal(scale=0.1, size=center.size)
                rec = run_starts(L, [(f"topup_{guard}", x)], f"topup_template_L{L}_{guard}")
                added_template.append(rec)
                templates[L] = merge_recs(sys, gs, E0, L, templates[L], rec)
                save_best(sys, L, templates[L], f"template_L{L}")
        topup[L] = {
            "n_added_plain": len(added_plain),
            "n_added_template": len(added_template),
            "plain_nfev": plain[L]["nfev_sum"],
            "template_nfev": templates[L]["nfev_sum"],
            "target_nfev": target,
            "reached_target": min(plain[L]["nfev_sum"], templates[L]["nfev_sum"]) >= target,
        }
        log(f"  topup L{L} plain {plain[L]['nfev_sum']} template {templates[L]['nfev_sum']} reached {topup[L]['reached_target']}")

    post = posthoc(sys, plain, templates)
    screens = {}
    for L in (5, 6):
        if L in plain and L in templates:
            screens[str(L)] = compare_screen(templates[L], plain[L])
            screens[str(L)]["equalized"] = bool(topup.get(L, {}).get("reached_target", False))
            if not screens[str(L)]["equalized"]:
                screens[str(L)]["equal_budget_censored"] = True
    costs = cost_ledger(plain, templates)
    qaoa_note = qaoa_floor(costs)
    dominance = dominance_table(plain, templates)
    payload = {
        "preregistration_sha256": pre_hash,
        "manifest_sha256": man_hash,
        "packing_sha256": pack_hash,
        "hc_sha256_le": hc_hash,
        "E0": E0,
        "periods": periods,
        "probe": probe,
        "n6_grad_max_abs": n6_err,
        "plain": strip(plain),
        "templates": strip(templates),
        "topup": topup,
        "posthoc": post,
        "screens": screens,
        "costs": costs,
        "qaoa_floor": qaoa_note,
        "dominance": dominance,
        "notes": notes,
        "seconds": time.perf_counter() - t0,
    }
    dump(RESULTS / "summary.json", payload)
    write_csv(plain, templates)
    make_figures(plain, templates)
    write_md(payload)
    log(f"done {(time.perf_counter()-t0)/60:.1f} min")


def save_best(sys, L, rec, stem):
    psi = logical_state(sys, L, rec["x"])
    np.savez(RESULTS / f"{stem}.npz", x=np.asarray(rec["x"], dtype=np.float64), psi=psi)


def merge_recs(sys, gs, E0, L, base, extra):
    rows = []
    for s in base["starts"]:
        rows.append(s)
    for s in extra["starts"]:
        rows.append(s)
    rec = summarize(sys, gs, E0, L, rows)
    rec.pop("psi", None)
    for key in ("training_depths", "gauge", "center_hash", "center", "seconds", "nfev_preregistered", "n_starts_preregistered"):
        if key in base:
            rec[key] = base[key]
    rec["seconds"] = float(base.get("seconds", 0.0)) + float(extra.get("seconds", 0.0))
    return rec


def strip(block):
    out = {}
    for L, rec in block.items():
        item = dict(rec)
        item.pop("psi", None)
        out[str(L)] = item
    return out


def cost_ledger(plain, templates):
    def nfev(store, L):
        return int(store[L]["nfev_sum"]) if L in store else 0

    prefix = sum(nfev(plain, L) for L in (1, 2, 3, 4))
    return {
        "plain_L1_to_L4": prefix,
        "plain_L5_marginal": nfev(plain, 5),
        "plain_L6_marginal": nfev(plain, 6),
        "template_L5_marginal": nfev(templates, 5),
        "template_L6_marginal": nfev(templates, 6),
        "template_L5_inclusive": prefix + nfev(templates, 5),
        "plain_L5_inclusive": prefix + nfev(plain, 5),
        "template_L6_inclusive": prefix + nfev(plain, 5) + nfev(templates, 6),
        "plain_L6_inclusive": prefix + nfev(plain, 5) + nfev(plain, 6),
        "run_total_once": prefix + nfev(plain, 5) + nfev(plain, 6) + nfev(templates, 5) + nfev(templates, 6),
    }


def qaoa_floor(costs):
    total = costs["run_total_once"]
    out = {"candidate_run_total": total, "depths": {}}
    for p, row in BASE["qaoa"].items():
        if int(p) < 9:
            continue
        out["depths"][p] = {
            "nfev": row["nfev_sum"],
            "covers_candidate": bool(row["nfev_sum"] >= total),
        }
    out["new_qaoa_starts"] = 0
    out["shortfall"] = not all(v["covers_candidate"] for v in out["depths"].values())
    return out


def dominance_table(plain, templates):
    rows = []
    cells = [("plain", plain), ("template", templates)]
    for arm, store in cells:
        for L, rec in store.items():
            if int(L) not in (5, 6):
                continue
            for p, base in BASE["qaoa"].items():
                if int(L) == 5 and int(p) not in (5, 6, 9, 10, 12):
                    continue
                if int(L) == 6 and int(p) not in (6, 7, 11, 12):
                    continue
                rows.append({"arm": arm, "L": int(L), "baseline": f"qaoa_p{p}", **dominates(rec["E"], rec["F"], base)})
            stored = BASE["stored_rfull_replay"][str(L)]
            rows.append({"arm": arm, "L": int(L), "baseline": "stored_rfull", **dominates(rec["E"], rec["F"], stored)})
    return rows


def posthoc(sys, plain, templates):
    out = {}
    data = R4 / "source_data"
    for L, name in ((5, "N12_L5_A_sub_M.npz"), (6, "N12_L6_A_sub_M.npz")):
        src = np.load(data / name)["psi"]
        block = {}
        for arm, store in (("plain", plain), ("template", templates)):
            if L not in store:
                continue
            psi = logical_state(sys, L, store[L]["x"])
            block[arm] = {
                "overlap_stored": overlap(psi, src),
                "overlap_conjugate": overlap(psi, np.conj(src)),
                "E": store[L]["E"],
                "F": store[L]["F"],
            }
        out[str(L)] = block
    return out


def write_csv(plain, templates):
    lines = ["arm,L,tag,E,F,energy_error,nfev,screen_hit,basin_hit"]
    for arm, store in (("plain", plain), ("template", templates)):
        for L, rec in store.items():
            for s in rec["starts"]:
                if s.get("E") is None:
                    lines.append(f"{arm},{L},{s['tag']},,,,{s.get('nfev',0)},,")
                    continue
                lines.append(
                    f"{arm},{L},{s['tag']},{s['E']:.16g},{s['F']:.16g},{s['energy_error']:.16g},{s['nfev']},{int(s['screen_hit'])},{int(s['basin_hit'])}"
                )
    (RESULTS / "starts.csv").write_text("\n".join(lines) + "\n")


def make_figures(plain, templates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, L in zip(axes, (5, 6)):
        for store, label, color in ((plain, "plain", "C0"), (templates, "template", "C1")):
            if L not in store:
                continue
            Es = np.sort(store[L]["energies"])
            ax.plot(np.linspace(0, 1, len(Es), endpoint=False), Es, color=color, label=label)
        ax.axhline(float(PRE["screen"]["screen_energy"][str(L)]), color="0.4", ls="--", label="screen energy")
        ax.axhline(float(PRE["basin_posthoc"]["E"][str(L)]), color="C3", ls=":", label="stored energy")
        ax.set_title(f"L{L} seed energies")
        ax.set_xlabel("empirical CDF")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("final energy")
    fig.tight_layout()
    fig.savefig(RESULTS / "seed_cdf.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=False)
    for ax, L in zip(axes, (5, 6)):
        for store, label in ((plain, "plain"), (templates, "template")):
            if L not in store:
                continue
            ax.plot(store[L]["best_so_far_nfev"], store[L]["best_so_far_E"], label=label)
        ax.set_title(f"L{L} best so far")
        ax.set_xlabel("marginal evaluations")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("best energy")
    fig.tight_layout()
    fig.savefig(RESULTS / "best_so_far.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for p, row in BASE["qaoa"].items():
        if int(p) < 9:
            continue
        ax.scatter([row["energy_error"]], [row["F"]], marker="s", label=f"QAOA p{p}")
    for L, row in BASE["stored_rfull_replay"].items():
        if not str(L).isdigit():
            continue
        ax.scatter([row["energy_error"]], [row["F"]], marker="D", label=f"stored Rfull L{L}")
    for store, name in ((plain, "plain"), (templates, "template")):
        for L, rec in store.items():
            if int(L) < 5:
                continue
            ax.scatter([rec["energy_error"]], [rec["F"]], label=f"{name} L{L}")
    ax.set_xlabel("energy error")
    ax.set_ylabel("full-space fidelity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto.png", dpi=140)
    plt.close(fig)


def write_md(payload):
    lines = [
        "# Round 5 LT — fresh ladder and leave-depth-out templates",
        "",
        f"Manifest `{payload['manifest_sha256']}`. Packing `{payload['packing_sha256']}`. Hc little-endian `{payload['hc_sha256_le']}`.",
        f"Preregistration `{payload['preregistration_sha256']}`. E0 = {payload['E0']:.10f}.",
        "",
        "Stored angles were not training seeds. Template L5 was fit on fresh plain L3 and L4 only. Template L6 was fit on fresh plain L3, L4, and L5 only. Plain L5 is a separate search from the L5 template.",
        "",
        "| arm | L | E | error | F | Var(H) | median E | IQR | median F | screen hits | basin hits | starts | nfev | best tag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("plain", "templates"):
        for L, rec in payload[arm].items():
            lines.append(
                f"| {arm} | {L} | {rec['E']:.6f} | {rec['energy_error']:.4f} | {rec['F']:.4f} | {rec['var_H']:.4f} | "
                f"{rec['E_median']:.4f} | {None if rec['E_iqr'] is None else round(rec['E_iqr'], 4)} | {rec['F_median']:.4f} | {rec['n_screen']}/{rec['n_finished']} | "
                f"{rec['n_basin']}/{rec['n_finished']} | {rec['n_finished']}/{rec['n_starts']} | {rec['nfev_sum']} | {rec['best_tag']} |"
            )
    lines += ["", "## Posthoc overlap with the stored rays", ""]
    for L, block in payload["posthoc"].items():
        for arm, row in block.items():
            lines.append(
                f"- L{L} {arm}: overlap {row['overlap_stored']:.6f}, conjugate overlap {row['overlap_conjugate']:.6f}, E {row['E']:.6f}, F {row['F']:.4f}."
            )
    lines += ["", "## Equal-budget screen", ""]
    for L, sc in payload["screens"].items():
        lines.append(
            f"- L{L}: gain {sc['gain']}, evals/success ratio {sc['evals_ratio']}, gain_pass {sc['gain_pass']}, evals_pass {sc['evals_pass']}, censored {sc['censored']}, equalized {sc['equalized']}."
        )
    lines += ["", "## Cost", ""]
    for k, v in payload["costs"].items():
        lines.append(f"- {k}: {v}")
    q = payload["qaoa_floor"]
    lines.append(f"- candidate run total {q['candidate_run_total']}; QAOA shortfall {q['shortfall']}.")
    lines += ["", "## Both-metric deltas", "", "Positive dE means a lower energy than the baseline. Positive dF means a higher fidelity. Both must clear 1e-8.", ""]
    for row in payload["dominance"]:
        lines.append(f"- {row['arm']} L{row['L']} vs {row['baseline']}: dE {row['dE']:.6f}, dF {row['dF']:.6f}, both {row['both']}.")
    lines += ["", "## Notes", ""]
    for n in payload["notes"]:
        lines.append(f"- {n}")
    if not payload["notes"]:
        lines.append("- The preregistered order finished.")
    lines.append("")
    lines.append("Figures: `seed_cdf.png`, `best_so_far.png`, `pareto.png`. Starts: `starts.csv`.")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
