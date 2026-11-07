"""Independent verifier and fresh searches. Budgets fixed before any energy is seen.

QAOA: 64 fresh starts at p=9,10,11,12, maxiter 400. A second wave of 192 starts
runs only when more than 40 minutes remain after that wave, aiming at 4x the
source inclusive eval count. No start is the source optimum.
Rfull: ascending bootstrap plus 12 restarts, then 48 extra starts at L=4,5,6.
Controls: stride4 and random packings 0 and 1, 6 starts at L=6, only if time remains.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from sim import (
    DATA,
    MANIFEST_SHA,
    PACKING_SHA,
    System,
    fidelity,
    load_manifest,
    load_packings,
    logical_state,
    overlap,
    parse_J,
    qaoa_energy_grad,
    qaoa_state,
    rfull_energy_grad,
    unpack_rfull,
)
from test_replay import main as algebra_and_directions

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
QAOA_SEED = 611084
RFULL_SEED = 611085
EXTRA_SEED = 611086
CONTROL_SEEDS = {"stride4": 611087, "random0": 611088, "random1": 611089}
N_QAOA = 64
N_QAOA_EXTRA = 192
N_BOOT = 12
N_EXTRA = 48
N_CONTROL = 6
MAXITER_Q = 400
MAXITER_R = 500
SOURCE_NFE = {9: 11518, 10: 16383, 11: 16962, 12: 16740}
SOURCE_P12_E = -8.325407549783158
SOURCE_L6_E = -8.321482847047687
SOURCE_L6_F = 0.6098414550958995
ERR_BAR = 0.292
F_BAR = 0.610
OLD_ERR = {5: 0.390, 6: 0.315}
OLD_F = {5: 0.412, 6: 0.485}

_SYS = None


def _init(J, B, packing):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    global _SYS
    _SYS = System(np.asarray(J), float(B), packing)


def _lbfgs(fun, x0, maxiter):
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
        options={"maxfun": int(maxiter), "maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-8},
    )
    return best["x"], int(getattr(res, "nfev", 0)), bool(res.success)


def _polish(task):
    kind = task["kind"]
    L = int(task["L"])
    x0 = np.asarray(task["x0"], dtype=np.float64)

    def fun(x):
        if kind == "qaoa":
            return qaoa_energy_grad(_SYS, x[:L], x[L:])
        return rfull_energy_grad(_SYS, L, x)

    try:
        x, nfev, success = _lbfgs(fun, x0, int(task["maxiter"]))
        if kind == "qaoa":
            E = _SYS.energy(qaoa_state(_SYS, x))
        else:
            E = _SYS.energy(logical_state(_SYS, L, x))
        return {"tag": task["tag"], "E": float(E), "x": x.tolist(), "nfev": nfev, "success": success, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"tag": task["tag"], "E": None, "x": x0.tolist(), "nfev": 0, "success": False, "error": repr(exc)}


def fourier_qaoa(rng, L):
    q = min(3, L)
    u = rng.normal(scale=0.3, size=q)
    v = rng.normal(scale=0.25, size=q)
    g = np.zeros(L)
    b = np.zeros(L)
    for i in range(1, L + 1):
        for k in range(1, q + 1):
            arg = (k - 0.5) * (i - 0.5) * np.pi / L
            g[i - 1] += u[k - 1] * np.sin(arg)
            b[i - 1] += v[k - 1] * np.cos(arg)
    return np.concatenate([g, b])


def qaoa_starts(p, rng, n):
    starts = []
    s = np.linspace(0, 1, p)
    kinds = ("fourier", "adiab", "perturb", "random")
    per = n // 4
    for kind_i, kind in enumerate(kinds):
        count = per if kind_i < 3 else n - 3 * per
        for i in range(count):
            if kind == "fourier":
                x = fourier_qaoa(rng, p)
            elif kind == "adiab":
                gmax = float(rng.uniform(0.2, 2.5))
                bmax = float(rng.uniform(0.05, 1.2))
                x = np.concatenate([gmax * s, bmax * (1 - s)])
            elif kind == "perturb":
                base = np.concatenate([0.35 * s, 0.45 * (1 - s)])
                x = base + rng.normal(scale=0.12, size=base.size)
            else:
                x = np.concatenate([rng.normal(scale=0.5, size=p), rng.uniform(0, np.pi / 2, size=p)])
            starts.append((f"{kind}_{i}", x))
    return starts


def resample(values, L, mode):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1 or L == 1:
        return np.full(L, float(values.reshape(-1)[0]))
    s_old = np.linspace(0, 1, values.size)
    s_new = np.linspace(0, 1, L)
    if mode == "cubic" and values.size >= 3:
        from scipy.interpolate import CubicSpline

        return np.asarray(CubicSpline(s_old, values, bc_type="natural")(s_new), dtype=np.float64)
    return np.interp(s_new, s_old, values)


def rfull_starts(L, prev, rng, n_random):
    starts = []
    if prev is not None:
        g, tk, tm, shell = unpack_rfull(L - 1, prev)
        if L == 2:
            sh = np.vstack([shell[0], shell[0]])
            starts.append(("duplicate", np.concatenate([[g, tk, tm], sh.reshape(-1)])))
        else:
            lin = np.column_stack([resample(shell[:, k], L, "linear") for k in range(3)])
            starts.append(("linear", np.concatenate([[g, tk, tm], lin.reshape(-1)])))
            if shell.shape[0] >= 3:
                cub = np.column_stack([resample(shell[:, k], L, "cubic") for k in range(3)])
                starts.append(("cubic", np.concatenate([[g, tk, tm], cub.reshape(-1)])))
        base = starts[0][1]
        starts.append(("perturb", base + rng.normal(scale=0.05, size=base.size)))
    for i in range(n_random):
        x = np.zeros(3 + 3 * L)
        x[0] = rng.uniform(-1.2, 1.2)
        x[1] = rng.uniform(-1.2, 1.2)
        x[2] = rng.uniform(-1.2, 1.2)
        x[3:] = rng.normal(scale=0.3, size=3 * L)
        starts.append((f"restart_{i}", x))
    return starts


def state_hash(psi) -> str:
    return hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()


def variance(sys, psi) -> float:
    hp = sys.apply_H(psi)
    e = float(np.real(np.vdot(psi, hp)))
    return float(np.real(np.vdot(hp, hp)) - e * e)


def iqr(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if a.size < 2:
        return None
    return float(np.percentile(a, 75) - np.percentile(a, 25))


def run_pool(J, B, packing, tasks, workers=4):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(J, B, packing)) as ex:
        return list(ex.map(_polish, tasks, chunksize=1))


def score_starts(sys, gs, kind, L, rows, E0):
    finished = []
    for row in rows:
        if row.get("E") is None:
            row["F"] = None
            continue
        psi = qaoa_state(sys, row["x"]) if kind == "qaoa" else logical_state(sys, L, row["x"])
        F = fidelity(gs, psi)
        row["F"] = F
        row["energy_error"] = row["E"] - E0
        finished.append(row)
    Es = [r["E"] for r in finished]
    Fs = [r["F"] for r in finished]
    best = min(finished, key=lambda r: r["E"])
    psi = qaoa_state(sys, best["x"]) if kind == "qaoa" else logical_state(sys, L, best["x"])
    running = []
    cur = np.inf
    for e in Es:
        cur = min(cur, e)
        running.append(cur)
    return {
        "E": sys.energy(psi),
        "F": fidelity(gs, psi),
        "energy_error": sys.energy(psi) - E0,
        "var_H": variance(sys, psi),
        "state_hash": state_hash(psi),
        "x": best["x"],
        "best_tag": best["tag"],
        "E_median": float(np.median(Es)),
        "E_iqr": iqr(Es),
        "F_median": float(np.median(Fs)),
        "F_iqr": iqr(Fs),
        "n_starts": len(rows),
        "n_finished": len(finished),
        "nfev_sum": int(sum(r.get("nfev") or 0 for r in rows)),
        "energies": Es,
        "fidelities": Fs,
        "tags": [r["tag"] for r in finished],
        "best_so_far": running,
        "n_err_le_0292": int(sum(e - E0 <= ERR_BAR + 1e-12 for e in Es)),
        "n_F_ge_0610": int(sum(f >= F_BAR - 1e-12 for f in Fs)),
        "n_both_pareto": int(sum((e - E0 <= ERR_BAR + 1e-12) and (f >= F_BAR - 1e-12) for e, f in zip(Es, Fs))),
        "n_E_below_source_p12": int(sum(e <= SOURCE_P12_E + 1e-9 for e in Es)),
        "psi": psi,
    }


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(type(obj))


def dump(payload):
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=2, default=_jsonable))


def main():
    import multiprocessing as mp

    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.perf_counter()
    deadline = t0 + 75 * 60
    RESULTS.mkdir(parents=True, exist_ok=True)
    logp = RESULTS / "run.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with logp.open("a") as fh:
            fh.write(line + "\n")

    def left():
        return deadline - time.perf_counter()

    log("direction, algebra, replay hashes")
    algebra_and_directions()
    manifest, _ = load_manifest()
    packings, _ = load_packings()
    J, B = parse_J(manifest, 12)
    packing = packings["sublattice"]["12"]
    sys = System(J, B, packing)
    log(f"rK={sys.rK:.10f} rM={sys.rM:.10f} scaleK={1/sys.rK:.10f} scaleM={1/sys.rM:.10f}")
    t_spec = time.perf_counter()
    dim = sys.dim
    H = np.diag(sys.hc).astype(np.float64)
    idx = np.arange(dim)
    for i in range(12):
        H[idx ^ (1 << i), idx] += B
    evals, evecs = eigh(H)
    gs = evecs[:, 0].astype(np.complex128)
    if np.sum(gs) < 0:
        gs = -gs
    E0 = float(evals[0])
    log(f"E0={E0:.10f} full_gap={float(evals[1]-evals[0]):.6e} spectrum_s={time.perf_counter()-t_spec:.1f}")
    if abs(E0 - (-8.6132024589353)) > 1e-9:
        raise RuntimeError(f"unexpected E0 {E0}")

    replay = {}
    for L, name in ((5, "N12_L5_A_sub_M"), (6, "N12_L6_A_sub_M")):
        cell = json.loads((DATA / f"{name}.json").read_text())
        src = np.load(DATA / f"{name}.npz")["psi"]
        psi = logical_state(sys, L, cell["x"])
        E = sys.energy(psi)
        F = fidelity(gs, psi)
        ov = overlap(psi, src)
        dE = abs(E - cell["E"])
        dF = abs(F - cell["F"])
        digest = state_hash(psi)
        src_digest = state_hash(src)
        aligned = psi * np.exp(-1j * np.angle(np.vdot(src, psi)))
        residual = float(np.max(np.abs(aligned - src)))
        ok = ov >= 1 - 1e-12 and dE < 1e-10 and dF < 1e-10 and src_digest == cell["state_hash"] and residual < 1e-12
        log(f"replay L{L} ov={ov:.16f} dE={dE:.3e} dF={dF:.3e} residual={residual:.3e} var={variance(sys, psi):.6f}")
        if not ok:
            raise RuntimeError(f"replay L{L} failed")
        replay[str(L)] = {
            "E": E, "source_E": cell["E"], "F": F, "source_F": cell["F"], "energy_error": E - E0,
            "var_H": variance(sys, psi), "source_var_H": cell["var_H"], "overlap": ov,
            "dE": dE, "dF": dF, "state_hash": digest, "source_state_hash": src_digest,
            "amplitude_residual": residual, "source_nfev": cell["nfev_sum"],
            "source_n_starts": cell["n_starts"], "source_median": cell["E_median"], "source_iqr": cell["E_iqr"],
        }
    pi = np.arange(12)
    pi[0], pi[3] = 3, 0
    J2 = np.zeros_like(J)
    for i in range(12):
        for j in range(12):
            J2[pi[i], pi[j]] = J[i, j]
    pack2 = [[int(pi[s]) for s in c] for c in packing]
    sys2 = System(J2, B, pack2)
    image = np.zeros(dim, dtype=np.int64)
    for i, dest in enumerate(pi):
        image |= ((np.arange(dim) >> i) & 1) << int(dest)
    gs2 = np.zeros_like(gs)
    gs2[image] = gs
    psi = logical_state(sys, 6, json.loads((DATA / "N12_L6_A_sub_M.json").read_text())["x"])
    psi2 = np.zeros_like(psi)
    psi2[image] = psi
    dE = abs(sys.energy(psi) - sys2.energy(psi2))
    dF = abs(fidelity(gs, psi) - fidelity(gs2, psi2))
    log(f"joint relabel dE={dE:.3e} dF={dF:.3e}")
    if dE > 1e-8 or dF > 1e-8:
        raise RuntimeError("relabel invariance failed")

    notes = [
        "Replay reproduced the stored L5 and L6 states. That match is not a fresh robustness result.",
        "Fresh QAOA and Rfull starts were drawn from new seeds and were not initialized at the stored angles.",
    ]
    rows = []
    qaoa = {}
    payload = {"replay": replay, "rows": rows, "notes": notes, "E0": E0, "qaoa": {}, "rfull": {}, "manifest": MANIFEST_SHA, "packing": PACKING_SHA}

    t_g = time.perf_counter()
    qaoa_energy_grad(sys, np.linspace(0.2, 0.5, 12), np.linspace(0.4, 0.1, 12))
    log(f"one p12 grad ms {(time.perf_counter()-t_g)*1000:.1f}")

    def one_qaoa_wave(p, n, seed, label):
        rng = np.random.default_rng(seed)
        starts = qaoa_starts(p, rng, n)
        tasks = [{"kind": "qaoa", "L": p, "tag": t, "x0": x.tolist(), "maxiter": MAXITER_Q} for t, x in starts]
        t1 = time.perf_counter()
        out = run_pool(J, B, packing, tasks)
        scored = score_starts(sys, gs, "qaoa", p, out, E0)
        psi = scored.pop("psi")
        scored["seconds"] = time.perf_counter() - t1
        scored["seed"] = seed
        scored["wave"] = label
        np.savez(RESULTS / f"qaoa_p{p}_{label}.npz", x=np.asarray(scored["x"], dtype=np.float64), psi=psi, energies=np.asarray(scored["energies"]))
        log(
            f"  {label} E={scored['E']:.6f} err={scored['energy_error']:.4f} F={scored['F']:.4f} "
            f"med={scored['E_median']:.4f} iqr={scored['E_iqr']:.4f} nfev={scored['nfev_sum']} "
            f"both={scored['n_both_pareto']}/{scored['n_finished']}"
        )
        return scored

    for p in (9, 10, 11, 12):
        if left() < 60:
            notes.append(f"stopped before QAOA p={p}")
            break
        log(f"QAOA p={p} wave1 {N_QAOA} starts seed {QAOA_SEED+p}")
        qaoa[p] = {"waves": [one_qaoa_wave(p, N_QAOA, QAOA_SEED + p, "wave1")]}
        payload["qaoa"][str(p)] = qaoa[p]
        dump(payload)

    if all(p in qaoa for p in (9, 10, 11, 12)) and left() > 40 * 60:
        log("second QAOA wave, 192 starts, predetermined because time remained")
        for p in (9, 10, 11, 12):
            if left() < 8 * 60:
                notes.append(f"4x wave stopped before p={p}")
                break
            qaoa[p]["waves"].append(one_qaoa_wave(p, N_QAOA_EXTRA, QAOA_SEED + 1000 + p, "wave2"))
            dump(payload)
    else:
        notes.append("4x QAOA wave was not run. Remaining time after the 64-start wave was under 40 minutes, or an earlier depth was skipped.")

    for p, block in qaoa.items():
        merged = merge_waves(block["waves"])
        merged["method"] = "qaoa_fresh"
        merged["L"] = p
        merged["P"] = 2 * p
        merged["source_nfev"] = SOURCE_NFE[p]
        merged["four_x_nfev_target"] = 4 * SOURCE_NFE[p]
        merged["nfev_over_source"] = merged["nfev_sum"] / SOURCE_NFE[p]
        merged["reached_4x"] = merged["nfev_sum"] >= 4 * SOURCE_NFE[p]
        block["merged"] = merged
        rows.append(merged)
        log(f"QAOA p={p} merged nfev {merged['nfev_sum']} / source {SOURCE_NFE[p]} / 4x {4*SOURCE_NFE[p]} both {merged['n_both_pareto']}")
    payload["notes"] = notes
    dump(payload)

    prev = None
    rfull = {}
    for L in range(1, 7):
        if left() < 40:
            notes.append(f"stopped before fresh Rfull L={L}")
            break
        log(f"fresh Rfull L={L} seed {RFULL_SEED+L}")
        rng = np.random.default_rng(RFULL_SEED + L)
        starts = rfull_starts(L, prev, rng, N_BOOT)
        tasks = [{"kind": "rfull", "L": L, "tag": t, "x0": x.tolist(), "maxiter": MAXITER_R} for t, x in starts]
        t1 = time.perf_counter()
        out = run_pool(J, B, packing, tasks)
        scored = score_starts(sys, gs, "rfull", L, out, E0)
        psi = scored.pop("psi")
        scored.update({"method": "Rfull_fresh", "L": L, "P": 3 + 3 * L, "seconds": time.perf_counter() - t1, "seeded_from_source": False, "seed": RFULL_SEED + L})
        rfull[L] = scored
        prev = np.asarray(scored["x"], dtype=np.float64)
        rows.append({k: v for k, v in scored.items()})
        np.savez(RESULTS / f"rfull_L{L}.npz", x=prev, psi=psi, energies=np.asarray(scored["energies"]))
        log(f"  E={scored['E']:.6f} err={scored['energy_error']:.4f} F={scored['F']:.4f} med={scored['E_median']:.4f} iqr={scored['E_iqr']}")
        payload["rfull"][str(L)] = scored
        dump(payload)

    for L in (4, 5, 6):
        if L not in rfull or left() < 30:
            notes.append(f"skipped extra Rfull L={L}")
            continue
        log(f"extra {N_EXTRA} Rfull L={L} seed {EXTRA_SEED+L}")
        rng = np.random.default_rng(EXTRA_SEED + L)
        base = np.asarray(rfull[L]["x"], dtype=np.float64)
        starts = [(f"extra_near_{i}", base + rng.normal(scale=0.1, size=base.size)) for i in range(N_EXTRA // 2)]
        wide = rfull_starts(L, None, rng, N_EXTRA // 2)
        starts += wide
        tasks = [{"kind": "rfull", "L": L, "tag": t, "x0": np.asarray(x).tolist(), "maxiter": MAXITER_R} for t, x in starts]
        out = run_pool(J, B, packing, tasks)
        scored = score_starts(sys, gs, "rfull", L, out, E0)
        psi = scored.pop("psi")
        adopted = scored["E"] < rfull[L]["E"] - 1e-10
        rec = {"method": "Rfull_extra", "L": L, "adopted_over_fresh12": adopted, "seeded_from_source": False, "seed": EXTRA_SEED + L}
        rec.update({k: scored[k] for k in scored})
        if adopted:
            np.savez(RESULTS / f"rfull_extra_L{L}.npz", x=np.asarray(scored["x"]), psi=psi)
        rows.append(rec)
        log(f"  extra E={scored['E']:.6f} F={scored['F']:.4f} adopted={adopted} med={scored['E_median']:.4f}")
        dump(payload)

    if left() > 3 * 60:
        for label, pack in (
            ("stride4", packings["stride4"]["12"]),
            ("random0", packings["random12"][0]),
            ("random1", packings["random12"][1]),
        ):
            if left() < 40:
                notes.append(f"skipped control {label}")
                break
            log(f"control {label} L=6 seed {CONTROL_SEEDS[label]}")
            rng = np.random.default_rng(CONTROL_SEEDS[label])
            starts = rfull_starts(6, None, rng, N_CONTROL)
            tasks = [{"kind": "rfull", "L": 6, "tag": t, "x0": x.tolist(), "maxiter": MAXITER_R} for t, x in starts]
            out = run_pool(J, B, pack, tasks)
            sys_c = System(J, B, pack)
            scored = score_starts(sys_c, gs, "rfull", 6, out, E0)
            psi = scored.pop("psi")
            rec = {"method": f"control_{label}", "L": 6, "P": 21, "seed": CONTROL_SEEDS[label]}
            rec.update(scored)
            rows.append(rec)
            np.savez(RESULTS / f"control_{label}.npz", x=np.asarray(scored["x"]), psi=psi)
            log(f"  {label} E={scored['E']:.6f} F={scored['F']:.4f} med={scored['E_median']:.4f}")
    else:
        notes.append("controls skipped for time")

    make_figures(qaoa, rfull, rows)
    payload["notes"] = notes
    payload["seconds"] = time.perf_counter() - t0
    payload["rows"] = rows
    dump(payload)
    write_md(replay, rows, notes, E0, qaoa)
    log(f"done {(time.perf_counter()-t0)/60:.1f} min")


def merge_waves(waves):
    Es, Fs, tags, nfev, best = [], [], [], 0, None
    for wave in waves:
        Es.extend(wave["energies"])
        Fs.extend(wave["fidelities"])
        tags.extend(wave["tags"])
        nfev += wave["nfev_sum"]
        if best is None or wave["E"] < best["E"]:
            best = wave
    running = []
    cur = np.inf
    for e in Es:
        cur = min(cur, e)
        running.append(cur)
    return {
        "E": best["E"], "F": best["F"], "energy_error": best["energy_error"], "var_H": best["var_H"],
        "state_hash": best["state_hash"], "x": best["x"], "best_tag": best["best_tag"],
        "E_median": float(np.median(Es)), "E_iqr": iqr(Es), "F_median": float(np.median(Fs)), "F_iqr": iqr(Fs),
        "n_starts": sum(w["n_starts"] for w in waves), "n_finished": sum(w["n_finished"] for w in waves),
        "nfev_sum": nfev, "energies": Es, "fidelities": Fs, "tags": tags, "best_so_far": running,
        "n_err_le_0292": int(sum(e <= SOURCE_L6_E + (ERR_BAR - (SOURCE_L6_E - (-8.6132024589353))) + 1e-9 for e in Es)),
        "n_F_ge_0610": int(sum(f >= F_BAR - 1e-12 for f in Fs)),
        "n_both_pareto": int(sum(((-8.6132024589353) + ERR_BAR + 1e-12 >= e) and (f >= F_BAR - 1e-12) for e, f in zip(Es, Fs))),
        "n_E_below_source_p12": int(sum(e <= SOURCE_P12_E + 1e-9 for e in Es)),
        "seconds": sum(w["seconds"] for w in waves),
    }


def make_figures(qaoa, rfull, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if 12 in qaoa:
        Es = np.sort(qaoa[12]["merged"]["energies"])
        ax.plot(np.linspace(0, 1, len(Es), endpoint=False), Es, label="fresh QAOA p12")
        ax.axhline(SOURCE_P12_E, color="k", ls="--", label="source p12 best")
        ax.axhline(SOURCE_L6_E, color="C3", ls=":", label="source Rfull L6")
    if 6 in rfull:
        Es = np.sort(rfull[6]["energies"])
        ax.plot(np.linspace(0, 1, len(Es), endpoint=False), Es, label="fresh Rfull L6, 12-start arm")
    ax.set_xlabel("empirical CDF")
    ax.set_ylabel("final energy of each start")
    ax.set_title("Seed-energy CDFs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "seed_energy_cdf.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if 12 in qaoa:
        ax.plot(qaoa[12]["merged"]["best_so_far"], label="fresh QAOA p12 best so far")
    if 6 in rfull:
        y = rfull[6]["best_so_far"]
        ax.plot(y, color="C1", marker="o", markevery=max(1, len(y) // 8), label="fresh Rfull L6 best so far")
    ax.axhline(SOURCE_P12_E, color="k", ls="--", label="source p12")
    ax.axhline(SOURCE_L6_E, color="C3", ls=":", label="source Rfull L6")
    ax.set_xlabel("finished start index")
    ax.set_ylabel("best energy so far")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "best_so_far.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter([0.2877949091521419], [0.514718554056666], marker="s", s=60, label="source QAOA p12")
    ax.scatter([0.2917196118876131], [SOURCE_L6_F], marker="D", s=60, label="source Rfull L6")
    ax.axvline(ERR_BAR, color="0.5", ls=":", label="error 0.292")
    ax.axhline(F_BAR, color="0.5", ls="--", label="F 0.610")
    for rec in rows:
        if rec.get("method") in ("qaoa_fresh", "Rfull_fresh", "Rfull_extra") or str(rec.get("method", "")).startswith("control"):
            if "energy_error" in rec and "F" in rec:
                ax.scatter([rec["energy_error"]], [rec["F"]], label=f"{rec['method']} L{rec['L']}")
    ax.set_xlabel("energy error")
    ax.set_ylabel("full-space ground-state fidelity")
    ax.set_title("Energy-selected states")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto_energy_fidelity.png", dpi=140)
    plt.close(fig)


def write_md(replay, rows, notes, E0, qaoa):
    lines = [
        "# Round 4 V — independent sublattice + block-mean replay",
        "",
        f"Manifest `{MANIFEST_SHA}`. Packing `{PACKING_SHA}`. E0 = {E0:.10f}.",
        "",
        "The stored L5 and L6 states were reproduced by a new logical simulator, and the physical packing matched that logical state. Reproducing a stored vector is not a fresh multi-start result.",
        "",
        "| L | replay E | source E | dE | replay F | source F | dF | overlap | amplitude residual |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for L, r in replay.items():
        lines.append(
            f"| {L} | {r['E']:.10f} | {r['source_E']:.10f} | {r['dE']:.2e} | {r['F']:.8f} | {r['source_F']:.8f} | {r['dF']:.2e} | {r['overlap']:.14f} | {r['amplitude_residual']:.2e} |"
        )
    lines += [
        "",
        "Source tradeoff on these same states: QAOA p12 energy error 0.2878 and F 0.5147, against Rfull L6 energy error 0.2917 and F 0.6098. The 0.004 energy edge is a difference between two states, not an uncertainty on either energy. A new energy-selected QAOA state removes the L6 Pareto lead only when its error is at most 0.292 and its fidelity is at least 0.610.",
        "",
        "| method | L | P | E | error | F | Var(H) | median E | IQR E | median F | starts | nfev | err≤0.292 | F≥0.610 | both |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "energy_error" not in r:
            continue
        lines.append(
            f"| {r['method']} | {r['L']} | {r.get('P','')} | {r['E']:.6f} | {r['energy_error']:.4f} | {r['F']:.4f} | {r.get('var_H', float('nan')):.4f} | "
            f"{r['E_median']:.4f} | {r['E_iqr']:.4f} | {r['F_median']:.4f} | {r['n_finished']}/{r['n_starts']} | {r['nfev_sum']} | "
            f"{r['n_err_le_0292']} | {r['n_F_ge_0610']} | {r['n_both_pareto']} |"
        )
    lines += ["", "## QAOA inclusive budget", ""]
    for p, block in qaoa.items():
        m = block["merged"]
        lines.append(
            f"- p={p}: nfev {m['nfev_sum']}, source nfev {m['source_nfev']}, ratio {m['nfev_over_source']:.2f}, 4x target {m['four_x_nfev_target']}, reached_4x={m['reached_4x']}, best tag `{m['best_tag']}`."
        )
    lines += ["", "## Notes", ""]
    for n in notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "Figures: `seed_energy_cdf.png`, `best_so_far.png`, `pareto_energy_fidelity.png`.",
        "Old baseline gates 0.390 (L5) and 0.315 (L6), and fidelities 0.412 and 0.485, are the earlier matched-depth numbers. They are not a substitute for the p12 comparison above.",
        "",
    ]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
