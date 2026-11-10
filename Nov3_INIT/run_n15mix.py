"""N=15 L=1 mixer-angle freeze. Frozen v4 center-4. Not a two-template copy."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import METHOD_ID, make_rng, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.bstar import maximize
from Nov3_INIT.circuit import Logical
from Nov3_INIT.n15mix import MIX_METHOD_ID, TEST, TRAIN, build_freeze, run_mix_equal4
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_N15MIX.md"
RES = ROOT / "results" / "n15mix"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
VECTORS = RES / "MIXER_VECTORS.json"
BAND = list(range(10888, 10904))
HOLD = list(range(10624, 10632))
N = 15
L = 1
Q = 1024
TFIX = 122
N_BOOT = 4000
HISTORICAL = 3211552
CELL_WALL_CAP = 600.0
CUT = list(range(7))
METHODS = (
    ("nov3_center4", "Nov3rd", METHOD_ID["equal4"]["Nov3rd"]),
    ("qaoa_center4", "QAOA", METHOD_ID["equal4"]["QAOA"]),
)
FORBIDDEN = (
    set(range(10432, 10440)) | set(range(10552, 10560)) | set(range(10600, 10616))
    | set(range(10648, 10664)) | set(range(10680, 10696)) | set(range(10536, 10552)) | set(range(10584, 10600))
    | set(range(10504, 10520)) | set(range(10632, 10648)) | set(range(10520, 10536))
    | set(range(10568, 10584)) | set(range(10480, 10488)) | set(range(10456, 10464))
    | set(range(10376, 10384)) | set(range(10368, 10376)) | set(range(10616, 10624)) | set(range(10664, 10680)) | set(range(10696, 10712)) | set(range(10712, 10728)) | set(range(10728, 10744)) | set(range(10744, 10760)) | set(range(10760, 10776)) | set(range(10776, 10792)) | set(range(10792, 10808)) | set(range(10808, 10824)) | set(range(10824, 10840)) | set(range(10840, 10856)) | set(range(10856, 10872)) | set(range(10872, 10888))
    | set(HOLD) | set(range(10560, 10568)) | set(range(10144, 10152))
    | set(range(11000, 11032)) | set(range(20000, 21000))
)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed in HOLD:
        raise RuntimeError(f"held-unused seed {seed}")
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the n15mix band 10888..10903")
    if seed in FORBIDDEN:
        raise RuntimeError(f"reserved seed {seed}")


def split_of(seed: int) -> str:
    seed = int(seed)
    if seed in TRAIN:
        return "train"
    if seed in TEST:
        return "test"
    raise RuntimeError(f"seed {seed} is not train or test")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("n15mix conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("n15mix seeds already appear under Nov3_INIT/results")
    if packing(N) != [[a, a + 5, a + 10] for a in range(5)]:
        raise RuntimeError("N=15 packing")
    if int(doc.get("T", -1)) != TFIX or int(doc.get("Q", -1)) != Q:
        raise RuntimeError("Q/T mismatch")
    if doc.get("primary_gate") != "MIXER_FREEZE_NEGATIVE_CI":
        raise RuntimeError("primary gate must stay mixer-freeze")
    return doc


def _bstar_worker(seed: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    allow(int(seed))
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    rec = maximize(make_J(int(seed), N), N)
    gs = rec.pop("gs")
    rec.pop("hc")
    rec.update({
        "N": N,
        "seed": int(seed),
        "split": split_of(seed),
        "seconds": time.perf_counter() - t0,
        "cpu_seconds": time.process_time() - cpu0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "prereg_sha256": prereg_sha(),
        "classical_not_a_quantum_query": True,
        "n15mix_band": True,
    })
    if rec.get("cut") != CUT:
        raise RuntimeError(f"unexpected cut {rec.get('cut')}")
    return {"rec": rec, "gs": gs}


def build_instances() -> None:
    _conf()
    pending = []
    for s in BAND:
        allow(s)
        if not (INST / f"N{N}_s{s}.json").exists():
            pending.append(s)
    if not pending:
        return
    import multiprocessing as mp

    INST.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(4) as pool:
        for got in pool.imap_unordered(_bstar_worker, pending, chunksize=1):
            seed = got["rec"]["seed"]
            np.savez(INST / f"N{N}_s{seed}.npz", gs=got["gs"])
            _dump(INST / f"N{N}_s{seed}.json", got["rec"])
            r = got["rec"]
            print(f"B* N=15 seed={seed} split={r['split']} b={r['Bstar']:.6f} gap={r['gap']:.4g} {r['seconds']:.1f}s", flush=True)


def _logical(seed: int):
    meta = json.loads((INST / f"N{N}_s{seed}.json").read_text())
    gs = np.load(INST / f"N{N}_s{seed}.npz")["gs"]
    if int(meta["N"]) != N or int(meta["seed"]) != int(seed) or gs.size != (1 << N) or meta.get("cut") != CUT:
        raise RuntimeError(f"B* invalid seed {seed}")
    return meta, Logical(N, seed, float(meta["Bstar"]), gs)


def _fit_path(arm: str, method: str, seed: int) -> Path:
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_q{Q}.json"


def _mix_path(seed: int) -> Path:
    return CELLS / f"nov3_mixfreeze_Nov3rd_N{N}_s{seed}_L{L}_q{Q}.json"


def _test_hits() -> list[Path]:
    hits = []
    if not CELLS.exists():
        return hits
    for seed in TEST:
        hits.extend(sorted(CELLS.glob(f"*s{seed}*")))
    return hits


def _one_fit(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method, mid = task["arm"], task["method"], int(task["method_id"])
    seed = int(task["seed"])
    split = split_of(seed)
    allow(seed)
    if split == "test" and not VECTORS.exists():
        raise RuntimeError("refusing test fit before transfer vectors are frozen")
    if split == "train" and seed not in TRAIN:
        raise RuntimeError("train task with non-train seed")
    path = _fit_path(arm, method, seed)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(seed)
        b = float(meta["Bstar"])
        rng_init = [None if s == 0 else make_rng(seed, N, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, N, b, L, mid, s, 0) for s in range(4)]

        def draw(method_name, ell, start_id, rng, _method=method):
            if _method == "QAOA":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _log=logical, _method=method, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _log.propagate(_method, L, x)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        e0 = float(meta["E0"])
        f_plus = float(meta["F_plus"])
        queries = dict(out["queries"])
        queries["transfer"] = 0
        if int(out["n_iter"]) != TFIX or int(queries["total"]) != Q:
            raise RuntimeError(f"ledger/T mismatch {out['n_iter']} {queries}")
        if queries["init"] != 4 or queries["calibration"] != 32 or queries["iteration"] != 976 or queries["candidate"] != 12:
            raise RuntimeError(f"query cats {queries}")
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_n15mix",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "n_params": n_params(method, L),
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q,
            "Q": Q,
            "T": int(out["n_iter"]),
            "split": split,
            "role": "independent_fit",
            "procedural_cap": Q,
            "same_Q_both_methods": True,
            "Bstar": b,
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / float(N)),
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "tails": out["tails"],
            "queries": queries,
            "x": [float(v) for v in out["x"]],
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
            "fidelity_posthoc": True,
            "no_gr_patch": True,
            "not_architecture_change": True,
            "primary_gate": "MIXER_FREEZE_NEGATIVE_CI",
            "do_not_switch_frozen_coords_after_test": True,
            "not_unsigned_mean_copy": True,
            "not_two_template_copy": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_n15mix",
            "arm": arm,
            "method": method,
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q,
            "Q": Q,
            "split": split,
            "role": "independent_fit",
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} n15mix {split} {arm} N=15 s={seed}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def _one_mix(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if not VECTORS.exists():
        raise RuntimeError("refusing mixer-freeze fit before freeze")
    seed = int(task["seed"])
    allow(seed)
    if seed not in TEST:
        raise RuntimeError(f"mixer-freeze fit on non-test seed {seed}")
    path = _mix_path(seed)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    theta_k = float(task["theta_K"])
    theta_m = float(task["theta_M"])
    try:
        meta, logical = _logical(seed)
        b = float(meta["Bstar"])
        rng_init = [None if s == 0 else make_rng(seed, N, b, L, MIX_METHOD_ID, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, N, b, L, MIX_METHOD_ID, s, 0) for s in range(4)]

        def propagate_full(x, _log=logical, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _log.propagate("Nov3rd", L, x)

        out = run_mix_equal4(propagate_full, theta_k, theta_m, Q, rng_init, rng_spsa)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        e0 = float(meta["E0"])
        f_plus = float(meta["F_plus"])
        queries = dict(out["queries"])
        queries["transfer"] = 0
        if int(out["n_iter"]) != TFIX or int(queries["total"]) != Q:
            raise RuntimeError(f"mix ledger/T mismatch {out['n_iter']} {queries}")
        if queries["init"] != 4 or queries["calibration"] != 32 or queries["iteration"] != 976 or queries["candidate"] != 12:
            raise RuntimeError(f"mix query cats {queries}")
        x = np.asarray(out["x"], dtype=np.float64)
        if abs(x[1] - theta_k) > 1e-12 or abs(x[2] - theta_m) > 1e-12:
            raise RuntimeError(f"mixer coords drifted {x}")
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_n15mix",
            "arm": "nov3_mixfreeze",
            "method": "Nov3rd",
            "method_id": MIX_METHOD_ID,
            "n_params": 4,
            "n_params_full": 6,
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q,
            "Q": Q,
            "T": int(out["n_iter"]),
            "split": "test",
            "role": "mix_freeze_fit",
            "procedural_cap": Q,
            "same_Q_both_methods": True,
            "Bstar": b,
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / float(N)),
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "tails": out["tails"],
            "queries": queries,
            "x": [float(v) for v in x],
            "x_free": [float(v) for v in out["x_free"]],
            "theta_K_frozen": theta_k,
            "theta_M_frozen": theta_m,
            "mixer_locked": True,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
            "fidelity_posthoc": True,
            "no_gr_patch": True,
            "not_architecture_change": True,
            "primary_gate": "MIXER_FREEZE_NEGATIVE_CI",
            "do_not_switch_frozen_coords_after_test": True,
            "not_unsigned_mean_copy": True,
            "not_two_template_copy": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_n15mix",
            "arm": "nov3_mixfreeze",
            "method": "Nov3rd",
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q,
            "Q": Q,
            "split": "test",
            "role": "mix_freeze_fit",
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} n15mix mixfreeze N=15 s={seed}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_train() -> None:
    import multiprocessing as mp

    _conf()
    if VECTORS.exists():
        raise RuntimeError("freeze already written; will not refit train under a locked freeze")
    if _test_hits():
        raise RuntimeError("test artifacts exist before train freeze")
    sha = prereg_sha()
    pending = []
    for seed in TRAIN:
        for arm, method, mid in METHODS:
            if not _fit_path(arm, method, seed).exists():
                pending.append({
                    "arm": arm,
                    "method": method,
                    "method_id": mid,
                    "seed": seed,
                    "prereg_sha256": sha,
                })
    print(f"n15mix train pending {len(pending)}", flush=True)
    wall0 = time.perf_counter()
    if pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one_fit, pending, chunksize=1))
    (RES / "wall_train.json").write_text(json.dumps({"fit_seconds": time.perf_counter() - wall0, "workers": 4, "blas_threads": 1, "split": "train"}) + "\n")


def freeze_vectors() -> None:
    _conf()
    if VECTORS.exists():
        raise RuntimeError(f"freeze already written: {VECTORS}")
    hits = _test_hits()
    if hits:
        raise RuntimeError(f"test artifacts exist before freeze: {hits[:3]}")
    rows = []
    for seed in TRAIN:
        for arm, method, _mid in METHODS:
            path = _fit_path(arm, method, seed)
            if not path.exists():
                raise RuntimeError(f"missing train cell {path.name}")
            rec = json.loads(path.read_text())
            if not rec.get("ok") or rec.get("split") != "train":
                raise RuntimeError(f"bad train cell {path.name}")
            rows.append(rec)
    freeze = build_freeze(rows, prereg_sha())
    _dump(VECTORS, freeze)
    payload = {
        "freeze_written": True,
        "freeze_sha256": hashlib.sha256(VECTORS.read_bytes()).hexdigest(),
        "primary_gate": freeze["primary_gate"],
        "frozen_names": freeze["frozen_names"],
        "theta_K_mean": freeze["theta_K_mean"],
        "theta_M_mean": freeze["theta_M_mean"],
        "do_not_switch_frozen_coords_after_test": True,
        "n_train": 8,
        "prereg_sha256": prereg_sha(),
        "not_two_template_copy": True,
    }
    _dump(RES / "STATUS.json", payload)
    print(
        f"n15mix froze mixer theta_K={freeze['theta_K_mean']:+.6f} "
        f"theta_M={freeze['theta_M_mean']:+.6f} stdK={freeze['theta_K_std']:.4f} "
        f"stdM={freeze['theta_M_std']:.4f}; test untouched",
        flush=True,
    )


def run_test() -> None:
    import multiprocessing as mp

    _conf()
    if not VECTORS.exists():
        raise RuntimeError("freeze missing; will not run test")
    freeze = json.loads(VECTORS.read_text())
    if freeze.get("prereg_sha256") != prereg_sha():
        raise RuntimeError("freeze prereg hash mismatch")
    if freeze.get("primary_gate") != "MIXER_FREEZE_NEGATIVE_CI":
        raise RuntimeError("refusing to run test under a non-mixer-freeze primary")
    if freeze.get("frozen_names") != ["theta_K", "theta_M"]:
        raise RuntimeError("frozen coords drifted")
    sha = prereg_sha()
    fit_pending = []
    for seed in TEST:
        for arm, method, mid in METHODS:
            if not _fit_path(arm, method, seed).exists():
                fit_pending.append({
                    "arm": arm,
                    "method": method,
                    "method_id": mid,
                    "seed": seed,
                    "prereg_sha256": sha,
                })
    mix_pending = []
    for seed in TEST:
        if not _mix_path(seed).exists():
            mix_pending.append({
                "seed": seed,
                "theta_K": freeze["theta_K_mean"],
                "theta_M": freeze["theta_M_mean"],
                "prereg_sha256": sha,
            })
    print(f"n15mix test fits {len(fit_pending)} mix {len(mix_pending)}", flush=True)
    wall0 = time.perf_counter()
    if fit_pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one_fit, fit_pending, chunksize=1))
    if mix_pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one_mix, mix_pending, chunksize=1))
    (RES / "wall_test.json").write_text(json.dumps({
        "fit_seconds": time.perf_counter() - wall0,
        "workers": 4,
        "blas_threads": 1,
        "split": "test",
        "n_fit": len(fit_pending),
        "n_mix": len(mix_pending),
    }) + "\n")


def _boot(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    rng = np.random.default_rng(3334)
    draws = rng.integers(0, n, size=(N_BOOT, n))
    means = values[draws].mean(axis=1)
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_neg": int(np.sum(values < 0.0)),
        "n_pos": int(np.sum(values > 0.0)),
        "values": [float(v) for v in values],
    }


def _index(rows: list[dict], role: str, split: str | None, arm: str, agg: str | None = None) -> dict[int, dict]:
    out = {}
    for r in rows:
        if not r.get("ok") or r.get("role") != role or r.get("arm") != arm:
            continue
        if split is not None and r.get("split") != split:
            continue
        if agg is not None and r.get("aggregator") != agg:
            continue
        out[int(r["seed"])] = r
    return out


def _contrast(a: dict[int, dict], b: dict[int, dict]) -> dict:
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return {"n": 0, "seeds": [], "dE": None, "dF": None, "errN": None, "wins": 0, "ci_entirely_lt0": False}
    e_a = np.array([a[s]["E"] for s in seeds])
    e_b = np.array([b[s]["E"] for s in seeds])
    f_a = np.array([a[s]["F"] for s in seeds])
    f_b = np.array([b[s]["F"] for s in seeds])
    err_a = np.array([a[s]["E_error_over_N"] for s in seeds])
    err_b = np.array([b[s]["E_error_over_N"] for s in seeds])
    d = _boot(e_a - e_b)
    return {
        "n": len(seeds),
        "seeds": seeds,
        "dE": d,
        "dF": _boot(f_a - f_b),
        "errN": _boot(err_a - err_b),
        "wins": int(np.sum(e_a < e_b)),
        "ci_entirely_lt0": bool(d["n"] == 8 and d["ci95"][1] < 0.0),
        "E_a": _boot(e_a),
        "E_b": _boot(e_b),
        "F_a": _boot(f_a),
        "F_b": _boot(f_b),
    }


def _fmt(ct: dict) -> str:
    d = ct.get("dE")
    if not d:
        return f"n={ct.get('n', 0)} incomplete"
    return (
        f"{d['mean']:+.5f} med {d['median']:+.5f} "
        f"[{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}] "
        f"tails [{d['min']:+.5f},{d['max']:+.5f}] {ct['wins']}/{ct['n']}"
    )


def summarize() -> None:
    if not VECTORS.exists():
        raise RuntimeError("freeze missing")
    freeze = json.loads(VECTORS.read_text())
    rows = [json.loads(p.read_text()) for p in sorted(CELLS.glob("*.json"))]
    fits = [r for r in rows if r.get("role") == "independent_fit"]
    mixes = [r for r in rows if r.get("role") == "mix_freeze_fit"]
    fails = [r for r in rows if not r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    train_n = _index(fits, "independent_fit", "train", "nov3_center4")
    train_q = _index(fits, "independent_fit", "train", "qaoa_center4")
    test_n = _index(fits, "independent_fit", "test", "nov3_center4")
    test_q = _index(fits, "independent_fit", "test", "qaoa_center4")
    mix_n = _index(mixes, "mix_freeze_fit", "test", "nov3_mixfreeze")
    ct_primary = _contrast(mix_n, test_q)
    ct_train = _contrast(train_n, train_q)
    ct_ceil = _contrast(test_n, test_q)
    ct_gap = _contrast(mix_n, test_n)
    complete = (
        not fails
        and ct_train["n"] == 8
        and ct_ceil["n"] == 8
        and ct_primary["n"] == 8
        and ct_gap["n"] == 8
        and len(fits) == 32
        and len(mixes) == 8
        and freeze.get("primary_gate") == "MIXER_FREEZE_NEGATIVE_CI"
        and freeze.get("frozen_names") == ["theta_K", "theta_M"]
    )
    if complete and ct_primary["ci_entirely_lt0"]:
        label = "MIXER_FREEZE_NEGATIVE_CI"
    else:
        label = "UNSTABLE"
    new_q = int(sum(int((r.get("queries") or {}).get("total", 0)) for r in rows if r.get("ok")))
    indep_q = int(sum(int((r.get("queries") or {}).get("total", 0)) for r in fits if r.get("ok")))
    mix_q = int(sum(int((r.get("queries") or {}).get("total", 0)) for r in mixes if r.get("ok")))
    payload = {
        "version": "N3INIT_n15mix",
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "not_architecture_change": True,
        "not_gr_retune": True,
        "not_unsigned_mean_copy": True,
        "not_sign_aligned_only_copy": True,
        "not_two_template_copy": True,
        "holdout_l2_fail_stands": True,
        "compiler_stream_closed": True,
        "pr111_closed": True,
        "pr117_l3_2xbudget_fail_stands": True,
        "pr118_n12_stands": True,
        "pr120_n9_unsigned_unstable_stands": True,
        "pr122_n9_two_template_stands": True,
        "pr124_n12_two_template_stands": True,
        "pr127_n18_two_template_stands": True,
        "pr128_n15_two_template_stands": True,
        "primary_gate": "MIXER_FREEZE_NEGATIVE_CI",
        "do_not_switch_frozen_coords_after_test": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r.get("arm"), "seed": r.get("seed"), "role": r.get("role"), "failure": r.get("failure")} for r in fails],
        "new_queries": new_q,
        "independent_queries": indep_q,
        "mix_queries": mix_q,
        "historical_queries_spent_once": HISTORICAL,
        "total_research_queries": HISTORICAL + new_q,
        "classical_bstar": {
            "n": len(inst),
            "wall_seconds": float(sum(r["seconds"] for r in inst)),
            "cpu_seconds": float(sum(r.get("cpu_seconds", 0.0) for r in inst)),
            "peak_rss_kib_max": int(max((r.get("peak_rss_kib", 0) for r in inst), default=0)),
            "not_a_quantum_query": True,
        },
        "label": label,
        "complete": complete,
        "theta_K_mean": freeze.get("theta_K_mean"),
        "theta_M_mean": freeze.get("theta_M_mean"),
        "theta_K_std": freeze.get("theta_K_std"),
        "theta_M_std": freeze.get("theta_M_std"),
        "contrast_primary_mixfreeze": ct_primary,
        "contrast_train": ct_train,
        "contrast_test_ceiling": ct_ceil,
        "contrast_constraint_gap": ct_gap,
        "held_unused": HOLD,
        "train": list(TRAIN),
        "test": list(TEST),
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N=15 L=1 mixer-angle freeze",
        "",
        "Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=15 L=1 Q=1024 T=122. Train 10888–10895, test 10896–10903. Mixer angles θ_K, θ_M frozen from train mean before test. Test mixer-freeze is 4-D SPSA on (γ,s1,s2,s3). Not a two-template copy. Not unsigned-mean copy. Not architecture. No GR. Not L=2/L=3.",
        "",
        f"Fits {len(fits)} ok {sum(1 for r in fits if r.get('ok'))}. Mixer-freeze fits {len(mixes)} ok {sum(1 for r in mixes if r.get('ok'))}. Charged {new_q} (independent {indep_q} + mix {mix_q}). Failures {len(fails)}.",
        f"Frozen mixer θ_K mean {freeze['theta_K_mean']:+.6f} std {freeze['theta_K_std']:.4f}; θ_M mean {freeze['theta_M_mean']:+.6f} std {freeze['theta_M_std']:.4f}.",
        "",
        f"**Label {label}.**",
        "",
        f"Primary mixer-freeze Nov3−QAOA: {_fmt(ct_primary)}",
        f"1 train independent Nov3−QAOA: {_fmt(ct_train)}",
        f"2 test ceiling independent Nov3−QAOA: {_fmt(ct_ceil)}",
        f"3 constraint gap mixer-freeze vs independent Nov3: {_fmt(ct_gap)}",
        "",
        f"New energy queries {new_q}. Historical {HISTORICAL}. Total {HISTORICAL + new_q}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s.",
        "",
        "Exploratory; not adoption; does not replace per-instance SPSA on the full 6-vector. Two-template ladder is closed. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118, PR120, PR122, PR124, PR127, PR128 stand.",
        "",
    ]
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r.get('role')} {r.get('arm')} s={r.get('seed')}: {r.get('failure')}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Independent and mixer-freeze ledgers equal Q=1024.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "train", "freeze", "test", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("n15mix preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "train":
        run_train()
    elif args.phase == "freeze":
        freeze_vectors()
    elif args.phase == "test":
        run_test()
    else:
        summarize()


if __name__ == "__main__":
    main()
