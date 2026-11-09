"""N=9 L=1 two-template transfer. Frozen v4 center-4. Not an unsigned-mean copy."""
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
from Nov3_INIT.n9tmpl import TEST, TRAIN, build_freeze
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_N9TMPL.md"
RES = ROOT / "results" / "n9tmpl"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
VECTORS = RES / "TRANSFER_VECTORS.json"
BAND = list(range(10744, 10760))
HOLD = list(range(10624, 10632))
N = 9
L = 1
Q = 1024
TFIX = 122
N_BOOT = 4000
HISTORICAL = 3080256
CELL_WALL_CAP = 600.0
CUT = list(range(4))
METHODS = (
    ("nov3_center4", "Nov3rd", METHOD_ID["equal4"]["Nov3rd"]),
    ("qaoa_center4", "QAOA", METHOD_ID["equal4"]["QAOA"]),
)
FORBIDDEN = (
    set(range(10432, 10440)) | set(range(10552, 10560)) | set(range(10600, 10616))
    | set(range(10648, 10664)) | set(range(10680, 10696)) | set(range(10536, 10552)) | set(range(10584, 10600))
    | set(range(10504, 10520)) | set(range(10632, 10648)) | set(range(10520, 10536))
    | set(range(10568, 10584)) | set(range(10480, 10488)) | set(range(10456, 10464))
    | set(range(10376, 10384)) | set(range(10368, 10376)) | set(range(10616, 10624)) | set(range(10664, 10680)) | set(range(10696, 10712)) | set(range(10712, 10728)) | set(range(10728, 10744))
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
        raise RuntimeError(f"seed {seed} is not in the n9tmpl band 10744..10759")
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
        raise RuntimeError("n9tmpl conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("n9tmpl seeds already appear under Nov3_INIT/results")
    if packing(N) != [[a, a + 3, a + 6] for a in range(3)]:
        raise RuntimeError("N=9 packing")
    if int(doc.get("T", -1)) != TFIX or int(doc.get("Q", -1)) != Q:
        raise RuntimeError("Q/T mismatch")
    if doc.get("primary_gate") != "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI":
        raise RuntimeError("primary gate must stay two-template")
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
        "n9tmpl_band": True,
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
            print(f"B* N=9 seed={seed} split={r['split']} b={r['Bstar']:.6f} gap={r['gap']:.4g} {r['seconds']:.1f}s", flush=True)


def _logical(seed: int):
    meta = json.loads((INST / f"N{N}_s{seed}.json").read_text())
    gs = np.load(INST / f"N{N}_s{seed}.npz")["gs"]
    if int(meta["N"]) != N or int(meta["seed"]) != int(seed) or gs.size != (1 << N) or meta.get("cut") != CUT:
        raise RuntimeError(f"B* invalid seed {seed}")
    return meta, Logical(N, seed, float(meta["Bstar"]), gs)


def _fit_path(arm: str, method: str, seed: int) -> Path:
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_q{Q}.json"


def _xfer_path(arm: str, method: str, seed: int, agg: str) -> Path:
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_xfer_{agg}_Q1.json"


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
            "version": "N3INIT_n9tmpl",
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
            "primary_gate": "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI",
            "do_not_switch_primary_after_test": True,
            "not_unsigned_mean_copy": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_n9tmpl",
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
    print(f"{'ok' if record['ok'] else 'FAIL'} n9tmpl {split} {arm} N=9 s={seed}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def _one_xfer(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if not VECTORS.exists():
        raise RuntimeError("refusing transfer eval before freeze")
    seed = int(task["seed"])
    allow(seed)
    if seed not in TEST:
        raise RuntimeError(f"transfer eval on non-test seed {seed}")
    arm, method = task["arm"], task["method"]
    agg = task["aggregator"]
    path = _xfer_path(arm, method, seed, agg)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    x = np.asarray(task["x"], dtype=np.float64)
    try:
        meta, logical = _logical(seed)
        e, psi = logical.propagate(method, L, x)
        f = logical.fidelity(psi)
        e0 = float(meta["E0"])
        f_plus = float(meta["F_plus"])
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_n9tmpl",
            "arm": arm,
            "method": method,
            "n_params": n_params(method, L),
            "N": N,
            "seed": seed,
            "L": L,
            "q": None,
            "Q": 1,
            "T": 0,
            "split": "test",
            "role": "transfer_eval",
            "aggregator": agg,
            "tag": agg,
            "secondary": bool(task.get("secondary", False)),
            "Bstar": float(meta["Bstar"]),
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": float(e),
            "E_error_over_N": float((e - e0) / float(N)),
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "queries": {
                "screen": 0,
                "init": 0,
                "calibration": 0,
                "iteration": 0,
                "candidate": 0,
                "monitoring": 0,
                "transfer": 1,
                "total": 1,
            },
            "x": [float(v) for v in x],
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
            "fidelity_posthoc": True,
            "no_gr_patch": True,
            "not_architecture_change": True,
            "primary_gate": "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI",
            "do_not_switch_primary_after_test": True,
            "not_unsigned_mean_copy": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_n9tmpl",
            "arm": arm,
            "method": method,
            "N": N,
            "seed": seed,
            "L": L,
            "Q": 1,
            "split": "test",
            "role": "transfer_eval",
            "aggregator": agg,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} n9tmpl xfer-{agg} {arm} N=9 s={seed}{extra} {record['seconds']:.1f}s", flush=True)
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
    print(f"n9tmpl train pending {len(pending)}", flush=True)
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
        "n_templates": freeze["n_templates"],
        "planned_transfer_queries": freeze["planned_transfer_queries"],
        "nov3_two_template": freeze["nov3_two_template"],
        "qaoa_two_template": freeze["qaoa_two_template"],
        "do_not_switch_primary_after_test": True,
        "n_train": 8,
        "prereg_sha256": prereg_sha(),
    }
    _dump(RES / "STATUS.json", payload)
    print(
        f"n9tmpl froze catalog n_templates={freeze['n_templates']} "
        f"transfer={freeze['planned_transfer_queries']} "
        f"nov3={freeze['nov3_two_template']} qaoa={freeze['qaoa_two_template']}; test untouched",
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
    if freeze.get("primary_gate") != "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI":
        raise RuntimeError("refusing to run test under a non-two-template primary")
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
    xfer_pending = []
    for seed in TEST:
        for item in freeze["catalog"]:
            xfer_pending.append({
                "arm": item["arm"],
                "method": item["method"],
                "seed": seed,
                "aggregator": item["tag"],
                "x": item["x"],
                "secondary": item["secondary"],
                "prereg_sha256": sha,
            })
    xfer_pending = [t for t in xfer_pending if not _xfer_path(t["arm"], t["method"], t["seed"], t["aggregator"]).exists()]
    print(f"n9tmpl test fits {len(fit_pending)} xfer {len(xfer_pending)}", flush=True)
    wall0 = time.perf_counter()
    if fit_pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one_fit, fit_pending, chunksize=1))
    if xfer_pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one_xfer, xfer_pending, chunksize=1))
    (RES / "wall_test.json").write_text(json.dumps({"fit_seconds": time.perf_counter() - wall0, "workers": 4, "blas_threads": 1, "split": "test", "n_fit": len(fit_pending), "n_xfer": len(xfer_pending)}) + "\n")


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


def _better_of(plus: dict[int, dict], minus: dict[int, dict]) -> tuple[dict[int, dict], dict[int, str], bool]:
    out: dict[int, dict] = {}
    chosen: dict[int, str] = {}
    seeds = sorted(set(plus) | set(minus))
    reduced = not (plus and minus)
    for s in seeds:
        cands = []
        if s in plus:
            cands.append(("plus", plus[s]))
        if s in minus:
            cands.append(("minus", minus[s]))
        if not cands:
            continue
        tag, rec = min(cands, key=lambda z: (z[1]["E"], z[0]))
        out[s] = rec
        chosen[s] = tag
    return out, chosen, reduced


def _fmt(ct: dict) -> str:
    d = ct.get("dE")
    if not d:
        return f"n={ct.get('n', 0)} incomplete"
    return (
        f"{d['mean']:+.5f} med {d['median']:+.5f} "
        f"[{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}] "
        f"tails [{d['min']:+.5f},{d['max']:+.5f}] {ct['wins']}/{ct['n']}"
    )


def _conc_line(block: dict, title: str) -> str:
    if block.get("empty") or "mean" not in block:
        return f"{title} EMPTY n=0 (no invented template)"
    parts = [f"{title} n={block['n']}"]
    for i, name in enumerate(block["names"]):
        parts.append(
            f"{name} mean {block['mean'][i]:+.4f} std {block['std'][i]:.4f} "
            f"range [{block['min'][i]:+.4f},{block['max'][i]:+.4f}]"
        )
    return "; ".join(parts)


def summarize() -> None:
    if not VECTORS.exists():
        raise RuntimeError("freeze missing")
    freeze = json.loads(VECTORS.read_text())
    rows = [json.loads(p.read_text()) for p in sorted(CELLS.glob("*.json"))]
    fits = [r for r in rows if r.get("role") == "independent_fit"]
    xfers = [r for r in rows if r.get("role") == "transfer_eval"]
    fails = [r for r in rows if not r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N9_s*.json")]
    train_n = _index(fits, "independent_fit", "train", "nov3_center4")
    train_q = _index(fits, "independent_fit", "train", "qaoa_center4")
    test_n = _index(fits, "independent_fit", "test", "nov3_center4")
    test_q = _index(fits, "independent_fit", "test", "qaoa_center4")
    n_plus = _index(xfers, "transfer_eval", "test", "nov3_center4", "cluster_plus")
    n_minus = _index(xfers, "transfer_eval", "test", "nov3_center4", "cluster_minus")
    q_plus = _index(xfers, "transfer_eval", "test", "qaoa_center4", "cluster_plus")
    q_minus = _index(xfers, "transfer_eval", "test", "qaoa_center4", "cluster_minus")
    n_unsigned = _index(xfers, "transfer_eval", "test", "nov3_center4", "unsigned_mean")
    n_aligned = _index(xfers, "transfer_eval", "test", "nov3_center4", "sign_aligned_mean")
    q_unsigned = _index(xfers, "transfer_eval", "test", "qaoa_center4", "unsigned_mean")
    better_n, chosen_n, reduced_n = _better_of(n_plus, n_minus)
    better_q, chosen_q, reduced_q = _better_of(q_plus, q_minus)
    ct_primary = _contrast(better_n, better_q)
    ct_train = _contrast(train_n, train_q)
    ct_ceil = _contrast(test_n, test_q)
    ct_vs_ind_q = _contrast(better_n, test_q)
    ct_gap = _contrast(better_n, test_n)
    ct_plus_vs_u = _contrast(n_plus, q_unsigned)
    ct_plus_vs_ind = _contrast(n_plus, test_q)
    ct_minus_vs_u = _contrast(n_minus, q_unsigned)
    ct_minus_vs_ind = _contrast(n_minus, test_q)
    ct_unsigned = _contrast(n_unsigned, q_unsigned)
    ct_aligned = _contrast(n_aligned, q_unsigned)
    n_xfer_expected = int(freeze.get("planned_transfer_queries", 0))
    complete = (
        not fails
        and ct_train["n"] == 8
        and ct_ceil["n"] == 8
        and ct_primary["n"] == 8
        and len(fits) == 32
        and len(xfers) == n_xfer_expected
        and freeze.get("primary_gate") == "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI"
    )
    if complete and ct_primary["ci_entirely_lt0"]:
        label = "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI"
    else:
        label = "UNSTABLE"
    new_q = int(sum(int((r.get("queries") or {}).get("total", 0)) for r in rows if r.get("ok")))
    indep_q = int(sum(int((r.get("queries") or {}).get("total", 0)) for r in fits if r.get("ok")))
    xfer_q = int(sum(int((r.get("queries") or {}).get("total", 0)) for r in xfers if r.get("ok")))
    payload = {
        "version": "N3INIT_n9tmpl",
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "not_architecture_change": True,
        "not_gr_retune": True,
        "not_unsigned_mean_copy": True,
        "not_sign_aligned_only_copy": True,
        "holdout_l2_fail_stands": True,
        "compiler_stream_closed": True,
        "pr111_closed": True,
        "pr117_l3_2xbudget_fail_stands": True,
        "pr118_n12_stands": True,
        "pr120_n9_unsigned_unstable_stands": True,
        "primary_gate": "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI",
        "do_not_switch_primary_after_test": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r.get("arm"), "seed": r.get("seed"), "role": r.get("role"), "tag": r.get("aggregator"), "failure": r.get("failure")} for r in fails],
        "new_queries": new_q,
        "independent_queries": indep_q,
        "transfer_queries": xfer_q,
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
        "nov3_two_template": freeze.get("nov3_two_template"),
        "qaoa_two_template": freeze.get("qaoa_two_template"),
        "nov3_reduces_to_single_cluster": reduced_n,
        "qaoa_reduces_to_single_cluster": reduced_q,
        "chosen_nov3": {str(k): v for k, v in chosen_n.items()},
        "chosen_qaoa": {str(k): v for k, v in chosen_q.items()},
        "contrast_primary_two_template": ct_primary,
        "contrast_train": ct_train,
        "contrast_test_ceiling": ct_ceil,
        "contrast_two_template_nov3_vs_ind_qaoa": ct_vs_ind_q,
        "contrast_transfer_gap": ct_gap,
        "contrast_cluster_plus_vs_unsigned_qaoa": ct_plus_vs_u,
        "contrast_cluster_plus_vs_ind_qaoa": ct_plus_vs_ind,
        "contrast_cluster_minus_vs_unsigned_qaoa": ct_minus_vs_u,
        "contrast_cluster_minus_vs_ind_qaoa": ct_minus_vs_ind,
        "contrast_unsigned_replication": ct_unsigned,
        "contrast_sign_aligned_replication": ct_aligned,
        "concentration_train_nov3_unsigned": freeze.get("nov3_unsigned"),
        "concentration_train_qaoa_unsigned": freeze.get("qaoa_unsigned"),
        "nov3_cluster_plus": freeze.get("nov3_cluster_plus"),
        "nov3_cluster_minus": freeze.get("nov3_cluster_minus"),
        "qaoa_cluster_plus": freeze.get("qaoa_cluster_plus"),
        "qaoa_cluster_minus": freeze.get("qaoa_cluster_minus"),
        "held_unused": HOLD,
        "train": list(TRAIN),
        "test": list(TEST),
    }
    _dump(RES / "summary.json", payload)
    n_u = freeze["nov3_unsigned"]
    q_u = freeze["qaoa_unsigned"]
    s1 = [v[3] for v in n_u["vectors"]]
    beta = [v[1] for v in q_u["vectors"]]
    lines = [
        "# N=9 L=1 two-template (bimodal cluster-mean) transfer",
        "",
        "Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. N=9 L=1 Q=1024 T=122. Train 10744–10751, test 10752–10759. Cluster means frozen from train before test. Primary is better-of-two min of frozen cluster templates. Not unsigned-mean copy. Not sign-aligned-only copy. Not architecture. No GR. Not L=2/L=3.",
        "",
        f"Fits {len(fits)} ok {sum(1 for r in fits if r.get('ok'))}. Transfer evals {len(xfers)} ok {sum(1 for r in xfers if r.get('ok'))}. Charged {new_q} (independent {indep_q} + transfer {xfer_q}). Failures {len(fails)}.",
        f"Nov3 clusters: plus n={freeze['nov3_cluster_plus'].get('n', 0)} minus n={freeze['nov3_cluster_minus'].get('n', 0)} two_template={freeze['nov3_two_template']}. QAOA clusters: plus n={freeze['qaoa_cluster_plus'].get('n', 0)} minus n={freeze['qaoa_cluster_minus'].get('n', 0)} two_template={freeze['qaoa_two_template']}.",
        f"Nov3 reduces to single cluster: {reduced_n}. QAOA reduces to single cluster: {reduced_q}.",
        "",
        f"**Label {label}.**",
        "",
        f"Primary two-template Nov3−QAOA: {_fmt(ct_primary)}",
        f"1 train independent Nov3−QAOA: {_fmt(ct_train)}",
        f"2 test ceiling independent Nov3−QAOA: {_fmt(ct_ceil)}",
        f"3 two-template Nov3 vs independent test QAOA: {_fmt(ct_vs_ind_q)}",
        f"4 transfer gap two-template Nov3 vs independent Nov3: {_fmt(ct_gap)}",
        f"5a cluster+ Nov3 vs unsigned-mean QAOA: {_fmt(ct_plus_vs_u)}",
        f"5b cluster+ Nov3 vs independent QAOA: {_fmt(ct_plus_vs_ind)}",
        f"5c cluster− Nov3 vs unsigned-mean QAOA: {_fmt(ct_minus_vs_u)}",
        f"5d cluster− Nov3 vs independent QAOA: {_fmt(ct_minus_vs_ind)}",
        f"6a unsigned-mean Nov3 vs unsigned-mean QAOA (replication only): {_fmt(ct_unsigned)}",
        f"6b sign-aligned-mean Nov3 vs unsigned-mean QAOA (replication only): {_fmt(ct_aligned)}",
        "",
        _conc_line(n_u, "Train Nov3 unsigned"),
        _conc_line(freeze["nov3_cluster_plus"], "Train Nov3 cluster+"),
        _conc_line(freeze["nov3_cluster_minus"], "Train Nov3 cluster−"),
        _conc_line(q_u, "Train QAOA unsigned"),
        _conc_line(freeze["qaoa_cluster_plus"], "Train QAOA cluster+"),
        _conc_line(freeze["qaoa_cluster_minus"], "Train QAOA cluster−"),
        "",
        f"Train s1 n_neg {sum(1 for v in s1 if v < 0)}/8 n_pos {sum(1 for v in s1 if v > 0)}/8 n_zero {sum(1 for v in s1 if v == 0)}/8 std {n_u['std'][3]:.4f}.",
        f"Train QAOA β n_neg {sum(1 for v in beta if v < 0)}/8 n_pos {sum(1 for v in beta if v > 0)}/8 std {q_u['std'][1]:.4f}.",
        f"Chosen Nov3 templates: {chosen_n}. Chosen QAOA templates: {chosen_q}.",
        "",
        f"New energy queries {new_q}. Historical {HISTORICAL}. Total {HISTORICAL + new_q}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s.",
        "",
        "Unsigned-mean and sign-aligned-mean are secondary only; do not reopen the unsigned-mean ladder. L=2 holdout FAIL stands. PR111 closed. PR117 L3_2XBUDGET_FAIL stands; no q=4096. PR118 and PR120 stand. Exploratory; not adoption.",
        "",
    ]
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r.get('role')} {r.get('arm')} {r.get('aggregator')} s={r.get('seed')}: {r.get('failure')}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Independent ledger equals Q=1024. Transfer evals charge 1 query each.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "train", "freeze", "test", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("n9tmpl preregistration missing")
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
