"""v1 train, freeze, and test. Refuses to fit before the preregistration file is in place."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import METHOD_ID, make_rng, run_equal4, run_halve
from Nov3_INIT.bstar import maximize
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import make_J

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V1.md"
RES = ROOT / "results" / "v1"
CELLS = RES / "cells"
INST = RES / "instances"
TRAIN = [10018, 10019, 10020, 10021]
TEST = [10022, 10023, 10024, 10025]
NS = (9, 12)
LS = (1, 3)
QS = (128, 512, 1024)
ARMS = ("equal4", "halve8to2")
METHODS = ("Nov3rd", "QAOA")
BOOT_SEED = 3334
N_BOOT = 4000


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")
    if seed not in TRAIN and seed not in TEST:
        raise RuntimeError(f"seed {seed} is outside the disjoint dev panel 10018..10025")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def build_instances() -> None:
    sha = prereg_sha()
    for n in NS:
        for seed in TRAIN + TEST:
            allow(seed)
            dest = INST / f"N{n}_s{seed}.json"
            npz = INST / f"N{n}_s{seed}.npz"
            if dest.exists() and npz.exists():
                continue
            t0 = time.perf_counter()
            rec = maximize(make_J(seed, n), n)
            gs = rec.pop("gs")
            rec.pop("hc")
            rec.update({
                "N": n,
                "seed": seed,
                "seconds": time.perf_counter() - t0,
                "prereg_sha256": sha,
                "classical_not_a_quantum_query": True,
            })
            INST.mkdir(parents=True, exist_ok=True)
            np.savez(npz, gs=gs)
            _dump(dest, rec)
            print(
                f"B* N={n} seed={seed} b={rec['Bstar']:.6f} S={rec['S_A']:.4f} "
                f"gap={rec['gap']:.4g} even={rec['delta_even']:.4f} F+={rec['F_plus']:.3f} "
                f"{rec['seconds']:.1f}s",
                flush=True,
            )


def _load(n: int, seed: int):
    meta = json.loads((INST / f"N{n}_s{seed}.json").read_text())
    gs = np.load(INST / f"N{n}_s{seed}.npz")["gs"]
    return meta, Logical(n, seed, meta["Bstar"], gs)


def _cell_path(split: str, arm: str, method: str, n: int, seed: int, L: int, q: int) -> Path:
    return CELLS / f"{split}_{arm}_{method}_N{n}_s{seed}_L{L}_q{q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    split, arm, method = task["split"], task["arm"], task["method"]
    n, seed, L, q = int(task["N"]), int(task["seed"]), int(task["L"]), int(task["q"])
    allow(seed)
    path = _cell_path(split, arm, method, n, seed, L, q)
    if path.exists():
        return {"skipped": True, "path": str(path)}
    t0 = time.perf_counter()
    meta, logical = _load(n, seed)
    Q = q * L
    b = float(meta["Bstar"])
    mid = METHOD_ID[arm][method]
    nstart = 4 if arm == "equal4" else 8
    rng_init = []
    rng_spsa = []
    for s in range(nstart):
        rng_spsa.append(make_rng(seed, n, b, L, mid, s, 0))
        rng_init.append(None if (arm == "equal4" and s == 0) or (arm == "halve8to2" and s == 0) else make_rng(seed, n, b, L, mid, s, 2))

    def propagate(x, _logical=logical, _method=method, _L=L):
        return _logical.propagate(_method, _L, x)

    runner = run_equal4 if arm == "equal4" else run_halve
    try:
        out = runner(propagate, method, L, Q, rng_init, rng_spsa)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        denom = 1.0 - f_plus
        eta = float((f - f_plus) / denom) if denom > 1e-12 else float("nan")
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_v1",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "split": split,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": Q,
            "Bstar": b,
            "E0": meta["E0"],
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "F": f,
            "eta": eta,
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * n),
            "x": [float(v) for v in out["x"]],
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "kept_start_ids": [int(s) for s in out["kept_start_ids"]],
            "screen_energies": None if out["screen_energies"] is None else [float(v) for v in out["screen_energies"]],
            "tails": out["tails"],
            "queries": out["queries"],
            "a_gain_selected": float(out["a_gain"]),
            "seconds": time.perf_counter() - t0,
            "prereg_sha256": task["prereg_sha256"],
            "energy_selection_only": True,
            "fidelity_posthoc": True,
            "not_the_frozen_1000_benchmark": True,
            "not_an_L1_to_L3_transfer": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_v1",
            "arm": arm,
            "method": method,
            "split": split,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": q * L,
            "seconds": time.perf_counter() - t0,
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    flag = "ok" if record["ok"] else "FAIL"
    extra = ""
    if record["ok"]:
        extra = f" E={record['E']:.4f} F={record['F']:.3f}"
    print(f"{flag} {split} {arm} {method} N={n} s={seed} L={L} q={q}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def tasks_for(split: str) -> list[dict]:
    seeds = TRAIN if split == "train" else TEST
    sha = prereg_sha()
    rows = []
    for seed in seeds:
        for n in NS:
            for L in LS:
                for q in QS:
                    for arm in ARMS:
                        for method in METHODS:
                            rows.append({
                                "split": split,
                                "arm": arm,
                                "method": method,
                                "N": n,
                                "seed": seed,
                                "L": L,
                                "q": q,
                                "prereg_sha256": sha,
                            })
    return rows


def _pool(tasks: list[dict]) -> None:
    import multiprocessing as mp

    pending = []
    for task in tasks:
        path = _cell_path(task["split"], task["arm"], task["method"], task["N"], task["seed"], task["L"], task["q"])
        if not path.exists():
            pending.append(task)
    print(f"pending {len(pending)} of {len(tasks)}", flush=True)
    if not pending:
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        list(pool.imap_unordered(_one, pending, chunksize=1))


def run_split(split: str) -> None:
    if split == "test":
        freeze = RES / "FREEZE.json"
        if not freeze.exists():
            raise RuntimeError("test half is refused until FREEZE.json exists")
        doc = json.loads(freeze.read_text())
        if doc.get("prereg_sha256") != prereg_sha():
            raise RuntimeError("FREEZE.json was not computed from the current preregistration")
    elif split != "train":
        raise RuntimeError(split)
    _pool(tasks_for(split))


def _records(split: str) -> list[dict]:
    rows = []
    for path in sorted(CELLS.glob(f"{split}_*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def _boot_mean(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    draws = rng.integers(0, n, size=(N_BOOT, n))
    means = values[draws].mean(axis=1)
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def _pair_table(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Fixed loop order is part of the preregistered bootstrap stream."""
    cells = []
    for q in QS:
        for n in NS:
            for L in LS:
                block = [r for r in rows if r.get("ok") and r["q"] == q and r["N"] == n and r["L"] == L]
                for arm in ARMS:
                    nov = {r["seed"]: r for r in block if r["arm"] == arm and r["method"] == "Nov3rd"}
                    qaoa = {r["seed"]: r for r in block if r["arm"] == arm and r["method"] == "QAOA"}
                    seeds = sorted(set(nov) & set(qaoa))
                    if not seeds:
                        continue
                    e_n = np.array([nov[s]["E"] for s in seeds])
                    e_q = np.array([qaoa[s]["E"] for s in seeds])
                    f_n = np.array([nov[s]["F"] for s in seeds])
                    f_q = np.array([qaoa[s]["F"] for s in seeds])
                    d_e = e_n - e_q
                    d_f = f_n - f_q
                    cells.append({
                        "arm": arm,
                        "N": n,
                        "L": L,
                        "q": q,
                        "seeds": seeds,
                        "E_nov3": _boot_mean(e_n, rng),
                        "E_qaoa": _boot_mean(e_q, rng),
                        "F_nov3": _boot_mean(f_n, rng),
                        "F_qaoa": _boot_mean(f_q, rng),
                        "dE": _boot_mean(d_e, rng),
                        "dF": _boot_mean(d_f, rng),
                        "energy_wins_nov3": int(np.sum(d_e < 0)),
                        "energy_ties": int(np.sum(d_e == 0)),
                        "energy_losses_nov3": int(np.sum(d_e > 0)),
                        "fidelity_wins_nov3_posthoc": int(np.sum(d_f > 0)),
                        "nov3_escape": int(sum(nov[s]["escaped"] for s in seeds)),
                        "qaoa_escape": int(sum(qaoa[s]["escaped"] for s in seeds)),
                        "queries_each": int(nov[seeds[0]]["queries"]["total"]),
                        "seconds_sum": float(sum(nov[s]["seconds"] + qaoa[s]["seconds"] for s in seeds)),
                    })
    return cells


def freeze() -> dict:
    rows = _records("train")
    expected = len(TRAIN) * len(NS) * len(LS) * len(QS) * len(ARMS) * len(METHODS)
    ok = [r for r in rows if r.get("ok")]
    if len(rows) != expected or len(ok) != expected:
        raise RuntimeError(f"train grid incomplete: files {len(rows)} ok {len(ok)} expected {expected}")
    by_q = {}
    for q in QS:
        d_self = []
        gap_h = []
        gap_e = []
        for seed in TRAIN:
            for n in NS:
                for L in LS:
                    def get(arm, method, _seed=seed, _n=n, _L=L, _q=q):
                        hit = [r for r in ok if r["seed"] == _seed and r["N"] == _n and r["L"] == _L and r["q"] == _q and r["arm"] == arm and r["method"] == method]
                        if len(hit) != 1:
                            raise RuntimeError("missing train cell")
                        return hit[0]["E"]
                    en_h = get("halve8to2", "Nov3rd")
                    en_e = get("equal4", "Nov3rd")
                    eq_h = get("halve8to2", "QAOA")
                    eq_e = get("equal4", "QAOA")
                    d_self.append(en_h - en_e)
                    gap_h.append(en_h - eq_h)
                    gap_e.append(en_e - eq_e)
        mean_self = float(np.mean(d_self))
        mean_gap_h = float(np.mean(gap_h))
        mean_gap_e = float(np.mean(gap_e))
        adopt = "halve8to2" if (mean_self < 0.0 and mean_gap_h <= mean_gap_e) else "equal4"
        by_q[str(q)] = {
            "adopt": adopt,
            "n": len(d_self),
            "mean_d_self_E": mean_self,
            "mean_gap_halve": mean_gap_h,
            "mean_gap_equal4": mean_gap_e,
        }
    doc = {
        "version": "N3INIT_v1",
        "prereg_sha256": prereg_sha(),
        "train_seeds": TRAIN,
        "test_seeds_not_used": TEST,
        "rule": "adopt halve8to2 at q only if mean Nov3 energy improves and the Nov3-QAOA energy gap does not worsen",
        "by_q": by_q,
    }
    _dump(RES / "FREEZE.json", doc)
    print(json.dumps(by_q, indent=2), flush=True)
    return doc


def summarize() -> None:
    sha = prereg_sha()
    rng = np.random.default_rng(BOOT_SEED)
    payload = {"version": "N3INIT_v1", "prereg_sha256": sha, "bootstrap": {"seed": BOOT_SEED, "n": N_BOOT}}
    lines = [
        "# N3INIT v1 results",
        "",
        "Original Nov3 versus QAOA at the same allocation. Energy selects. Fidelity is post hoc.",
        "This is not the frozen 1000-seed benchmark and it does not retune that recipe's gains.",
        "",
    ]
    freeze_path = RES / "FREEZE.json"
    if freeze_path.exists():
        doc = json.loads(freeze_path.read_text())
        payload["freeze"] = doc
        lines.append("## Train-half freeze")
        lines.append("")
        for q, row in doc["by_q"].items():
            lines.append(
                f"- q={q}: recommend `{row['adopt']}` "
                f"(mean Nov3 halve−equal E {row['mean_d_self_E']:+.4f}; "
                f"gap halve {row['mean_gap_halve']:+.4f}, equal4 {row['mean_gap_equal4']:+.4f}; n={row['n']})"
            )
        lines.append("")
    for split in ("train", "test"):
        rows = _records(split)
        if not rows:
            continue
        fails = [r for r in rows if not r.get("ok")]
        payload[split] = {"n_files": len(rows), "n_fail": len(fails), "cells": _pair_table([r for r in rows if r.get("ok")], rng)}
        lines.append(f"## {split}")
        lines.append("")
        lines.append(f"Files {len(rows)}, failures {len(fails)}.")
        lines.append("")
        lines.append("| arm | N | L | q | n | mean E Nov3 | mean E QAOA | mean ΔE | ΔE CI | energy wins | mean F Nov3 | mean F QAOA | mean ΔF | F wins | Nov3 escape |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for cell in payload[split]["cells"]:
            ci = cell["dE"]["ci95"]
            lines.append(
                f"| {cell['arm']} | {cell['N']} | {cell['L']} | {cell['q']} | {cell['dE']['n']} | "
                f"{cell['E_nov3']['mean']:.4f} | {cell['E_qaoa']['mean']:.4f} | {cell['dE']['mean']:+.4f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {cell['energy_wins_nov3']}/{cell['dE']['n']} | "
                f"{cell['F_nov3']['mean']:.3f} | {cell['F_qaoa']['mean']:.3f} | {cell['dF']['mean']:+.3f} | "
                f"{cell['fidelity_wins_nov3_posthoc']}/{cell['dF']['n']} | {cell['nov3_escape']}/{cell['dE']['n']} |"
            )
        lines.append("")
        seconds = sum(r.get("seconds", 0.0) for r in rows)
        queries = sum(r.get("queries", {}).get("total", 0) for r in rows if r.get("ok"))
        lines.append(f"Charged quantum queries {queries}. Sum of cell wall times {seconds:.1f}s.")
        lines.append("")
    _dump(RES / "summary.json", payload)
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "train", "freeze", "test", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("preregistration file is missing")
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "train":
        run_split("train")
    elif args.phase == "freeze":
        freeze()
    elif args.phase == "test":
        run_split("test")
    else:
        summarize()


if __name__ == "__main__":
    main()
