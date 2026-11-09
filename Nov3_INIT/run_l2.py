"""L=2 evaluation of equal4 and center4. Does not adopt a new arm."""
from __future__ import annotations

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

from Nov3_INIT.alloc import make_rng, run_equal4, x0_center4
from Nov3_INIT.run_v1 import QS, TEST, TRAIN, _dump, _load, allow

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_L2.md"
RES = ROOT / "results" / "l2"
CELLS = RES / "cells"
NS = (9, 12)
L = 2


def sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _path(split, arm, method, n, seed, q) -> Path:
    return CELLS / f"{split}_{arm}_{method}_N{n}_s{seed}_L2_q{q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    split, arm, method = task["split"], task["arm"], task["method"]
    n, seed, q = int(task["N"]), int(task["seed"]), int(task["q"])
    allow(seed)
    path = _path(split, arm, method, n, seed, q)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    meta, logical = _load(n, seed)
    b = float(meta["Bstar"])
    mid = 0 if method == "QAOA" else 1
    rng_init = [None if s == 0 else make_rng(seed, n, b, L, mid, s, 2) for s in range(4)]
    rng_spsa = [make_rng(seed, n, b, L, mid, s, 0) for s in range(4)]
    draw = x0_center4 if arm == "center4" else None

    def propagate(x, _logical=logical, _method=method):
        return _logical.propagate(_method, L, x)

    try:
        out = run_equal4(propagate, method, L, q * L, rng_init, rng_spsa, x0_fn=draw)
        if arm == "center4":
            ref = json.loads(_path(split, "equal4", "Nov3rd", n, seed, q).read_text())
            for row in out["tails"]:
                if row["start_id"] == 0:
                    continue
                prev = ref["tails"][row["start_id"]]
                if abs(row["E"] - prev["E"]) > 1e-8:
                    raise RuntimeError(f"start {row['start_id']} diverged from equal4")
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_L2",
            "arm": arm,
            "method": method,
            "split": split,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": q * L,
            "E": out["E"],
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * n),
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "tails": out["tails"],
            "queries": out["queries"],
            "x": [float(v) for v in out["x"]],
            "seconds": time.perf_counter() - t0,
            "prereg_sha256": task["prereg_sha256"],
            "no_new_adoption": True,
        }
    except Exception as exc:
        record = {"ok": False, "failure": repr(exc), "split": split, "arm": arm, "method": method, "N": n, "seed": seed, "q": q, "seconds": time.perf_counter() - t0}
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} L2 {split} {arm} {method} N={n} s={seed} q={q}{extra}", flush=True)
    return record


def _tasks(arms) -> list[dict]:
    digest = sha()
    rows = []
    for split, seeds in (("train", TRAIN), ("test", TEST)):
        for seed in seeds:
            for n in NS:
                for q in QS:
                    for arm, method in arms:
                        rows.append({"split": split, "arm": arm, "method": method, "N": n, "seed": seed, "q": q, "prereg_sha256": digest})
    return rows


def run() -> None:
    import multiprocessing as mp

    # equal4 first so center4 can check starts 1-3.
    for arms in ((("equal4", "Nov3rd"), ("equal4", "QAOA")), (("center4", "Nov3rd"),)):
        pending = [t for t in _tasks(arms) if not _path(t["split"], t["arm"], t["method"], t["N"], t["seed"], t["q"]).exists()]
        print(f"L2 pending {len(pending)}", flush=True)
        if pending:
            with mp.get_context("fork").Pool(4) as pool:
                list(pool.imap_unordered(_one, pending, chunksize=1))
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    lines = ["# N3INIT L2 evaluation", "", "No arm is adopted from this table. `center4` remains the v2 recommendation.", ""]
    lines.append("| split | N | q | n | mean E equal | mean E center | mean E QAOA | wins equal | wins center | mean F equal | mean F center | mean F QAOA |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in ("train", "test"):
        for n in NS:
            for q in QS:
                def grab(arm, method, split=split, n=n, q=q):
                    hit = [r for r in rows if r.get("ok") and r["split"] == split and r["N"] == n and r["q"] == q and r["arm"] == arm and r["method"] == method]
                    return hit
                eq, ce, qa = grab("equal4", "Nov3rd"), grab("center4", "Nov3rd"), grab("equal4", "QAOA")
                if not eq:
                    continue
                e_e = np.array([r["E"] for r in eq])
                e_c = np.array([r["E"] for r in ce])
                e_q = np.array([r["E"] for r in qa])
                lines.append(
                    f"| {split} | {n} | {q} | {len(eq)} | {e_e.mean():.4f} | {e_c.mean():.4f} | {e_q.mean():.4f} | "
                    f"{int(np.sum(e_e < e_q))}/{len(eq)} | {int(np.sum(e_c < e_q))}/{len(eq)} | "
                    f"{np.mean([r['F'] for r in eq]):.3f} | {np.mean([r['F'] for r in ce]):.3f} | {np.mean([r['F'] for r in qa]):.3f} |"
                )
    fails = sum(1 for r in rows if not r.get("ok"))
    lines += ["", f"Files {len(rows)}, failures {fails}, new queries {sum(r.get('queries', {}).get('total', 0) for r in rows if r.get('ok'))}, cell time {sum(r.get('seconds', 0.0) for r in rows):.1f}s."]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    if not PREREG.exists():
        raise RuntimeError("L2 preregistration missing")
    run()
