"""v3: eight equal SPSA blocks, coherent center held fixed. Both methods."""
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

from Nov3_INIT.alloc import iterations_spread8, make_rng, run_spread8
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import LS, NS, QS, TEST, TRAIN, _dump, _load, allow, prereg_sha

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V3.md"
RES = ROOT / "results" / "v3"
CELLS = RES / "cells"
V1 = ROOT / "results" / "v1" / "cells"
V2 = ROOT / "results" / "v2" / "cells"
FREEZE = RES / "FREEZE_V3.json"
METHOD_ID = {"Nov3rd": 82, "QAOA": 83}


def prereg3_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _read(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text())


def _path(split: str, method: str, n: int, seed: int, L: int, q: int) -> Path:
    return CELLS / f"{split}_spread8_{method}_N{n}_s{seed}_L{L}_q{q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    split, method = task["split"], task["method"]
    n, seed, L, q = int(task["N"]), int(task["seed"]), int(task["L"]), int(task["q"])
    allow(seed)
    path = _path(split, method, n, seed, L, q)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    meta, logical = _load(n, seed)
    b = float(meta["Bstar"])
    mid = METHOD_ID[method]
    rng_init = [None if s == 0 else make_rng(seed, n, b, L, mid, s, 2) for s in range(8)]
    rng_spsa = [make_rng(seed, n, b, L, mid, s, 0) for s in range(8)]

    def propagate(x, _logical=logical, _method=method, _L=L):
        return _logical.propagate(_method, _L, x)

    try:
        out = run_spread8(propagate, method, L, q * L, rng_init, rng_spsa)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        denom = 1.0 - f_plus
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_v3",
            "arm": "spread8",
            "method": method,
            "method_id": mid,
            "split": split,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": q * L,
            "Bstar": b,
            "E0": meta["E0"],
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "F": f,
            "eta": float((f - f_plus) / denom) if denom > 1e-12 else float("nan"),
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * n),
            "x": [float(v) for v in out["x"]],
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "tails": out["tails"],
            "queries": out["queries"],
            "seconds": time.perf_counter() - t0,
            "prereg_v3_sha256": task["prereg_sha256"],
            "third_look_at_test_half": split == "test",
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_v3",
            "arm": "spread8",
            "method": method,
            "split": split,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "seconds": time.perf_counter() - t0,
            "prereg_v3_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} v3 {split} {method} N={n} s={seed} L={L} q={q}{extra}", flush=True)
    return record


def _tasks(split: str) -> list[dict]:
    seeds = TRAIN if split == "train" else TEST
    sha = prereg3_sha()
    return [
        {"split": split, "method": method, "N": n, "seed": seed, "L": L, "q": q, "prereg_sha256": sha}
        for seed in seeds for n in NS for L in LS for q in QS for method in ("Nov3rd", "QAOA")
    ]


def _pool(split: str) -> None:
    import multiprocessing as mp

    pending = [t for t in _tasks(split) if not _path(t["split"], t["method"], t["N"], t["seed"], t["L"], t["q"]).exists()]
    print(f"v3 pending {len(pending)}", flush=True)
    if pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one, pending, chunksize=1))


def _rows(split: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(CELLS.glob(f"{split}_*.json"))]


def _v1_qaoa(split, n, seed, L, q):
    return _read(V1, f"{split}_equal4_QAOA_N{n}_s{seed}_L{L}_q{q}.json")


def _v2_nov3(split, n, seed, L, q):
    return _read(V2, f"{split}_center4_Nov3rd_N{n}_s{seed}_L{L}_q{q}.json")


def freeze() -> None:
    rows = [r for r in _rows("train") if r.get("ok")]
    expected = len(TRAIN) * len(NS) * len(LS) * len(QS) * 2
    if len(rows) != expected:
        raise RuntimeError(f"v3 train incomplete {len(rows)}/{expected}")
    by_q = {}
    for q in QS:
        d_self, gap_s, gap_c = [], [], []
        for seed in TRAIN:
            for n in NS:
                for L in LS:
                    nov = next(r for r in rows if r["seed"] == seed and r["N"] == n and r["L"] == L and r["q"] == q and r["method"] == "Nov3rd")
                    qaoa = next(r for r in rows if r["seed"] == seed and r["N"] == n and r["L"] == L and r["q"] == q and r["method"] == "QAOA")
                    center = _v2_nov3("train", n, seed, L, q)
                    base_q = _v1_qaoa("train", n, seed, L, q)
                    d_self.append(nov["E"] - center["E"])
                    gap_s.append(nov["E"] - qaoa["E"])
                    gap_c.append(center["E"] - base_q["E"])
        ms, gs, gc = float(np.mean(d_self)), float(np.mean(gap_s)), float(np.mean(gap_c))
        by_q[str(q)] = {
            "adopt": "spread8" if (ms < 0.0 and gs <= gc) else "center4",
            "n": len(d_self),
            "mean_d_self_E": ms,
            "mean_gap_spread8": gs,
            "mean_gap_center4": gc,
        }
    _dump(FREEZE, {
        "version": "N3INIT_v3",
        "prereg_v3_sha256": prereg3_sha(),
        "prereg_v1_sha256": prereg_sha(),
        "train_seeds": TRAIN,
        "test_is_a_third_look": True,
        "by_q": by_q,
    })
    print(json.dumps(by_q, indent=2), flush=True)


def summarize() -> None:
    doc = json.loads(FREEZE.read_text()) if FREEZE.exists() else None
    lines = ["# N3INIT v3 spread8", "", "Eight equal blocks. Nov3 start 0 stays at the v2 center. The test half is a third look.", ""]
    if doc:
        lines.append("## Train-half freeze")
        lines.append("")
        for q, row in doc["by_q"].items():
            lines.append(
                f"- q={q}: recommend `{row['adopt']}` "
                f"(mean Nov3 spread−center E {row['mean_d_self_E']:+.4f}; "
                f"gap spread {row['mean_gap_spread8']:+.4f}, center {row['mean_gap_center4']:+.4f})"
            )
        lines.append("")
    for split in ("train", "test"):
        rows = [r for r in _rows(split) if r.get("ok")]
        if not rows:
            continue
        lines += [f"## {split}", "", "| N | L | q | n | mean E Nov3 | mean E QAOA | mean ΔE | energy wins | mean F Nov3 | mean F QAOA | mean E center |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for q in QS:
            for n in NS:
                for L in LS:
                    nov = [r for r in rows if r["N"] == n and r["L"] == L and r["q"] == q and r["method"] == "Nov3rd"]
                    qaoa = [r for r in rows if r["N"] == n and r["L"] == L and r["q"] == q and r["method"] == "QAOA"]
                    if not nov:
                        continue
                    seeds = sorted(r["seed"] for r in nov)
                    e_n = np.array([next(r["E"] for r in nov if r["seed"] == s) for s in seeds])
                    e_q = np.array([next(r["E"] for r in qaoa if r["seed"] == s) for s in seeds])
                    e_c = np.array([_v2_nov3(split, n, s, L, q)["E"] for s in seeds])
                    f_n = np.array([next(r["F"] for r in nov if r["seed"] == s) for s in seeds])
                    f_q = np.array([next(r["F"] for r in qaoa if r["seed"] == s) for s in seeds])
                    lines.append(
                        f"| {n} | {L} | {q} | {len(seeds)} | {e_n.mean():.4f} | {e_q.mean():.4f} | {e_n.mean()-e_q.mean():+.4f} | "
                        f"{int(np.sum(e_n < e_q))}/{len(seeds)} | {f_n.mean():.3f} | {f_q.mean():.3f} | {e_c.mean():.4f} |"
                    )
        lines.append("")
        lines.append(f"New queries {sum(r['queries']['total'] for r in rows)}. Cell time {sum(r['seconds'] for r in rows):.1f}s. Failures {sum(1 for r in _rows(split) if not r.get('ok'))}.")
        lines.append("")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("train", "freeze", "test", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("v3 preregistration missing")
    assert iterations_spread8(128) == 2 and iterations_spread8(3072) == 186
    if args.phase == "train":
        _pool("train")
    elif args.phase == "freeze":
        freeze()
    elif args.phase == "test":
        if not FREEZE.exists() or json.loads(FREEZE.read_text())["prereg_v3_sha256"] != prereg3_sha():
            raise RuntimeError("test refused until FREEZE_V3 matches the preregistration")
        _pool("test")
    else:
        summarize()


if __name__ == "__main__":
    main()
