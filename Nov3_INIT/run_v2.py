"""v2: change only Nov3 start 0. Starts 1-3 must reproduce v1 equal4."""
from __future__ import annotations

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
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import (
    ARMS,
    LS,
    METHODS,
    NS,
    QS,
    TEST,
    TRAIN,
    _boot_mean,
    _dump,
    _load,
    allow,
    prereg_sha,
)

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V2.md"
RES = ROOT / "results" / "v2"
CELLS = RES / "cells"
V1 = ROOT / "results" / "v1" / "cells"
FREEZE = RES / "FREEZE_V2.json"


def prereg2_sha() -> str:
    return __import__("hashlib").sha256(PREREG.read_bytes()).hexdigest()


def _v1(split: str, arm: str, method: str, n: int, seed: int, L: int, q: int) -> dict:
    path = V1 / f"{split}_{arm}_{method}_N{n}_s{seed}_L{L}_q{q}.json"
    return json.loads(path.read_text())


def _path(split: str, n: int, seed: int, L: int, q: int) -> Path:
    return CELLS / f"{split}_center4_Nov3rd_N{n}_s{seed}_L{L}_q{q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    split = task["split"]
    n, seed, L, q = int(task["N"]), int(task["seed"]), int(task["L"]), int(task["q"])
    allow(seed)
    path = _path(split, n, seed, L, q)
    if path.exists():
        return {"skipped": True}
    ref = _v1(split, "equal4", "Nov3rd", n, seed, L, q)
    if not ref.get("ok"):
        raise RuntimeError("v1 equal4 Nov3 cell missing or failed")
    t0 = time.perf_counter()
    meta, logical = _load(n, seed)
    Q = q * L
    b = float(meta["Bstar"])
    rng_init = [None if s == 0 else make_rng(seed, n, b, L, 1, s, 2) for s in range(4)]
    rng_spsa = [make_rng(seed, n, b, L, 1, s, 0) for s in range(4)]

    def propagate(x, _logical=logical, _L=L):
        return _logical.propagate("Nov3rd", _L, x)

    try:
        out = run_equal4(propagate, "Nov3rd", L, Q, rng_init, rng_spsa, x0_fn=x0_center4)
        mismatches = []
        for row in out["tails"]:
            if row["start_id"] == 0:
                continue
            prev = ref["tails"][row["start_id"]]
            if abs(row["E"] - prev["E"]) > 1e-8 or abs(row["e_init"] - prev["e_init"]) > 1e-8:
                mismatches.append(row["start_id"])
        if mismatches:
            raise RuntimeError(f"starts 1-3 did not reproduce v1 equal4: {mismatches}")
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        denom = 1.0 - f_plus
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_v2",
            "arm": "center4",
            "method": "Nov3rd",
            "method_id": 1,
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
            "eta": float((f - f_plus) / denom) if denom > 1e-12 else float("nan"),
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * n),
            "x": [float(v) for v in out["x"]],
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "tails": out["tails"],
            "queries": out["queries"],
            "starts_1_to_3_match_equal4": True,
            "qaoa_baseline": "v1 equal4, already charged, not rerun",
            "seconds": time.perf_counter() - t0,
            "prereg_v1_sha256": prereg_sha(),
            "prereg_v2_sha256": task["prereg_sha256"],
            "second_look_at_test_half": split == "test",
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_v2",
            "arm": "center4",
            "method": "Nov3rd",
            "split": split,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": q * L,
            "seconds": time.perf_counter() - t0,
            "prereg_v2_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    flag = "ok" if record["ok"] else "FAIL"
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{flag} v2 {split} N={n} s={seed} L={L} q={q}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def _tasks(split: str) -> list[dict]:
    seeds = TRAIN if split == "train" else TEST
    sha = prereg2_sha()
    return [{"split": split, "N": n, "seed": seed, "L": L, "q": q, "prereg_sha256": sha} for seed in seeds for n in NS for L in LS for q in QS]


def _pool(split: str) -> None:
    import multiprocessing as mp

    pending = [t for t in _tasks(split) if not _path(t["split"], t["N"], t["seed"], t["L"], t["q"]).exists()]
    print(f"v2 pending {len(pending)}", flush=True)
    if not pending:
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        list(pool.imap_unordered(_one, pending, chunksize=1))


def _rows(split: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(CELLS.glob(f"{split}_*.json"))]


def freeze() -> None:
    rows = _rows("train")
    expected = len(TRAIN) * len(NS) * len(LS) * len(QS)
    ok = [r for r in rows if r.get("ok")]
    if len(ok) != expected:
        raise RuntimeError(f"v2 train incomplete {len(ok)}/{expected}")
    by_q = {}
    for q in QS:
        d_self, gap_c, gap_e = [], [], []
        for seed in TRAIN:
            for n in NS:
                for L in LS:
                    hit = [r for r in ok if r["seed"] == seed and r["N"] == n and r["L"] == L and r["q"] == q]
                    base = _v1("train", "equal4", "Nov3rd", n, seed, L, q)
                    qaoa = _v1("train", "equal4", "QAOA", n, seed, L, q)
                    d_self.append(hit[0]["E"] - base["E"])
                    gap_c.append(hit[0]["E"] - qaoa["E"])
                    gap_e.append(base["E"] - qaoa["E"])
        mean_self, mean_c, mean_e = float(np.mean(d_self)), float(np.mean(gap_c)), float(np.mean(gap_e))
        by_q[str(q)] = {
            "adopt": "center4" if (mean_self < 0.0 and mean_c <= mean_e) else "equal4",
            "n": len(d_self),
            "mean_d_self_E": mean_self,
            "mean_gap_center4": mean_c,
            "mean_gap_equal4": mean_e,
        }
    doc = {
        "version": "N3INIT_v2",
        "prereg_v2_sha256": prereg2_sha(),
        "prereg_v1_sha256": prereg_sha(),
        "train_seeds": TRAIN,
        "test_is_a_second_look": True,
        "by_q": by_q,
    }
    _dump(FREEZE, doc)
    print(json.dumps(by_q, indent=2), flush=True)


def summarize() -> None:
    import numpy as np

    rng = np.random.default_rng(3334)
    lines = [
        "# N3INIT v2 center4",
        "",
        "Only Nov3 start 0 changed. QAOA numbers are v1 equal4, already charged.",
        "The test half is a second look.",
        "",
    ]
    if FREEZE.exists():
        doc = json.loads(FREEZE.read_text())
        lines.append("## Train-half freeze")
        lines.append("")
        for q, row in doc["by_q"].items():
            lines.append(
                f"- q={q}: recommend `{row['adopt']}` "
                f"(mean Nov3 center−equal E {row['mean_d_self_E']:+.4f}; "
                f"gap center {row['mean_gap_center4']:+.4f}, equal4 {row['mean_gap_equal4']:+.4f})"
            )
        lines.append("")
    payload = {"freeze": json.loads(FREEZE.read_text()) if FREEZE.exists() else None}
    for split in ("train", "test"):
        rows = [r for r in _rows(split) if r.get("ok")]
        if not rows:
            continue
        lines.append(f"## {split}")
        lines.append("")
        lines.append("| N | L | q | n | mean E center | mean E equal4 | mean Δ self | mean E QAOA | energy wins center | energy wins equal4 | mean F center | mean F equal4 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        cells = []
        for q in QS:
            for n in NS:
                for L in LS:
                    seeds = sorted({r["seed"] for r in rows if r["N"] == n and r["L"] == L and r["q"] == q})
                    if not seeds:
                        continue
                    e_c = np.array([next(r["E"] for r in rows if r["seed"] == s and r["N"] == n and r["L"] == L and r["q"] == q) for s in seeds])
                    e_e = np.array([_v1(split, "equal4", "Nov3rd", n, s, L, q)["E"] for s in seeds])
                    e_q = np.array([_v1(split, "equal4", "QAOA", n, s, L, q)["E"] for s in seeds])
                    f_c = np.array([next(r["F"] for r in rows if r["seed"] == s and r["N"] == n and r["L"] == L and r["q"] == q) for s in seeds])
                    f_e = np.array([_v1(split, "equal4", "Nov3rd", n, s, L, q)["F"] for s in seeds])
                    d_self = _boot_mean(e_c - e_e, rng)
                    d_c = _boot_mean(e_c - e_q, rng)
                    d_e = _boot_mean(e_e - e_q, rng)
                    cells.append({"N": n, "L": L, "q": q, "d_self": d_self, "d_center": d_c, "d_equal": d_e})
                    lines.append(
                        f"| {n} | {L} | {q} | {len(seeds)} | {float(e_c.mean()):.4f} | {float(e_e.mean()):.4f} | "
                        f"{d_self['mean']:+.4f} | {float(e_q.mean()):.4f} | {int(np.sum(e_c < e_q))}/{len(seeds)} | "
                        f"{int(np.sum(e_e < e_q))}/{len(seeds)} | {float(f_c.mean()):.3f} | {float(f_e.mean()):.3f} |"
                    )
        lines.append("")
        seconds = sum(r["seconds"] for r in rows)
        queries = sum(r["queries"]["total"] for r in rows)
        lines.append(f"New Nov3 queries {queries}. Cell time {seconds:.1f}s. Failures {sum(1 for r in _rows(split) if not r.get('ok'))}.")
        lines.append("")
        payload[split] = cells
    _dump(RES / "summary.json", payload)
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("train", "freeze", "test", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("v2 preregistration missing")
    # Touch the v1 symbols so a partial import surfaces a mismatch before the pool starts.
    assert ARMS[0] == "equal4" and METHODS[1] == "QAOA"
    if args.phase == "train":
        _pool("train")
    elif args.phase == "freeze":
        freeze()
    elif args.phase == "test":
        if not FREEZE.exists() or json.loads(FREEZE.read_text())["prereg_v2_sha256"] != prereg2_sha():
            raise RuntimeError("test refused until FREEZE_V2 matches the v2 preregistration")
        _pool("test")
    else:
        summarize()


if __name__ == "__main__":
    main()
