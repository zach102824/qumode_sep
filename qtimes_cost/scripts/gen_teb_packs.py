#!/usr/bin/env python3
"""M0 — generate TEB / Hamiltonian packs data/data_n{n}.npz.

Self-contained: uses qtimes_cost.lib.sat_gen only.

Defaults:
  n<=12: train 10000 / val 200 / test 100, dense E stored
  n>=15: train 0 / val 50 / test 100; dense E stored if 2^n * count fits
         (--no_E to skip dense energy tables)

Smoke:
  python -m qtimes_cost.scripts.gen_teb_packs --ns 6 --ntrain 200 --nval 20 --ntest 20
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow `python -m qtimes_cost.scripts.gen_teb_packs` from repo root
_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qtimes_cost.lib.sat_gen import (  # noqa: E402
    TEB,
    TRB,
    VAB,
    clause_count,
    instances,
    pack_clauses,
)


def default_counts(n: int):
    if n <= 12:
        return 10_000, 200, 100
    if n <= 15:
        return 0, 50, 100
    # 18, 21 — eval only, smaller
    return 0, 20, 50


def should_store_E(n: int, ntrain: int, nval: int, ntest: int, force_no_E: bool) -> bool:
    if force_no_E:
        return False
    # Train n (6,9) and mid eval (12,15) keep dense E. n>=18: clauses-only by default
    # (2^18=262k ok-ish, but sat-filter gens are slow; 2^21 is heavy).
    if n >= 18:
        return False
    total = ntrain + nval + ntest
    bytes_est = total * (1 << n)
    return bytes_est <= 2_000_000_000


def build_dataset(
    n: int,
    out_dir: Path,
    ntrain: int,
    nval: int,
    ntest: int,
    store_E: bool,
    overwrite: bool = False,
):
    out_path = out_dir / f"data_n{n}.npz"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return out_path
    m = clause_count(n)
    t0 = time.time()
    splits = {}
    for name, count, base in [
        ("train", ntrain, TRB),
        ("val", nval, VAB),
        ("test", ntest, TEB),
    ]:
        if count <= 0:
            # empty placeholders for schema stability
            splits[f"{name}_seeds"] = np.zeros((0,), dtype=np.int64)
            splits[f"{name}_nsol"] = np.zeros((0,), dtype=np.int32)
            splits[f"{name}_cls_v"] = np.zeros((0, m, 4), dtype=np.uint8)
            splits[f"{name}_cls_s"] = np.zeros((0, m, 4), dtype=np.int8)
            if store_E:
                splits[f"{name}_E"] = np.zeros((0, 1 << n), dtype=np.uint8)
            print(f"n={n} {name}: empty")
            continue
        # For large n without E we cannot filter on E.min()==0 cheaply;
        # still use compute_E when store_E True (unique-GS filter).
        compute_E = store_E
        insts = instances(n, count, base, compute_E=compute_E)
        seeds = np.array([ins["seed"] for ins in insts], dtype=np.int64)
        nsols = np.array([ins["nsol"] for ins in insts], dtype=np.int32)
        cls_v, cls_s = pack_clauses([ins["cl"] for ins in insts], m, n)
        splits[f"{name}_seeds"] = seeds
        splits[f"{name}_nsol"] = nsols
        splits[f"{name}_cls_v"] = cls_v
        splits[f"{name}_cls_s"] = cls_s
        if store_E:
            Es = np.stack([ins["E"] for ins in insts]).astype(np.uint8)
            splits[f"{name}_E"] = Es
            inv = (2.0 ** n) / np.maximum(nsols, 1)
            print(
                f"n={n} {name}: {count} inst in {time.time()-t0:.1f}s "
                f"(rand med 1/p={np.median(inv):.2f}, B90={np.quantile(inv,0.9):.2f})"
            )
        else:
            print(f"n={n} {name}: {count} inst (no dense E) in {time.time()-t0:.1f}s")
    meta_flags = np.array([1 if store_E else 0], dtype=np.int32)
    np.savez_compressed(
        out_path,
        n=np.int32(n),
        m=np.int32(m),
        store_E=meta_flags,
        **splits,
    )
    print(f"Saved {out_path} ({time.time()-t0:.1f}s total, store_E={store_E})")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=str, default="6,9,12,15,18,21", help="comma list of n")
    ap.add_argument("--ntrain", type=int, default=None, help="override train count for all n")
    ap.add_argument("--nval", type=int, default=None)
    ap.add_argument("--ntest", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="output dir (default: qtimes_cost/data)")
    ap.add_argument("--no_E", action="store_true", help="never store dense energy tables")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    for n in ns:
        if n % 3 != 0:
            raise SystemExit(f"n={n} not multiple of 3 (fixed K=3 packing)")
    out_dir = Path(args.out) if args.out else _PKG / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in ns:
        dtr, dva, dte = default_counts(n)
        ntrain = dtr if args.ntrain is None else args.ntrain
        nval = dva if args.nval is None else args.nval
        ntest = dte if args.ntest is None else args.ntest
        # train ns should always get E
        store_E = should_store_E(n, ntrain, nval, ntest, args.no_E)
        if n in (6, 9) and not store_E:
            print(f"WARNING: n={n} train pack without E — ansatz train will fail")
        build_dataset(n, out_dir, ntrain, nval, ntest, store_E, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
