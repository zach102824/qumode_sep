#!/usr/bin/env python3
"""Build sealed beat_qaoa train packs: 250/n for n∈{6,9,12,15}, disjoint from scale1000.

- n6/n9: subsample sealed train_seeds
- n12: use all sealed train (128) + generate extras to reach 250
- n15: generate 250 (sealed train empty)
Uses sat_gen base BEAT_TRB=61_000_000_000 to avoid collision with TRB/TEB/VAB.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from Nov_3rd import settings as S
from explore_OCT29.lib.sat_gen import instances

OUT = S.OUT / "beat_qaoa_12h" / "train_packs"
BEAT_TRB = 61_000_000_000
TARGET = 250
NS = (6, 9, 12, 15)


def scale_seeds(n: int) -> set[int]:
    p = S.SCALE1000_DIR / f"data_n{n}.npz"
    d = np.load(p, allow_pickle=True)
    return set(map(int, np.asarray(d["scale1000_seeds"]).ravel().tolist()))


def sealed_train(n: int):
    p = S.SEALED_DATA / f"data_n{n}.npz"
    d = np.load(p, allow_pickle=True)
    seeds = np.asarray(d["train_seeds"]).ravel()
    E = np.asarray(d["train_E"])
    if seeds.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 1 << n), dtype=np.int16)
    return seeds.astype(np.int64), E


def gen_extra(n: int, need: int, forbid: set[int]) -> tuple[np.ndarray, np.ndarray]:
    seeds, Es = [], []
    # instances() yields sat ones; filter forbid
    for rec in instances(n, need + len(forbid) + 50, BEAT_TRB, compute_E=True):
        s = int(rec["seed"])
        if s in forbid:
            continue
        seeds.append(s)
        Es.append(np.asarray(rec["E"], dtype=np.int16))
        if len(seeds) >= need:
            break
    if len(seeds) < need:
        raise RuntimeError(f"n{n}: only got {len(seeds)}/{need} extra")
    return np.asarray(seeds[:need], dtype=np.int64), np.stack(Es[:need], axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"target_per_n": TARGET, "base": BEAT_TRB, "ns": {}, "finished": None}
    for n in NS:
        forbid = scale_seeds(n)
        seeds0, E0 = sealed_train(n)
        # drop any accidental overlap
        keep = [i for i, s in enumerate(seeds0.tolist()) if int(s) not in forbid]
        seeds0 = seeds0[keep] if keep else np.zeros(0, dtype=np.int64)
        E0 = E0[keep] if keep else np.zeros((0, 1 << n), dtype=np.int16)

        if len(seeds0) >= TARGET:
            rng = np.random.default_rng(1000 + n)
            pick = rng.choice(len(seeds0), size=TARGET, replace=False)
            pick.sort()
            seeds = seeds0[pick]
            E = E0[pick]
            src = "sealed_subsample"
        else:
            need = TARGET - len(seeds0)
            forbid2 = forbid | set(map(int, seeds0.tolist()))
            print(f"n{n}: have {len(seeds0)}, generating {need} …", flush=True)
            t0 = time.time()
            s_ex, E_ex = gen_extra(n, need, forbid2)
            print(f"n{n}: generated in {time.time()-t0:.1f}s", flush=True)
            seeds = np.concatenate([seeds0, s_ex]) if len(seeds0) else s_ex
            E = np.concatenate([E0, E_ex], axis=0) if len(E0) else E_ex
            src = f"sealed_{len(seeds0)}+gen_{need}"

        inter = sorted(set(map(int, seeds.tolist())) & forbid)
        assert not inter, (n, inter[:5])
        out = OUT / f"train_n{n}.npz"
        np.savez_compressed(
            out,
            n=n,
            m=int(round(9.93 * n)),
            train_seeds=seeds,
            train_E=E.astype(np.int16),
            source=src,
            beat_base=BEAT_TRB,
        )
        manifest["ns"][str(n)] = {
            "path": str(out),
            "N": int(len(seeds)),
            "source": src,
            "disjoint_scale1000": True,
            "seed_min": int(seeds.min()),
            "seed_max": int(seeds.max()),
        }
        print(f"wrote {out} N={len(seeds)} src={src}", flush=True)

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DONE", manifest["finished"])


if __name__ == "__main__":
    main()
