"""Fast shared disk cache for clean16 4-SAT instances (train 51e9, test 52e9, val 53e9).
Stores E as uint8 (since m <= 199 < 255), V as uint8 (count, m, 4), SGN as int8 (count, m, 4), seeds as int64.
"""
from __future__ import annotations
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import sim

TRB = 51_000_000_000
TEB = 52_000_000_000
VAB = 53_000_000_000

CACHE_DIR = os.path.join(ROOT, "results", "oct23_exp", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _build_stream(n: int, max_count: int, base: int, tag: str, known_seeds=None):
    cpath = os.path.join(CACHE_DIR, f"{tag}_n{n}_c{max_count}.npz")
    if os.path.exists(cpath):
        return cpath
    t0 = time.time()
    m = int(round(9.93 * n))
    seeds = np.zeros(max_count, dtype=np.int64)
    E_arr = np.zeros((max_count, 1 << n), dtype=np.uint8)
    V_arr = np.zeros((max_count, m, 4), dtype=np.uint8)
    S_arr = np.zeros((max_count, m, 4), dtype=np.int8)
    nsol = np.zeros(max_count, dtype=np.int32)

    if known_seeds is not None:
        assert len(known_seeds) == max_count
        for idx, sd in enumerate(known_seeds):
            m_, cl, E = sim.gen_instance(n, int(sd))
            assert m_ == m and E.min() == 0
            seeds[idx] = int(sd)
            E_arr[idx] = E.astype(np.uint8)
            nsol[idx] = int((E == 0).sum())
            for j, (v, s) in enumerate(cl):
                V_arr[idx, j] = v
                S_arr[idx, j] = s
    else:
        idx = 0
        k = 0
        while idx < max_count:
            sd = int(base) + 10_000_000 * n + k
            k += 1
            m_, cl, E = sim.gen_instance(n, sd)
            if E.min() == 0:
                seeds[idx] = sd
                E_arr[idx] = E.astype(np.uint8)
                nsol[idx] = int((E == 0).sum())
                for j, (v, s) in enumerate(cl):
                    V_arr[idx, j] = v
                    S_arr[idx, j] = s
                idx += 1

    tmp = f"{cpath}.tmp.{os.getpid()}.npz"
    np.savez(tmp, seeds=seeds, E=E_arr, V=V_arr, SGN=S_arr, nsol=nsol, m=np.int32(m), n=np.int32(n))
    os.replace(tmp, cpath)
    print(f"[cache] Built {os.path.basename(cpath)} in {time.time()-t0:.1f}s", flush=True)
    return cpath


def load_split(split: str, n: int, count: int):
    """Return dict with keys: seeds (B,), E (B, 2^n) uint8, V (B, m, 4) uint8, SGN (B, m, 4) int8, nsol (B,), m (int)."""
    if split == "test":
        assert count <= 100
        wfile = os.path.join(ROOT, "results", "clean16", f"W_n{n}.json")
        kseeds = json.load(open(wfile))["seeds"] if os.path.exists(wfile) else None
        cpath = _build_stream(n, 100, TEB, "test", known_seeds=kseeds)
    elif split == "val":
        max_c = max(200, count)
        cpath = _build_stream(n, max_c, VAB, "val")
    elif split == "train":
        max_c = 10000 if n in (8, 12) else max(1000, count)
        if count > max_c:
            max_c = count
        cpath = _build_stream(n, max_c, TRB, "train")
    else:
        raise ValueError(split)
    Z = np.load(cpath)
    return {
        "seeds": Z["seeds"][:count],
        "E": Z["E"][:count],
        "V": Z["V"][:count],
        "SGN": Z["SGN"][:count],
        "nsol": Z["nsol"][:count],
        "m": int(Z["m"]),
        "n": int(Z["n"]),
    }


def load_walksat(n: int):
    wfile = os.path.join(ROOT, "results", "clean16", f"W_n{n}.json")
    return json.load(open(wfile))


if __name__ == "__main__":
    for n in (8, 12, 16, 20):
        load_split("test", n, 100)
    load_split("val", 12, 200)
    load_split("val", 8, 200)
    load_split("train", 8, 10000)
    load_split("train", 12, 10000)
    print("ALL CACHES READY!")
