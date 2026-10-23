"""Pre-generate and cache 10,000 train, 200 val, 100 test 4-SAT instances for n=6 and n=8
using clean16 seed bases (TRB=51e9, TEB=52e9, VAB=53e9), plus exact 1, 2, 3 block flat state preps."""
import os, sys, time, json
import numpy as np

ROOT = "/workspace/qumode_sep/oct_explore"
sys.path.insert(0, ROOT)
import sim

OUT_DIR = os.path.join(ROOT, "results", "oct23_exp")
os.makedirs(OUT_DIR, exist_ok=True)

TRB, TEB, VAB = 51_000_000_000, 52_000_000_000, 53_000_000_000

def build_dataset(n, ntrain=10000, nval=200, ntest=100):
    t0 = time.time()
    out_path = os.path.join(OUT_DIR, f"data_n{n}.npz")
    if os.path.exists(out_path):
        print(f"Dataset {out_path} already exists.")
        return
    m = int(round(9.93 * n))
    splits = {}
    for name, count, base in [("train", ntrain, TRB), ("val", nval, VAB), ("test", ntest, TEB)]:
        insts = sim.instances(n, count, base)
        seeds = np.array([ins["seed"] for ins in insts], dtype=np.int64)
        Es = np.stack([ins["E"] for ins in insts]).astype(np.uint8)
        nsols = np.array([ins["nsol"] for ins in insts], dtype=np.int32)
        cls_v = np.zeros((count, m, 4), dtype=np.uint8)
        cls_s = np.zeros((count, m, 4), dtype=np.int8)
        for i, ins in enumerate(insts):
            _, cl, _ = sim.gen_instance(n, int(ins["seed"]))
            for c_idx, (v, s) in enumerate(cl):
                cls_v[i, c_idx] = v
                cls_s[i, c_idx] = s
        splits[f"{name}_seeds"] = seeds
        splits[f"{name}_E"] = Es
        splits[f"{name}_nsol"] = nsols
        splits[f"{name}_cls_v"] = cls_v
        splits[f"{name}_cls_s"] = cls_s
        print(f"n={n} {name}: {count} instances generated in {time.time()-t0:.1f}s (rand med 1/p = {np.median((2.0**n)/nsols):.2f}, B90 = {np.quantile((2.0**n)/nsols, 0.9):.2f})")
    np.savez_compressed(out_path, n=n, m=m, **splits)
    print(f"Saved {out_path} ({time.time()-t0:.1f}s total)")

if __name__ == "__main__":
    build_dataset(6, 10000, 200, 100)
    build_dataset(8, 10000, 200, 100)
