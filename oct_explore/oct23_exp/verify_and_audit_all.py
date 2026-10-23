"""Verify and audit every JSON in results/oct23_exp/:
1. Move any smoke_* or mpw4_* (which used warn_contrad) or stale pre-21:06 un-reproducible JSONs to stale_archive/.
2. For every remaining JSON, re-evaluate with current part1_2cav.py / gnn_5block.py at NF_chk, verify max|p_repro - p_saved| < 1e-4,
   and compute:
   - presolve_00_frac: fraction of test instances where Succ[b, 0, 0] > 0
   - zero_oracle_med, zero_oracle_B90: Med / B90 of 1/p with gamma_l = 0 and w_{c,l} = 0
   - unsolved_count, unsolved_med, unsolved_B90: conditional Med / B90 of 1/p on instances with Succ[b, 0, 0] == 0
"""
import os, sys, glob, json, shutil, subprocess
import numpy as np

RES_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp"
ARCH_DIR = os.path.join(RES_DIR, "stale_archive")
os.makedirs(ARCH_DIR, exist_ok=True)

# 1. Move smoke_* and mpw4* and *_metrics.json to stale_archive
for p in sorted(glob.glob(os.path.join(RES_DIR, "*.json"))):
    name = os.path.basename(p)[:-5]
    if name == "preps":
        continue
    if name.startswith("smoke_") or "mpw4" in name or name.endswith("_metrics"):
        shutil.move(p, os.path.join(ARCH_DIR, os.path.basename(p)))
        npz = p.replace(".json", "_theta.npz")
        if os.path.exists(npz):
            shutil.move(npz, os.path.join(ARCH_DIR, os.path.basename(npz)))
        print(f"Archived {name}")

