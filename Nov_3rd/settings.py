"""Nov_3rd LOCKED settings — all paths inside this package.

Zach 2026-11-03 (t2442–t2448, t2495):
  Motif Nov1st-final X-basis sandwich
  QAOA-warm shared shell SNAP
  b=3 bits per cavity, Binary W=H^{⊗3}
  Train n=6 only, 1000 samples; L-BFGS only
  L ∈ {1,4,8,12,16,20,24,28,32}
  Eval order (t2495): ALL L on n∈{6,9,12,15} FIRST, then n=18 across L, then n=21.
"""
from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent
OUT = PKG / "runs"
# all data local to Nov_3rd/
SEALED_DATA = PKG / "data" / "sealed"
SCALE1000_DIR = PKG / "data" / "scale1000"
QAOA_DIR = PKG / "qaoa_angles"

RECIPE_NAME = "Nov_3rd"
MOTIF = "nov1st_final_x_sandwich"

B_BITS = 3
BITS_PER_CAVITY = 3
NF = 8
W_FRAME = "BinH"

SNAP_HEAD = "shell"  # Hamming-shell SNAP (Zach name)
SNAP_SHARED_CAVITIES = True
SNAP_SHARED_LAYERS = False
QAOA_WARM = True
OPTIMIZER = "LBFGS_only"

TRAIN_NS = (6,)
MAX_TRAIN = 1000
MAX_VAL = 40

L_LIST = (1, 4, 8, 12, 16, 20, 24, 28, 32)

# Eval phasing (t2495u): core first for every L, then extend.
EVAL_NS_CORE = (6, 9, 12, 15)       # phase 1 — gate ladder / STOP
EVAL_NS_PASS2 = (18,)               # phase 2 — after all L pass core
EVAL_NS_PASS3 = (21,)               # phase 3 — after pass2 (pack may be missing)
EVAL_NS_SIM = EVAL_NS_CORE          # primary sim / gate set
EVAL_N_LO, EVAL_N_HI = 6, 21
EVAL_NS = tuple(range(EVAL_N_LO, EVAL_N_HI + 1))

SCALE_B = 1000
SEEDS = (0, 1, 2)

LBFGS_ROUNDS = 40
LBFGS_ITER = 8
LBFGS_BATCH = 64
LBFGS_LR = 0.25

NUM_THREADS = 8

# QAOA train init — Boulebnane–Montanaro arXiv:2208.06909 §Parameter optimization
# all β=0.01, all γ=-0.01; single start (nstart=1). Angle convention: see qaoa_init.
QAOA_INIT_BETA = 0.01
QAOA_INIT_GAMMA = -0.01
QAOA_NSTART = 1


def qaoa_angles_path(L: int) -> Path:
    return QAOA_DIR / f"L{L}" / "angles.json"


def qaoa_results_path(L: int) -> Path:
    return QAOA_DIR / f"L{L}" / "results.json"


def assert_locked(**kwargs):
    checks = {
        "b": B_BITS,
        "W_frame": W_FRAME,
        "snap_head": SNAP_HEAD,
        "optimizer": OPTIMIZER,
        "train_ns": TRAIN_NS,
        "max_train": MAX_TRAIN,
        "qaoa_warm": QAOA_WARM,
        "snap_shared_layers": SNAP_SHARED_LAYERS,
        "L_list": L_LIST,
        "eval_ns": EVAL_NS,
        "eval_ns_core": EVAL_NS_CORE,
    }
    for k, expected in checks.items():
        if k in kwargs and kwargs[k] != expected:
            raise ValueError(
                f"Nov_3rd lock violation: {k}={kwargs[k]!r} (locked to {expected!r})"
            )


FORBIDDEN = (
    "Adam",
    "Gray",
    "gray",
    "per_cavity",
    "gauss",
    "free_head",
    "max_train=200",
    "train_ns=(6, 9)",
    "train_ns=(9, 12)",
)
