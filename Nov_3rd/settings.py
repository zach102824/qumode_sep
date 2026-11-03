"""Nov_3rd LOCKED settings — all paths inside this package.

Zach 2026-11-03 (t2442–t2448):
  Motif Nov1st-final X-basis sandwich
  QAOA-warm shared shell SNAP
  b=3 bits per cavity, Binary W=H^{⊗3}
  Train n=6 only, 1000 samples; L-BFGS only
  L ∈ {1,4,8,12,16,20,24,28,32}
  Eval n from 6 to 18 (scale1000 B=1000)
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

SNAP_HEAD = "shell"
SNAP_SHARED_CAVITIES = True
SNAP_SHARED_LAYERS = False
QAOA_WARM = True
OPTIMIZER = "LBFGS_only"

TRAIN_NS = (6,)
MAX_TRAIN = 1000
MAX_VAL = 40

L_LIST = (1, 4, 8, 12, 16, 20, 24, 28, 32)

EVAL_N_LO, EVAL_N_HI = 6, 18
EVAL_NS = tuple(range(EVAL_N_LO, EVAL_N_HI + 1))
EVAL_NS_SIM = tuple(n for n in EVAL_NS if n % B_BITS == 0)

SCALE_B = 1000
SEEDS = (0, 1, 2)

LBFGS_ROUNDS = 40
LBFGS_ITER = 8
LBFGS_BATCH = 64
LBFGS_LR = 0.25


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
