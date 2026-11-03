"""Nov_3rd LOCKED settings — do not import Nov/ explore defaults that conflict.

Zach 2026-11-03 (t2442 / t2443):
  Motif Nov1st-final X-basis sandwich
  QAOA-warm shared shell SNAP
  b=3 (hardware), Binary W=H^{⊗3}
  Train n=6 only, 1000 samples
  Optimizer = L-BFGS only (no Adam)
  Eval scale1000 B=1000; primary L=16
"""
from __future__ import annotations

from pathlib import Path

# ---- paths (this package only) ----
PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
EXPLORE = ROOT / "explore_OCT29"
OUT = PKG / "runs"
SEALED_DATA = EXPLORE / "data"
SCALE1000_DIR = EXPLORE / "data" / "scale1000"
QAOA_RUNS = EXPLORE / "runs"

# ---- hard locks ----
RECIPE_NAME = "Nov_3rd"
MOTIF = "nov1st_final_x_sandwich"  # |0> -> [Uc^X -> bare SNAP]^L -> final W -> Fock
B_BITS = 3
NF = 8  # Fock trunc; computational NU=8 for b=3
W_FRAME = "BinH"  # Binary H^{⊗b}; NOT Gray
SNAP_HEAD = "shell"  # θ_k = φ_pc(k)
SNAP_SHARED_CAVITIES = True  # same SNAP on every cavity
SNAP_SHARED_LAYERS = False  # φ,γ free per layer
QAOA_WARM = True
OPTIMIZER = "LBFGS_only"  # forbidden: Adam
TRAIN_NS = (6,)  # only n=6
MAX_TRAIN = 1000
MAX_VAL = 40
L_PRIMARY = 16
EVAL_NS = (6, 9, 12, 15, 18)
SCALE_B = 1000
SEEDS = (0, 1, 2)

# L-BFGS schedule (not Adam steps)
LBFGS_ROUNDS = 40
LBFGS_ITER = 8
LBFGS_BATCH = 64
LBFGS_LR = 0.25

# QAOA warm angles
def qaoa_angles_path(L: int = L_PRIMARY) -> Path:
    return QAOA_RUNS / f"qaoa_L{L}_scale1000" / "angles.json"


def qaoa_results_path(L: int = L_PRIMARY) -> Path:
    return QAOA_RUNS / f"qaoa_L{L}_scale1000" / "results.json"


def assert_locked(**kwargs):
    """Raise if a run tries to override a locked knob."""
    checks = {
        "b": B_BITS,
        "W_frame": W_FRAME,
        "snap_head": SNAP_HEAD,
        "optimizer": OPTIMIZER,
        "train_ns": TRAIN_NS,
        "max_train": MAX_TRAIN,
        "qaoa_warm": QAOA_WARM,
        "snap_shared_layers": SNAP_SHARED_LAYERS,
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
