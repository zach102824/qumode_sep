"""Train-only mixer-angle freeze. SPSA on (gamma, s1, s2, s3) only.

Must not read test fits. Does not partition by sign(s1). Does not freeze gamma/shells.
"""
from __future__ import annotations

import numpy as np

from Nov3_INIT.alloc import (
    ALPHA_EXP,
    A_SPALL,
    GAMMA_EXP,
    Ledger,
    iterations_equal4,
    sigmas_of,
    spsa_block,
    x0_center4,
)

NOV3_NAMES = ("gamma", "theta_K", "theta_M", "s1", "s2", "s3")
FREE_IDX = (0, 3, 4, 5)
FROZEN_IDX = (1, 2)
FROZEN_NAMES = ("theta_K", "theta_M")
TRAIN = tuple(range(10888, 10896))
TEST = tuple(range(10896, 10904))
MIX_METHOD_ID = 10


def stitch(x_free: np.ndarray, theta_k: float, theta_m: float) -> np.ndarray:
    x_free = np.asarray(x_free, dtype=np.float64)
    if x_free.size != 4:
        raise ValueError(f"free dim {x_free.size} != 4")
    x = np.empty(6, dtype=np.float64)
    x[0] = x_free[0]
    x[1] = float(theta_k)
    x[2] = float(theta_m)
    x[3:6] = x_free[1:4]
    return x


def split_free(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.array([x[i] for i in FREE_IDX], dtype=np.float64)


def sigmas_free() -> np.ndarray:
    sig6 = sigmas_of("Nov3rd", 1)
    return np.array([sig6[i] for i in FREE_IDX], dtype=np.float64)


def x0_free(start_id: int, rng) -> np.ndarray:
    return split_free(x0_center4("Nov3rd", 1, start_id, rng))


def collect_nov3(rows: list[dict]) -> tuple[list[int], np.ndarray]:
    seeds, xs = [], []
    for r in sorted(rows, key=lambda z: z["seed"]):
        if r.get("arm") != "nov3_center4" or not r.get("ok"):
            continue
        if r["seed"] not in TRAIN:
            raise ValueError(f"freeze saw non-train seed {r['seed']}")
        if r["seed"] in TEST:
            raise ValueError(f"freeze peeked test seed {r['seed']}")
        if r.get("split") != "train":
            raise ValueError(f"freeze saw split={r.get('split')}")
        x = np.asarray(r["x"], dtype=np.float64)
        if x.size != 6:
            raise ValueError(f"Nov3 dim {x.size} != 6")
        seeds.append(int(r["seed"]))
        xs.append(x)
    if len(seeds) != 8 or seeds != list(TRAIN):
        raise ValueError(f"Nov3 freeze seeds {seeds} != {list(TRAIN)}")
    return seeds, np.stack(xs, axis=0)


def build_freeze(train_rows: list[dict], prereg_sha: str) -> dict:
    for r in train_rows:
        if r["seed"] in TEST or r["seed"] not in TRAIN:
            raise ValueError(f"build_freeze got illegal seed {r['seed']}")
        if r.get("split") != "train":
            raise ValueError(f"build_freeze got split={r.get('split')}")
    seeds, stack = collect_nov3(train_rows)
    theta_k = stack[:, 1]
    theta_m = stack[:, 2]
    freeze = {
        "frozen_before_test": True,
        "do_not_peek_test": True,
        "do_not_switch_frozen_coords_after_test": True,
        "primary_gate": "MIXER_FREEZE_NEGATIVE_CI",
        "frozen_names": list(FROZEN_NAMES),
        "frozen_idx": list(FROZEN_IDX),
        "free_names": ["gamma", "s1", "s2", "s3"],
        "free_idx": list(FREE_IDX),
        "not_unsigned_mean_copy": True,
        "not_two_template_copy": True,
        "not_sign_aligned_copy": True,
        "theta_K_mean": float(theta_k.mean()),
        "theta_M_mean": float(theta_m.mean()),
        "theta_K_std": float(theta_k.std(ddof=1)),
        "theta_M_std": float(theta_m.std(ddof=1)),
        "theta_K_values": theta_k.tolist(),
        "theta_M_values": theta_m.tolist(),
        "nov3_vectors": stack.tolist(),
        "nov3_seeds": seeds,
        "n_train": 8,
        "caveat": "n=8 train vectors; only mixer angles are frozen",
        "prereg_sha256": prereg_sha,
    }
    freeze["x_mixer"] = [freeze["theta_K_mean"], freeze["theta_M_mean"]]
    return freeze


def run_mix_equal4(propagate_full, theta_k: float, theta_m: float, Q: int, rngs_init, rngs_spsa) -> dict:
    """v4 equal4 schedule in the 4-D free subspace. Does not edit alloc.py."""
    t_iter = iterations_equal4(Q)
    sig = sigmas_free()
    ledger = Ledger()
    best = None
    tails = []

    def propagate(x_free):
        return propagate_full(stitch(x_free, theta_k, theta_m))

    for start_id, (rng_i, rng_s) in enumerate(zip(rngs_init, rngs_spsa)):
        x0 = x0_free(start_id, rng_i)
        block = spsa_block(
            propagate,
            x0,
            sig,
            t_iter,
            rng_s,
            ledger,
            alpha=ALPHA_EXP,
            gamma=GAMMA_EXP,
            A=A_SPALL,
        )
        tails.append({"start_id": start_id, "e_init": block["e_init"], "E": block["E"], "which": block["which"], "a_gain": block["a_gain"]})
        if best is None or block["E"] < best["E"] or (block["E"] == best["E"] and start_id < best["start_id"]):
            best = dict(block)
            best["start_id"] = start_id
    if ledger.total() != Q or ledger.monitoring != 0 or ledger.screen != 0:
        raise RuntimeError(f"mix equal4 ledger {ledger.as_dict()} != Q {Q}")
    best["queries"] = ledger.as_dict()
    best["n_iter"] = t_iter
    best["tails"] = tails
    best["x_free"] = np.asarray(best["x"], dtype=np.float64).copy()
    best["x"] = stitch(best["x_free"], theta_k, theta_m)
    return best
