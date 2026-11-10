"""Train-only SK-geometry freeze: mean θ_K and θ_M. Must not read test fits."""
from __future__ import annotations

import numpy as np

NOV3_NAMES = ("gamma", "theta_K", "theta_M", "s1", "s2", "s3")
TRAIN = tuple(range(10872, 10880))
TEST = tuple(range(10880, 10888))
FROZEN_IDX = (1, 2)
FROZEN_NAMES = ("theta_K", "theta_M")


def coord_stats(stack):
    stack = np.asarray(stack, dtype=np.float64)
    return {
        "mean": stack.mean(axis=0).tolist(),
        "median": np.median(stack, axis=0).tolist(),
        "var": stack.var(axis=0, ddof=1).tolist() if stack.shape[0] > 1 else [0.0] * stack.shape[1],
        "std": stack.std(axis=0, ddof=1).tolist() if stack.shape[0] > 1 else [0.0] * stack.shape[1],
        "min": stack.min(axis=0).tolist(),
        "max": stack.max(axis=0).tolist(),
        "range": (stack.max(axis=0) - stack.min(axis=0)).tolist(),
        "n": int(stack.shape[0]),
        "dim": int(stack.shape[1]),
    }


def collect_nov3(rows):
    seeds, xs = [], []
    for r in sorted(rows, key=lambda z: z["seed"]):
        if r.get("arm") != "nov3_center4" or r.get("status") != "ok":
            continue
        if r["seed"] not in TRAIN:
            raise ValueError(f"freeze saw non-train seed {r['seed']}")
        if r["seed"] in TEST:
            raise ValueError(f"freeze peeked test seed {r['seed']}")
        x = np.asarray(r["x"], dtype=np.float64)
        if x.size != 6:
            raise ValueError(f"Nov3 dim {x.size} != 6")
        seeds.append(int(r["seed"]))
        xs.append(x)
    if len(seeds) != 8 or seeds != list(TRAIN):
        raise ValueError(f"Nov3 freeze seeds {seeds} != {list(TRAIN)}")
    return seeds, np.stack(xs, axis=0)


def project(x, freeze):
    y = np.asarray(x, dtype=np.float64).copy()
    for i, name in zip(FROZEN_IDX, FROZEN_NAMES):
        y[i] = float(freeze["frozen_mean"][name])
    return y


def build_freeze(train_rows, prereg_sha):
    for r in train_rows:
        if r["seed"] in TEST or r["seed"] not in TRAIN:
            raise ValueError(f"build_freeze got illegal seed {r['seed']}")
        if r.get("split") != "train":
            raise ValueError(f"build_freeze got split={r.get('split')}")
    seeds, stack = collect_nov3(train_rows)
    stats = coord_stats(stack)
    stats["names"] = list(NOV3_NAMES)
    stats["seeds"] = seeds
    stats["vectors"] = stack.tolist()
    frozen_mean = {name: float(stats["mean"][i]) for i, name in zip(FROZEN_IDX, FROZEN_NAMES)}
    frozen_std = {name: float(stats["std"][i]) for i, name in zip(FROZEN_IDX, FROZEN_NAMES)}
    return {
        "frozen_before_test": True,
        "do_not_peek_test": True,
        "not_two_template": True,
        "not_unsigned_full_vector_primary": True,
        "primary_rule": "geom_spsa_tk_tm_frozen",
        "n_train": 8,
        "caveat": "n=8 train vectors; freeze is mean θ_K, θ_M only",
        "prereg_sha256": prereg_sha,
        "frozen_indices": list(FROZEN_IDX),
        "frozen_names": list(FROZEN_NAMES),
        "frozen_mean": frozen_mean,
        "frozen_std": frozen_std,
        "nov3_unsigned": stats,
        "layout": "LE x; Nov3 identical to FT",
        "does_not_reopen_pr119": True,
    }
