"""Train-only mean/median freeze. Must not read test fits."""
from __future__ import annotations

import numpy as np

NOV3_NAMES = ("gamma", "theta_K", "theta_M", "s1", "s2", "s3")
QAOA_NAMES = ("gamma", "beta")
TRAIN = tuple(range(10600, 10608))
TEST = tuple(range(10608, 10616))


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


def pack_arm(rows, arm, names):
    seeds = []
    xs = []
    xfts = []
    for r in sorted(rows, key=lambda z: z["seed"]):
        if r.get("arm") != arm or r.get("status") != "ok":
            continue
        if r["seed"] not in TRAIN:
            raise ValueError(f"freeze saw non-train seed {r['seed']}")
        if r["seed"] in TEST:
            raise ValueError(f"freeze peeked test seed {r['seed']}")
        x = np.asarray(r["x"], dtype=np.float64)
        if x.size != len(names):
            raise ValueError(f"{arm} dim {x.size} != {len(names)}")
        seeds.append(int(r["seed"]))
        xs.append(x)
        xfts.append(np.asarray(r["x_ft"], dtype=np.float64))
    if len(seeds) != 8 or seeds != list(TRAIN):
        raise ValueError(f"{arm} freeze seeds {seeds} != {list(TRAIN)}")
    stack = np.stack(xs, axis=0)
    stats = coord_stats(stack)
    stats["names"] = list(names)
    stats["seeds"] = seeds
    stats["vectors"] = stack.tolist()
    stats["vectors_ft"] = np.stack(xfts, axis=0).tolist()
    return stats


def build_freeze(train_rows, prereg_sha):
    for r in train_rows:
        if r["seed"] in TEST or r["seed"] not in TRAIN:
            raise ValueError(f"build_freeze got illegal seed {r['seed']}")
        if r.get("split") != "train":
            raise ValueError(f"build_freeze got split={r.get('split')}")
    return {
        "frozen_before_test": True,
        "do_not_peek_test": True,
        "aggregator_not_chosen_on_test": True,
        "primary_aggregator": "mean",
        "also_frozen": "median",
        "n_train": 8,
        "caveat": "n=8 train vectors; concentration diagnostics are descriptive",
        "prereg_sha256": prereg_sha,
        "nov3": pack_arm(train_rows, "nov3_center4", NOV3_NAMES),
        "qaoa": pack_arm(train_rows, "qaoa_center4", QAOA_NAMES),
        "layout": "LE x; Nov3 identical to FT; QAOA p=1 stacked equals interleaved",
    }
