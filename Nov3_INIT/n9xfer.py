"""Train-only mean/median freeze. Must not read test fits."""
from __future__ import annotations

import numpy as np

NOV3_NAMES = ("gamma", "theta_K", "theta_M", "s1", "s2", "s3")
QAOA_NAMES = ("gamma", "beta")
TRAIN = tuple(range(10696, 10704))
TEST = tuple(range(10704, 10712))
PR116_N15_TRAIN_STD = {
    "theta_K": 0.0511,
    "theta_M": 0.0233,
    "s1": 0.8058,
    "beta": 0.4015,
}
PR118_N12_TRAIN_STD = {
    "theta_K": 0.0573,
    "theta_M": 0.0152,
    "s1": 0.8265,
    "beta": 0.5523,
}


def coord_stats(stack: np.ndarray) -> dict:
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


def pack_arm(rows: list[dict], arm: str, names: tuple[str, ...]) -> dict:
    seeds = []
    xs = []
    for r in sorted(rows, key=lambda z: z["seed"]):
        if r.get("arm") != arm or not r.get("ok"):
            continue
        if r["seed"] not in TRAIN:
            raise ValueError(f"freeze saw non-train seed {r['seed']}")
        if r["seed"] in TEST:
            raise ValueError(f"freeze peeked test seed {r['seed']}")
        if r.get("split") != "train":
            raise ValueError(f"freeze saw split={r.get('split')}")
        x = np.asarray(r["x"], dtype=np.float64)
        if x.size != len(names):
            raise ValueError(f"{arm} dim {x.size} != {len(names)}")
        seeds.append(int(r["seed"]))
        xs.append(x)
    if len(seeds) != 8 or seeds != list(TRAIN):
        raise ValueError(f"{arm} freeze seeds {seeds} != {list(TRAIN)}")
    stack = np.stack(xs, axis=0)
    stats = coord_stats(stack)
    stats["names"] = list(names)
    stats["seeds"] = seeds
    stats["vectors"] = stack.tolist()
    return stats


def build_freeze(train_rows: list[dict], prereg_sha: str) -> dict:
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
        "do_not_switch_primary_to_median": True,
        "n_train": 8,
        "caveat": "n=8 train vectors; concentration diagnostics are descriptive",
        "prereg_sha256": prereg_sha,
        "nov3": pack_arm(train_rows, "nov3_center4", NOV3_NAMES),
        "qaoa": pack_arm(train_rows, "qaoa_center4", QAOA_NAMES),
        "pr116_n15_train_std_descriptive_not_paired": PR116_N15_TRAIN_STD,
        "pr118_n12_train_std_descriptive_not_paired": PR118_N12_TRAIN_STD,
    }
