"""Train-only two-template freeze. Must not read test fits.

Nov3 clusters by sign(s1); QAOA clusters by sign(β). Empty clusters stay empty.
Sign-aligned secondary flips ONLY s1 when s1<0. No extra gauge.
"""
from __future__ import annotations

import numpy as np

NOV3_NAMES = ("gamma", "theta_K", "theta_M", "s1", "s2", "s3")
QAOA_NAMES = ("gamma", "beta")
TRAIN = tuple(range(10776, 10784))
TEST = tuple(range(10784, 10792))
S1_IDX = 3
BETA_IDX = 1
GAUGE = {
    "map": "flip_s1_only_if_s1_lt_0",
    "covariant_discrete_gauge_in_saved_recipe": False,
    "theta_K_theta_M_unchanged": True,
    "s1_index": S1_IDX,
    "layout": list(NOV3_NAMES),
}


def coord_stats(stack: np.ndarray) -> dict:
    stack = np.asarray(stack, dtype=np.float64)
    if stack.size == 0:
        raise ValueError("coord_stats on empty stack")
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


def collect_arm(rows: list[dict], arm: str, names: tuple[str, ...]) -> tuple[list[int], np.ndarray]:
    seeds, xs = [], []
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
    return seeds, np.stack(xs, axis=0)


def pack_unsigned(seeds: list[int], stack: np.ndarray, names: tuple[str, ...]) -> dict:
    stats = coord_stats(stack)
    stats["names"] = list(names)
    stats["seeds"] = seeds
    stats["vectors"] = stack.tolist()
    stats["empty"] = False
    return stats


def align_s1(x: np.ndarray) -> tuple[np.ndarray, bool]:
    y = np.asarray(x, dtype=np.float64).copy()
    flipped = bool(y[S1_IDX] < 0.0)
    if flipped:
        y[S1_IDX] = -y[S1_IDX]
    return y, flipped


def pack_sign_aligned(seeds: list[int], stack: np.ndarray) -> dict:
    aligned = []
    flipped_seeds = []
    for seed, x in zip(seeds, stack):
        y, flipped = align_s1(x)
        aligned.append(y)
        if flipped:
            flipped_seeds.append(int(seed))
    aligned = np.stack(aligned, axis=0)
    stats = coord_stats(aligned)
    stats["names"] = list(NOV3_NAMES)
    stats["seeds"] = seeds
    stats["vectors_aligned"] = aligned.tolist()
    stats["n_flipped"] = len(flipped_seeds)
    stats["seeds_flipped"] = flipped_seeds
    stats["gauge"] = dict(GAUGE)
    stats["empty"] = False
    stats["secondary_only"] = True
    return stats


def pack_cluster(seeds: list[int], stack: np.ndarray, names: tuple[str, ...], coord_idx: int, sign: str) -> dict:
    col = stack[:, int(coord_idx)]
    if sign == "plus":
        mask = col > 0.0
        chosen = f"{names[coord_idx]}>0"
    elif sign == "minus":
        mask = col < 0.0
        chosen = f"{names[coord_idx]}<0"
    else:
        raise ValueError(sign)
    n_pos = int((col > 0.0).sum())
    n_neg = int((col < 0.0).sum())
    n_zero = int((col == 0.0).sum())
    sub_seeds = [int(s) for s, keep in zip(seeds, mask) if keep]
    if int(mask.sum()) == 0:
        return {
            "empty": True,
            "do_not_invent_template": True,
            "names": list(names),
            "seeds": [],
            "n": 0,
            "dim": int(stack.shape[1]),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_zero": n_zero,
            "sign": sign,
            "chosen": chosen,
            "coord": names[coord_idx],
        }
    sub = stack[mask]
    stats = coord_stats(sub)
    stats["names"] = list(names)
    stats["seeds"] = sub_seeds
    stats["vectors"] = sub.tolist()
    stats["empty"] = False
    stats["n_pos"] = n_pos
    stats["n_neg"] = n_neg
    stats["n_zero"] = n_zero
    stats["sign"] = sign
    stats["chosen"] = chosen
    stats["coord"] = names[coord_idx]
    return stats


def two_template_plan(plus: dict, minus: dict) -> dict:
    nonempty = []
    if not plus.get("empty"):
        nonempty.append("plus")
    if not minus.get("empty"):
        nonempty.append("minus")
    return {
        "n_nonempty": len(nonempty),
        "templates": nonempty,
        "reduces_to_single_cluster": len(nonempty) == 1,
        "both_empty": len(nonempty) == 0,
        "two_template": len(nonempty) == 2,
    }


def template_catalog(freeze: dict) -> list[dict]:
    """Locked list of transfer vectors. Empty clusters omitted."""
    out = []
    mapping = (
        ("nov3_center4", "Nov3rd", "cluster_plus", freeze["nov3_cluster_plus"]),
        ("nov3_center4", "Nov3rd", "cluster_minus", freeze["nov3_cluster_minus"]),
        ("nov3_center4", "Nov3rd", "unsigned_mean", freeze["nov3_unsigned"]),
        ("nov3_center4", "Nov3rd", "sign_aligned_mean", freeze["nov3_sign_aligned"]),
        ("qaoa_center4", "QAOA", "cluster_plus", freeze["qaoa_cluster_plus"]),
        ("qaoa_center4", "QAOA", "cluster_minus", freeze["qaoa_cluster_minus"]),
        ("qaoa_center4", "QAOA", "unsigned_mean", freeze["qaoa_unsigned"]),
    )
    for arm, method, tag, block in mapping:
        if block.get("empty"):
            continue
        out.append({
            "arm": arm,
            "method": method,
            "tag": tag,
            "x": block["mean"] if tag != "sign_aligned_mean" else block["mean"],
            "secondary": tag in {"unsigned_mean", "sign_aligned_mean"},
        })
    return out


def build_freeze(train_rows: list[dict], prereg_sha: str) -> dict:
    for r in train_rows:
        if r["seed"] in TEST or r["seed"] not in TRAIN:
            raise ValueError(f"build_freeze got illegal seed {r['seed']}")
        if r.get("split") != "train":
            raise ValueError(f"build_freeze got split={r.get('split')}")
    n_seeds, n_stack = collect_arm(train_rows, "nov3_center4", NOV3_NAMES)
    q_seeds, q_stack = collect_arm(train_rows, "qaoa_center4", QAOA_NAMES)
    nov3_plus = pack_cluster(n_seeds, n_stack, NOV3_NAMES, S1_IDX, "plus")
    nov3_minus = pack_cluster(n_seeds, n_stack, NOV3_NAMES, S1_IDX, "minus")
    qaoa_plus = pack_cluster(q_seeds, q_stack, QAOA_NAMES, BETA_IDX, "plus")
    qaoa_minus = pack_cluster(q_seeds, q_stack, QAOA_NAMES, BETA_IDX, "minus")
    freeze = {
        "frozen_before_test": True,
        "do_not_peek_test": True,
        "aggregator_not_chosen_on_test": True,
        "primary_gate": "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI",
        "primary_rule": "test better-of-two Nov3 minus better-of-two QAOA; min of frozen cluster energies; energy-only",
        "do_not_switch_primary_after_test": True,
        "not_unsigned_mean_copy": True,
        "not_sign_aligned_only_copy": True,
        "n_train": 8,
        "caveat": "n=8 train vectors; clusters may be 4/4 or unbalanced; empty clusters stay empty",
        "prereg_sha256": prereg_sha,
        "gauge": dict(GAUGE),
        "nov3_unsigned": pack_unsigned(n_seeds, n_stack, NOV3_NAMES),
        "nov3_sign_aligned": pack_sign_aligned(n_seeds, n_stack),
        "nov3_cluster_plus": nov3_plus,
        "nov3_cluster_minus": nov3_minus,
        "nov3_two_template": two_template_plan(nov3_plus, nov3_minus),
        "qaoa_unsigned": pack_unsigned(q_seeds, q_stack, QAOA_NAMES),
        "qaoa_cluster_plus": qaoa_plus,
        "qaoa_cluster_minus": qaoa_minus,
        "qaoa_two_template": two_template_plan(qaoa_plus, qaoa_minus),
    }
    freeze["catalog"] = template_catalog(freeze)
    freeze["n_templates"] = len(freeze["catalog"])
    freeze["planned_transfer_queries"] = 8 * len(freeze["catalog"])
    freeze["unsigned_and_sign_aligned_are_secondary"] = True
    return freeze
