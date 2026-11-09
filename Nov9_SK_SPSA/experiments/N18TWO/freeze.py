"""Train-only two-template cluster-mean freeze. Must not read test fits.

Nov3 clusters by sign(s1); QAOA clusters by sign(beta). Empty clusters are
recorded and do not invent a template. Sign-alignment (secondary) flips ONLY
s1 when s1<0; no covariant discrete gauge is documented in the saved recipe.
"""
from __future__ import annotations

import numpy as np

NOV3_NAMES = ("gamma", "theta_K", "theta_M", "s1", "s2", "s3")
QAOA_NAMES = ("gamma", "beta")
TRAIN = tuple(range(10728, 10736))
TEST = tuple(range(10736, 10744))
S1_IDX = 3
BETA_IDX = 1
GAUGE = {
    "map": "flip_s1_only_if_s1_lt_0",
    "covariant_discrete_gauge_in_saved_recipe": False,
    "theta_K_theta_M_unchanged": True,
    "s1_index": S1_IDX,
    "layout": list(NOV3_NAMES),
}


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


def collect_arm(rows, arm, names):
    seeds, xs, xfts = [], [], []
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
    return seeds, np.stack(xs, axis=0), np.stack(xfts, axis=0)


def pack_unsigned(seeds, stack, xfts, names):
    stats = coord_stats(stack)
    stats["names"] = list(names)
    stats["seeds"] = seeds
    stats["vectors"] = stack.tolist()
    stats["vectors_ft"] = xfts.tolist()
    return stats


def align_s1(x):
    y = np.asarray(x, dtype=np.float64).copy()
    flipped = bool(y[S1_IDX] < 0)
    if flipped:
        y[S1_IDX] = -y[S1_IDX]
    return y, flipped


def pack_sign_aligned(seeds, stack):
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
    return stats


def pack_one_cluster(seeds, stack, mask, names, label):
    n = int(np.asarray(mask).sum())
    if n == 0:
        return {
            "empty": True,
            "n": 0,
            "dim": len(names),
            "mean": None,
            "names": list(names),
            "seeds": [],
            "vectors": [],
            "label": label,
        }
    sub = stack[mask]
    stats = coord_stats(sub)
    stats["empty"] = False
    stats["names"] = list(names)
    stats["seeds"] = [int(s) for s, keep in zip(seeds, mask) if keep]
    stats["vectors"] = sub.tolist()
    stats["label"] = label
    return stats


def pack_signed_clusters(seeds, stack, names, coord_idx, coord_name):
    vals = stack[:, coord_idx]
    pos = vals > 0
    neg = vals < 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    n_zero = int((vals == 0).sum())
    return {
        "coord": coord_name,
        "index": int(coord_idx),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "pos": pack_one_cluster(seeds, stack, pos, names, f"{coord_name}>0"),
        "neg": pack_one_cluster(seeds, stack, neg, names, f"{coord_name}<0"),
        "n_templates": int(n_pos > 0) + int(n_neg > 0),
    }


def stage_b_vectors(freeze):
    """Nonempty cluster templates plus unsigned/aligned secondaries."""
    out = []
    for arm, key, agg_pos, agg_neg in (
        ("nov3_center4", "nov3_clusters", "cluster_pos", "cluster_neg"),
        ("qaoa_center4", "qaoa_clusters", "cluster_pos", "cluster_neg"),
    ):
        block = freeze[key]
        if not block["pos"].get("empty"):
            out.append((arm, agg_pos, block["pos"]["mean"]))
        if not block["neg"].get("empty"):
            out.append((arm, agg_neg, block["neg"]["mean"]))
    out.append(("nov3_center4", "unsigned_mean", freeze["nov3_unsigned"]["mean"]))
    out.append(("nov3_center4", "sign_aligned_mean", freeze["nov3_sign_aligned"]["mean"]))
    out.append(("qaoa_center4", "unsigned_mean", freeze["qaoa_unsigned"]["mean"]))
    return out


def build_freeze(train_rows, prereg_sha):
    for r in train_rows:
        if r["seed"] in TEST or r["seed"] not in TRAIN:
            raise ValueError(f"build_freeze got illegal seed {r['seed']}")
        if r.get("split") != "train":
            raise ValueError(f"build_freeze got split={r.get('split')}")
    n_seeds, n_stack, n_xft = collect_arm(train_rows, "nov3_center4", NOV3_NAMES)
    q_seeds, q_stack, q_xft = collect_arm(train_rows, "qaoa_center4", QAOA_NAMES)
    nov3_clusters = pack_signed_clusters(n_seeds, n_stack, NOV3_NAMES, S1_IDX, "s1")
    qaoa_clusters = pack_signed_clusters(q_seeds, q_stack, QAOA_NAMES, BETA_IDX, "beta")
    return {
        "frozen_before_test": True,
        "do_not_peek_test": True,
        "aggregator_not_chosen_on_test": True,
        "primary_rule": "better_of_two_cluster_means",
        "primary_contrast": "min_Nov3_cluster_means_minus_min_QAOA_cluster_means",
        "unsigned_and_aligned_are_secondary": True,
        "n_train": 8,
        "caveat": "n=8 train vectors; empty clusters do not invent a template",
        "prereg_sha256": prereg_sha,
        "gauge": dict(GAUGE),
        "nov3_unsigned": pack_unsigned(n_seeds, n_stack, n_xft, NOV3_NAMES),
        "nov3_sign_aligned": pack_sign_aligned(n_seeds, n_stack),
        "nov3_clusters": nov3_clusters,
        "qaoa_unsigned": pack_unsigned(q_seeds, q_stack, q_xft, QAOA_NAMES),
        "qaoa_clusters": qaoa_clusters,
        "nov3_n_templates": nov3_clusters["n_templates"],
        "qaoa_n_templates": qaoa_clusters["n_templates"],
        "layout": "LE x; Nov3 identical to FT; QAOA p=1 stacked equals interleaved",
        "not_a_posthoc_of_pr119_test": True,
        "does_not_reopen_pr119": True,
    }
