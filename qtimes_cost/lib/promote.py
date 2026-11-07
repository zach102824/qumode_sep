"""Val-only promotion gate for the cloud_day1 farm.

Lower S_comb is better. A healthier n6→n9 median ratio is never enough
on its own: n9 median must hold, and a very large ratio is vetoed unless
n9 itself improved clearly.
"""
from __future__ import annotations

# Previous box noon_explore r014 (superseded). Small-val, B90 not reported.
BOX_R014 = {
    "tag": "r014",
    "L": 10,
    "leakpen": 0.1,
    "lr": 1e-4,
    "alpha0": "learned",
    "med6": 18.0,
    "med9": 23.0,
    "ratio_9_6": 23.0 / 18.0,
    "note": "earlier box BEST; optimistic small val",
}

# Current box BEST. A win claim has to beat the reval40 numbers, not the small val.
BOX_NEW = {
    "tag": "box_L10_lr5e-5",
    "L": 10,
    "leakpen": 0.1,
    "lr": 5e-5,
    "alpha0": "learned",
    "small_val_med6": 10.2,
    "small_val_med9": 18.0,
    "reval40_med6": 14.5,
    "reval40_med9": 19.5,
    "ratio_reval": 19.5 / 14.5,
    "note": "box NEW BEST; reval40 is the bar for a win",
}


# Cloud day-1 L=12 trial 0, the parent for the next fork. Val-40 is the promote bar.
CLOUD_L12 = {
    "tag": "cloud_L12_trial0",
    "L": 12,
    "leakpen": 0.1,
    "lr": 5e-5,
    "alpha0": "learned",
    "val40_med6": 12.006427764892578,
    "val40_med9": 16.979470252990723,
    "val40_S": 3.3937208125844736,
    "test_med6": 15.118804454803467,
    "test_med9": 16.78520965576172,
    "test_S": 3.578523508847792,
    "note": "saved BEST before the L=12/14/16 fork",
}


def ratio_9_6(med6: float, med9: float) -> float:
    return float(med9) / max(float(med6), 1e-9)


def beats_box_reval(
    med6: float,
    med9: float,
    *,
    med6_bar: float = BOX_NEW["reval40_med6"],
    med9_bar: float = BOX_NEW["reval40_med9"],
) -> dict:
    """Full win only if both reval medians beat the box and the ratio is healthy.

    Search-val numbers must not be passed here.
    """
    ratio = ratio_9_6(med6, med9)
    healthy = bool(0.75 <= ratio <= 2.0)
    beats6 = bool(med6 < med6_bar)
    beats9 = bool(med9 < med9_bar)
    return {
        "med6": float(med6),
        "med9": float(med9),
        "ratio_9_6": ratio,
        "ratio_healthy": healthy,
        "beats_n6": beats6,
        "beats_n9": beats9,
        "full_win": bool(beats6 and beats9 and healthy),
    }


def fork_promote(
    val: dict,
    test: dict,
    champ: dict,
    *,
    test_caps: dict,
    min_improve: float = 1e-4,
) -> tuple[bool, str]:
    """Promote only after val-40 and test.

    Both val medians and val S_comb must beat the champion. Test medians must
    stay under the fixed caps taken from the original cloud L=12 test
    (a modest slack). Ratio on val must stay in [0.75, 2].
    """
    med6 = float(val["med6"])
    med9 = float(val["med9"])
    s_comb = float(val["s_comb"])
    if not (med6 == med6 and med9 == med9 and s_comb == s_comb):
        return False, "nonfinite"
    ratio = ratio_9_6(med6, med9)
    if not (0.75 <= ratio <= 2.0):
        return False, "ratio"
    if med6 >= float(champ["med6"]):
        return False, "val_med6"
    if med9 >= float(champ["med9"]):
        return False, "val_med9"
    if s_comb + min_improve >= float(champ["s_comb"]):
        return False, "val_S"
    t6 = float(test["med6"])
    t9 = float(test["med9"])
    if not (t6 == t6 and t9 == t9):
        return False, "test_nonfinite"
    if t6 > float(test_caps["med6"]) or t9 > float(test_caps["med9"]):
        return False, "test_wreck"
    return True, "promote"


def promotion_decision(
    cand: dict,
    champ: dict | None,
    *,
    min_improve: float = 1e-4,
    n9_slack: float = 0.05,
    max_healthy_ratio: float = 2.5,
    max_leak9: float = 0.35,
) -> tuple[bool, str]:
    """Return (keep, reason).

    cand/champ need s_comb, med6, med9. leak9 is optional.
    """
    if champ is None:
        if cand.get("s_comb") is None:
            return False, "empty"
        return True, "first"

    s_c = float(cand["s_comb"])
    s_b = float(champ["s_comb"])
    med9_c = float(cand["med9"])
    med9_b = float(champ["med9"])
    med6_c = float(cand["med6"])
    med6_b = float(champ["med6"])
    if not (s_c == s_c and med9_c == med9_c and med6_c == med6_c):  # NaN
        return False, "nonfinite"

    ratio_c = ratio_9_6(med6_c, med9_c)
    ratio_b = ratio_9_6(med6_b, med9_b)
    s_better = s_c + min_improve < s_b
    n9_better = med9_c < med9_b
    # Ratio improved while n9 did not: do not promote, even if S moved.
    ratio_only = (ratio_c + 1e-6 < ratio_b) and (not n9_better)
    if ratio_only and med9_c > med9_b * (1.0 - 1e-6):
        return False, "ratio_only"

    if not s_better:
        return False, "s_comb"

    n9_ceiling = med9_b * (1.0 + n9_slack) + 0.5
    if med9_c > n9_ceiling:
        return False, "n9_regressed"

    n9_clear = med9_c <= med9_b * 0.90
    if ratio_c > max_healthy_ratio and not n9_clear:
        return False, "unhealthy_ratio"

    leak9 = cand.get("leak9")
    if leak9 is not None and float(leak9) > max_leak9 and not n9_clear:
        return False, "leak"

    return True, "promote"
