"""Decision rule for the shallow Walsh SNAP vs plain Fourier QAOA seal.

R_Fourier = Medε_SNAP / Medε_Fourier on frozen n=12.
Medε is the median over seeds of each seed's median-over-instances excess
energy ⟨H⟩−E₀. The instance bootstrap resamples instances inside each seed
and keeps the seeds fixed.

PASS on a depth when all three hold:
  - R_Fourier ≤ 0.85
  - at least 4 of 5 seeds have their own n12 ratio ≤ 0.85
  - the instance-bootstrap 95% interval ends strictly below 0.95

KILL when Fourier takes the depth: median R_Fourier > 0.98, or at least 4 of
5 seeds have ratio > 0.98. needs_edit is the soft remainder. L≥8 is not a
depth this rule can accept.
"""
from __future__ import annotations

import math

import numpy as np

BEAT_R = 0.85
CI_UPPER = 0.95
KILL_R = 0.98
N_SEEDS = 5
MIN_BEAT_SEEDS = 4
BOOT_DRAWS = 4000
BOOT_SEED = 5209
DEPTHS = (2, 4)


def ratio(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else (0.0 if num == 0 else float("-inf"))
    return float(num) / float(den)


def decide_depth(r_med: float, seed_rs: list[float], ci_hi: float, n_seeds: int = N_SEEDS) -> str:
    """PASS / needs_edit / KILL for one depth at frozen n12."""
    if int(n_seeds) != N_SEEDS or len(seed_rs) != N_SEEDS:
        return "INCOMPLETE"
    n_beat = sum(1 for r in seed_rs if float(r) <= BEAT_R)
    n_taken = sum(1 for r in seed_rs if float(r) > KILL_R)
    if (not math.isfinite(r_med)) or float(r_med) > KILL_R or n_taken >= MIN_BEAT_SEEDS:
        return "KILL"
    ci_ok = math.isfinite(ci_hi) and float(ci_hi) < CI_UPPER
    if float(r_med) <= BEAT_R and n_beat >= MIN_BEAT_SEEDS and ci_ok:
        return "PASS"
    return "needs_edit"


def decide_overall(by_depth: dict[int, str]) -> str:
    if set(by_depth) != set(DEPTHS):
        return "INCOMPLETE"
    vals = [by_depth[L] for L in DEPTHS]
    if any(v == "INCOMPLETE" for v in vals):
        return "INCOMPLETE"
    if any(v == "KILL" for v in vals):
        return "KILL"
    if any(v == "needs_edit" for v in vals):
        return "needs_edit"
    if all(v == "PASS" for v in vals):
        return "PASS"
    return "INCOMPLETE"


def paired_counts(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 1:
        raise ValueError("paired shape")
    delta = a - b
    return {
        "win": int(np.sum(delta < 0)),
        "tie": int(np.sum(delta == 0)),
        "loss": int(np.sum(delta > 0)),
        "n": int(a.size),
    }


def _median_ratio(methods: list[np.ndarray], bases: list[np.ndarray]) -> float:
    med_m = float(np.median([float(np.median(a)) for a in methods]))
    med_b = float(np.median([float(np.median(b)) for b in bases]))
    return ratio(med_m, med_b)


def bootstrap_ratio(
    eps_method,
    eps_base,
    *,
    n_boot: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
) -> dict:
    """Instance bootstrap of the median-over-seeds ratio. Seeds stay fixed."""
    methods = [np.asarray(a, dtype=np.float64).reshape(-1) for a in eps_method]
    bases = [np.asarray(b, dtype=np.float64).reshape(-1) for b in eps_base]
    if len(methods) != len(bases) or len(methods) < 2:
        raise ValueError("bootstrap seeds")
    width = methods[0].size
    for a, b in zip(methods, bases):
        if a.shape != b.shape or int(a.size) != width:
            raise ValueError("bootstrap instance axis")
    point = _median_ratio(methods, bases)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.float64)
    n_seed = len(methods)
    for t in range(int(n_boot)):
        ms, bs = [], []
        for s in range(n_seed):
            idx = rng.integers(0, width, size=width)
            ms.append(methods[s][idx])
            bs.append(bases[s][idx])
        draws[t] = _median_ratio(ms, bs)
    ok = draws[np.isfinite(draws)]
    if ok.size < int(n_boot):
        raise RuntimeError(f"nonfinite bootstrap draws {int(n_boot) - int(ok.size)}")
    lo, hi = np.quantile(ok, [0.025, 0.975])
    return {
        "R": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(ok.size),
        "n_seeds": int(n_seed),
        "n_inst": int(width),
        "seed": int(seed),
        "mode": "instance",
    }


def self_test() -> None:
    five = [0.30, 0.32, 0.33, 0.34, 0.35]
    if decide_depth(0.32, five, 0.40) != "PASS":
        raise AssertionError("clear shallow beat must PASS")
    if decide_depth(0.32, five, 0.95) != "needs_edit":
        raise AssertionError("CI upper == 0.95 is not under 0.95")
    if decide_depth(0.32, five, 0.949) != "PASS":
        raise AssertionError("CI upper 0.949 clears the strict bound")
    three = [0.40, 0.50, 0.60, 0.90, 0.92]
    if decide_depth(0.60, three, 0.80) != "needs_edit":
        raise AssertionError("3/5 BEAT is soft")
    four = [0.40, 0.50, 0.60, 0.80, 0.90]
    if decide_depth(0.60, four, 0.90) != "PASS":
        raise AssertionError("4/5 BEAT with R and CI inside the bar is PASS")
    if decide_depth(0.90, five, 0.40) != "needs_edit":
        raise AssertionError("median R above 0.85 is soft when Fourier has not taken it")
    if decide_depth(0.98, five, 0.40) != "needs_edit":
        raise AssertionError("R == 0.98 is not yet KILL")
    if decide_depth(0.9800001, five, 0.40) != "KILL":
        raise AssertionError("R > 0.98 is KILL")
    if decide_depth(1.20, five, 0.40) != "KILL":
        raise AssertionError("Fourier ahead on the median is KILL")
    taken = [1.01, 1.05, 1.10, 1.20, 0.40]
    if decide_depth(0.90, taken, 0.80) != "KILL":
        raise AssertionError("4/5 seeds with R>0.98 is KILL")
    soft_taken = [1.01, 1.05, 1.10, 0.70, 0.40]
    if decide_depth(0.90, soft_taken, 0.99) != "needs_edit":
        raise AssertionError("3/5 seeds above 0.98 with median 0.90 is soft")
    if decide_depth(0.32, five[:4], 0.40, n_seeds=4) != "INCOMPLETE":
        raise AssertionError("the seal is five seeds")
    if decide_overall({2: "PASS", 4: "PASS"}) != "PASS":
        raise AssertionError("both depths")
    if decide_overall({2: "PASS", 4: "needs_edit"}) != "needs_edit":
        raise AssertionError("one soft depth")
    if decide_overall({2: "PASS", 4: "KILL"}) != "KILL":
        raise AssertionError("one kill depth")
    if decide_overall({2: "KILL", 4: "needs_edit"}) != "KILL":
        raise AssertionError("kill dominates soft")
    if decide_overall({2: "PASS"}) != "INCOMPLETE":
        raise AssertionError("L=4 required")
    if decide_overall({2: "PASS", 8: "PASS"}) != "INCOMPLETE":
        raise AssertionError("L=8 is outside the seal")

    # Paired bootstrap point equals the ratio of median-over-seeds medians.
    rng = np.random.default_rng(0)
    snap = [rng.normal(loc=0.08, scale=0.01, size=32) for _ in range(5)]
    fou = [rng.normal(loc=0.26, scale=0.01, size=32) for _ in range(5)]
    boot = bootstrap_ratio(snap, fou, n_boot=200, seed=1)
    if abs(boot["R"] - _median_ratio(snap, fou)) > 1e-12:
        raise AssertionError("bootstrap point drifted from the table ratio")
    if not (boot["lo"] <= boot["R"] <= boot["hi"]):
        raise AssertionError("point outside its own interval")
    if boot["n_inst"] != 32 or boot["n_seeds"] != 5:
        raise AssertionError("bootstrap shape")
