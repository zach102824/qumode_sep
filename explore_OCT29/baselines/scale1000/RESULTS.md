# Classical WalkSAT — scale1000 B=1000

_Generated 2026-10-30 09:22:15 +0800 (Asia/Shanghai)_

**Algo:** SKC WalkSAT (`explore_OCT29/lib/walksat.py`, pure Python), p=0.5, **R=20** runs/instance
**Pack:** `explore_OCT29/data/scale1000/data_n{9,12,15,18}.npz` split=`scale1000_*`
**Across-inst stats:** median / B90 of *per-instance median* over R runs
**Out JSON:** `baselines/scale1000/walksat_n{n}.json`

## Metric note (do not mix)

| side | cost metric | meaning |
|------|-------------|---------|
| **classical WalkSAT** | median **flips** (also report **evals**) | flips until first satisfying assignment; evals ≈ break-count queries |
| **quantum Arm C** | median **1/p** | inverse success probability of measuring a solution (L=16 layers) |

These are **not** the same unit. Scaling exponents `b` below are within-metric only (`cost ~ b^n`).

## Ladder — flips

| n | Med flips | B90 flips | mean flips | Med evals | B90 evals | B | wall_s |
|---|-----------|-----------|------------|-----------|-----------|---|--------|
| 9 | **13.000** | 37.050 | 18.303 | 53.000 | 149.200 | 1000 | 10.9 |
| 12 | **26.500** | 69.550 | 34.831 | 107.000 | 279.200 | 1000 | 20.6 |
| 15 | **44.750** | 123.550 | 62.796 | 180.000 | 495.200 | 1000 | 31.2 |
| 18 | **75.500** | 210.650 | 102.248 | 303.000 | 843.600 | 1000 | 48.9 |

## b fits (flips Med unless noted)

- **9→18** b_Med≈**1.2159**, b_B90≈**1.2130**
- **9→15** b_Med≈**1.2288**
- **9→12** b_Med≈**1.2679**
- **12→15** b_Med≈**1.1908**
- **15→18** b_Med≈**1.1905**
- **9→18 evals** b_Med≈**1.2138**, b_B90≈**1.2123**

## Side-by-side vs Arm C (same packs; different metrics)

Arm C B=1000 Med = median **1/p** (from `runs/armC_B1000`, n18 still running at write time).

| n | WalkSAT Med flips | WalkSAT B90 flips | Arm C Med 1/p | Arm C B90 1/p |
|---|-------------------|-------------------|---------------|---------------|
| 9 | 13.000 | 37.050 | 1.441 | 3.314 |
| 12 | 26.500 | 69.550 | 1.991 | 6.252 |
| 15 | 44.750 | 123.550 | 2.921 | 10.959 |
| 18 | 75.500 | 210.650 | *(pending)* | *(pending)* |

- Classical **b_Med(flips) 9→18 ≈ 1.216**
- Quantum Arm C **b_Med(1/p) 9→15 ≈ 1.125** (n18 incomplete on B=1000)

## Notes

- `scale1000_*` and `test_*` keys in these packs are identical (same seeds/clauses); runner used `--split scale1000`.
- Ran single-thread `nice -n 19` so as not to starve `eval_armC_B1000` (6 threads).
- Script: `scripts/run_walksat_bars.py` now accepts `--split`.
