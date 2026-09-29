# jp: SPSA vs SPSA-Adam (same gradient estimator, Adam update)

Updated Asia/Shanghai: 2026-09-29 17:12 CST. Fleets run 17:05–17:09 CST (09:05–09:09 UTC), 8 workers, OMP_NUM_THREADS=1.

## What changed

`--optimizer spsa_adam` (`run_spsa_adam` in `spsa_gibbs.py`, `--adam-lr`, default 0.05). Each step is the SPSA step
unchanged: same η-refresh hook (steps 1, 6, 11, …), same c_k = 0.15 / k^0.101, same Rademacher draw
`rng.choice([-1, 1], n)` (the rng stream is consumed identically), 2 cost evals per step + 1 final eval. Only the
update differs: Adam (lr 0.05, β1 0.9, β2 0.999, eps 1e-8, bias-corrected) on ĝ instead of x −= a_k ĝ; the SPSA gain
(a, A, α) is not used. **Eval counts are identical to SPSA** (401 per 200-step trial; 802 / 1203 / 1604 for
growth to L = 2 / 3 / 4). With `--grow`, each stage starts a **fresh Adam state** (m = v = 0, bias correction restarts),
like the η controller restart. Default `spsa` path is bit-for-bit unchanged (golden-hash test vs commit 2c59b35).

Tests: `noiseless/tests/test_spsa_adam.py` (9 tests: eval count, same perturbations / rng stream as SPSA, same η
schedule, determinism, first-step = lr·sign(ĝ), grow, runner/CLI, SPSA golden hashes). Full suite: 76 passed.

## lr smoke (2 instances × 5 trials, L=4, 200 steps; 10 trials each)

| optimizer | success | mean p(GS) | median p(GS) |
|---|---|---|---|
| SPSA | 0.8 | 0.098 | 0.087 |
| Adam lr 0.02 | 0.6 | 0.073 | 0.059 |
| **Adam lr 0.05** | 1.0 | 0.154 | 0.151 |
| Adam lr 0.1 | 1.0 | 0.185 | 0.171 |

0.05 was not "clearly bad", so the full runs use the planned default 0.05 (0.1 looked a bit better on this tiny smoke;
not pursued).

## Setup

jp, binary encoding, λ1 = λ = 0 Gibbs, 20 four_sat × 25 trials = 500 trials per cell, `--seed 20260917`
(same per-trial seeds and x0 as the SPSA baselines).

```
python noiseless/run_u_sweep.py --u-names jp --layers 2,3,4 --trials 25 --steps 200 --workers 8 --seed 20260917 \
  --lambda1 0 --encoding binary --optimizer spsa_adam --adam-lr 0.05 --tag jp_adam_A
python noiseless/run_u_sweep.py ... --grow --grow-steps-per-stage 200 --tag jp_adam_C_grow_ps200
python noiseless/analyze_adam.py      # -> adam_stats_summary.json + tables
```

SPSA references: A = `jp_L2_20260929T041311Z`, `jp_L3_20260929T041643Z`, `jp_B0_20260928T042134Z`; C =
`jp_enc_C_bin_grow_ps200_20260929T084629Z`. Adam: `jp_adam_A_20260929T090642Z`, `jp_adam_C_grow_ps200_20260929T090938Z`.

## Results

| arm | L | evals/trial | success | mean p(GS) | median p(GS) | frac p(GS)>0.5 | best-of-25 | wall/trial (s) |
|---|---|---|---|---|---|---|---|---|
| A (SPSA) | 2 | 401 | 0.456 | 0.1297 | 0.0833 | 0.032 | 0.3061 | 0.30 |
| A_adam | 2 | 401 | 0.462 | 0.1384 | 0.0825 | 0.038 | 0.3163 | 0.31 |
| C (SPSA grow) | 2 | 802 | 0.486 | 0.2051 | 0.1642 | **0.106** | **0.3385** | 0.47 |
| C_adam | 2 | 802 | **0.510** | **0.2223** | **0.2284** | 0.086 | 0.2881 | 0.46 |
| A (SPSA) | 3 | 401 | 0.774 | 0.1581 | 0.1270 | 0.026 | 0.3646 | 0.42 |
| A_adam | 3 | 401 | 0.760 | 0.1839 | 0.1443 | 0.046 | 0.3696 | 0.44 |
| C (SPSA grow) | 3 | 1203 | 0.602 | 0.3052 | 0.2639 | 0.132 | **0.4948** | 0.89 |
| C_adam | 3 | 1203 | **0.772** | **0.3409** | **0.3207** | **0.148** | 0.4867 | 0.89 |
| A (SPSA) | 4 | 401 | **0.932** | 0.1717 | 0.1434 | 0.018 | 0.3631 | 0.53 |
| A_adam | 4 | 401 | 0.930 | 0.2080 | 0.1700 | 0.048 | 0.3987 | 0.58 |
| C (SPSA grow) | 4 | 1604 | 0.752 | 0.3819 | 0.3864 | **0.354** | **0.5802** | 1.44 |
| C_adam | 4 | 1604 | 0.852 | **0.3973** | **0.3957** | 0.288 | 0.5626 | 1.43 |

Per-stage (C_adam, final L=4): L1 0.114 / 0.092 → L2 0.528 / 0.230 → L3 0.778 / 0.341 → L4 0.852 / 0.397
(SPSA C: 0.146 / 0.105 → 0.546 / 0.223 → 0.656 / 0.320 → 0.752 / 0.382).

### Paired (same Hamiltonian + trial + seed + x0, n = 500)

| comparison | L | frac Adam higher | mean Δp(GS) | median Δp(GS) | SPSA-fail→succ | SPSA-succ→fail |
|---|---|---|---|---|---|---|
| A_adam vs A | 2 / 3 / 4 | 0.53 / 0.68 / 0.65 | +0.009 / +0.026 / +0.036 | +0.001 / +0.018 / +0.024 | 59 / 50 / 17 | 56 / 57 / 18 |
| C_adam vs C | 2 / 3 / 4 | 0.63 / 0.49 / 0.41 | +0.017 / +0.036 / +0.015 | ≈0 / ≈0 / −0.002 | 39 / **93** / **60** | 27 / 8 / 10 |

### Failure modes (argmax energy level for failed trials; level 1 = first excited)

| arm | L | n fail | frac first-excited | mean p(GS) in failures |
|---|---|---|---|---|
| A / A_adam | 2 | 272 / 269 | 0.96 / 0.93 | 0.065 / 0.068 |
| A / A_adam | 3 | 113 / 120 | 0.96 / 0.92 | 0.079 / 0.088 |
| A / A_adam | 4 | 34 / 35 | 0.88 / 0.97 | 0.081 / 0.113 |
| C / C_adam | 2 | 257 / 245 | 1.00 / 0.98 | 0.082 / 0.116 |
| C / C_adam | 3 | 199 / 114 | 1.00 / 0.97 | 0.062 / 0.054 |
| C / C_adam | 4 | 124 / 74 | 1.00 / 0.97 | 0.019 / 0.037 |

## Findings

- **Random init (A): Adam = SPSA on success, modestly better p(GS).** Success within noise (0.462 / 0.760 / 0.930 vs
  0.456 / 0.774 / 0.932). Mean p(GS) +0.009 / +0.026 / +0.036 (L4 0.208 vs 0.172, +21%), and Adam wins the paired
  comparison in 65–68% of trials at L3/L4. For reference, the L4 gain from Adam at 401 evals (0.208) is roughly what SPSA
  gets from 400 steps / 801 evals (A400 L4: 0.213).
- **Growth (C): Adam recovers much of the success growth had lost.** L3 success 0.602 → **0.772** (≈ random-init
  0.774), L4 0.752 → **0.852** (random-init still better at 0.932), L2 0.486 → 0.510. The flips are one-sided: 93 vs 8
  (L3) and 60 vs 10 (L4) SPSA-fail→succ vs succ→fail. Mean / median p(GS) go up slightly too (L3 0.341 / 0.321 vs
  0.305 / 0.264; L4 0.397 / 0.396 vs 0.382 / 0.386), but frac p(GS)>0.5 and best-of-25 dip a little at L4 (0.288 vs
  0.354; 0.563 vs 0.580). So Adam finds the GS basin more often but polishes the winners slightly less sharply in the
  same number of evals. The gain comes in the later stages (L1 stage success is actually lower with Adam: 0.11–0.13 vs
  0.15), i.e. Adam's per-coordinate normalized steps help later stages leave the L=1 basin.
- **Failure mode unchanged for growth:** C_adam failures still sit on the first-excited level (97–98%, vs 100% for
  SPSA C), with collapsed GS weight (median p(GS) in failures ≈ 0 at L3/L4, like SPSA C); there are just fewer of them.
  Random-init failures are also mostly first-excited (88–97%) but keep p(GS) ≈ 0.07–0.11.
- **Cost:** identical eval counts; wall time equal within noise (the Adam update is negligible next to the 2 circuit
  evals): A_adam fleet 85 s (1500 trials), C_adam fleet 176 s (SPSA C: 178 s).

## Caveats

- Single lr (0.05) chosen a priori; the 10-trial smoke hints 0.1 may be better at L4. No lr schedule / β tuning.
- Adam state resets per growth stage (documented choice); carrying moments across stages was not tested.
