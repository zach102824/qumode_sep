# jp gate: SPSA step-budget scan (200 / 400 / 800 steps), Gibbs cost only

Updated Asia/Shanghai: 2026-09-29 15:30 CST. New fleets created 06:26–07:23 UTC (14:26–15:23 CST).

## Setup

- Gate `jp`, noiseless ECD VQE, 20 `four_sat` Hamiltonians × 25 trials = 500 trials per cell.
- Pure Gibbs cost: λ1=0, λ=0, no β cap, no adaptive λ; SPSA; seed 20260917 (per-trial seeds identical across step budgets).
- Command: `python noiseless/run_u_sweep.py --u-names jp --layers {L} --trials 25 --steps {S} --workers 1 --seed 20260917 --lambda1 0 --optimizer spsa --tag jp_L{L}_s{S}`.
- Stored `args` of all new fleets are identical to the 200-step fleets except `steps` (the old L*=4 file `jp_B0_20260928T042134Z.json` predates the `optimizer` field; it is SPSA).
- SPSA gain schedule does **not** depend on the step budget: a_k = a/(k+A)^0.602, c_k = c/k^0.101 with A=10, c=0.15, a = 0.2·sqrt(37/n_params) (L*=2: 0.3041, L*=3: 0.2483, L*=4: 0.2151; same in every file). η refresh cadence (every 5 steps) is also step-independent. A longer run therefore just continues the same decaying schedule; with identical seeds, the first 200 SPSA steps of a 400/800-step trial follow the same schedule and random draws as the 200-step trial.

Source files (200-step values recomputed from full JSONs with the same code):

- L2_s200: `jp_L2_20260929T041311Z.json`
- L2_s400: `jp_L2_s400_20260929T062619Z.json`
- L2_s800: `jp_L2_s800_20260929T065200Z.json`
- L3_s200: `jp_L3_20260929T041643Z.json`
- L3_s400: `jp_L3_s400_20260929T063319Z.json`
- L3_s800: `jp_L3_s800_20260929T070550Z.json`
- L4_s200: `jp_B0_20260928T042134Z.json`
- L4_s400: `jp_L4_s400_20260929T064228Z.json`
- L4_s800: `jp_L4_s800_20260929T072352Z.json`

## Results

Best-of-25 = per-Hamiltonian max p(GS) over 25 trials, then mean over 20 H. Mean trial-max |β| = mean over trials of max_k |β_k|. Wall = mean per-trial time (1 worker, OMP_NUM_THREADS=1).

| L* | steps | success | mean p(GS) | median p(GS) | best-of-25 mean p(GS) | mean \|β\| | mean trial-max \|β\| | wall / trial (s) |
|---|---|---|---|---|---|---|---|---|
| 2 | 200 | 0.456 | 0.1297 | 0.0833 | 0.3061 | 2.185 | 3.444 | 0.30 |
| 2 | 400 | 0.486 | 0.1487 | 0.1058 | 0.3479 | 2.213 | 3.548 | 0.57 |
| 2 | 800 | 0.462 | 0.1624 | 0.1193 | 0.3765 | 2.237 | 3.642 | 1.14 |
| 3 | 200 | 0.774 | 0.1581 | 0.1270 | 0.3646 | 1.928 | 3.371 | 0.42 |
| 3 | 400 | 0.804 | 0.1900 | 0.1516 | 0.4271 | 1.941 | 3.470 | 0.84 |
| 3 | 800 | 0.820 | 0.2207 | 0.1758 | 0.4844 | 1.956 | 3.582 | 1.66 |
| 4 | 200 | 0.932 | 0.1717 | 0.1434 | 0.3631 | 1.847 | 3.388 | 0.53 |
| 4 | 400 | 0.946 | 0.2133 | 0.1818 | 0.4358 | 1.855 | 3.458 | 1.09 |
| 4 | 800 | 0.950 | 0.2558 | 0.2210 | 0.4863 | 1.864 | 3.539 | 2.16 |

## Paired comparison vs 200 steps (same ham_file + trial + seed, n=500 per L*)

| L* | vs | frac p(GS) higher than 200-step | mean Δp(GS) | median Δp(GS) | fail@200 → succeed | succeed@200 → fail | net |
|---|---|---|---|---|---|---|---|
| 2 | 400 | 0.870 | +0.0190 | +0.0053 | 26 | 11 | +15 |
| 2 | 800 | 0.872 | +0.0327 | +0.0083 | 32 | 29 | +3 |
| 3 | 400 | 0.970 | +0.0319 | +0.0201 | 22 | 7 | +15 |
| 3 | 800 | 0.968 | +0.0626 | +0.0385 | 33 | 10 | +23 |
| 4 | 400 | 0.992 | +0.0416 | +0.0317 | 8 | 1 | +7 |
| 4 | 800 | 0.992 | +0.0841 | +0.0665 | 10 | 1 | +9 |

(No ties in p(GS) in any pairing.)

## Conclusions

- **p(GS) keeps improving with steps at every depth; 200 steps is not converged.** 200→800 raises mean p(GS) by +25% (L*=2), +40% (L*=3), +49% (L*=4) and best-of-25 by ~+0.07–0.12 absolute. Gains from 400→800 are about as large as from 200→400, so the optimizer is still climbing at 800.
- **Deeper circuits benefit most:** the paired fraction improving is 0.87 (L*=2), 0.97 (L*=3), 0.99 (L*=4); at L*=4, 800 steps beats 200 steps on 496/500 trials.
- **Success rate (argmax = ground state) barely moves:** L*=3 0.774→0.820, L*=4 0.932→0.950, and L*=2 is flat/noisy (0.456→0.486→0.462). At L*=2 more steps sharpen p(GS) but flip argmax in both directions (32 gained, 29 lost), i.e. the shallow ansatz's failures are landscape/expressivity-limited, not step-limited. Depth remains the main lever for success; steps are the lever for p(GS).
- **|β| is essentially step-independent:** mean |β| changes by ≤0.05 and trial-max |β| drifts up by only ~0.15–0.2 from 200→800, so longer Gibbs-only optimization does not run away in displacement.
- **Cost is linear in steps** (≈1.4 ms/step at L*=2, 2.1 ms at L*=3, 2.7 ms at L*=4 per trial).
