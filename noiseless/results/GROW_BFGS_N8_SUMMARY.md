# n=8 jp ECD: layer growth + BFGS (L 1→4) vs growth + Adam, fixed-L BFGS, SPSA

Updated 2026-09-30 16:45 CST (Asia/Shanghai). Fleets ran 16:31–16:38 CST, sequentially, 8 workers, OMP_NUM_THREADS=1.

## Setup

jp, binary encoding, λ1 = λ = 0 Gibbs cost, 20 `four_sat` instances × 25 trials = 500 trials, `--seed 20260917`
(same per-trial seeds / L=1 x0 as every reference arm). Growth L=1→2→3→4 (start 1); each new layer is appended
probability-transparent + Gaussian kick σ on its 8 params; each stage is BFGS (scipy, finite-difference gradient,
gtol 1e-5) warm-started from the previous stage, `maxiter` 500 per stage, η handled with
`--bfgs-eta-mode restart` (BFGS to convergence at fixed η → refresh η with the usual sampled-tail controller/EMA →
restart with carried inverse Hessian until η moves < 1 %, a restart makes no iteration, or 500 iterations are used;
η controller restarted each stage). See `noiseless/README.md` ("Growth + BFGS").

```
python noiseless/run_u_sweep.py --u-names jp --layers 4 --trials 25 --workers 8 --seed 20260917 --lambda1 0 \
  --encoding binary --optimizer bfgs --grow --grow-steps-per-stage 500 --bfgs-eta-mode restart \
  [--grow-kick-sigma 0.5] --tag jp_grow_bfgs_restart[_kick05]
python noiseless/analyze_grow_bfgs.py   # -> grow_bfgs_n8_stats.json
```

Fleet wall: σ=0.05 170 s, σ=0.5 243 s (8 workers).

## Results (L=4, 500 trials each)

"Stuck" = final p(GS) < 1e-6. Best-of-25 = max p(GS) over the 25 trials of an instance, averaged (min) over the 20 instances.

| arm | success | mean p(GS) ± se | median p(GS) | frac p>0.5 | frac stuck | mean best-of-25 (min) | max p(GS) | evals/trial | wall/trial | stage mean p(GS) L1 / L2 / L3 / L4 |
|---|---|---|---|---|---|---|---|---|---|---|
| **grow+BFGS, kick 0.05** | 0.764 | 0.4545 ± 0.0137 | 0.5187 | 0.528 | 0.236 | 0.633 (0.286) | 0.994 | 2764 | 2.69 s | 0.105 / 0.226 / 0.347 / 0.454 |
| **grow+BFGS, kick 0.5** | 0.884 | **0.5425 ± 0.0119** | **0.5615** | **0.648** | 0.116 | **0.766** (0.416) | 0.994 | 3927 | 3.84 s | 0.105 / 0.246 / 0.412 / 0.543 |
| grow+Adam tuned (lr 0.5,0.2,0.05,0.02) | **0.978** | 0.5421 ± 0.0090 | 0.5179 | 0.594 | 0.002 | 0.683 (0.354) | 0.993 | 1604 | 1.38 s | 0.136 / 0.231 / 0.428 / 0.542 |
| grow+Adam untuned (C_adam, lr 0.05) | 0.852 | 0.3973 ± 0.0111 | 0.3957 | 0.288 | 0.096 | 0.563 (0.274) | 0.986 | 1604 | 1.43 s | 0.092 / 0.230 / 0.341 / 0.397 |
| fixed-L4 BFGS, random init (legacy η) | 0.976 | 0.5203 ± 0.0089 | 0.5019 | 0.506 | 0.004 | 0.744 (0.427) | 0.995 | 2557 | 3.14 s* | — |
| SPSA 200 steps, L4, random init | 0.932 | 0.1717 ± 0.0046 | 0.1434 | 0.018 | 0.004 | 0.363 (0.202) | 0.729 | 401 | 0.53 s* | — |

\* fixed-L4 BFGS and SPSA-200 were run with 1 worker; the other arms with 8 workers (per-trial wall is comparable).
Reference files: `grow_tune_F_lr_sched_20260929T093305Z`, `jp_adam_C_grow_ps200_20260929T090938Z` (L=4 records),
`jp_bfgs_B0_20260929T021402Z`, `jp_B0_20260928T042134Z`.

Paired Δp(GS) on identical (instance, trial, seed):

| comparison | mean Δp ± se | median Δp | frac higher | flips f→s / s→f |
|---|---|---|---|---|
| grow+BFGS k0.5 − grow+Adam tuned | +0.0004 ± 0.0097 | +0.033 | 0.71 | 8 / 55 |
| grow+BFGS k0.5 − fixed-L4 BFGS | +0.022 ± 0.011 | +0.060 | 0.63 | 10 / 56 |
| grow+BFGS k0.05 − grow+Adam tuned | −0.088 ± 0.011 | +0.012 | 0.62 | 8 / 115 |
| grow+BFGS k0.5 − grow+BFGS k0.05 | +0.088 ± 0.009 | ≈ 0 | 0.40 | 60 / 0 |

Conditional on not being stuck: grow+BFGS k0.05 mean / median p(GS) 0.595 / 0.562, k0.5 0.614 / 0.578, success
**1.000** for both; tuned Adam 0.543 / 0.518 (success 0.980), fixed-L4 BFGS 0.522 / 0.502 (0.980).

### Per-stage BFGS details (means over 500 trials, stages L1 / L2 / L3 / L4)

| arm | stage success | frac stuck | BFGS iterations (max) | evals | wall s | restarts | termination (L4) |
|---|---|---|---|---|---|---|---|
| kick 0.05 | 0.106 / 0.456 / 0.720 / 0.764 | 0.260 / 0.250 / 0.246 / 0.236 | 14 / 17 / 26 / 36 (46 / 72 / 200 / 500) | 167 / 365 / 794 / 1439 | 0.06 / 0.23 / 0.71 / 1.68 | 2.1 / 1.2 / 1.4 / 1.2 | η converged 437, x stationary 62, maxiter 1 |
| kick 0.5 | 0.106 / 0.546 / 0.820 / 0.884 | 0.260 / 0.198 / 0.154 / 0.116 | 14 / 27 / 37 / 55 (46 / 214 / 154 / 272) | 167 / 551 / 1107 / 2103 | 0.06 / 0.34 / 0.99 / 2.43 | 2.1 / 1.9 / 2.5 / 2.5 | x stationary 377, η converged 123 |

No precision-loss stops end a stage (restart mode); only 1 of 4000 stages hit the 500-iteration cap.

## Interpretation

- **Does BFGS growth beat Adam growth?** On mean p(GS): tie (kick 0.5: 0.5425 vs 0.5421, paired +0.000 ± 0.010).
  On median / typical trial: yes (0.562 vs 0.518; 71 % of paired trials higher, median Δ +0.033; frac p>0.5
  0.65 vs 0.59). On success: no (0.884 vs 0.978). BFGS growth is bimodal: every non-stuck trial succeeds with
  p(GS) ≈ 0.61 on average — the best polishing of any arm — but 11.6 % (kick 0.5) / 23.6 % (kick 0.05) of trials are
  stuck at p(GS) ≈ 1e-25, and those are *all* of its failures. It also costs 2.4× the evals and 2.8× the wall of
  tuned Adam growth.
- **Where the failures come from:** 26 % of trials are already stuck after the L=1 BFGS stage (L=1 BFGS converges,
  gradient < gtol, to a stationary point with zero GS weight; L=1 is identical for both kicks). The transparent
  insertion preserves that point, so with kick 0.05 almost none escape (0.260 → 0.236); kick 0.5 rescues about
  half (0.260 → 0.116). Tuned Adam's large early lr (0.5) walks out of these basins (0.2 % stuck). Fixed-L4 BFGS
  from random init rarely gets stuck (0.4 %), so the trap is specific to starting at L=1.
- **Growth vs fixed-L BFGS:** grow+BFGS kick 0.5 is +0.022 ± 0.011 mean / +0.06 median over random-init L4 BFGS
  with higher best-of-25 (0.766 vs 0.744), but lower success (0.884 vs 0.976).
- **Is the ansatz expressive enough?** Yes for this task: max p(GS) seen is 0.994–0.995 (instances 007, 019), and
  best-of-25 p(GS) > 0.5 on 19/20 instances for grow+BFGS k0.5 (mean 0.766). Taking the best over all six arms per
  instance: mean 0.777, min 0.427 (four_sat_018; its best is 0.416–0.427 in every strong arm, so that instance looks
  limited by the L=4 ansatz / local optima rather than by the optimizer). What limits things is the optimizer
  getting trapped, not the ansatz.
- Suggested next steps (not run): growth + BFGS with a larger L=1 escape (kick ≥ 1 at the first insertion only, or
  a few Adam lr-0.5 steps before BFGS in each stage, or multi-start at L=1 and keep non-stuck), which should combine
  Adam's ~0 % stuck rate with BFGS's polishing (≈ 0.61 conditional mean).

## Files

`noiseless/analyze_grow_bfgs.py` → `grow_bfgs_n8_stats.json`; fleet summaries
`jp_grow_bfgs_restart_20260930T083350Z_summary.json`, `jp_grow_bfgs_restart_kick05_20260930T083752Z_summary.json`
(per-trial JSONs gitignored).
