# BFGS vs SPSA on jp-gate noiseless ECD VQE (L*=4, 200 iterations)

Date: 2026-09-29. Only the optimizer differs between the paired fleets.

## Setup

- Fleets: U=jp, L*=4 (32 parameters), 20 Hamiltonians (`Hamiltonians/four_sat`, 8 spins,
  unique GS) × 25 trials = 500 trials per fleet, `--steps 200`, `--seed 20260917`,
  `--lambda1 0`, `--workers 1`.
- λ=0: Gibbs cost only. λ=2: Gibbs + 2·Σ max(abs(β)−2.1, 0)².
- SPSA fleets (existing): `jp_B0_20260928T042134Z.json`, `jp_lam2_20260928T051924Z.json`.
- BFGS fleets (new): `jp_bfgs_B0_20260929T021402Z.json`, `jp_bfgs_lam2_20260929T025135Z.json`
  (`--optimizer bfgs`). All stored args other than `optimizer` match the SPSA runs
  exactly (checked by `noiseless/analyze_bfgs_vs_spsa.py`).
- BFGS: `scipy.optimize.minimize(method="BFGS")`, default settings (finite-difference
  gradient, gtol 1e-5), `maxiter=200`. Same x0 as the SPSA trial with the same seed
  (x0 is the first draw from the trial rng). η is refreshed from x0 before the start and
  then after every 5th BFGS iteration (same cadence as SPSA steps 1, 6, 11, ...).
  `nfev` counts every cost evaluation (including the final one). η refreshes themselves
  are not counted, same as for SPSA.
- Code check: with the new code the SPSA path reproduces stored SPSA records bit-exactly
  (p(GS) and final x checked on sample trials from both fleets); all 20 tests in
  `noiseless/tests` pass.
- "Success" = most likely bitstring equals the ground state. β statistics use the final x.
  "abs(β)>2.1" is computed from the final x with β_max=2.1 for all four fleets (the λ=0
  records store 0 there because no cap was set).

## Fleet results

| metric | SPSA λ=0 | BFGS λ=0 | SPSA λ=2 | BFGS λ=2 |
|---|---|---|---|---|
| trials | 500 | 500 | 500 | 500 |
| success rate | 0.932 | 0.976 | 0.774 | 0.916 |
| mean p(GS) | 0.1717 | 0.5203 | 0.1174 | 0.2974 |
| median p(GS) | 0.1434 | 0.5019 | 0.0824 | 0.2124 |
| mean abs(β) | 1.847 | 1.933 | 1.570 | 1.463 |
| mean trial-max abs(β) | 3.388 | 4.367 | 2.170 | 2.189 |
| frac of β with abs(β)>2.1 | 0.396 | 0.431 | 0.277 | 0.433 |
| frac of trials with any abs(β)>2.1 | 0.960 | 0.934 | 0.854 | 0.914 |
| mean nfev / trial | 401 | 2557 | 401 | 3658 |
| median nfev / trial | 401 | 2245 | 401 | 3912 |
| mean nit | 200.0 | 68.0 | 200.0 | 73.4 |
| frac stopped before 200 it | 0.000 | 0.992 | 0.000 | 0.994 |
| mean wall s / trial | 0.53 | 3.14 | 0.53 | 4.50 |
| total wall min (fleet) | 4.4 | 26.1 | 4.4 | 37.5 |

BFGS stop reasons (scipy status and message):

BFGS λ=0 status: 0 'Optimization terminated successfully.': 404; 2 'Desired error not necessarily achieved due to precision loss.': 92; 1 'Maximum number of iterations has been exceeded.': 4

BFGS λ=2 status: 0 'Optimization terminated successfully.': 307; 2 'Desired error not necessarily achieved due to precision loss.': 190; 1 'Maximum number of iterations has been exceeded.': 3

## Paired per-trial comparison

Trials matched by (ham_file, trial, seed), so each pair starts from the same x0.

| paired (BFGS vs SPSA) | λ=0 | λ=2 |
|---|---|---|
| matched trials | 500 | 500 |
| unmatched | 0 | 0 |
| BFGS higher p(GS) | 0.988 | 0.982 |
| tie (abs Δ ≤ 1e-6) | 0.004 | 0.000 |
| BFGS lower p(GS) | 0.008 | 0.018 |
| mean Δp(GS) (BFGS−SPSA) | +0.3485 | +0.1801 |
| median Δp(GS) | +0.3389 | +0.1327 |
| both succeed | 0.926 | 0.726 |
| only BFGS succeeds | 0.050 | 0.190 |
| only SPSA succeeds | 0.006 | 0.048 |
| neither succeeds | 0.018 | 0.036 |

## Per-instance success (successes / 25)

| instance | SPSA λ=0 | BFGS λ=0 | SPSA λ=2 | BFGS λ=2 |
|---|---|---|---|---|
| four_sat_000.npz | 23/25 | 25/25 | 14/25 | 20/25 |
| four_sat_001.npz | 24/25 | 25/25 | 21/25 | 23/25 |
| four_sat_002.npz | 25/25 | 25/25 | 22/25 | 25/25 |
| four_sat_003.npz | 22/25 | 24/25 | 15/25 | 20/25 |
| four_sat_004.npz | 22/25 | 25/25 | 13/25 | 22/25 |
| four_sat_005.npz | 25/25 | 25/25 | 25/25 | 25/25 |
| four_sat_006.npz | 23/25 | 24/25 | 17/25 | 22/25 |
| four_sat_007.npz | 25/25 | 25/25 | 25/25 | 25/25 |
| four_sat_008.npz | 25/25 | 25/25 | 25/25 | 25/25 |
| four_sat_009.npz | 23/25 | 25/25 | 21/25 | 24/25 |
| four_sat_010.npz | 21/25 | 23/25 | 18/25 | 20/25 |
| four_sat_011.npz | 24/25 | 24/25 | 20/25 | 24/25 |
| four_sat_012.npz | 23/25 | 25/25 | 19/25 | 21/25 |
| four_sat_013.npz | 23/25 | 25/25 | 16/25 | 24/25 |
| four_sat_014.npz | 22/25 | 24/25 | 19/25 | 22/25 |
| four_sat_015.npz | 21/25 | 24/25 | 23/25 | 23/25 |
| four_sat_016.npz | 23/25 | 23/25 | 15/25 | 20/25 |
| four_sat_017.npz | 23/25 | 23/25 | 16/25 | 24/25 |
| four_sat_018.npz | 24/25 | 24/25 | 18/25 | 24/25 |
| four_sat_019.npz | 25/25 | 25/25 | 25/25 | 25/25 |

## Convergence details

- Almost all BFGS trials stopped before 200 iterations: mean nit 68.0 (λ=0) and 73.4 (λ=2);
  only 4 and 3 trials hit the 200-iteration limit.
- Status 2 ("precision loss", i.e. the line search failed) is common: 92/500 at λ=0 and
  190/500 at λ=2. Every one of these stops happened at an iteration count that is a
  multiple of 5, which is exactly where η is refreshed. So these stops come from the
  objective changing under BFGS (its stored f and gradient at the current point were
  computed with the old η), not from a real optimum. Those trials still mostly succeed
  (95.7% at λ=0, 89.5% at λ=2) but have lower p(GS) (mean 0.395 and 0.232) than the
  status-0 trials (0.546 and 0.340).
- Status-0 stops ("converged") also land on multiples of 5 more often than chance
  (25.5% at λ=0, 42.7% at λ=2 vs about 20% expected), so η refreshes likely also trigger
  some of the gradient-tolerance stops.
- Cost evaluations per BFGS iteration: 37.6 (λ=0) and 49.8 (λ=2) on average (33 for the
  finite-difference gradient plus line-search evaluations), vs 2 per SPSA step.

## Caveats

- Cost budget is not matched. Per iteration BFGS uses about 19× (λ=0) to 25× (λ=2) more
  cost evaluations than SPSA (roughly 17× from the gradient alone). Per trial, because
  BFGS stops early, it used 6.4× (λ=0) and 9.1× (λ=2) more evaluations than SPSA's 401.
  On hardware each evaluation is a shot batch, so BFGS is much more expensive in shots.
  An SPSA run given the same number of evaluations was not tested.
- This is noiseless and the cost is exact. Finite-difference gradients with the default
  step (~1.5e-8) would not work with shot noise.
- η changes during the run, which breaks BFGS's fixed-objective assumption; see the
  status-2 stops above. Results would likely change with a different η schedule (for
  example freezing η, or restarting BFGS after each refresh).
- Wall time: BFGS 3.14 s/trial (λ=0) and 4.50 s/trial (λ=2) vs 0.53 s/trial for SPSA,
  on the same 1-CPU box, one worker.
- With λ=2 the soft cap is less effective under BFGS: final mean trial-max abs(β) is about the
  same (2.19 vs 2.17) but the fraction of β above 2.1 is higher (0.433 vs 0.277), so
  more β end slightly above the cap, where the quadratic penalty is small.

Reproduce tables:

```bash
python noiseless/analyze_bfgs_vs_spsa.py \
  --spsa-b0 noiseless/results/jp_B0_20260928T042134Z.json \
  --bfgs-b0 noiseless/results/jp_bfgs_B0_20260929T021402Z.json \
  --spsa-lam2 noiseless/results/jp_lam2_20260928T051924Z.json \
  --bfgs-lam2 noiseless/results/jp_bfgs_lam2_20260929T025135Z.json \
  --json-out noiseless/results/bfgs_vs_spsa_jp_stats.json
```
