# Scaling study: RY-only product ansatz vs HEA vs WalkSAT/SA on planted 4-SAT (family F1)

Date: 2026-09-29 (Asia/Shanghai). Tag `main`. Code: `Hamiltonians/four_sat_scaling.py`,
`noiseless/scaling_ansatz.py`, `noiseless/classical_baselines.py`, `noiseless/run_scaling.py`,
`noiseless/analyze_scaling.py`, sweep scripts `noiseless/run_scaling_sweep.sh`, `noiseless/run_scaling_sweep2.sh`.
Tests: `tests/test_four_sat_scaling.py`. Per-trial checkpoints (gitignored):
`noiseless/results/scaling_runs/main/*.jsonl`; committed aggregates:
`noiseless/results/scaling_main_summary.json`, `noiseless/results/scaling_main_analysis_summary.json`,
`noiseless/results/scaling_timing_summary.json`.

## Setup

* **Instances (F1)**: n ∈ {8,12,16,20,24,28}, 20 instances per n, seed `20260929 + 1000 n + idx`.
  Planted x* (weight U[2,n−2], trivial patterns rejected) → one local-rigidity clause per site
  (same rule as the n=8 generator) → random x*-compatible clauses up to m_base = round(r·n),
  r ~ U[14/8, 23/8] = U[1.75, 2.875] (the n=8 generator's clause window / 8) → greedy top-up:
  exact enumeration of all other models (numba brute force over 2^n), repeatedly add the compatible
  4-clause killing the most remaining models (kill counts on a random 2048-model subsample when more
  remain), until x* is unique. Uniqueness verified twice per instance: brute-force model count = 1 and
  a CDCL proof (python-sat Minisat22) that F ∧ (x ≠ x*) is UNSAT. Per-n manifests in
  `Hamiltonians/four_sat_scaling/nNN/`. Differences vs n=8 script: one candidate per instance (no
  4000-candidate basin-score selection); final m includes the top-up.
* **Quantum arms** (conventions of `noiseless/hea.py` / `run_ecd_vs_qaoa.py`): start |0⟩^n, RY-only
  params init U[0,π), SPSA 200 steps (a = 0.2·sqrt(37/n_params), c = 0.15, A = 10, α = 0.602,
  γ = 0.101, 401 cost evals), Gibbs cost with the SampledTailEta η controller (refresh every 5 steps),
  success = argmax bitstring == GS, p(GS) = Born probability of GS; 25 trials/instance, seed
  `20260917 + 1e6·inst + 1e5·arm + 1e3·n + trial`.
  * `ry0`: RY-only product, n params. <H>, p(GS), argmax analytic (clause by clause). Gibbs cost exact
    by enumeration for n ≤ 20; for n = 24, 28 the `ry0` arm used the **sampled** estimator
    (4096 samples/eval, common random numbers per SPSA step, η from sample quantiles).
  * `ry0_split`: RY-only with the **exact** Gibbs cost for any n ≤ 28 via the hi/lo factorization
    Σ_h p_hi[h] Σ_l p_lo[l] e^{−ηE[h,l]} over a uint8 spectrum (numba); η from the exact energy-level
    distribution. This is the headline RY-only series at n = 24, 28 (see validation).
  * `hea1`, `hea2`: HEA L=1,2 on the 2 × n/2 snake lattice (3n/2−2 CZ per layer in 3 sublayers,
    final RY layer; n(L+1) params), exact statevector, n ≤ 20. Bit-for-bit equal to `hea.py` at n=8
    (tested: states, energies, cost, η, and a full 30-step SPSA trajectory).
* **Classical baselines** (25 runs/instance): budget unit B0 = 401 = cost evaluations of one SPSA run.
  Accounting: 1 evaluation = energy (or energy change) of one full assignment; WalkSAT (SKC, noise 0.5)
  charges 4 per step (break counts of the 4 variables of the chosen violated clause) + 1 for the start;
  SA (single-flip Metropolis, geometric T 2.0 → 0.05 over the budget) charges 1 per proposal. Success
  = GS visited within budget. Budgets B0 × {1, 10, 100, 1000} (×1000 ≈ shot-matched to 1000 shots
  per Gibbs-cost estimate). WalkSAT is anytime (one run at max budget); SA re-run per budget.

## Instances: m and m/n

| n | m mean [min,max] | m/n mean [min,max] | m_base mean | top-up mean | #sol after base (median) |
|---|---|---|---|---|---|
| 8 | 27.6 [25,30] | 3.45 [3.12,3.75] | 17.9 | 9.7 | 76 |
| 12 | 41.6 [36,46] | 3.47 [3.00,3.83] | 28.6 | 13.1 | 680 |
| 16 | 53.0 [43,62] | 3.31 [2.69,3.88] | 36.4 | 16.6 | 6628 |
| 20 | 69.2 [57,80] | 3.46 [2.85,4.00] | 48.8 | 20.4 | 43830 |
| 24 | 79.0 [64,93] | 3.29 [2.67,3.88] | 54.5 | 24.5 | 529760 |
| 28 | 93.2 [72,109] | 3.33 [2.57,3.89] | 64.5 | 28.7 | 5089802 |

Base density matches the n=8 window by construction; the unique-solution top-up adds ≈ n clauses,
so the final m/n ≈ 3.3–3.5 is flat in n (the committed n=8 family has m/n = 2.375, all m = 19).

## Results per n

| arm | n | params | cost | inst | trials | success | mean p(GS) | median p(GS) | mean <H> | s/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| ry0 | 8 | 8 | exact | 20 | 500 | 0.992 | 0.975 | 0.989 | 0.035 | 0.0 |
| ry0 | 12 | 12 | exact | 20 | 500 | 0.882 | 0.582 | 0.664 | 0.783 | 0.0 |
| ry0 | 16 | 16 | exact | 20 | 500 | 0.278 | 0.0827 | 4.91e-06 | 1.927 | 0.4 |
| ry0 | 20 | 20 | exact | 20 | 500 | 0.034 | 0.00532 | 4.5e-15 | 1.953 | 11.8 |
| ry0 | 24 | 24 | sampled | 20 | 500 | 0.000 | 1.78e-10 | 2.22e-22 | 1.483 | 0.3 |
| ry0 | 28 | 28 | sampled | 20 | 500 | 0.000 | 1.6e-13 | 8.87e-25 | 1.740 | 0.5 |
| ry0_split | 16 | 16 | exact_split | 20 | 500 | 0.278 | 0.0827 | 5.06e-06 | 1.932 | 0.0 |
| ry0_split | 20 | 20 | exact_split | 20 | 500 | 0.034 | 0.00532 | 4.5e-15 | 1.953 | 0.4 |
| ry0_split | 24 | 24 | exact_split | 20 | 500 | 0.000 | 5.53e-07 | 1.42e-21 | 1.617 | 5.3 |
| ry0_split | 28 | 28 | exact_split | 20 | 500 | 0.000 | 3.48e-11 | 5.93e-25 | 1.735 | 83.0 |
| ry0_sampled | 16 | 16 | sampled | 20 | 500 | 0.024 | 0.015 | 2.37e-16 | 1.179 | 0.2 |
| ry0_sampled | 20 | 20 | sampled | 20 | 500 | 0.004 | 0.00177 | 1.04e-20 | 1.312 | 0.3 |
| hea1 | 8 | 16 | exact | 20 | 500 | 0.996 | 0.933 | 0.942 | 0.121 | 0.1 |
| hea1 | 12 | 24 | exact | 20 | 500 | 0.808 | 0.428 | 0.432 | 1.227 | 0.3 |
| hea1 | 16 | 32 | exact | 20 | 500 | 0.104 | 0.0362 | 7.36e-09 | 1.839 | 3.2 |
| hea1 | 20 | 40 | exact | 20 | 500 | 0.006 | 0.00217 | 6.68e-13 | 1.973 | 75.3 |
| hea2 | 8 | 24 | exact | 20 | 500 | 1.000 | 0.882 | 0.896 | 0.214 | 0.1 |
| hea2 | 12 | 36 | exact | 20 | 500 | 0.768 | 0.336 | 0.31 | 1.472 | 0.4 |
| hea2 | 16 | 48 | exact | 20 | 500 | 0.040 | 0.0124 | 1.01e-08 | 1.950 | 4.7 |
| hea2 | 20 | 60 | exact | 20 | 500 | 0.002 | 0.000667 | 5.5e-11 | 2.460 | 108.7 |

`ry0` rows at n = 24, 28 are the sampled-cost runs; the exact-cost RY-only numbers there are the
`ry0_split` rows. Instances with ≥1 successful trial: RY-only 20/20 (n≤16), 12/20 (n=20), 0/20
(n=24,28); HEA L=1 17/20 (n=16), 3/20 (n=20); HEA L=2 12/20 (n=16), 1/20 (n=20).

### Classical baselines: success probability within budget (evals)

| n | inst | WalkSAT@401 | WalkSAT@4010 | WalkSAT@40100 | WalkSAT@401000 | SA@401 | SA@4010 | SA@40100 | SA@401000 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 20 | 0.802 | 1.000 | 1.000 | 1.000 | 0.686 | 0.998 | 1.000 | 1.000 |
| 12 | 20 | 0.402 | 0.986 | 1.000 | 1.000 | 0.154 | 0.816 | 1.000 | 1.000 |
| 16 | 20 | 0.116 | 0.688 | 0.998 | 1.000 | 0.014 | 0.162 | 0.852 | 1.000 |
| 20 | 20 | 0.028 | 0.316 | 0.920 | 1.000 | 0.002 | 0.074 | 0.316 | 0.916 |
| 24 | 20 | 0.010 | 0.128 | 0.676 | 0.996 | 0.004 | 0.002 | 0.040 | 0.404 |
| 28 | 20 | 0.004 | 0.088 | 0.458 | 0.942 | 0.000 | 0.002 | 0.010 | 0.078 |

## Fitted per-qubit decay c (log10 y = a + n·log10 c)

Headline RY-only series `ry0_exact` = per-state exact cost (n ≤ 20) + factorized exact cost
(n = 24, 28). Success fits skip points with zero successes.

| series | n range | slope log10/qubit | c (per-qubit factor) |
|---|---|---|---|
| ry0_exact mean_p_gs | 8-28 (6 pts) | -0.5107 | 0.3086 |
| ry0_exact median_p_gs | 8-28 (6 pts) | -1.3726 | 0.0424 |
| ry0_exact success | 8-20 (4 pts) | -0.1224 | 0.7544 |
| ry0_exact mean_p_gs | 8-20 (4 pts) | -0.1909 | 0.6443 |
| ry0_exact median_p_gs | 8-20 (4 pts) | -1.2040 | 0.0625 |
| ry0_exact success | 8-20 (4 pts) | -0.1224 | 0.7544 |
| ry0_exact mean_p_gs | 12-20 (3 pts) | -0.2549 | 0.5560 |
| ry0_exact median_p_gs | 12-20 (3 pts) | -1.7712 | 0.0169 |
| ry0_exact success | 12-20 (3 pts) | -0.1767 | 0.6657 |
| ry0_exact mean_p_gs | 16-28 (4 pts) | -0.8027 | 0.1575 |
| ry0_exact median_p_gs | 16-28 (4 pts) | -1.5814 | 0.0262 |
| ry0_exact success | 16-20 (2 pts) | -0.2281 | 0.5914 |
| hea1 mean_p_gs | 8-20 (4 pts) | -0.2243 | 0.5966 |
| hea1 median_p_gs | 8-20 (4 pts) | -1.1054 | 0.0785 |
| hea1 success | 8-20 (4 pts) | -0.1888 | 0.6475 |
| hea1 mean_p_gs | 12-20 (3 pts) | -0.2868 | 0.5166 |
| hea1 median_p_gs | 12-20 (3 pts) | -1.4763 | 0.0334 |
| hea1 success | 12-20 (3 pts) | -0.2662 | 0.5418 |
| hea1 mean_p_gs | 16-20 (2 pts) | -0.3055 | 0.4949 |
| hea1 median_p_gs | 16-20 (2 pts) | -1.0105 | 0.0976 |
| hea1 success | 16-20 (2 pts) | -0.3097 | 0.4901 |
| hea2 mean_p_gs | 8-20 (4 pts) | -0.2699 | 0.5371 |
| hea2 median_p_gs | 8-20 (4 pts) | -0.9530 | 0.1114 |
| hea2 success | 8-20 (4 pts) | -0.2345 | 0.5828 |
| hea2 mean_p_gs | 12-20 (3 pts) | -0.3378 | 0.4594 |
| hea2 median_p_gs | 12-20 (3 pts) | -1.2189 | 0.0604 |
| hea2 success | 12-20 (3 pts) | -0.3230 | 0.4753 |
| hea2 mean_p_gs | 16-20 (2 pts) | -0.3173 | 0.4816 |
| hea2 median_p_gs | 16-20 (2 pts) | -0.5663 | 0.2715 |
| hea2 success | 16-20 (2 pts) | -0.3253 | 0.4729 |
| ry0_sampled_cost mean_p_gs | 16-28 (4 pts) | -0.9978 | 0.1005 |
| ry0_sampled_cost median_p_gs | 16-28 (4 pts) | -0.6739 | 0.2119 |
| ry0_sampled_cost success | 16-20 (2 pts) | -0.1945 | 0.6389 |
| ry0_sampled_cost mean_p_gs | 16-20 (2 pts) | -0.2324 | 0.5856 |
| ry0_sampled_cost median_p_gs | 16-20 (2 pts) | -1.0892 | 0.0814 |
| ry0_sampled_cost success | 16-20 (2 pts) | -0.1945 | 0.6389 |
| walksat success @401 | 8-28 (6 pts) | -0.1210 | 0.7568 |
| walksat success @4010 | 8-28 (6 pts) | -0.0591 | 0.8727 |
| walksat success @40100 | 8-28 (6 pts) | -0.0160 | 0.9638 |
| walksat success @401000 | 8-28 (6 pts) | -0.0010 | 0.9978 |
| sa success @401 | 8-24 (5 pts) | -0.1589 | 0.6936 |
| sa success @4010 | 8-28 (6 pts) | -0.1547 | 0.7003 |
| sa success @40100 | 8-28 (6 pts) | -0.1045 | 0.7862 |
| sa success @401000 | 8-28 (6 pts) | -0.0483 | 0.8948 |

## Exact vs sampled validation (paired: same seeds, x0, SPSA Δ)

| test path | n | trials | exact success | test success | exact mean p(GS) | test mean p(GS) | exact median | test median | same final ML bitstring | Pearson r(log10 p) |
|---|---|---|---|---|---|---|---|---|---|---|
| ry0_sampled | 16 | 500 | 0.278 | 0.024 | 0.0827 | 0.015 | 4.91e-06 | 2.37e-16 | 0.206 | 0.275 |
| ry0_sampled | 20 | 500 | 0.034 | 0.004 | 0.00532 | 0.00177 | 4.5e-15 | 1.04e-20 | 0.394 | 0.459 |
| ry0_split | 16 | 500 | 0.278 | 0.278 | 0.0827 | 0.0827 | 4.91e-06 | 5.06e-06 | 0.998 | 0.985 |
| ry0_split | 20 | 500 | 0.034 | 0.034 | 0.00532 | 0.00532 | 4.5e-15 | 4.5e-15 | 1.000 | 1.000 |

* **Sampled Gibbs cost does NOT reproduce the exact-cost optimizer** (n=16: success 0.278 → 0.024;
  n=20: 0.034 → 0.004). The same held with 65 536 samples/eval (5 inst × 8 trials at n=16: exact
  0.225, 4096 samples 0.025, 65 536 samples 0.025). Mechanism: the exact cost at η ≈ 10 is driven by the
  exponentially small p(GS) (and near-GS) mass, which no finite sample sees; the sampled estimator
  optimizes the lowest sampled energy levels instead (final <H> 1.18 vs 1.93 — lower energy, lower
  p(GS)) and its η ratchets to ~40. So the exact-cost RY-only/HEA numbers are an *infinite-shot*
  idealization, and a finite-shot implementation would do markedly worse at n ≥ 16.
* **Factorized exact path (`ry0_split`) agrees with the per-state exact cost**: identical success and
  mean p(GS) at n = 16, 20; same final argmax bitstring in 99.8% / 100% of trials (the only difference
  is the η quantile's sub-state interpolation at level boundaries).

## Takeaways

* All ansätze fall off a cliff between n = 12 and n = 20: RY-only success 0.99 → 0.88 → 0.28 → 0.034,
  then 0/500 at n = 24, 28; HEA L=1 0.996 → 0.81 → 0.10 → 0.006; HEA L=2 1.0 → 0.77 → 0.04 → 0.002.
  Median p(GS) collapses much faster than the mean (mean is carried by a few lucky trials).
* Adding entangling layers does not help here: RY-only > HEA L=1 > HEA L=2 in mean p(GS) at every n,
  and in success for every n ≥ 12 (at n = 8 all three are ≥ 0.99). Same 200-step SPSA budget; more
  params → smaller SPSA a and a harder landscape.
* Per-qubit decay of mean p(GS) over n = 12–20: RY-only c ≈ 0.56, HEA L=1 ≈ 0.52, HEA L=2 ≈ 0.46.
  Success over n = 12–20: 0.67 / 0.54 / 0.48.
* WalkSAT at the literal 401-eval budget is comparable to the ansätze at small n (0.80 at n=8,
  0.40 at n=12, 0.12 at n=16, 0.03 at n=20; c ≈ 0.76, about the RY-only success rate of decay, but it
  stays nonzero through n = 28), and decays far more slowly with more budget (c ≈ 0.87 at 4010 evals,
  0.96 at 40 100); at 40 100 evals (≈ 100 shots per cost eval) it still solves 46% at n = 28, and 94% at
  401 000. SA is weaker than WalkSAT at every budget.

## Wall time (8 workers, box with 8 CPUs)

Phase 1 smoke (1 instance, 2 trials, 8 concurrent jobs), s/trial: HEA L=1 n=16 4.3, n=20 74;
HEA L=2 n=16 5.5, n=20 108; RY-only n=20 exact 14, n=24/28 sampled 0.5; classical n=28 2.0 s per
instance (25 runs × 4 budgets × 2 algorithms). Projection for the full sweep (20 × 25 per cell):
HEA L=1 ≈ 1.3 h, HEA L=2 ≈ 1.9 h, RY-only + baselines ≈ 0.2 h → ≈ 3.5 h; no cuts needed.
Actual: RY-only + classical + sampled validation 14 min; HEA L=1 82 min; HEA L=2 119 min; RY-only
exact-factorized (added after validation; n=28 ≈ 83 s/trial) 93 min; generation ≈ 10 min.
Total ≈ 5.3 h of sweep, well inside the 12 h budget. No trims (all 20 instances × 25 trials in every cell).
