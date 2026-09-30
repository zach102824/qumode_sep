# ECD (jp) vs RY-only / HEA L=1 at n = 16 (family F1)

Date: 2026-09-30 (Asia/Shanghai). Instances: `Hamiltonians/four_sat_scaling/n16` (20 planted unique 4-SAT,
m/n 2.69–3.88), 25 trials/instance, seed 20260917, exact (infinite-shot) Gibbs cost with the sampled-tail η
controller (λ1 = λ = 0). ECD fleet ran 10:58–11:55 CST (**56.7 min wall**, 8 workers, OMP_NUM_THREADS=1).
RY-only / HEA L=1 rows are the EXISTING scaling-study checkpoints (`noiseless/results/scaling_runs/{main,steps400,steps600,steps800}`,
plain SPSA, no new runs); bins / best-of-25 recomputed from those jsonl files.

## Setup

**Layout (same chip as n = 8).** 2 data transmons (d, e) + 2 cavities (A, B); the middle transmon is the coupler
and is not simulated, the inter-cavity jp gate |n,m⟩ → (−i)^{(n+m) mod 2}|n,m⟩ is applied as an ideal gate.
Each cavity encodes **7 bits as the binary digits of its Fock number n = 0..127**; bits MSB-first
(q_d, q_e | n_A[6:0] | n_B[6:0]), i.e. exactly the n = 8 map with k = 3 → 7. 4·128·128 = 2^16.

**Truncation / leakage.** Each cavity is simulated with **nf = 160** levels (> 128) so ECD displacements are not
distorted over the encoded range. The n = 8 code has no out-of-encoding levels (it truncates at exactly 8), so a rule
was needed: Fock n ≥ 128 states are invalid and get energy **E_leak = E_max(valid) + 1** (worse than any bitstring;
the Gibbs cost treats leaked mass as essentially lost, the η quantiles see it as the top level). p(GS) is the Born
probability of the GS in the full truncated space; an argmax in a leaked state counts as failure (never happened).
Every trial's final x was re-evaluated at nf = 224: max |Δp| over all basis states 3.5e-10 (L=2) / 3.5e-13 (L=4),
success identical in 1000/1000 trials → nf = 160 is converged.

**Recipe.** jp, binary, SPSA-Adam, layer growth from L=1 (transparent layer + kick 0.05), c = 0.15, fresh Adam/η per stage.
- **ECD L=2** (16 params, matched to RY-only 16 params): growth 1→2, **400 + 400 steps, lr 0.5, 0.05** (1602 evals).
  lr choice: the tuned 4-stage schedule 0.5/0.2/0.05/0.02 is "large lr early to find the basin, small lr late to polish";
  with two stages we keep its first (0.5) and its polish-phase (0.05) values.
- **ECD L=4** (32 params, matched to HEA L=1 32 params): growth 1→4, **200×4 steps, lr 0.5/0.2/0.05/0.02** (tuned default, 1604 evals).

Code: `noiseless/ecd_kbit.py` (k-bit encoding, truncation, fast displacement via cached eigendecomposition),
`noiseless/run_ecd_kbit.py` (resumable runner, `--tag main`), `noiseless/analyze_ecd_vs_hea_n16.py` (this table →
`noiseless/results/ecd_vs_hea_n16_summary.json`), tests `noiseless/tests/test_ecd_kbit.py` (n = 8 path bit-for-bit incl.
stored tuned fleet records; n = 16 bit map == clause-violation counts; truncation convergence).

## Results (500 trials per row; bins are fractions of trials by final p(GS))

| arm | SPSA steps | params | evals | trials | success | mean p(GS) | median p(GS) | best-of-25 mean | inst ≥1 succ | >0.5 | 0.1–0.5 | 0.01–0.1 | <0.01 | mean <H> | s/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ECD jp L=2 | 400+400 | 16 | 1602 | 500 | 0.054 | 0.0104 | 8e-44 | 0.0143 | 2/20 | 0.000 | 0.048 | 0.016 | 0.936 | 1.979 | 21.3 |
| ECD jp L=4 | 200x4 | 32 | 1604 | 500 | 0.090 | 0.0316 | 3.59e-40 | 0.0393 | 2/20 | 0.046 | 0.022 | 0.030 | 0.902 | 1.802 | 32.7 |
| RY-only | 200 | 16 | 401 | 500 | 0.278 | 0.0827 | 4.91e-06 | 0.657 | 20/20 | 0.058 | 0.156 | 0.078 | 0.708 | 1.927 | 0.4 |
| RY-only | 400 | 16 | 801 | 500 | 0.388 | 0.259 | 2.54e-05 | 0.931 | 20/20 | 0.306 | 0.060 | 0.032 | 0.602 | 1.361 | 1.2 |
| RY-only | 600 | 16 | 1201 | 500 | 0.452 | 0.357 | 0.000233 | 0.983 | 20/20 | 0.382 | 0.046 | 0.036 | 0.536 | 1.123 | 1.7 |
| RY-only | 800 | 16 | 1601 | 500 | 0.498 | 0.418 | 0.0226 | 0.995 | 20/20 | 0.432 | 0.054 | 0.028 | 0.486 | 1.002 | 2.5 |
| HEA L=1 | 200 | 32 | 401 | 500 | 0.104 | 0.0362 | 7.36e-09 | 0.499 | 17/20 | 0.036 | 0.038 | 0.056 | 0.870 | 1.839 | 3.2 |
| HEA L=1 | 400 | 32 | 801 | 500 | 0.242 | 0.104 | 1.21e-05 | 0.736 | 19/20 | 0.100 | 0.096 | 0.072 | 0.732 | 1.742 | 6.7 |
| HEA L=1 | 600 | 32 | 1201 | 500 | 0.346 | 0.181 | 8.99e-05 | 0.844 | 20/20 | 0.188 | 0.118 | 0.068 | 0.626 | 1.601 | 9.8 |
| HEA L=1 | 800 | 32 | 1601 | 500 | 0.418 | 0.255 | 0.000516 | 0.915 | 20/20 | 0.286 | 0.098 | 0.066 | 0.550 | 1.467 | 13.1 |

s/trial: all with 8 concurrent workers. Evals ≈ 1600 per trial is the matched budget (ECD vs RY/HEA at 800 steps).

**Parameter-matched comparisons (≈1600 evals):**
- ECD L=2 vs RY-only (16 params, 800 steps): success **0.054 vs 0.498**, mean p(GS) 0.010 vs 0.418, best-of-25 mean 0.014 vs 0.995, instances solved 2/20 vs 20/20.
- ECD L=4 vs HEA L=1 (32 params, 800 steps): success **0.090 vs 0.418**, mean p(GS) 0.032 vs 0.255, best-of-25 mean 0.039 vs 0.915, instances solved 2/20 vs 20/20.
- ECD is worse than even the 200-step (401-eval) RY-only / HEA rows on every success metric, although its mean ⟨H⟩ (1.80–1.98) is similar to theirs at 200 steps.

## Leakage and |β| (ECD)

| arm | leakage mean | median | max | trials > 1e-3 | max\|β\| per trial: mean / median / max | mean \|β\| (all) | nf 160 vs 224 max\|Δp\| |
|---|---|---|---|---|---|---|---|
| ECD L=2 | 9.6e-07 | 1.9e-31 | 4.8e-04 | 0/500 | 5.57 / 5.62 / 19.2 | 2.16 | 3.5e-10 |
| ECD L=4 | 2.4e-08 | 4.1e-31 | 1.2e-05 | 0/500 | 5.42 / 5.23 / 17.1 | 1.39 | 3.5e-13 |

Leakage above Fock 127 is negligible (typical ~1e-31, worst 4.8e-4 < 1e-3). |β| stays moderate: per-trial max |β| is
~5 typically (one displacement does most of the work, the rest are small); the largest is 19.

## Why ECD fails at n = 16 (diagnosis)

1. **Success only where the GS sits at low Fock numbers.** All ECD successes come from two instances:
   inst 8 (GS Fock (n_A*, n_B*) = (2, 1): success 0.96 at both depths) and inst 16 ((7, 4): 0.12 at L=2, 0.84 at L=4).
   Every other instance has max(n_A*, n_B*) ≥ 26 and ECD gets **0/25** on all of them, with p(GS) typically 1e-20 to 1e-66.
   The GS targets are spread over 0..127 (median n_A* = 50, n_B* = 47), because the planted bitstrings are uniform.
2. **The optimized states stay at low photon number.** Final per-cavity photon distributions (re-simulated from
   final x): median ⟨n_A⟩ ≈ 5.2, ⟨n_B⟩ ≈ 4.1 with sd ≈ 2.3 / 2.1 (effective number of occupied levels ≈ 5–6),
   versus required n* ≈ 50. The marginal probability of the required Fock number, P_A(n_A*), has median 6e-29 (L=4);
   since p(GS) ≤ min(P_A(n_A*), P_B(n_B*)), the GS is unreachable from these states. Only 10.6 % (L=4) of trials have
   both marginals > 0.01, essentially all on inst 8/16.
3. **The Gibbs landscape doesn't pull the cavities up.** In binary encoding the low Fock levels n = 0..~10 already cover
   every value of the low 3–4 bits of each cavity. So a narrow low-photon state can lower ⟨H⟩ (optimized ⟨H⟩ ≈ 1.8,
   similar to HEA at 200 steps) by getting the low bits and transmon bits right, while the high bits (n ≥ 16, 32, 64) stay 0.
   Setting a high bit means moving the cavity to n ≳ 16–64 with a displacement of |β|/2 ≈ √n* ≈ 4–11, but that
   produces a Poisson-like spread (sd ≈ √n* ≈ 7 at n* = 50) that scrambles the low bits. The Gibbs cost sees this as an
   energy increase, so SPSA-Adam stays in the low-photon basin.
4. **Even a perfectly aimed displacement cannot concentrate on one Fock level.** A coherent-like state centered on n* = 50
   puts at most ≈ 1/(√(2π)·7) ≈ 0.06 on n* in each cavity, so ≲ 3e-3 on the joint GS. Getting a single high Fock state
   needs many ECD layers (Fock-state preparation depth grows with n*), far beyond L = 2–4. At n = 8 (Fock ≤ 7) this
   barrier did not exist, which is why the same recipe reached 0.98 success there.

**Conclusion.** With a binary 7-bit-per-cavity Fock encoding, shallow ECD (L = 2, 4) at a ~1600-eval budget solves only
the 2/20 instances whose GS has both cavities at Fock ≤ 7, and loses clearly to both the RY-only product ansatz and HEA L=1
at matched parameter count and budget. Truncation and leakage are not the issue (converged, leakage ≪ 1e-3). The
limitation is structural: the encoding needs sharp high-Fock states, while few-layer ECD produces broad low-photon ones.

## Per-instance diagnostics

GS Fock target, success, p(GS), and the median over 25 trials of the marginal probability of the target Fock number and of the per-cavity mean ± sd photon number.

### ECD L=2
| inst | GS (n_A*, n_B*) | success | mean p(GS) | max p(GS) | med P_A(n_A*) | med P_B(n_B*) | med ⟨n_A⟩±sd | med ⟨n_B⟩±sd | ⟨H⟩ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (51, 15) | 0.00 | 2.19e-28 | 5.47e-27 | 7.0e-33 | 1.4e-03 | 0.0±0.1 | 15.1±3.9 | 2.04 |
| 1 | (47, 3) | 0.00 | 3.59e-19 | 8.97e-18 | 2.7e-26 | 2.2e-11 | 5.9±2.4 | 0.0±0.0 | 1.67 |
| 2 | (93, 51) | 0.00 | 4.93e-46 | 1.23e-44 | 3.8e-33 | 6.2e-33 | 9.3±3.1 | 0.0±0.1 | 1.83 |
| 3 | (127, 31) | 0.00 | 1.85e-44 | 4.64e-43 | 3.4e-33 | 2.7e-28 | 0.0±0.0 | 1.7±1.2 | 1.19 |
| 4 | (0, 74) | 0.00 | 1.15e-18 | 2.04e-17 | 4.6e-02 | 1.4e-32 | 3.1±1.8 | 10.3±3.2 | 1.75 |
| 5 | (32, 18) | 0.00 | 1.97e-04 | 4.91e-03 | 9.0e-33 | 1.6e-31 | 0.0±0.1 | 0.0±0.1 | 1.35 |
| 6 | (49, 42) | 0.00 | 6.78e-26 | 1.58e-24 | 8.2e-27 | 1.3e-32 | 6.4±2.5 | 0.3±0.5 | 2.49 |
| 7 | (26, 50) | 0.00 | 1.55e-12 | 3.89e-11 | 7.1e-07 | 2.5e-32 | 8.4±2.9 | 4.8±2.2 | 1.97 |
| 8 | (2, 1) | 0.96 | 1.91e-01 | 1.99e-01 | 2.4e-01 | 3.7e-01 | 1.9±1.4 | 1.0±1.0 | 1.66 |
| 9 | (108, 61) | 0.00 | 7.06e-45 | 1.76e-43 | 5.4e-33 | 2.7e-28 | 6.7±2.7 | 9.6±3.2 | 2.76 |
| 10 | (75, 53) | 0.00 | 5.96e-22 | 1.49e-20 | 1.5e-33 | 7.9e-20 | 0.1±0.2 | 11.7±3.4 | 2.01 |
| 11 | (53, 44) | 0.00 | 7.08e-07 | 1.77e-05 | 3.4e-20 | 2.6e-32 | 10.7±5.8 | 0.9±1.3 | 2.53 |
| 12 | (88, 0) | 0.00 | 9.52e-04 | 2.38e-02 | 5.7e-33 | 7.6e-03 | 5.4±2.3 | 4.9±2.2 | 3.50 |
| 13 | (111, 115) | 0.00 | 5.64e-66 | 3.13e-65 | 3.7e-33 | 7.0e-33 | 0.0±0.2 | 0.0±0.1 | 1.03 |
| 14 | (36, 107) | 0.00 | 9.49e-41 | 2.37e-39 | 2.6e-10 | 2.2e-33 | 10.0±3.2 | 0.0±0.1 | 2.61 |
| 15 | (115, 124) | 0.00 | 1.89e-66 | 1.68e-65 | 5.5e-33 | 2.2e-33 | 5.0±2.2 | 6.0±2.5 | 1.92 |
| 16 | (7, 4) | 0.12 | 1.64e-02 | 5.76e-02 | 1.5e-01 | 3.9e-03 | 6.7±2.6 | 0.6±0.8 | 2.01 |
| 17 | (2, 58) | 0.00 | 1.21e-39 | 1.24e-38 | 1.3e-03 | 8.9e-33 | 10.7±3.3 | 5.3±2.3 | 1.82 |
| 18 | (64, 8) | 0.00 | 9.84e-32 | 1.88e-30 | 6.1e-33 | 2.6e-03 | 0.0±0.0 | 12.3±3.5 | 2.17 |
| 19 | (29, 100) | 0.00 | 3.27e-49 | 4.46e-48 | 1.4e-32 | 4.8e-33 | 0.1±0.3 | 1.6±1.1 | 1.25 |

### ECD L=4
| inst | GS (n_A*, n_B*) | success | mean p(GS) | max p(GS) | med P_A(n_A*) | med P_B(n_B*) | med ⟨n_A⟩±sd | med ⟨n_B⟩±sd | ⟨H⟩ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (51, 15) | 0.00 | 2.94e-27 | 4.07e-26 | 3.3e-32 | 6.6e-04 | 0.1±0.2 | 6.2±2.6 | 1.84 |
| 1 | (47, 3) | 0.00 | 4.13e-04 | 1.03e-02 | 5.2e-18 | 2.8e-02 | 7.4±3.0 | 1.7±2.1 | 1.74 |
| 2 | (93, 51) | 0.00 | 1.02e-50 | 2.54e-49 | 9.7e-33 | 1.5e-32 | 9.0±2.7 | 0.0±0.2 | 1.59 |
| 3 | (127, 31) | 0.00 | 6.16e-49 | 1.54e-47 | 5.3e-33 | 4.2e-29 | 0.0±0.0 | 1.1±0.5 | 1.01 |
| 4 | (0, 74) | 0.00 | 5.57e-19 | 7.10e-18 | 5.6e-02 | 4.3e-30 | 3.0±1.7 | 10.8±3.8 | 1.53 |
| 5 | (32, 18) | 0.00 | 1.07e-03 | 7.57e-03 | 2.1e-32 | 1.5e-27 | 0.0±0.1 | 0.0±0.1 | 1.34 |
| 6 | (49, 42) | 0.00 | 3.83e-13 | 9.58e-12 | 6.2e-24 | 3.4e-32 | 6.1±2.3 | 0.3±0.5 | 2.28 |
| 7 | (26, 50) | 0.00 | 2.03e-13 | 5.08e-12 | 2.4e-06 | 5.9e-29 | 8.4±2.9 | 4.7±2.2 | 1.92 |
| 8 | (2, 1) | 0.96 | 5.46e-01 | 6.45e-01 | 5.5e-01 | 9.2e-01 | 2.0±1.4 | 1.1±0.6 | 0.92 |
| 9 | (108, 61) | 0.00 | 2.69e-45 | 6.30e-44 | 6.8e-33 | 1.2e-14 | 2.4±1.5 | 15.5±4.4 | 2.20 |
| 10 | (75, 53) | 0.00 | 5.20e-21 | 1.30e-19 | 5.5e-33 | 4.6e-17 | 0.7±0.9 | 11.5±3.4 | 1.96 |
| 11 | (53, 44) | 0.00 | 6.47e-11 | 1.33e-09 | 1.9e-15 | 4.3e-32 | 11.1±5.5 | 0.6±1.0 | 2.53 |
| 12 | (88, 0) | 0.00 | 5.25e-08 | 1.31e-06 | 1.9e-32 | 8.1e-03 | 5.6±2.3 | 4.5±2.1 | 2.48 |
| 13 | (111, 115) | 0.00 | 9.65e-66 | 3.99e-65 | 3.9e-33 | 8.8e-33 | 0.0±0.2 | 0.0±0.1 | 1.05 |
| 14 | (36, 107) | 0.00 | 6.23e-19 | 1.02e-17 | 3.6e-10 | 2.8e-33 | 10.3±3.1 | 0.0±0.2 | 2.46 |
| 15 | (115, 124) | 0.00 | 1.36e-65 | 6.51e-65 | 6.8e-33 | 4.4e-33 | 4.4±2.1 | 5.3±2.4 | 1.56 |
| 16 | (7, 4) | 0.84 | 8.50e-02 | 1.23e-01 | 1.5e-01 | 2.1e-01 | 6.8±2.6 | 4.1±2.1 | 2.30 |
| 17 | (2, 58) | 0.00 | 2.64e-36 | 5.73e-35 | 3.9e-03 | 2.8e-32 | 9.5±3.2 | 5.2±2.2 | 1.73 |
| 18 | (64, 8) | 0.00 | 1.78e-09 | 4.45e-08 | 1.5e-32 | 5.1e-03 | 3.0±1.9 | 14.1±3.8 | 2.51 |
| 19 | (29, 100) | 0.00 | 3.03e-46 | 7.57e-45 | 2.1e-32 | 3.8e-33 | 0.0±0.2 | 1.1±0.5 | 1.08 |

## Reproduce
```
PYTHONPATH=. OMP_NUM_THREADS=1 python noiseless/run_ecd_kbit.py --layers 2 --steps-schedule 400,400 --lr-schedule 0.5,0.05 --tag main --workers 8
PYTHONPATH=. OMP_NUM_THREADS=1 python noiseless/run_ecd_kbit.py --layers 4 --steps-schedule 200,200,200,200 --lr-schedule 0.5,0.2,0.05,0.02 --tag main --workers 8
PYTHONPATH=. python noiseless/analyze_ecd_vs_hea_n16.py
```
