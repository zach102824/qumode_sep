# Independent check of PRX Quantum 5, 030348 (random 8-SAT QAOA)

Exact statevector, paper ensemble (Poisson `m`, literals with replacement),
published angles, satisfiable instances only. TTS = `1/p_succ`. Paper Table II
quotes `log2(median TTS) = a + b n` fitted on **n = 12…20**.

## Checkpoint: the paper is right at small n

| n | p | N | our mean p_succ | our median TTS | Table II median TTS | ratio |
|---|---|---|---|---|---|---|
| 8 | 14 | 40 | 0.369 | 2.81 | 2.92 (extrapolated) | 0.96 |
| 12 | 14 | 40 | 0.176 | 6.10 | 7.20 | 0.85 |
| 12 | 60 | 40 | **0.558** | 1.51 | 1.72 | 0.88 |

The paper’s only raw number of this kind is *“p = 60, k = 8, n = 12, the mean
success probability was approximately 0.56”*. We get **0.5576**.

n = 8 is outside their fit range. Using n = 8 with their p = 14 line still
hits our median TTS to ~4%. Random assignment at n = 8 is ~0.007; p = 14 QAOA
is ~0.37.

## n = 20: Table II is pessimistic, not too good to be true

| n | p | N | our mean p_succ | our median p_succ (95% boot CI) | our median TTS (95% CI) | Table II TTS | ratio |
|---|---|---|---|---|---|---|---|
| 16 | 14 | 40 | 0.124 | 0.106 | 9.41 [7.3, 14] | 17.8 | 0.53 |
| 16 | 60 | 40 | 0.372 | 0.411 | 2.45 [1.9, 4.7] | 3.97 | 0.62 |
| 20 | 14 | 40 | 0.056 | 0.039 [0.022, 0.061] | 25.7 [16, 46] | 43.9 | 0.59 |
| 20 | 60 | **120** | 0.293 | 0.232 [0.171, 0.375] | **4.31 [2.66, 5.85]** | **9.18** | **0.47** |

At n = 20, p = 60 the published fit (median TTS ≈ 9.2, implied p_succ ≈ 0.11)
lies **outside** a 120-instance bootstrap CI. Two independent batches (40 then
80) both sit near median p_succ 0.20–0.30. So this is not a 40-shot fluke.

The p_succ distribution is heavy-tailed (n = 20, p = 60: min 8e-4, p10 ≈ 0.03,
median 0.23, max 0.68). That makes a least-squares fit on log median TTS
sensitive to small-n points. Our own fit on n = 12, 16, 20:

- p = 14: `log2 T ≈ -0.64 + 0.260 n` (paper 0.326)
- p = 60: `log2 T ≈ -1.70 + 0.190 n` (paper 0.302)

WalkSATlm in Table II is `−0.309 + 0.325 n` (n = 20 → T ≈ 73 formula evals).
Our n = 20 p = 14 median is 26 shots; p = 60 is 4.3 shots. The headline that
p ≳ 14 is in the same league as WalkSATlm **still holds**; the Table II QAOA
slopes overstate the exponent.

## What was not wrong

- Ensemble, SAT filter, and half-angle mixer all check out (n = 12 mean matches).
- n = 20 Hilbert space is 2^20; we used exact Born probabilities, not shots.
- QAOA is far above random: at n = 20 random p_succ ~ 2×10^{-6}; p = 1 already
  ~1×10^{-4}; p = 60 ~ 0.23.

## What *was* easy to get wrong (and would look like “n = 20 is fake”)

1. **Cost-phase sign.** Eq. (13) as written with the published positive γ is
   *worse* than random. The working sign is `exp(+i γ H / 2)` (γ → −γ in the
   PDF). That matches `lib/generalized_binomial_sum.py` in their data repo.
2. **Forgetting the `1/2` in `Rx(β)`.** Standard Farhi `exp(-i β X)` is a
   different parametrization than Yao `Rx(β)`.
3. **Extrapolating Table II to n = 8 and calling the mismatch a bug.** The fit
   domain is n = 12…20. At n = 8 it still happens to work for p = 14; the
   p = 60 intercept would give TTS < 1, which is meaningless.

## Files

- `eval_n8_k8_*.json`, `eval_n12_*.json`, `eval_n16_*.json`, `eval_n20_*.json`
- `comparison_summary.json` — pooled table
- `../tests/test_ksat_qaoa.py` — energy table, mixer vs Kronecker `Rx`, p = 1
  beats random, Poisson random-assignment formula
