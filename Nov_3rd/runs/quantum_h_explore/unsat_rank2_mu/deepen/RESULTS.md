# RESULTS — Round 5 deepen, unsat λ=0.75 + μ=−0.25 Walsh freeze

**Verdict: PASS**

frozen n=12 L=16 Walsh R=0.342 (instance 95% CI 0.320–0.373) and n=15 R=0.375 (CI 0.340–0.412) both stay BEAT. 5/5 and 5/5 seeds beat. Paired majorities 5/5 and 5/5. Drop vs μ=0 unsat Walsh 0.598. n=18 probe L=16 Walsh R=0.411 Med ε=0.0007 on B=16 (peak 0.001, product gap 0.107); exploratory, not gated. PR #23 point estimates were n=12 R=0.330 and n=15 R=0.355.

## Setup

Same gate and Hamiltonian as PR #23. Intra-cavity XX is scored from the X histogram. Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, use those frozen angles. n=21 was not loaded.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −0.25
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from `beat_qaoa_12h/train_packs` (base 51e9). Selection holdout B=256 per n (PR #23 used 128). Sealed transfer B=128 at n=12 and B=64 at n=15 (PR #23 used 64 and 32). Seeds [0, 1, 2, 3, 4]. L=[2, 16]. L-BFGS rounds 18/48, max_iter 8/8, lr=0.35. That is the PR #23 optimizer budget on a larger selection holdout. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. Early stop is 8 rounds without a 10⁻⁶ improvement.

PR #23 point estimates, 3 seeds: n=12 L=16 Walsh R=0.330 on B=64, n=15 L=16 Walsh R=0.355 on B=32. μ=0 comparison remains the published PR #19 ratio of medians (n=12 L=16 Walsh R=0.940), not a retrain.

n=18 probe B=16. One Lanczos-plus-product ground state took 6.1s, under the 90s feasibility limit. The probe is reported and is not part of PASS.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|.

Worst |ΔE0| Lanczos vs dense = 1.95e-14. Histogram vs ⟨ψ|H|ψ⟩ = 4.88e-15. μ=0 histogram vs Pauli field = 3.55e-15.

| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |
|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|
| 6 | train | 128 | 0.095 | 0.314 | 0.120 | 0.135 | 0.510 | 3.3e-15 |
| 6 | hold | 256 | 0.098 | 0.320 | 0.125 | 0.132 | 0.507 | 3.9e-15 |
| 9 | train | 128 | 0.028 | 0.380 | 0.110 | 0.126 | 0.508 | 8.0e-14 |
| 9 | hold | 256 | 0.028 | 0.368 | 0.107 | 0.126 | 0.508 | 7.0e-14 |
| 12 | transfer | 128 | 0.008 | 0.408 | 0.110 | 0.121 | 0.501 | 2.1e-10 |
| 15 | transfer | 64 | 0.002 | 0.411 | 0.108 | 0.125 | 0.504 | 3.1e-09 |
| 18 | probe | 16 | 0.001 | 0.388 | 0.107 | 0.126 | 0.504 | 7.8e-09 |

Train n=6 C_n Walsh mass in degrees 1–4 is 0.120 (degree 3 is 0.031, degree ≥5 is 4.690e-34).

## Transfer table

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The 95% CI resamples instances inside each seed and keeps every seed (4000 draws, seed 5136). Paired W/T/L counts instances with ε_method < ε_Fourier, exact ties, and losses, summed over seeds. μ=0 R is the PR #19 unsat seal. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. n=18 is the probe.

| method | L | n | role | B | Med ε | seed std | R | 95% CI | paired W/T/L | mark | μ=0 R | drop |
|:-------|--:|--:|:-----|--:|------:|---------:|--:|:-------|:-------------|:-----|------:|-----:|
| fourier_qaoa | 2 | 6 | hold | 256 | 0.0316 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 9 | hold | 256 | 0.0309 | 0.0001 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 12 | transfer | 128 | 0.0294 | 0.0001 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 15 | transfer | 64 | 0.0322 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 18 | probe | 16 | 0.0314 | 0.0000 | — | — | — |  | — | — |
| power_p | 2 | 6 | hold | 256 | 0.0257 | 0.0007 | 0.813 | 0.785–0.838 | 1277/0/3 | BEAT | 0.915 | +0.102 |
| power_p | 2 | 9 | hold | 256 | 0.0286 | 0.0002 | 0.928 | 0.907–0.938 | 1212/0/68 | BEAT | 0.961 | +0.033 |
| power_p | 2 | 12 | transfer | 128 | 0.0292 | 0.0004 | 0.993 | 0.964–1.003 | 394/0/246 | TIE | 1.000 | +0.007 |
| power_p | 2 | 15 | transfer | 64 | 0.0336 | 0.0007 | 1.044 | 0.992–1.050 | 96/0/224 | UNDER | 0.994 | -0.050 |
| power_p | 2 | 18 | probe | 16 | 0.0329 | 0.0006 | 1.047 | 1.001–1.085 | 16/0/64 | UNDER | — | — |
| walsh | 2 | 6 | hold | 256 | 0.0142 | 0.0002 | 0.451 | 0.432–0.470 | 1280/0/0 | BEAT | 0.670 | +0.219 |
| walsh | 2 | 9 | hold | 256 | 0.0142 | 0.0005 | 0.460 | 0.448–0.484 | 1280/0/0 | BEAT | 0.705 | +0.245 |
| walsh | 2 | 12 | transfer | 128 | 0.0142 | 0.0006 | 0.483 | 0.473–0.504 | 640/0/0 | BEAT | 0.761 | +0.278 |
| walsh | 2 | 15 | transfer | 64 | 0.0150 | 0.0008 | 0.468 | 0.458–0.500 | 320/0/0 | BEAT | 0.733 | +0.265 |
| walsh | 2 | 18 | probe | 16 | 0.0157 | 0.0008 | 0.499 | 0.465–0.546 | 80/0/0 | BEAT | — | — |
| fourier_qaoa | 16 | 6 | hold | 256 | 0.0022 | 0.0009 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 9 | hold | 256 | 0.0017 | 0.0009 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 12 | transfer | 128 | 0.0015 | 0.0010 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 15 | transfer | 64 | 0.0015 | 0.0010 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 18 | probe | 16 | 0.0016 | 0.0011 | — | — | — |  | — | — |
| power_p | 16 | 6 | hold | 256 | 0.0007 | 0.0008 | 0.291 | 0.273–0.319 | 1280/0/0 | BEAT | 0.978 | +0.687 |
| power_p | 16 | 9 | hold | 256 | 0.0006 | 0.0008 | 0.346 | 0.331–0.371 | 1280/0/0 | BEAT | 0.951 | +0.605 |
| power_p | 16 | 12 | transfer | 128 | 0.0006 | 0.0009 | 0.409 | 0.389–0.435 | 640/0/0 | BEAT | 0.990 | +0.581 |
| power_p | 16 | 15 | transfer | 64 | 0.0007 | 0.0010 | 0.436 | 0.413–0.490 | 320/0/0 | BEAT | 0.992 | +0.556 |
| power_p | 16 | 18 | probe | 16 | 0.0008 | 0.0010 | 0.480 | 0.447–0.578 | 80/0/0 | BEAT | — | — |
| walsh | 16 | 6 | hold | 256 | 0.0006 | 0.0007 | 0.250 | 0.233–0.274 | 1280/0/0 | BEAT | 0.925 | +0.675 |
| walsh | 16 | 9 | hold | 256 | 0.0005 | 0.0007 | 0.294 | 0.277–0.324 | 1280/0/0 | BEAT | 0.895 | +0.601 |
| walsh | 16 | 12 | transfer | 128 | 0.0005 | 0.0008 | 0.342 | 0.320–0.373 | 640/0/0 | BEAT | 0.940 | +0.598 |
| walsh | 16 | 15 | transfer | 64 | 0.0006 | 0.0008 | 0.375 | 0.340–0.412 | 320/0/0 | BEAT | 0.914 | +0.539 |
| walsh | 16 | 18 | probe | 16 | 0.0007 | 0.0009 | 0.411 | 0.381–0.496 | 80/0/0 | BEAT | — | — |

Per-seed median ε, per-seed R, and per-seed paired wins:

- `fourier_qaoa` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0316, 0.0316, 0.0316, 0.0316, 0.0316
- `fourier_qaoa` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0310, 0.0310, 0.0308, 0.0308, 0.0309
- `fourier_qaoa` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0296, 0.0296, 0.0294, 0.0294, 0.0294
- `fourier_qaoa` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0322, 0.0322, 0.0321, 0.0321, 0.0322
- `fourier_qaoa` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0314, 0.0314, 0.0314, 0.0314, 0.0314
- `power_p` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0263, 0.0263, 0.0255, 0.0246, 0.0257 R 0.830, 0.830, 0.808, 0.778, 0.814 (median of those R = 0.814); paired wins 256/256, 256/256, 256/256, 253/256, 256/256
- `power_p` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0287, 0.0287, 0.0285, 0.0282, 0.0286 R 0.926, 0.926, 0.924, 0.916, 0.928 (median of those R = 0.926); paired wins 250/256, 250/256, 237/256, 237/256, 238/256
- `power_p` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0286, 0.0286, 0.0293, 0.0293, 0.0292 R 0.966, 0.966, 0.997, 0.997, 0.993 (median of those R = 0.993); paired wins 95/128, 95/128, 68/128, 68/128, 68/128
- `power_p` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0325, 0.0325, 0.0337, 0.0338, 0.0336 R 1.012, 1.012, 1.049, 1.054, 1.044 (median of those R = 1.044); paired wins 36/64, 36/64, 9/64, 5/64, 10/64
- `power_p` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0319, 0.0319, 0.0330, 0.0331, 0.0329 R 1.015, 1.015, 1.051, 1.057, 1.047 (median of those R = 1.047); paired wins 7/16, 7/16, 1/16, 0/16, 1/16
- `walsh` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0146, 0.0142, 0.0142, 0.0144, 0.0142 R 0.462, 0.450, 0.448, 0.456, 0.449 (median of those R = 0.450); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0152, 0.0142, 0.0139, 0.0145, 0.0140 R 0.492, 0.458, 0.452, 0.471, 0.454 (median of those R = 0.458); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0154, 0.0142, 0.0141, 0.0147, 0.0141 R 0.521, 0.480, 0.478, 0.501, 0.479 (median of those R = 0.480); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0167, 0.0150, 0.0148, 0.0157, 0.0150 R 0.518, 0.468, 0.461, 0.488, 0.467 (median of those R = 0.468); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0173, 0.0157, 0.0154, 0.0164, 0.0156 R 0.550, 0.499, 0.491, 0.522, 0.498 (median of those R = 0.499); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `fourier_qaoa` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0022, 0.0035, 0.0021, 0.0021, 0.0038
- `fourier_qaoa` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0017, 0.0028, 0.0015, 0.0015, 0.0034
- `fourier_qaoa` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0015, 0.0027, 0.0014, 0.0014, 0.0035
- `fourier_qaoa` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0015, 0.0028, 0.0014, 0.0014, 0.0035
- `fourier_qaoa` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0029, 0.0015, 0.0015, 0.0038
- `power_p` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0006, 0.0020, 0.0005, 0.0007, 0.0023 R 0.278, 0.556, 0.247, 0.317, 0.599 (median of those R = 0.317); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0006, 0.0018, 0.0005, 0.0006, 0.0023 R 0.336, 0.653, 0.305, 0.373, 0.672 (median of those R = 0.373); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0006, 0.0018, 0.0005, 0.0006, 0.0024 R 0.404, 0.686, 0.364, 0.433, 0.689 (median of those R = 0.433); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `power_p` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0007, 0.0020, 0.0006, 0.0007, 0.0026 R 0.431, 0.710, 0.412, 0.464, 0.754 (median of those R = 0.464); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `power_p` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0008, 0.0022, 0.0007, 0.0008, 0.0029 R 0.476, 0.750, 0.447, 0.511, 0.763 (median of those R = 0.511); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `walsh` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0005, 0.0016, 0.0005, 0.0006, 0.0021 R 0.232, 0.443, 0.239, 0.272, 0.536 (median of those R = 0.272); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0005, 0.0014, 0.0004, 0.0005, 0.0021 R 0.279, 0.508, 0.291, 0.318, 0.606 (median of those R = 0.318); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0005, 0.0014, 0.0005, 0.0005, 0.0022 R 0.323, 0.513, 0.346, 0.361, 0.624 (median of those R = 0.361); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0005, 0.0015, 0.0005, 0.0006, 0.0024 R 0.356, 0.550, 0.384, 0.399, 0.676 (median of those R = 0.399); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0006, 0.0017, 0.0006, 0.0007, 0.0027 R 0.398, 0.578, 0.429, 0.437, 0.711 (median of those R = 0.437); paired wins 16/16, 16/16, 16/16, 16/16, 16/16

Walsh L=16 n=12: R=0.342, instance CI 0.320–0.373, seed-resample CI 0.323–0.624, pooled paired wins 640/640 (Wilson 95% 0.994–1.000; pairs repeat instances across seeds). PR #23 point estimate 0.330.
Walsh L=16 n=15: R=0.375, instance CI 0.340–0.412, seed-resample CI 0.356–0.676, pooled paired wins 320/320 (Wilson 95% 0.988–1.000; pairs repeat instances across seeds). PR #23 point estimate 0.355.
Walsh L=16 n=18 probe: R=0.411, instance CI 0.381–0.496, pooled paired wins 80/80 (Wilson 95% 0.954–1.000). Exploratory.

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| n | role | γ | Med ε | Walsh L=16 Med ε |
|--:|:-----|--:|------:|------------------:|
| 6 | hold | 0.0000 | 0.1322 | 0.0006 |
| 9 | hold | 0.0000 | 0.1258 | 0.0005 |
| 12 | transfer | 0.0000 | 0.1202 | 0.0005 |
| 15 | transfer | 0.0000 | 0.1245 | 0.0006 |
| 18 | probe | 0.0000 | 0.1242 | 0.0007 |

## Rule

PASS if frozen L=16 Walsh at n=12 and at n=15 both have ratio-of-medians ≤ 0.98, the instance-bootstrap 95% interval lies inside that BEAT line, every seed beats its own Fourier, every seed wins a strict majority of instances, the n=12 drop versus PR #19's 0.940 is at least 0.02 on both the ratio of medians and the median per-seed ratio, and the n=12 and n=15 ground states have peak < 0.5 and product gap ≥ 0.02. power_p does not gate PASS. n=18 does not gate PASS.

A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. needs_edit also covers a BEAT whose interval, seed list, or paired counts do not all sit inside the bar. KILL if a ground state at n=6, 9, or 12 is a bitstring; if n=12 loses to Fourier and is not below 0.940; or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05).

power_p control at n=12 L=16: R=0.409 (BEAT), instance CI 0.389–0.435, published μ=0 power_p R=0.990, drop +0.581.

Optimizer time summed over fits 450.5s. Branch `cursor/unsat-walsh-mu-deepen-5136`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

