# RESULTS — unsat λ=0.75 + μ=−0.25 XX, Walsh freeze {6,9} → n=12/15

**Verdict: PASS**

frozen n=12 L=16 Walsh R=0.330 is a BEAT and clearly below the μ=0 unsat Walsh ratio 0.940 (drop 0.610, seed-median R=0.330). 3/3 seeds beat. n=15 R=0.355.

## Setup

Same gate as the μ=0 unsat Walsh seal (PR #19). The intra-cavity XX term is scored, not compiled into U_c. Angles for n=12 and n=15 were not in the loss and not in the checkpoint. n=21 was not loaded.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −0.25
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from `beat_qaoa_12h/train_packs` (base 51e9). Holdout B=128 per n and transfer B=64 (n=12), B=32 (n=15) are the scale1000 satisfiable prefix (base 54e9). Seeds [0, 1, 2]. L=[2, 16]. L-BFGS rounds 18/48, max_iter 8/8, lr=0.35. That is the PR #19 budget. Early stop is 8 rounds without a 10⁻⁶ improvement. Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier.

The μ=0 comparison is the published PR #19 ratio of medians, not a retrain. n=12 L=16 Walsh there is R=0.940. A clear drop is at least 0.02 on that ratio and on the median of the three per-seed ratios. Absolute residuals are not on the same scale once μ changes E_0.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian, diagonal plus single-X edges plus intra-cavity XX edges. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state found from |0⟩, |1⟩, |+⟩, the best bitstring, and three random starts, polished with L-BFGS.

Worst |ΔE0| Lanczos vs dense = 1.95e-14. Histogram vs ⟨ψ|H|ψ⟩ = 4.88e-15. μ=0 histogram vs Pauli field = 3.55e-15.

| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |
|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|
| 6 | train | 128 | 0.095 | 0.314 | 0.120 | 0.135 | 0.510 | 3.3e-15 |
| 6 | hold | 128 | 0.098 | 0.321 | 0.123 | 0.137 | 0.512 | 3.2e-15 |
| 9 | train | 128 | 0.028 | 0.380 | 0.110 | 0.126 | 0.508 | 8.0e-14 |
| 9 | hold | 128 | 0.029 | 0.365 | 0.104 | 0.126 | 0.507 | 7.0e-14 |
| 12 | transfer | 64 | 0.008 | 0.410 | 0.112 | 0.119 | 0.499 | 9.6e-11 |
| 15 | transfer | 32 | 0.002 | 0.420 | 0.106 | 0.125 | 0.505 | 3.0e-09 |

Train n=6 C_n Walsh mass in degrees 1–4 is 0.120 (degree 3 is 0.031, degree ≥5 is 4.690e-34). Equal degree weights still reproduce the scalar pulse up to a global phase when the mass above degree 4 vanishes. The XX term is not on the diagonal, so it does not change that split.

## Transfer table

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. μ=0 R is the published PR #19 unsat seal. Drop = (μ=0 R) − R. Positive means this run is lower. n=6 and n=9 are the selection holdout. n=12 and n=15 are not.

| method | L | n | role | params | Med ε | seed std | R | mark | μ=0 R | drop |
|:-------|--:|--:|:-----|-------:|------:|---------:|--:|:-----|------:|-----:|
| fourier_qaoa | 2 | 6 | hold | 4 | 0.0322 | 0.0000 | — |  | — | — |
| fourier_qaoa | 2 | 9 | hold | 4 | 0.0320 | 0.0000 | — |  | — | — |
| fourier_qaoa | 2 | 12 | transfer | 4 | 0.0290 | 0.0000 | — |  | — | — |
| fourier_qaoa | 2 | 15 | transfer | 4 | 0.0332 | 0.0000 | — |  | — | — |
| power_p | 2 | 6 | hold | 9 | 0.0275 | 0.0000 | 0.854 | BEAT | 0.915 | +0.061 |
| power_p | 2 | 9 | hold | 9 | 0.0299 | 0.0001 | 0.933 | BEAT | 0.961 | +0.028 |
| power_p | 2 | 12 | transfer | 9 | 0.0281 | 0.0001 | 0.970 | BEAT | 1.000 | +0.030 |
| power_p | 2 | 15 | transfer | 9 | 0.0332 | 0.0000 | 0.999 | TIE | 0.994 | -0.005 |
| walsh | 2 | 6 | hold | 15 | 0.0150 | 0.0001 | 0.466 | BEAT | 0.670 | +0.204 |
| walsh | 2 | 9 | hold | 15 | 0.0146 | 0.0008 | 0.456 | BEAT | 0.705 | +0.249 |
| walsh | 2 | 12 | transfer | 15 | 0.0142 | 0.0010 | 0.490 | BEAT | 0.761 | +0.271 |
| walsh | 2 | 15 | transfer | 15 | 0.0160 | 0.0011 | 0.483 | BEAT | 0.733 | +0.250 |
| fourier_qaoa | 16 | 6 | hold | 32 | 0.0021 | 0.0009 | — |  | — | — |
| fourier_qaoa | 16 | 9 | hold | 32 | 0.0017 | 0.0007 | — |  | — | — |
| fourier_qaoa | 16 | 12 | transfer | 32 | 0.0015 | 0.0008 | — |  | — | — |
| fourier_qaoa | 16 | 15 | transfer | 32 | 0.0016 | 0.0008 | — |  | — | — |
| power_p | 16 | 6 | hold | 65 | 0.0007 | 0.0009 | 0.320 | BEAT | 0.978 | +0.658 |
| power_p | 16 | 9 | hold | 65 | 0.0006 | 0.0008 | 0.366 | BEAT | 0.951 | +0.585 |
| power_p | 16 | 12 | transfer | 65 | 0.0007 | 0.0008 | 0.430 | BEAT | 0.990 | +0.560 |
| power_p | 16 | 15 | transfer | 65 | 0.0007 | 0.0008 | 0.459 | BEAT | 0.992 | +0.533 |
| walsh | 16 | 6 | hold | 113 | 0.0005 | 0.0007 | 0.251 | BEAT | 0.925 | +0.674 |
| walsh | 16 | 9 | hold | 113 | 0.0005 | 0.0006 | 0.278 | BEAT | 0.895 | +0.617 |
| walsh | 16 | 12 | transfer | 113 | 0.0005 | 0.0006 | 0.330 | BEAT | 0.940 | +0.610 |
| walsh | 16 | 15 | transfer | 113 | 0.0006 | 0.0006 | 0.355 | BEAT | 0.914 | +0.559 |

Per-seed median ε, then per-seed R where a baseline exists:

- `fourier_qaoa` L=2 n=6 seeds [0, 1, 2]: 0.0322, 0.0322, 0.0322
- `fourier_qaoa` L=2 n=9 seeds [0, 1, 2]: 0.0320, 0.0320, 0.0320
- `fourier_qaoa` L=2 n=12 seeds [0, 1, 2]: 0.0290, 0.0290, 0.0290
- `fourier_qaoa` L=2 n=15 seeds [0, 1, 2]: 0.0332, 0.0332, 0.0332
- `power_p` L=2 n=6 seeds [0, 1, 2]: 0.0276, 0.0275, 0.0275 R 0.856, 0.854, 0.854 (median of those R = 0.854)
- `power_p` L=2 n=9 seeds [0, 1, 2]: 0.0301, 0.0299, 0.0299 R 0.939, 0.933, 0.933 (median of those R = 0.933)
- `power_p` L=2 n=12 seeds [0, 1, 2]: 0.0283, 0.0281, 0.0281 R 0.975, 0.970, 0.970 (median of those R = 0.970)
- `power_p` L=2 n=15 seeds [0, 1, 2]: 0.0332, 0.0332, 0.0332 R 0.998, 1.000, 1.000 (median of those R = 1.000)
- `walsh` L=2 n=6 seeds [0, 1, 2]: 0.0152, 0.0150, 0.0149 R 0.471, 0.466, 0.464 (median of those R = 0.466)
- `walsh` L=2 n=9 seeds [0, 1, 2]: 0.0154, 0.0146, 0.0137 R 0.479, 0.456, 0.429 (median of those R = 0.456)
- `walsh` L=2 n=12 seeds [0, 1, 2]: 0.0151, 0.0142, 0.0131 R 0.520, 0.490, 0.450 (median of those R = 0.490)
- `walsh` L=2 n=15 seeds [0, 1, 2]: 0.0171, 0.0160, 0.0148 R 0.514, 0.483, 0.446 (median of those R = 0.483)
- `fourier_qaoa` L=16 n=6 seeds [0, 1, 2]: 0.0021, 0.0036, 0.0019
- `fourier_qaoa` L=16 n=9 seeds [0, 1, 2]: 0.0017, 0.0028, 0.0015
- `fourier_qaoa` L=16 n=12 seeds [0, 1, 2]: 0.0015, 0.0028, 0.0014
- `fourier_qaoa` L=16 n=15 seeds [0, 1, 2]: 0.0016, 0.0028, 0.0014
- `power_p` L=16 n=6 seeds [0, 1, 2]: 0.0007, 0.0021, 0.0005 R 0.320, 0.603, 0.273 (median of those R = 0.320)
- `power_p` L=16 n=9 seeds [0, 1, 2]: 0.0006, 0.0019, 0.0005 R 0.366, 0.667, 0.303 (median of those R = 0.366)
- `power_p` L=16 n=12 seeds [0, 1, 2]: 0.0007, 0.0019, 0.0005 R 0.430, 0.700, 0.377 (median of those R = 0.430)
- `power_p` L=16 n=15 seeds [0, 1, 2]: 0.0007, 0.0020, 0.0006 R 0.459, 0.727, 0.427 (median of those R = 0.459)
- `walsh` L=16 n=6 seeds [0, 1, 2]: 0.0005, 0.0017, 0.0005 R 0.251, 0.467, 0.243 (median of those R = 0.251)
- `walsh` L=16 n=9 seeds [0, 1, 2]: 0.0005, 0.0015, 0.0004 R 0.278, 0.528, 0.273 (median of those R = 0.278)
- `walsh` L=16 n=12 seeds [0, 1, 2]: 0.0005, 0.0015, 0.0005 R 0.330, 0.526, 0.328 (median of those R = 0.330)
- `walsh` L=16 n=15 seeds [0, 1, 2]: 0.0006, 0.0016, 0.0005 R 0.355, 0.587, 0.373 (median of those R = 0.373)

## One-parameter cost pulse

SNAP phase 0. A stack of such layers is one pulse along C_n, because they commute. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| n | role | γ | Med ε | Walsh L=16 Med ε |
|--:|:-----|--:|------:|------------------:|
| 6 | hold | 0.0000 | 0.1364 | 0.0005 |
| 9 | hold | 0.0000 | 0.1255 | 0.0005 |
| 12 | transfer | 0.0000 | 0.1186 | 0.0005 |
| 15 | transfer | 0.0000 | 0.1245 | 0.0006 |

## Rule

PASS if the frozen n=12 L=16 Walsh ratio of medians is ≤ 0.98 and at least 0.02 below PR #19's 0.940, the median of the three per-seed ratios drops by that same 0.02, at least two seeds beat their own Fourier, the n=12 ground state has peak < 0.5 and product gap ≥ 0.02, and n=15 L=16 Walsh R stays below 1.10 with a non-bitstring ground state. power_p does not gate PASS.

A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. KILL if a gated ground state (n=6, 9, or 12) is a bitstring; if n=12 loses to Fourier and is not below 0.940; or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05).

power_p control at n=12 L=16: R=0.430 (BEAT), published μ=0 power_p R=0.990, drop +0.560. PR #19's scalar head was a TIE at 0.990. This control says whether XX moves the scalar pulse, separate from Walsh.

Optimizer time summed over fits 270.9s. Branch `cursor/unsat-walsh-rank2-mu-5948`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

