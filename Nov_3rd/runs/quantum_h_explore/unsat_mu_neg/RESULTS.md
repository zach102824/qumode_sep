# RESULTS — unsat Walsh μ sweep below −0.50, L=16 freeze n=12

**Verdict: PASS**

μ=-1.00 frozen n=12 L=16 Walsh R=0.097 is a BEAT and clearly below the μ=−0.50 ratio 0.150 (drop 0.053, seed-median R=0.097). 3/3 seeds beat. n=12 peak 0.001, product gap 0.028. n=15 R=0.119. μ=−1.00 is the negative edge of this grid (further edge). A μ below −1.00 was not trained.

## Setup

Same gate as PR #26. The intra-cavity XX term is scored from the X histogram. It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. n=21 was not loaded.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
μ ∈ {−1.00, −0.85, −0.75, −0.65, −0.50}
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from the PR #26 unsat diagonals (those packs are base 51e9 / scale1000 base 54e9). Holdout B=128. Freeze n=12 B=64, n=15 B=32. Seeds [0, 1, 2]. L=16 only. L-BFGS rounds 48, max_iter 8, lr=0.35. Early stop is 8 rounds without a 10⁻⁶ improvement. That is the PR #26 L=16 budget. Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. μ=−0.50 is the PR #26 checkpoint, not a retrain.

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this μ). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. A μ beats −0.50 clearly when both the ratio of medians and the median of the three per-seed ratios drop by at least 0.02. Absolute residuals are not compared across μ.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|. At μ=−0.50 the n=12 product gap on the PR #26 holdout was 0.065.

Worst |ΔE0| Lanczos vs dense over the grid = 3.20e-14. Histogram vs ⟨ψ|H|ψ⟩ = 8.88e-15.

| μ | n | role | peak Med | prod gap Med |
|--:|--:|:-----|---------:|-------------:|
| -1.00 | 6 | hold | 0.042 | 0.033 |
| -1.00 | 9 | hold | 0.007 | 0.029 |
| -1.00 | 12 | transfer | 0.001 | 0.028 |
| -1.00 | 15 | transfer | 0.000 | 0.028 |
| -0.85 | 6 | hold | 0.046 | 0.041 |
| -0.85 | 9 | hold | 0.009 | 0.035 |
| -0.85 | 12 | transfer | 0.002 | 0.035 |
| -0.85 | 15 | transfer | 0.000 | 0.034 |
| -0.75 | 6 | hold | 0.050 | 0.048 |
| -0.75 | 9 | hold | 0.010 | 0.041 |
| -0.75 | 12 | transfer | 0.002 | 0.041 |
| -0.75 | 15 | transfer | 0.000 | 0.040 |
| -0.65 | 6 | hold | 0.055 | 0.056 |
| -0.65 | 9 | hold | 0.011 | 0.048 |
| -0.65 | 12 | transfer | 0.002 | 0.048 |
| -0.65 | 15 | transfer | 0.001 | 0.048 |
| -0.50 | 6 | hold | 0.065 | 0.074 |
| -0.50 | 9 | hold | 0.014 | 0.063 |
| -0.50 | 12 | transfer | 0.003 | 0.065 |
| -0.50 | 15 | transfer | 0.001 | 0.063 |

## Freeze table

n=6 and n=9 are the selection holdout. n=12 and n=15 are not. power_p is the scalar-γ control and does not choose μ.

| μ | head | n=6 R | n=9 R | n=12 R | n=12 mark | n=15 R | n=12 Med ε |
|--:|:-----|------:|------:|-------:|:----------|-------:|-----------:|
| -1.00 | walsh | 0.090 | 0.084 | 0.097 | BEAT | 0.119 | 0.0002 |
| -1.00 | power_p | 0.116 | 0.118 | 0.142 | BEAT | 0.170 | 0.0002 |
| -0.85 | walsh | 0.089 | 0.081 | 0.102 | BEAT | 0.130 | 0.0002 |
| -0.85 | power_p | 0.100 | 0.103 | 0.132 | BEAT | 0.160 | 0.0002 |
| -0.75 | walsh | 0.109 | 0.097 | 0.124 | BEAT | 0.136 | 0.0002 |
| -0.75 | power_p | 0.148 | 0.143 | 0.173 | BEAT | 0.185 | 0.0003 |
| -0.65 | walsh | 0.098 | 0.095 | 0.130 | BEAT | 0.152 | 0.0002 |
| -0.65 | power_p | 0.131 | 0.146 | 0.193 | BEAT | 0.221 | 0.0003 |
| -0.50 | walsh | 0.114 | 0.117 | 0.150 | BEAT | 0.176 | 0.0002 |
| -0.50 | power_p | 0.143 | 0.151 | 0.185 | BEAT | 0.217 | 0.0003 |

Per-seed n=12 Walsh R, seeds [0, 1, 2]:

- μ=-1.00: 0.077, 0.207, 0.097 (median of those R = 0.097); ratio of medians = 0.097
- μ=-0.85: 0.102, 0.251, 0.102 (median of those R = 0.102); ratio of medians = 0.102
- μ=-0.75: 0.115, 0.234, 0.124 (median of those R = 0.124); ratio of medians = 0.124
- μ=-0.65: 0.130, 0.155, 0.122 (median of those R = 0.130); ratio of medians = 0.130
- μ=-0.50: 0.154, 0.257, 0.131 (median of those R = 0.154); ratio of medians = 0.150

## Reading

Frozen n=12 Walsh R is 0.097, 0.102, 0.124, 0.130, 0.150 as μ runs from -1.00 to -0.50.
The ratio rises at every step toward μ=−0.50, so the minimum on this grid is the negative edge.
Grid points at least 0.02 below μ=−0.50 on the ratio of medians: -1.00 at 0.097, -0.85 at 0.102, -0.75 at 0.124, -0.65 at 0.130.
μ=−1.00 still wins and stays quantum, so this is a further edge. A μ below −1.00 was not trained.

power_p stays above Walsh at every μ. The μ move is not confined to the Walsh head.
power_p's lowest n=12 R on this grid is μ=-0.85 at 0.132, so it does not pick the same point as Walsh.

n=12 product gap by μ: -1.00:0.028, -0.85:0.035, -0.75:0.041, -0.65:0.048, -0.50:0.065.
n=12 product-gap minimum: -1.00 min 0.019 (1/64 under 0.02, peak max 0.002); -0.85 min 0.024 (0/64 under 0.02, peak max 0.003); -0.75 min 0.027 (0/64 under 0.02, peak max 0.003); -0.65 min 0.032 (0/64 under 0.02, peak max 0.004); -0.50 min 0.041 (0/64 under 0.02, peak max 0.006).

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| μ | n | γ | Med ε | Walsh L=16 Med ε |
|--:|--:|--:|------:|------------------:|
| -1.00 | 6 | 0.0000 | 0.0695 | 0.0002 |
| -1.00 | 9 | 0.0000 | 0.0634 | 0.0002 |
| -1.00 | 12 | 0.0000 | 0.0592 | 0.0002 |
| -1.00 | 15 | 0.0000 | 0.0607 | 0.0002 |
| -0.85 | 6 | 0.0000 | 0.0769 | 0.0002 |
| -0.85 | 9 | 0.0000 | 0.0702 | 0.0002 |
| -0.85 | 12 | 0.0000 | 0.0653 | 0.0002 |
| -0.85 | 15 | 0.0000 | 0.0676 | 0.0002 |
| -0.75 | 6 | 0.0000 | 0.0825 | 0.0003 |
| -0.75 | 9 | 0.0000 | 0.0757 | 0.0002 |
| -0.75 | 12 | 0.0000 | 0.0702 | 0.0002 |
| -0.75 | 15 | 0.0000 | 0.0733 | 0.0002 |
| -0.65 | 6 | 0.0000 | 0.0894 | 0.0003 |
| -0.65 | 9 | 0.0000 | 0.0822 | 0.0002 |
| -0.65 | 12 | 0.0000 | 0.0759 | 0.0002 |
| -0.65 | 15 | 0.0000 | 0.0796 | 0.0003 |
| -0.50 | 6 | 0.0000 | 0.1030 | 0.0003 |
| -0.50 | 9 | 0.0000 | 0.0945 | 0.0002 |
| -0.50 | 12 | 0.0000 | 0.0870 | 0.0002 |
| -0.50 | 15 | 0.0000 | 0.0924 | 0.0003 |

## Rule

Operating μ is the quantum grid point with the lowest frozen n=12 L=16 Walsh ratio, provided that ratio is at least 0.02 below the in-run μ=−0.50 checkpoint on both the ratio of medians and the median per-seed ratio, R≤0.98, at least two seeds beat their own Fourier, n=12 peak < 0.5 and product gap ≥ 0.02, and n=15 R stays below 1.10 with a non-bitstring ground state. Otherwise μ=−0.50 stays.

PASS if a μ beats −0.50 under that rule, or if the sweep shows −0.50 is still optimal, including when a lower ratio is a quantumness break. needs_edit if the grid or the PR #26 anchor is incomplete, or a clearly lower μ fails a transfer gate. KILL if the operating point's ground state (n=6, 9, or 12) is a bitstring, Walsh there loses to Fourier, or the L=16 advantage reverses. power_p does not gate the call. If μ=−1.00 wins, say so: the grid is still on a further edge.

Optimizer time summed over new fits 1155.7s (μ=−0.50 resumed, wall 274.2s). Branch `cursor/unsat-walsh-mu-neg-b37d`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

