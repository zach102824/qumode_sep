# RESULTS — quantumness-safe μ on unsat Walsh, L=16 freeze n=12

**Verdict: PASS**

Operate at μ=-0.50. That is the SAFE champ: frozen n=12 L=16 Walsh R=0.150 (seed-median R=0.154) is the lowest ratio whose median product gap is at least 0.05 (0.065) and whose instance fraction at gap ≥ 0.02 is at least 95% (64/64). The unconstrained argmin is the EDGE champ μ=-0.90 at R=0.092 (median gap 0.032, 64/64 instances ≥ 0.02). The EDGE champ misses the safe bar: median product gap 0.032 is under 0.05.

## Setup

Same gate as PR #26. The intra-cavity XX term is scored from the X histogram. It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. n=21 was not loaded. λ=0.75 is fixed.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
μ ∈ {−1.00, −0.95, −0.90, −0.85, −0.80, −0.75, −0.60, −0.50}
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from the PR #26 unsat diagonals (those packs are base 51e9 / scale1000 base 54e9). Holdout B=128. Freeze n=12 B=64, n=15 B=32. Seeds [0, 1, 2]. L=16 only. L-BFGS rounds 48, max_iter 8, lr=0.35. Early stop is 8 rounds without a 10⁻⁶ improvement. That is the PR #26 L=16 budget. Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. μ ∈ {−1.00, −0.85, −0.75, −0.50} are the PR #28 checkpoints, not retrains. μ ∈ {−0.95, −0.90, −0.80, −0.60} are new fits on the same diagonals.

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this μ). Lower R is the improvement direction from PR #26 and PR #28. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The SAFE champ is the argmin of frozen n=12 Walsh R among grid points with median product gap ≥ 0.05 and at least 95% of the 64 n=12 instances at gap ≥ 0.02. The EDGE champ is the unconstrained argmin of that R. The operating point is SAFE. A tie on R prefers the thicker median gap.

## Recommendation

SAFE champ: μ=-0.50, n=12 Walsh R=0.150.
EDGE champ: μ=-0.90, n=12 Walsh R=0.092.
Operating μ: -0.50.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0| on the frozen n=12 transfer set (B=64). The safe bar is the median of that set and the fraction of its instances, not the training set.

Worst |ΔE0| Lanczos vs dense over the grid = 4.44e-14. Histogram vs ⟨ψ|H|ψ⟩ = 8.88e-15.

| μ | n | role | peak Med | prod gap Med |
|--:|--:|:-----|---------:|-------------:|
| -1.00 | 6 | hold | 0.042 | 0.033 |
| -1.00 | 9 | hold | 0.007 | 0.029 |
| -1.00 | 12 | transfer | 0.001 | 0.028 |
| -1.00 | 15 | transfer | 0.000 | 0.028 |
| -0.95 | 6 | hold | 0.043 | 0.035 |
| -0.95 | 9 | hold | 0.008 | 0.031 |
| -0.95 | 12 | transfer | 0.001 | 0.030 |
| -0.95 | 15 | transfer | 0.000 | 0.030 |
| -0.90 | 6 | hold | 0.044 | 0.038 |
| -0.90 | 9 | hold | 0.008 | 0.033 |
| -0.90 | 12 | transfer | 0.002 | 0.032 |
| -0.90 | 15 | transfer | 0.000 | 0.032 |
| -0.85 | 6 | hold | 0.046 | 0.041 |
| -0.85 | 9 | hold | 0.009 | 0.035 |
| -0.85 | 12 | transfer | 0.002 | 0.035 |
| -0.85 | 15 | transfer | 0.000 | 0.034 |
| -0.80 | 6 | hold | 0.048 | 0.044 |
| -0.80 | 9 | hold | 0.009 | 0.038 |
| -0.80 | 12 | transfer | 0.002 | 0.037 |
| -0.80 | 15 | transfer | 0.000 | 0.037 |
| -0.75 | 6 | hold | 0.050 | 0.048 |
| -0.75 | 9 | hold | 0.010 | 0.041 |
| -0.75 | 12 | transfer | 0.002 | 0.041 |
| -0.75 | 15 | transfer | 0.000 | 0.040 |
| -0.60 | 6 | hold | 0.058 | 0.061 |
| -0.60 | 9 | hold | 0.012 | 0.052 |
| -0.60 | 12 | transfer | 0.003 | 0.053 |
| -0.60 | 15 | transfer | 0.001 | 0.052 |
| -0.50 | 6 | hold | 0.065 | 0.074 |
| -0.50 | 9 | hold | 0.014 | 0.063 |
| -0.50 | 12 | transfer | 0.003 | 0.065 |
| -0.50 | 15 | transfer | 0.001 | 0.063 |

## Freeze table

n=6 and n=9 are the selection holdout. n=12 and n=15 are not. power_p is the scalar-γ control and does not choose μ. The champ column marks SAFE, EDGE, or both.

| μ | champ | head | n=12 R | seed-med R | gap med | ≥0.02 | min gap | peak | n=6 R | n=9 R | n=15 R |
|--:|:------|:-----|-------:|-----------:|--------:|------:|--------:|-----:|------:|------:|-------:|
| -1.00 |  | walsh | 0.097 | 0.097 | 0.028 | 63/64 | 0.019 | 0.001 | 0.090 | 0.084 | 0.119 |
| -1.00 |  | power_p | 0.142 | 0.142 |  |  |  |  | 0.116 | 0.118 | 0.170 |
| -0.95 |  | walsh | 0.106 | 0.106 | 0.030 | 64/64 | 0.021 | 0.001 | 0.091 | 0.086 | 0.133 |
| -0.95 |  | power_p | 0.156 | 0.156 |  |  |  |  | 0.123 | 0.129 | 0.187 |
| -0.90 | EDGE | walsh | 0.092 | 0.092 | 0.032 | 64/64 | 0.022 | 0.002 | 0.084 | 0.074 | 0.117 |
| -0.90 |  | power_p | 0.132 | 0.132 |  |  |  |  | 0.109 | 0.111 | 0.159 |
| -0.85 |  | walsh | 0.102 | 0.102 | 0.035 | 64/64 | 0.024 | 0.002 | 0.089 | 0.081 | 0.130 |
| -0.85 |  | power_p | 0.132 | 0.132 |  |  |  |  | 0.100 | 0.103 | 0.160 |
| -0.80 |  | walsh | 0.110 | 0.110 | 0.037 | 64/64 | 0.025 | 0.002 | 0.092 | 0.084 | 0.135 |
| -0.80 |  | power_p | 0.149 | 0.149 |  |  |  |  | 0.120 | 0.120 | 0.174 |
| -0.75 |  | walsh | 0.124 | 0.124 | 0.041 | 64/64 | 0.027 | 0.002 | 0.109 | 0.097 | 0.136 |
| -0.75 |  | power_p | 0.173 | 0.173 |  |  |  |  | 0.148 | 0.143 | 0.185 |
| -0.60 |  | walsh | 0.155 | 0.155 | 0.053 | 64/64 | 0.035 | 0.003 | 0.115 | 0.113 | 0.177 |
| -0.60 |  | power_p | 0.213 | 0.213 |  |  |  |  | 0.147 | 0.162 | 0.246 |
| -0.50 | SAFE | walsh | 0.150 | 0.154 | 0.065 | 64/64 | 0.041 | 0.003 | 0.114 | 0.117 | 0.176 |
| -0.50 |  | power_p | 0.185 | 0.191 |  |  |  |  | 0.143 | 0.151 | 0.217 |

Per-seed n=12 Walsh R, seeds [0, 1, 2]:

- μ=-1.00: 0.077, 0.207, 0.097 (median of those R = 0.097); ratio of medians = 0.097
- μ=-0.95: 0.089, 0.200, 0.106 (median of those R = 0.106); ratio of medians = 0.106
- μ=-0.90: 0.090, 0.220, 0.092 (median of those R = 0.092); ratio of medians = 0.092
- μ=-0.85: 0.102, 0.251, 0.102 (median of those R = 0.102); ratio of medians = 0.102
- μ=-0.80: 0.102, 0.214, 0.110 (median of those R = 0.110); ratio of medians = 0.110
- μ=-0.75: 0.115, 0.234, 0.124 (median of those R = 0.124); ratio of medians = 0.124
- μ=-0.60: 0.137, 0.203, 0.155 (median of those R = 0.155); ratio of medians = 0.155
- μ=-0.50: 0.154, 0.257, 0.131 (median of those R = 0.154); ratio of medians = 0.150

## Reading

Frozen n=12 Walsh R by μ: -1.00:0.097, -0.95:0.106, -0.90:0.092, -0.85:0.102, -0.80:0.110, -0.75:0.124, -0.60:0.155, -0.50:0.150.
Unconstrained argmin on the ratio of medians is μ=-0.90 at 0.092.
That minimum is inside the grid, so R is not monotone in μ on this spacing.
Per-seed argmin of n=12 Walsh R: seed 0 μ=-1.00 at 0.077; seed 1 μ=-0.95 at 0.200; seed 2 μ=-0.90 at 0.092.
The median of the three per-seed ratios picks the same μ as the ratio of medians, at 0.092.
power_p stays above Walsh at every μ.
power_p's lowest n=12 R is μ=-0.85 at 0.132.
Walsh's unconstrained minimum and the power_p minimum are different grid points.
The 95% instance bar removes no grid point. The median-gap bar removes 6 grid points.
Other points that clear both bars sit at a higher ratio: μ=-0.60 at 0.155.
n=12 product gap: -1.00 med 0.028 (63/64 ≥ 0.02, min 0.019); -0.95 med 0.030 (64/64 ≥ 0.02, min 0.021); -0.90 med 0.032 (64/64 ≥ 0.02, min 0.022); -0.85 med 0.035 (64/64 ≥ 0.02, min 0.024); -0.80 med 0.037 (64/64 ≥ 0.02, min 0.025); -0.75 med 0.041 (64/64 ≥ 0.02, min 0.027); -0.60 med 0.053 (64/64 ≥ 0.02, min 0.035); -0.50 med 0.065 (64/64 ≥ 0.02, min 0.041).
Clears both bars: -0.60 R=0.155, -0.50 R=0.150.
Misses the safe bar: -1.00 R=0.097 (median product gap 0.028 is under 0.05); -0.95 R=0.106 (median product gap 0.030 is under 0.05); -0.90 R=0.092 (median product gap 0.032 is under 0.05); -0.85 R=0.102 (median product gap 0.035 is under 0.05); -0.80 R=0.110 (median product gap 0.037 is under 0.05); -0.75 R=0.124 (median product gap 0.041 is under 0.05).

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| μ | n | γ | Med ε | Walsh L=16 Med ε |
|--:|--:|--:|------:|------------------:|
| -1.00 | 6 | 0.0000 | 0.0695 | 0.0002 |
| -1.00 | 9 | 0.0000 | 0.0634 | 0.0002 |
| -1.00 | 12 | 0.0000 | 0.0592 | 0.0002 |
| -1.00 | 15 | 0.0000 | 0.0607 | 0.0002 |
| -0.95 | 6 | 0.0000 | 0.0719 | 0.0002 |
| -0.95 | 9 | 0.0000 | 0.0655 | 0.0002 |
| -0.95 | 12 | 0.0000 | 0.0611 | 0.0002 |
| -0.95 | 15 | 0.0000 | 0.0627 | 0.0002 |
| -0.90 | 6 | 0.0000 | 0.0745 | 0.0002 |
| -0.90 | 9 | 0.0000 | 0.0677 | 0.0001 |
| -0.90 | 12 | 0.0000 | 0.0631 | 0.0002 |
| -0.90 | 15 | 0.0000 | 0.0651 | 0.0002 |
| -0.85 | 6 | 0.0000 | 0.0769 | 0.0002 |
| -0.85 | 9 | 0.0000 | 0.0702 | 0.0002 |
| -0.85 | 12 | 0.0000 | 0.0653 | 0.0002 |
| -0.85 | 15 | 0.0000 | 0.0676 | 0.0002 |
| -0.80 | 6 | 0.0000 | 0.0794 | 0.0002 |
| -0.80 | 9 | 0.0000 | 0.0728 | 0.0002 |
| -0.80 | 12 | 0.0000 | 0.0677 | 0.0002 |
| -0.80 | 15 | 0.0000 | 0.0703 | 0.0002 |
| -0.75 | 6 | 0.0000 | 0.0825 | 0.0003 |
| -0.75 | 9 | 0.0000 | 0.0757 | 0.0002 |
| -0.75 | 12 | 0.0000 | 0.0702 | 0.0002 |
| -0.75 | 15 | 0.0000 | 0.0733 | 0.0002 |
| -0.60 | 6 | 0.0000 | 0.0935 | 0.0003 |
| -0.60 | 9 | 0.0000 | 0.0859 | 0.0002 |
| -0.60 | 12 | 0.0000 | 0.0792 | 0.0003 |
| -0.60 | 15 | 0.0000 | 0.0833 | 0.0003 |
| -0.50 | 6 | 0.0000 | 0.1030 | 0.0003 |
| -0.50 | 9 | 0.0000 | 0.0945 | 0.0002 |
| -0.50 | 12 | 0.0000 | 0.0870 | 0.0002 |
| -0.50 | 15 | 0.0000 | 0.0924 | 0.0003 |

## Rule

Operating μ is the SAFE champ: lowest frozen n=12 L=16 Walsh ratio among points with median product gap ≥ 0.05, at least 95% of the B=64 n=12 instances at gap ≥ 0.02, and ground-state peak < 0.5 on n=6, n=9, and n=12. EDGE is reported beside it and does not replace SAFE when the gap thins. power_p does not choose μ. n=15 is reported and does not choose μ. n=21 is not loaded.

PASS when the full grid names both champs and SAFE still beats or ties Fourier. needs_edit when a grid point, a resumed checkpoint, or the gap tail is missing, or when no μ clears both bars. KILL when the SAFE champ's Walsh ratio loses to Fourier or its ground state is a bitstring.

Optimizer time summed over new fits 1008.6s (resumed checkpoints 1138.7s, wall 1198.5s). Branch `cursor/unsat-walsh-mu-safe-fa2b`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

