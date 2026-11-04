# RESULTS — unsat Walsh μ sweep, L=16 freeze n=12

**Verdict: PASS**

μ=-0.50 frozen n=12 L=16 Walsh R=0.150 is a BEAT and clearly below the μ=−0.25 ratio 0.330 (drop 0.180, seed-median R=0.154). 3/3 seeds beat. n=12 peak 0.003, product gap 0.065. n=15 R=0.176.

## Setup

Same gate as PR #23. The intra-cavity XX term is scored from the X histogram. It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. n=21 was not loaded.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
μ ∈ {−0.50, −0.35, −0.25, −0.15, −0.05, 0}
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from the PR #23 unsat diagonals (those packs are base 51e9 / scale1000 base 54e9). Holdout B=128. Freeze n=12 B=64, n=15 B=32. Seeds [0, 1, 2]. L=16 only. L-BFGS rounds 48, max_iter 8, lr=0.35. Early stop is 8 rounds without a 10⁻⁶ improvement. That is the PR #23 L=16 budget. Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. μ=−0.25 is the PR #23 checkpoint, not a retrain.

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this μ). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. A μ beats −0.25 clearly when both the ratio of medians and the median of the three per-seed ratios drop by at least 0.02. Absolute residuals are not compared across μ.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|.

Worst |ΔE0| Lanczos vs dense over the grid = 1.28e-13. Histogram vs ⟨ψ|H|ψ⟩ = 6.22e-15.

| μ | n | role | peak Med | prod gap Med |
|--:|--:|:-----|---------:|-------------:|
| -0.50 | 6 | hold | 0.065 | 0.074 |
| -0.50 | 9 | hold | 0.014 | 0.063 |
| -0.50 | 12 | transfer | 0.003 | 0.065 |
| -0.50 | 15 | transfer | 0.001 | 0.063 |
| -0.35 | 6 | hold | 0.081 | 0.100 |
| -0.35 | 9 | hold | 0.021 | 0.085 |
| -0.35 | 12 | transfer | 0.005 | 0.090 |
| -0.35 | 15 | transfer | 0.001 | 0.084 |
| -0.25 | 6 | hold | 0.098 | 0.123 |
| -0.25 | 9 | hold | 0.029 | 0.104 |
| -0.25 | 12 | transfer | 0.008 | 0.112 |
| -0.25 | 15 | transfer | 0.002 | 0.106 |
| -0.15 | 6 | hold | 0.124 | 0.140 |
| -0.15 | 9 | hold | 0.042 | 0.128 |
| -0.15 | 12 | transfer | 0.014 | 0.135 |
| -0.15 | 15 | transfer | 0.005 | 0.124 |
| -0.05 | 6 | hold | 0.167 | 0.144 |
| -0.05 | 9 | hold | 0.066 | 0.138 |
| -0.05 | 12 | transfer | 0.025 | 0.145 |
| -0.05 | 15 | transfer | 0.012 | 0.127 |
| +0.00 | 6 | hold | 0.202 | 0.142 |
| +0.00 | 9 | hold | 0.086 | 0.132 |
| +0.00 | 12 | transfer | 0.038 | 0.138 |
| +0.00 | 15 | transfer | 0.019 | 0.120 |

## Freeze table

n=6 and n=9 are the selection holdout. n=12 and n=15 are not. power_p is the scalar-γ control and does not choose μ.

| μ | head | n=6 R | n=9 R | n=12 R | n=12 mark | n=15 R | n=12 Med ε |
|--:|:-----|------:|------:|-------:|:----------|-------:|-----------:|
| -0.50 | walsh | 0.114 | 0.117 | 0.150 | BEAT | 0.176 | 0.0002 |
| -0.50 | power_p | 0.143 | 0.151 | 0.185 | BEAT | 0.217 | 0.0003 |
| -0.35 | walsh | 0.186 | 0.208 | 0.253 | BEAT | 0.306 | 0.0004 |
| -0.35 | power_p | 0.246 | 0.279 | 0.332 | BEAT | 0.400 | 0.0005 |
| -0.25 | walsh | 0.251 | 0.278 | 0.330 | BEAT | 0.355 | 0.0005 |
| -0.25 | power_p | 0.320 | 0.366 | 0.430 | BEAT | 0.459 | 0.0007 |
| -0.15 | walsh | 0.433 | 0.502 | 0.556 | BEAT | 0.565 | 0.0010 |
| -0.15 | power_p | 0.529 | 0.577 | 0.675 | BEAT | 0.686 | 0.0012 |
| -0.05 | walsh | 0.741 | 0.768 | 0.778 | BEAT | 0.787 | 0.0019 |
| -0.05 | power_p | 0.842 | 0.866 | 0.878 | BEAT | 0.905 | 0.0021 |
| +0.00 | walsh | 0.803 | 0.856 | 0.871 | BEAT | 0.892 | 0.0024 |
| +0.00 | power_p | 0.968 | 0.976 | 0.982 | TIE | 0.975 | 0.0028 |

Per-seed n=12 Walsh R, seeds [0, 1, 2]:

- μ=-0.50: 0.154, 0.257, 0.131 (median of those R = 0.154); ratio of medians = 0.150
- μ=-0.35: 0.253, 0.432, 0.219 (median of those R = 0.253); ratio of medians = 0.253
- μ=-0.25: 0.330, 0.526, 0.328 (median of those R = 0.330); ratio of medians = 0.330
- μ=-0.15: 0.556, 0.697, 0.554 (median of those R = 0.556); ratio of medians = 0.556
- μ=-0.05: 0.778, 0.826, 0.784 (median of those R = 0.784); ratio of medians = 0.778
- μ=+0.00: 0.871, 0.856, 0.914 (median of those R = 0.871); ratio of medians = 0.871

## Reading

Frozen n=12 Walsh R is 0.150, 0.253, 0.330, 0.556, 0.778, 0.871 as μ runs from -0.50 to +0.00. The ratio rises at every step. 2 grid points clear the 0.02 bar against μ=−0.25 (-0.50 at 0.150, -0.35 at 0.253). The operating point is the lowest of those, the negative edge of this grid. A μ below that edge was not trained.

power_p follows the same direction and stays above Walsh at every μ. The μ move is not confined to the Walsh head.

The in-sweep μ=0 Walsh ratio is 0.871. PR #19 published 0.940 for that diagonal; that seal early-stopped some L=16 seeds. This sweep uses the PR #23 round cap (48, early stop only after 8 stalled rounds).

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| μ | n | γ | Med ε | Walsh L=16 Med ε |
|--:|--:|--:|------:|------------------:|
| -0.50 | 6 | 0.0000 | 0.1030 | 0.0003 |
| -0.50 | 9 | 0.0000 | 0.0945 | 0.0002 |
| -0.50 | 12 | 0.0000 | 0.0870 | 0.0002 |
| -0.50 | 15 | 0.0000 | 0.0924 | 0.0003 |
| -0.35 | 6 | 0.0000 | 0.1215 | 0.0004 |
| -0.35 | 9 | 0.0000 | 0.1112 | 0.0004 |
| -0.35 | 12 | 0.0000 | 0.1036 | 0.0004 |
| -0.35 | 15 | 0.0000 | 0.1089 | 0.0005 |
| -0.25 | 6 | 0.0000 | 0.1364 | 0.0005 |
| -0.25 | 9 | 0.0000 | 0.1255 | 0.0005 |
| -0.25 | 12 | 0.0000 | 0.1186 | 0.0005 |
| -0.25 | 15 | 0.0000 | 0.1245 | 0.0006 |
| -0.15 | 6 | 0.0000 | 0.1556 | 0.0009 |
| -0.15 | 9 | 0.0000 | 0.1439 | 0.0008 |
| -0.15 | 12 | 0.0000 | 0.1385 | 0.0010 |
| -0.15 | 15 | 0.0000 | 0.1457 | 0.0010 |
| -0.05 | 6 | 0.0000 | 0.1799 | 0.0016 |
| -0.05 | 9 | 0.0000 | 0.1691 | 0.0015 |
| -0.05 | 12 | 0.0000 | 0.1646 | 0.0019 |
| -0.05 | 15 | 0.0000 | 0.1739 | 0.0021 |
| +0.00 | 6 | 0.0000 | 0.1976 | 0.0019 |
| +0.00 | 9 | 0.0000 | 0.1873 | 0.0018 |
| +0.00 | 12 | 0.0000 | 0.1828 | 0.0024 |
| +0.00 | 15 | 0.0000 | 0.1907 | 0.0026 |

## Rule

Operating μ is the quantum grid point with the lowest frozen n=12 L=16 Walsh ratio, provided that ratio is at least 0.02 below the in-run μ=−0.25 checkpoint on both the ratio of medians and the median per-seed ratio, R≤0.98, at least two seeds beat their own Fourier, n=12 peak < 0.5 and product gap ≥ 0.02, and n=15 R stays below 1.10 with a non-bitstring ground state. Otherwise μ=−0.25 stays.

PASS if that switch happens, or if the sweep shows −0.25 is still optimal. needs_edit if the grid or the PR #23 anchor is incomplete, or a clearly lower μ fails a transfer gate. KILL if the operating point's ground state (n=6, 9, or 12) is a bitstring, Walsh there loses to Fourier, or the L=16 advantage reverses. power_p does not gate the call.

Optimizer time summed over new fits 1339.1s (μ=−0.25 resumed, wall 256.8s). Branch `cursor/unsat-walsh-mu-sweep-e2e2`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

