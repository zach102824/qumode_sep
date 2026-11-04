# RESULTS — unsat Walsh λ sweep at μ=−1.00, L=16 freeze n=12

**Verdict: PASS**

λ=1.00 has the lowest frozen n=12 L=16 Walsh R among cells with median product gap ≥ 0.02 (R=0.076, median gap 0.021, λ=0.75 is 0.097). 3/3 seeds beat. n=12 peak 0.001. n=15 R=0.096.

## Setup

Same gate as PR #26 and PR #28. The intra-cavity XX term is scored from the X histogram. It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. n=21 was not loaded.

```text
H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
μ = −1.00
λ ∈ {0.50, 0.65, 0.75, 0.90, 1.00, 1.25}
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − λ Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from the PR #26 unsat diagonals (those packs are base 51e9 / scale1000 base 54e9). Holdout B=128. Freeze n=12 B=64, n=15 B=32. Seeds [0, 1, 2]. L=16 only. L-BFGS rounds 48, max_iter 8, lr=0.35. Early stop is 8 rounds without a 10⁻⁶ improvement. That is the PR #26 L=16 budget. Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. λ=0.75 is the PR #28 checkpoint at μ=−1.00, not a retrain.

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this λ). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The operating λ is the lowest frozen n=12 Walsh R among cells with median product gap ≥ 0.02. If the lowest R on the grid misses that gap, both that unconstrained winner and the safe winner are named. Absolute residuals are not compared across λ.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|. % gap<0.02 is the fraction of instances in that split under the median-gap gate.

Worst |ΔE0| Lanczos vs dense over the grid = 4.26e-14. Histogram vs ⟨ψ|H|ψ⟩ = 1.60e-14.

| λ | n | role | peak Med | prod gap Med | % gap<0.02 |
|--:|--:|:-----|---------:|-------------:|-----------:|
| 0.50 | 6 | hold | 0.050 | 0.048 | 0.0 (0/128) |
| 0.50 | 9 | hold | 0.009 | 0.040 | 0.0 (0/128) |
| 0.50 | 12 | transfer | 0.002 | 0.039 | 0.0 (0/64) |
| 0.50 | 15 | transfer | 0.000 | 0.039 | 0.0 (0/32) |
| 0.65 | 6 | hold | 0.045 | 0.038 | 0.8 (1/128) |
| 0.65 | 9 | hold | 0.008 | 0.033 | 0.0 (0/128) |
| 0.65 | 12 | transfer | 0.001 | 0.032 | 0.0 (0/64) |
| 0.65 | 15 | transfer | 0.000 | 0.031 | 0.0 (0/32) |
| 0.75 | 6 | hold | 0.042 | 0.033 | 3.9 (5/128) |
| 0.75 | 9 | hold | 0.007 | 0.029 | 2.3 (3/128) |
| 0.75 | 12 | transfer | 0.001 | 0.028 | 1.6 (1/64) |
| 0.75 | 15 | transfer | 0.000 | 0.028 | 0.0 (0/32) |
| 0.90 | 6 | hold | 0.039 | 0.027 | 10.9 (14/128) |
| 0.90 | 9 | hold | 0.007 | 0.024 | 10.9 (14/128) |
| 0.90 | 12 | transfer | 0.001 | 0.023 | 7.8 (5/64) |
| 0.90 | 15 | transfer | 0.000 | 0.023 | 6.2 (2/32) |
| 1.00 | 6 | hold | 0.037 | 0.025 | 23.4 (30/128) |
| 1.00 | 9 | hold | 0.007 | 0.021 | 35.9 (46/128) |
| 1.00 | 12 | transfer | 0.001 | 0.021 | 29.7 (19/64) |
| 1.00 | 15 | transfer | 0.000 | 0.021 | 40.6 (13/32) |
| 1.25 | 6 | hold | 0.034 | 0.019 | 60.9 (78/128) |
| 1.25 | 9 | hold | 0.006 | 0.017 | 94.5 (121/128) |
| 1.25 | 12 | transfer | 0.001 | 0.016 | 93.8 (60/64) |
| 1.25 | 15 | transfer | 0.000 | 0.016 | 100.0 (32/32) |

## Freeze table

n=6 and n=9 are the selection holdout. n=12 and n=15 are not. power_p is the scalar-γ control and does not choose λ.

| λ | head | n=6 R | n=9 R | n=12 R | n=12 mark | n=15 R | n=12 Med ε |
|--:|:-----|------:|------:|-------:|:----------|-------:|-----------:|
| 0.50 | walsh | 0.111 | 0.103 | 0.122 | BEAT | 0.149 | 0.0003 |
| 0.50 | power_p | 0.122 | 0.121 | 0.145 | BEAT | 0.180 | 0.0004 |
| 0.65 | walsh | 0.101 | 0.085 | 0.103 | BEAT | 0.127 | 0.0002 |
| 0.65 | power_p | 0.121 | 0.116 | 0.138 | BEAT | 0.165 | 0.0003 |
| 0.75 | walsh | 0.090 | 0.084 | 0.097 | BEAT | 0.119 | 0.0002 |
| 0.75 | power_p | 0.116 | 0.118 | 0.142 | BEAT | 0.170 | 0.0002 |
| 0.90 | walsh | 0.079 | 0.076 | 0.088 | BEAT | 0.106 | 0.0001 |
| 0.90 | power_p | 0.117 | 0.120 | 0.149 | BEAT | 0.177 | 0.0002 |
| 1.00 | walsh | 0.067 | 0.064 | 0.076 | BEAT | 0.096 | 0.0001 |
| 1.00 | power_p | 0.102 | 0.104 | 0.123 | BEAT | 0.147 | 0.0001 |
| 1.25 | walsh | 0.069 | 0.067 | 0.089 | BEAT | 0.112 | 0.0001 |
| 1.25 | power_p | 0.117 | 0.127 | 0.166 | BEAT | 0.206 | 0.0002 |

Per-seed n=12 Walsh R, seeds [0, 1, 2]:

- λ=0.50: 0.114, 0.221, 0.125 (median of those R = 0.125); ratio of medians = 0.122
- λ=0.65: 0.103, 0.197, 0.080 (median of those R = 0.103); ratio of medians = 0.103
- λ=0.75: 0.077, 0.207, 0.097 (median of those R = 0.097); ratio of medians = 0.097
- λ=0.90: 0.075, 0.186, 0.088 (median of those R = 0.088); ratio of medians = 0.088
- λ=1.00: 0.077, 0.182, 0.068 (median of those R = 0.077); ratio of medians = 0.076
- λ=1.25: 0.089, 0.122, 0.078 (median of those R = 0.089); ratio of medians = 0.089

## Reading

Frozen n=12 Walsh R is 0.122, 0.103, 0.097, 0.088, 0.076, 0.089 as λ runs 0.50, 0.65, 0.75, 0.90, 1.00, 1.25.
n=12 product gap by λ: 0.50 gap 0.039, 0/64 under 0.02; 0.65 gap 0.032, 0/64 under 0.02; 0.75 gap 0.028, 1/64 under 0.02; 0.90 gap 0.023, 5/64 under 0.02; 1.00 gap 0.021, 19/64 under 0.02; 1.25 gap 0.016, 60/64 under 0.02.
λ=1.25 misses the median-gap gate and does not undercut the gap-safe minimum, so the unconstrained winner and the safe winner are the same cell.
The lowest frozen n=12 Walsh R with median gap ≥ 0.02 is λ=1.00 (R=0.076).
Frozen n=15 Walsh R is lowest at λ=1.00 (0.096).
power_p n=12 R is at least the Walsh R at every λ. power_p's lowest n=12 R is λ=1.00.

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| λ | n | γ | Med ε | Walsh L=16 Med ε |
|--:|--:|--:|------:|------------------:|
| 0.50 | 6 | 0.0000 | 0.0797 | 0.0005 |
| 0.50 | 9 | 0.0000 | 0.0712 | 0.0003 |
| 0.50 | 12 | 0.0000 | 0.0665 | 0.0003 |
| 0.50 | 15 | 0.0000 | 0.0675 | 0.0004 |
| 0.65 | 6 | 0.0000 | 0.0731 | 0.0003 |
| 0.65 | 9 | 0.0000 | 0.0662 | 0.0002 |
| 0.65 | 12 | 0.0000 | 0.0620 | 0.0002 |
| 0.65 | 15 | 0.0000 | 0.0632 | 0.0002 |
| 0.75 | 6 | 0.0000 | 0.0695 | 0.0002 |
| 0.75 | 9 | 0.0000 | 0.0634 | 0.0002 |
| 0.75 | 12 | 0.0000 | 0.0592 | 0.0002 |
| 0.75 | 15 | 0.0000 | 0.0607 | 0.0002 |
| 0.90 | 6 | 0.0000 | 0.0650 | 0.0002 |
| 0.90 | 9 | 0.0000 | 0.0598 | 0.0001 |
| 0.90 | 12 | 0.0000 | 0.0557 | 0.0001 |
| 0.90 | 15 | 0.0000 | 0.0573 | 0.0001 |
| 1.00 | 6 | 0.0000 | 0.0621 | 0.0001 |
| 1.00 | 9 | 0.0000 | 0.0576 | 0.0001 |
| 1.00 | 12 | 0.0000 | 0.0539 | 0.0001 |
| 1.00 | 15 | 0.0000 | 0.0553 | 0.0001 |
| 1.25 | 6 | 0.0000 | 0.0562 | 0.0001 |
| 1.25 | 9 | 0.0000 | 0.0524 | 0.0001 |
| 1.25 | 12 | 0.0000 | 0.0495 | 0.0001 |
| 1.25 | 15 | 0.0000 | 0.0508 | 0.0001 |

## Rule

Operating λ is the grid point with the lowest frozen n=12 L=16 Walsh ratio among cells whose median product gap is at least 0.02. The other gates stay: R≤0.98, at least two seeds beat their own Fourier, n=6/9/12 peak < 0.5, and n=15 R stays below 1.10 with a non-bitstring ground state. If the lowest ratio on the whole grid misses the gap gate, the report names that unconstrained winner and the safe winner. The safe winner is the operating point when it clears the gates.

PASS if that safe cell clears the gates, including when it is still λ=0.75. needs_edit if the grid or the PR #28 anchor is incomplete, no cell keeps median gap ≥ 0.02, or the gap-safe minimum fails a transfer gate. KILL if that cell's ground state (n=6, 9, or 12) is a bitstring, Walsh there loses to Fourier, or the L=16 advantage reverses. power_p does not gate the call.

Optimizer time summed over new fits 1427.0s (λ=0.75 resumed from PR #28, wall 287.2s). Branch `cursor/unsat-walsh-lambda-mu1-01fb`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

