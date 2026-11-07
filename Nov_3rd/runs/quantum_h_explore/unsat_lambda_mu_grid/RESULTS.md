# RESULTS — 2D safe grid λ×μ on unsat Walsh, L=16 freeze n=12

**Verdict: PASS**

Operate at λ=0.75, μ=-0.50. That is the SAFE champ: frozen n=12 L=16 Walsh R=0.150 (seed-median R=0.154) is the lowest ratio whose median product gap is at least 0.05 (0.065) and whose instance fraction at gap ≥ 0.02 is at least 95% (64/64). The EDGE champ is λ=0.75, μ=-0.90 at R=0.092 (median gap 0.032, 64/64 instances ≥ 0.02). The EDGE champ misses the SAFE bar: median product gap 0.032 is under 0.05.

## Setup

Same gate as PR #26. The intra-cavity XX term is scored from the X histogram. It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. n=21 was not loaded.

```text
H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
μ ∈ {−0.90, −0.75, −0.60, −0.50}
λ ∈ {0.75, 0.90, 1.00}
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − λ Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from the PR #26 unsat diagonals (those packs are base 51e9 / scale1000 base 54e9). Holdout B=128. Freeze n=12 B=64, n=15 B=32. Seeds [0, 1, 2]. L=16 only. L-BFGS rounds 48, max_iter 8, lr=0.35. Early stop is 8 rounds without a 10⁻⁶ improvement. That is the PR #26 L=16 budget. Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. λ=0.75 reuses the PR #31 checkpoints at these four μ values. λ=0.90 and λ=1.00 at μ=−0.50 reuse the PR #29 checkpoints. The other six cells are new fits on the same diagonals.

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this λ and μ). Lower R is the improvement direction from PR #26. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The SAFE champ is the argmin of frozen n=12 Walsh R among cells with median product gap ≥ 0.05 and at least 95% of the 64 n=12 instances at gap ≥ 0.02. The EDGE champ is the argmin of that R among cells with at least 95% of those instances at gap ≥ 0.02. The median-gap bar does not apply to EDGE. The operating point is SAFE. A tie on R prefers the thicker median gap, then the higher instance fraction. power_p does not choose the cell. n=15 does not choose the cell.

## Recommendation

SAFE champ: λ=0.75, μ=-0.50, n=12 Walsh R=0.150.
EDGE champ: λ=0.75, μ=-0.90, n=12 Walsh R=0.092.
Operating point: λ=0.75, μ=-0.50.

## Grid

One table. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. The champ column marks SAFE, EDGE, or both, on the Walsh row. Gap columns are the frozen n=12 product gap and do not depend on the head.

Worst |ΔE0| Lanczos vs dense over the grid = 5.51e-14. Histogram vs ⟨ψ|H|ψ⟩ = 1.07e-14.

| λ | μ | champ | head | n=12 R | seed-med R | gap med | ≥0.02 | min gap | peak | n=6 R | n=9 R | n=15 R |
|--:|--:|:------|:-----|-------:|-----------:|--------:|------:|--------:|-----:|------:|------:|-------:|
| 0.75 | -0.90 | EDGE | walsh | 0.092 | 0.092 | 0.032 | 64/64 | 0.022 | 0.002 | 0.084 | 0.074 | 0.117 |
| 0.75 | -0.90 |  | power_p | 0.132 | 0.132 |  |  |  |  | 0.109 | 0.111 | 0.159 |
| 0.90 | -0.90 |  | walsh | 0.096 | 0.096 | 0.027 | 63/64 | 0.018 | 0.001 | 0.079 | 0.078 | 0.119 |
| 0.90 | -0.90 |  | power_p | 0.139 | 0.139 |  |  |  |  | 0.106 | 0.114 | 0.165 |
| 1.00 | -0.90 |  | walsh | 0.087 | 0.087 | 0.024 | 60/64 | 0.017 | 0.001 | 0.068 | 0.069 | 0.112 |
| 1.00 | -0.90 |  | power_p | 0.135 | 0.136 |  |  |  |  | 0.100 | 0.109 | 0.161 |
| 0.75 | -0.75 |  | walsh | 0.124 | 0.124 | 0.041 | 64/64 | 0.027 | 0.002 | 0.109 | 0.097 | 0.136 |
| 0.75 | -0.75 |  | power_p | 0.173 | 0.173 |  |  |  |  | 0.148 | 0.143 | 0.185 |
| 0.90 | -0.75 |  | walsh | 0.110 | 0.112 | 0.033 | 64/64 | 0.022 | 0.002 | 0.094 | 0.084 | 0.124 |
| 0.90 | -0.75 |  | power_p | 0.167 | 0.169 |  |  |  |  | 0.135 | 0.136 | 0.185 |
| 1.00 | -0.75 |  | walsh | 0.106 | 0.106 | 0.029 | 63/64 | 0.020 | 0.002 | 0.089 | 0.081 | 0.119 |
| 1.00 | -0.75 |  | power_p | 0.170 | 0.170 |  |  |  |  | 0.135 | 0.142 | 0.191 |
| 0.75 | -0.60 |  | walsh | 0.155 | 0.155 | 0.053 | 64/64 | 0.035 | 0.003 | 0.115 | 0.113 | 0.177 |
| 0.75 | -0.60 |  | power_p | 0.213 | 0.213 |  |  |  |  | 0.147 | 0.162 | 0.246 |
| 0.90 | -0.60 |  | walsh | 0.115 | 0.115 | 0.042 | 64/64 | 0.028 | 0.002 | 0.095 | 0.089 | 0.133 |
| 0.90 | -0.60 |  | power_p | 0.168 | 0.168 |  |  |  |  | 0.132 | 0.134 | 0.190 |
| 1.00 | -0.60 |  | walsh | 0.142 | 0.143 | 0.036 | 64/64 | 0.024 | 0.002 | 0.103 | 0.103 | 0.164 |
| 1.00 | -0.60 |  | power_p | 0.206 | 0.208 |  |  |  |  | 0.140 | 0.163 | 0.233 |
| 0.75 | -0.50 | SAFE | walsh | 0.150 | 0.154 | 0.065 | 64/64 | 0.041 | 0.003 | 0.114 | 0.117 | 0.176 |
| 0.75 | -0.50 |  | power_p | 0.185 | 0.191 |  |  |  |  | 0.143 | 0.151 | 0.217 |
| 0.90 | -0.50 |  | walsh | 0.178 | 0.178 | 0.050 | 64/64 | 0.033 | 0.003 | 0.143 | 0.143 | 0.199 |
| 0.90 | -0.50 |  | power_p | 0.276 | 0.276 |  |  |  |  | 0.193 | 0.226 | 0.306 |
| 1.00 | -0.50 |  | walsh | 0.171 | 0.171 | 0.043 | 64/64 | 0.028 | 0.002 | 0.130 | 0.132 | 0.194 |
| 1.00 | -0.50 |  | power_p | 0.278 | 0.287 |  |  |  |  | 0.196 | 0.230 | 0.305 |

Per-seed n=12 Walsh R, seeds [0, 1, 2]:

- λ=0.75 μ=-0.90: 0.090, 0.220, 0.092 (median of those R = 0.092); ratio of medians = 0.092
- λ=0.90 μ=-0.90: 0.091, 0.185, 0.096 (median of those R = 0.096); ratio of medians = 0.096
- λ=1.00 μ=-0.90: 0.077, 0.230, 0.087 (median of those R = 0.087); ratio of medians = 0.087
- λ=0.75 μ=-0.75: 0.115, 0.234, 0.124 (median of those R = 0.124); ratio of medians = 0.124
- λ=0.90 μ=-0.75: 0.108, 0.246, 0.112 (median of those R = 0.112); ratio of medians = 0.110
- λ=1.00 μ=-0.75: 0.102, 0.286, 0.106 (median of those R = 0.106); ratio of medians = 0.106
- λ=0.75 μ=-0.60: 0.137, 0.203, 0.155 (median of those R = 0.155); ratio of medians = 0.155
- λ=0.90 μ=-0.60: 0.112, 0.197, 0.115 (median of those R = 0.115); ratio of medians = 0.115
- λ=1.00 μ=-0.60: 0.143, 0.230, 0.137 (median of those R = 0.143); ratio of medians = 0.142
- λ=0.75 μ=-0.50: 0.154, 0.257, 0.131 (median of those R = 0.154); ratio of medians = 0.150
- λ=0.90 μ=-0.50: 0.166, 0.232, 0.178 (median of those R = 0.178); ratio of medians = 0.178
- λ=1.00 μ=-0.50: 0.171, 0.260, 0.160 (median of those R = 0.171); ratio of medians = 0.171

## Reading

Frozen n=12 Walsh R by (λ, μ): (0.75,-0.90):0.092, (0.90,-0.90):0.096, (1.00,-0.90):0.087, (0.75,-0.75):0.124, (0.90,-0.75):0.110, (1.00,-0.75):0.106, (0.75,-0.60):0.155, (0.90,-0.60):0.115, (1.00,-0.60):0.142, (0.75,-0.50):0.150, (0.90,-0.50):0.178, (1.00,-0.50):0.171.
Unconstrained argmin on the ratio of medians is λ=1.00, μ=-0.90 at 0.087.
The 95% instance bar removes 1 grid point. The median-gap bar removes 9 grid points.
Clears both SAFE bars: λ=0.75 μ=-0.60 R=0.155, λ=0.75 μ=-0.50 R=0.150, λ=0.90 μ=-0.50 R=0.178.
Clears the EDGE instance bar: λ=0.75 μ=-0.90 R=0.092 gap=0.032, λ=0.90 μ=-0.90 R=0.096 gap=0.027, λ=0.75 μ=-0.75 R=0.124 gap=0.041, λ=0.90 μ=-0.75 R=0.110 gap=0.033, λ=1.00 μ=-0.75 R=0.106 gap=0.029, λ=0.75 μ=-0.60 R=0.155 gap=0.053, λ=0.90 μ=-0.60 R=0.115 gap=0.042, λ=1.00 μ=-0.60 R=0.142 gap=0.036, λ=0.75 μ=-0.50 R=0.150 gap=0.065, λ=0.90 μ=-0.50 R=0.178 gap=0.050, λ=1.00 μ=-0.50 R=0.171 gap=0.043.
power_p stays above Walsh at every cell.
power_p's lowest n=12 R is λ=0.75, μ=-0.90 at 0.132.
n=12 product gap: (0.75,-0.90) med 0.032 (64/64 ≥ 0.02, min 0.022); (0.90,-0.90) med 0.027 (63/64 ≥ 0.02, min 0.018); (1.00,-0.90) med 0.024 (60/64 ≥ 0.02, min 0.017); (0.75,-0.75) med 0.041 (64/64 ≥ 0.02, min 0.027); (0.90,-0.75) med 0.033 (64/64 ≥ 0.02, min 0.022); (1.00,-0.75) med 0.029 (63/64 ≥ 0.02, min 0.020); (0.75,-0.60) med 0.053 (64/64 ≥ 0.02, min 0.035); (0.90,-0.60) med 0.042 (64/64 ≥ 0.02, min 0.028); (1.00,-0.60) med 0.036 (64/64 ≥ 0.02, min 0.024); (0.75,-0.50) med 0.065 (64/64 ≥ 0.02, min 0.041); (0.90,-0.50) med 0.050 (64/64 ≥ 0.02, min 0.033); (1.00,-0.50) med 0.043 (64/64 ≥ 0.02, min 0.028).

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| λ | μ | n | γ | Med ε | Walsh L=16 Med ε |
|--:|--:|--:|--:|------:|------------------:|
| 0.75 | -0.90 | 6 | 0.0000 | 0.0745 | 0.0002 |
| 0.75 | -0.90 | 9 | 0.0000 | 0.0677 | 0.0001 |
| 0.75 | -0.90 | 12 | 0.0000 | 0.0631 | 0.0002 |
| 0.75 | -0.90 | 15 | 0.0000 | 0.0651 | 0.0002 |
| 0.90 | -0.90 | 6 | 0.0000 | 0.0691 | 0.0002 |
| 0.90 | -0.90 | 9 | 0.0000 | 0.0636 | 0.0001 |
| 0.90 | -0.90 | 12 | 0.0000 | 0.0592 | 0.0001 |
| 0.90 | -0.90 | 15 | 0.0000 | 0.0613 | 0.0002 |
| 1.00 | -0.90 | 6 | 0.0000 | 0.0661 | 0.0001 |
| 1.00 | -0.90 | 9 | 0.0000 | 0.0611 | 0.0001 |
| 1.00 | -0.90 | 12 | 0.0000 | 0.0572 | 0.0001 |
| 1.00 | -0.90 | 15 | 0.0000 | 0.0590 | 0.0001 |
| 0.75 | -0.75 | 6 | 0.0000 | 0.0825 | 0.0003 |
| 0.75 | -0.75 | 9 | 0.0000 | 0.0757 | 0.0002 |
| 0.75 | -0.75 | 12 | 0.0000 | 0.0702 | 0.0002 |
| 0.75 | -0.75 | 15 | 0.0000 | 0.0733 | 0.0002 |
| 0.90 | -0.75 | 6 | 0.0000 | 0.0763 | 0.0002 |
| 0.90 | -0.75 | 9 | 0.0000 | 0.0707 | 0.0001 |
| 0.90 | -0.75 | 12 | 0.0000 | 0.0653 | 0.0001 |
| 0.90 | -0.75 | 15 | 0.0000 | 0.0683 | 0.0002 |
| 1.00 | -0.75 | 6 | 0.0000 | 0.0730 | 0.0002 |
| 1.00 | -0.75 | 9 | 0.0000 | 0.0675 | 0.0001 |
| 1.00 | -0.75 | 12 | 0.0000 | 0.0629 | 0.0001 |
| 1.00 | -0.75 | 15 | 0.0000 | 0.0653 | 0.0001 |
| 0.75 | -0.60 | 6 | 0.0000 | 0.0935 | 0.0003 |
| 0.75 | -0.60 | 9 | 0.0000 | 0.0859 | 0.0002 |
| 0.75 | -0.60 | 12 | 0.0000 | 0.0792 | 0.0003 |
| 0.75 | -0.60 | 15 | 0.0000 | 0.0833 | 0.0003 |
| 0.90 | -0.60 | 6 | 0.0000 | 0.0856 | 0.0002 |
| 0.90 | -0.60 | 9 | 0.0000 | 0.0793 | 0.0001 |
| 0.90 | -0.60 | 12 | 0.0000 | 0.0730 | 0.0001 |
| 0.90 | -0.60 | 15 | 0.0000 | 0.0773 | 0.0002 |
| 1.00 | -0.60 | 6 | 0.0000 | 0.0813 | 0.0002 |
| 1.00 | -0.60 | 9 | 0.0000 | 0.0753 | 0.0001 |
| 1.00 | -0.60 | 12 | 0.0000 | 0.0699 | 0.0002 |
| 1.00 | -0.60 | 15 | 0.0000 | 0.0737 | 0.0002 |
| 0.75 | -0.50 | 6 | 0.0000 | 0.1030 | 0.0003 |
| 0.75 | -0.50 | 9 | 0.0000 | 0.0945 | 0.0002 |
| 0.75 | -0.50 | 12 | 0.0000 | 0.0870 | 0.0002 |
| 0.75 | -0.50 | 15 | 0.0000 | 0.0924 | 0.0003 |
| 0.90 | -0.50 | 6 | 0.0000 | 0.0939 | 0.0003 |
| 0.90 | -0.50 | 9 | 0.0000 | 0.0865 | 0.0002 |
| 0.90 | -0.50 | 12 | 0.0000 | 0.0798 | 0.0002 |
| 0.90 | -0.50 | 15 | 0.0000 | 0.0846 | 0.0003 |
| 1.00 | -0.50 | 6 | 0.0000 | 0.0888 | 0.0002 |
| 1.00 | -0.50 | 9 | 0.0000 | 0.0818 | 0.0002 |
| 1.00 | -0.50 | 12 | 0.0000 | 0.0757 | 0.0002 |
| 1.00 | -0.50 | 15 | 0.0000 | 0.0800 | 0.0002 |

## Rule

Operating point is the SAFE champ: lowest frozen n=12 L=16 Walsh ratio among cells with median product gap ≥ 0.05, at least 95% of the B=64 n=12 instances at gap ≥ 0.02, and ground-state peak < 0.5 on n=6, n=9, and n=12. EDGE is the lowest ratio among cells that clear the 95% instance bar and the same peak gate. EDGE does not replace SAFE when the median gap is under 0.05. A tie prefers the thicker median gap. power_p does not choose the cell. n=15 is reported and does not choose the cell. n=21 is not loaded.

PASS when the full grid names both champs and SAFE still beats Fourier (R≤0.98, at least two seeds beat their own Fourier, n=15 R stays under 1.10, no reversal). needs_edit when a grid point, a resumed checkpoint, or a gap tail is missing, when no cell clears a champ's bar, or when SAFE misses a transfer gate. KILL when the SAFE champ's Walsh ratio loses to Fourier or the L=16 advantage reverses.

Optimizer time summed over new fits 1674.9s (resumed checkpoints 1547.3s, wall 1979.1s). Branch `cursor/unsat-walsh-lam-mu-grid-1cdb`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

