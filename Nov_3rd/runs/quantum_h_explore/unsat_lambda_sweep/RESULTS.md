# RESULTS — unsat Walsh λ sweep at μ=−0.50, L=16 freeze n=12

**Verdict: PASS**

λ=0.75 remains the operating point. Frozen n=12 L=16 Walsh R=0.150 matches the PR #26 checkpoint and no other quantum λ on the grid drops it by 0.02. Lowest other quantum n=12 R is λ=0.65 at 0.161 (drop -0.011 against 0.150, bar 0.02).

## Setup

Same gate as PR #26. The intra-cavity XX term is scored from the X histogram. It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. n=21 was not loaded.

```text
H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
μ = −0.50
λ ∈ {0.50, 0.65, 0.75, 0.90, 1.00}
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − λ Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from the PR #26 unsat diagonals (those packs are base 51e9 / scale1000 base 54e9). Holdout B=128. Freeze n=12 B=64, n=15 B=32. Seeds [0, 1, 2]. L=16 only. L-BFGS rounds 48, max_iter 8, lr=0.35. Early stop is 8 rounds without a 10⁻⁶ improvement. That is the PR #26 L=16 budget. Fourier stores 2L Zhou coefficients. power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. λ=0.75 is the PR #26 checkpoint, not a retrain.

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this λ). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. A λ beats 0.75 clearly when both the ratio of medians and the median per-seed ratio are at least 0.02 below the in-run λ=0.75 ratio of medians. Absolute residuals are not compared across λ.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|.

Worst |ΔE0| Lanczos vs dense over the grid = 1.60e-14. Histogram vs ⟨ψ|H|ψ⟩ = 7.99e-15.

| λ | n | role | peak Med | prod gap Med |
|--:|--:|:-----|---------:|-------------:|
| 0.50 | 6 | hold | 0.088 | 0.130 |
| 0.50 | 9 | hold | 0.023 | 0.109 |
| 0.50 | 12 | transfer | 0.006 | 0.111 |
| 0.50 | 15 | transfer | 0.002 | 0.106 |
| 0.65 | 6 | hold | 0.072 | 0.091 |
| 0.65 | 9 | hold | 0.017 | 0.077 |
| 0.65 | 12 | transfer | 0.004 | 0.079 |
| 0.65 | 15 | transfer | 0.001 | 0.076 |
| 0.75 | 6 | hold | 0.065 | 0.074 |
| 0.75 | 9 | hold | 0.014 | 0.063 |
| 0.75 | 12 | transfer | 0.003 | 0.065 |
| 0.75 | 15 | transfer | 0.001 | 0.063 |
| 0.90 | 6 | hold | 0.057 | 0.056 |
| 0.90 | 9 | hold | 0.012 | 0.048 |
| 0.90 | 12 | transfer | 0.003 | 0.050 |
| 0.90 | 15 | transfer | 0.001 | 0.049 |
| 1.00 | 6 | hold | 0.053 | 0.047 |
| 1.00 | 9 | hold | 0.011 | 0.042 |
| 1.00 | 12 | transfer | 0.002 | 0.043 |
| 1.00 | 15 | transfer | 0.001 | 0.042 |

## Freeze table

n=6 and n=9 are the selection holdout. n=12 and n=15 are not. power_p is the scalar-γ control and does not choose λ.

| λ | head | n=6 R | n=9 R | n=12 R | n=12 mark | n=15 R | n=12 Med ε |
|--:|:-----|------:|------:|-------:|:----------|-------:|-----------:|
| 0.50 | walsh | 0.129 | 0.137 | 0.168 | BEAT | 0.210 | 0.0005 |
| 0.50 | power_p | 0.163 | 0.174 | 0.209 | BEAT | 0.254 | 0.0006 |
| 0.65 | walsh | 0.117 | 0.124 | 0.161 | BEAT | 0.199 | 0.0003 |
| 0.65 | power_p | 0.149 | 0.154 | 0.196 | BEAT | 0.235 | 0.0004 |
| 0.75 | walsh | 0.114 | 0.117 | 0.150 | BEAT | 0.176 | 0.0002 |
| 0.75 | power_p | 0.143 | 0.151 | 0.185 | BEAT | 0.217 | 0.0003 |
| 0.90 | walsh | 0.143 | 0.143 | 0.178 | BEAT | 0.199 | 0.0002 |
| 0.90 | power_p | 0.193 | 0.226 | 0.276 | BEAT | 0.306 | 0.0004 |
| 1.00 | walsh | 0.130 | 0.132 | 0.171 | BEAT | 0.194 | 0.0002 |
| 1.00 | power_p | 0.196 | 0.230 | 0.278 | BEAT | 0.305 | 0.0003 |

Per-seed n=12 Walsh R, seeds [0, 1, 2]:

- λ=0.50: 0.168, 0.360, 0.158 (median of those R = 0.168); ratio of medians = 0.168
- λ=0.65: 0.155, 0.323, 0.161 (median of those R = 0.161); ratio of medians = 0.161
- λ=0.75: 0.154, 0.257, 0.131 (median of those R = 0.154); ratio of medians = 0.150
- λ=0.90: 0.166, 0.232, 0.178 (median of those R = 0.178); ratio of medians = 0.178
- λ=1.00: 0.171, 0.260, 0.160 (median of those R = 0.171); ratio of medians = 0.171

## Reading

Frozen n=12 Walsh R is 0.168, 0.161, 0.150, 0.178, 0.171 as λ runs 0.50, 0.65, 0.75, 0.90, 1.00.
No grid point clears the 0.02 bar against λ=0.75. The operating point stays 0.75 (n=12 R=0.150).
λ=0.75 is the lowest frozen n=12 Walsh R on this grid, and both neighbors are higher.
Frozen n=15 Walsh R is lowest at λ=0.75 (0.176).
power_p n=12 R is at least the Walsh R at every λ. power_p's lowest n=12 R is λ=0.75.

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| λ | n | γ | Med ε | Walsh L=16 Med ε |
|--:|--:|--:|------:|------------------:|
| 0.50 | 6 | 0.0000 | 0.1242 | 0.0006 |
| 0.50 | 9 | 0.0000 | 0.1122 | 0.0005 |
| 0.50 | 12 | 0.0000 | 0.1032 | 0.0005 |
| 0.50 | 15 | 0.0000 | 0.1084 | 0.0006 |
| 0.65 | 6 | 0.0000 | 0.1104 | 0.0004 |
| 0.65 | 9 | 0.0000 | 0.1005 | 0.0003 |
| 0.65 | 12 | 0.0000 | 0.0927 | 0.0003 |
| 0.65 | 15 | 0.0000 | 0.0981 | 0.0004 |
| 0.75 | 6 | 0.0000 | 0.1030 | 0.0003 |
| 0.75 | 9 | 0.0000 | 0.0945 | 0.0002 |
| 0.75 | 12 | 0.0000 | 0.0870 | 0.0002 |
| 0.75 | 15 | 0.0000 | 0.0924 | 0.0003 |
| 0.90 | 6 | 0.0000 | 0.0939 | 0.0003 |
| 0.90 | 9 | 0.0000 | 0.0865 | 0.0002 |
| 0.90 | 12 | 0.0000 | 0.0798 | 0.0002 |
| 0.90 | 15 | 0.0000 | 0.0846 | 0.0003 |
| 1.00 | 6 | 0.0000 | 0.0888 | 0.0002 |
| 1.00 | 9 | 0.0000 | 0.0818 | 0.0002 |
| 1.00 | 12 | 0.0000 | 0.0757 | 0.0002 |
| 1.00 | 15 | 0.0000 | 0.0800 | 0.0002 |

## Rule

Operating λ is the quantum grid point with the lowest frozen n=12 L=16 Walsh ratio, provided that ratio is at least 0.02 below the in-run λ=0.75 checkpoint on both the ratio of medians and the median per-seed ratio, R≤0.98, at least two seeds beat their own Fourier, n=12 peak < 0.5 and product gap ≥ 0.02, and n=15 R stays below 1.10 with a non-bitstring ground state. Otherwise λ=0.75 stays.

PASS if that switch happens, or if the sweep shows 0.75 is still optimal. needs_edit if the grid or the PR #26 anchor is incomplete, or a clearly lower λ fails a transfer gate. KILL if the operating point's ground state (n=6, 9, or 12) is a bitstring, Walsh there loses to Fourier, or the L=16 advantage reverses. power_p does not gate the call.

Optimizer time summed over new fits 985.5s (λ=0.75 resumed, wall 274.2s). Branch `cursor/unsat-walsh-lambda-sweep-eec8`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

