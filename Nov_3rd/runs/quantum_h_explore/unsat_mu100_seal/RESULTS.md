# RESULTS — Round 7 seal, unsat λ=0.75 + μ=−1.00 Walsh freeze

**Verdict: PASS**

frozen n=12 L=16 Walsh R=0.096 (instance 95% CI 0.091–0.102) and n=15 R=0.106 (CI 0.100–0.117) both stay BEAT. 5/5 and 5/5 seeds beat. Paired majorities 5/5 and 5/5. Drop vs μ=0 unsat Walsh 0.844. Instance product-gap tails: n=12 frac gap<0.02 = 0.008 (1/128), frac gap<0.05 = 1.000; n=15 frac gap<0.02 = 0.000 (0/64), frac gap<0.05 = 1.000. At least 95% of each sealed size has gap ≥ 0.02. n=18 probe L=16 Walsh R=0.120 Med ε=0.0002 on B=16 (peak 0.000, product gap 0.026); exploratory, not gated. PR #28 point estimates were n=12 R=0.097 and n=15 R=0.119.

## Setup

Same gate as the PR #27 seal: BinH U_c^X, Hamming-shell SNAP. The Hamiltonian is the PR #28 edge, μ=−1.00. Intra-cavity XX is scored from the X histogram. Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, use those frozen angles. n=21 was not loaded. PR #27 sealed μ=−0.50 at n=12 L=16 Walsh R=0.147 with product gap 0.064. PR #28 moved the same H to μ=−1.00 (n=12 R=0.097 on B=64, median product gap 0.028, 1/64 instances under 0.02). This run keeps that edge and asks the larger sample to clear a 95% instance gap.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −1.00
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from `beat_qaoa_12h/train_packs` (base 51e9). Selection holdout B=256 per n (PR #28 used 128). Sealed transfer B=128 at n=12 and B=64 at n=15 (PR #28 used 64 and 32). Seeds [0, 1, 2, 3, 4]. L=[2, 16]. L-BFGS rounds 18/48, max_iter 8/8, lr=0.35. That is the PR #25 and PR #27 optimizer budget on a larger selection holdout. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. Early stop is 8 rounds without a 10⁻⁶ improvement.

PR #28 point estimates, 3 seeds: n=12 L=16 Walsh R=0.097 on B=64, n=15 L=16 Walsh R=0.119 on B=32. μ=0 comparison remains the published PR #19 ratio of medians (n=12 L=16 Walsh R=0.940), not a retrain.

n=18 probe B=16. One Lanczos-plus-product ground state took 3.8s, under the 90s feasibility limit. The probe is reported and is not part of PASS.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|.

Worst |ΔE0| Lanczos vs dense = 3.20e-14. Histogram vs ⟨ψ|H|ψ⟩ = 8.88e-15. μ=0 histogram vs Pauli field = 3.55e-15.

| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |
|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|
| 6 | train | 128 | 0.041 | 0.120 | 0.034 | 0.071 | 1.196 | 1.5e-14 |
| 6 | hold | 256 | 0.042 | 0.118 | 0.034 | 0.069 | 1.194 | 6.9e-15 |
| 9 | train | 128 | 0.008 | 0.133 | 0.030 | 0.063 | 1.195 | 3.9e-14 |
| 9 | hold | 256 | 0.008 | 0.129 | 0.029 | 0.064 | 1.196 | 3.6e-14 |
| 12 | transfer | 128 | 0.001 | 0.134 | 0.028 | 0.060 | 1.191 | 6.1e-14 |
| 15 | transfer | 64 | 0.000 | 0.134 | 0.027 | 0.061 | 1.190 | 7.7e-13 |
| 18 | probe | 16 | 0.000 | 0.130 | 0.026 | 0.061 | 1.190 | 6.5e-12 |

Instance product-gap tails. A count is the number of formulas with (E_prod − E_0) / |E_0| strictly below the bar. The Round 7 seal needs at least 95% of the n=12 instances and at least 95% of the n=15 instances at or above 0.02. The fraction below 0.05 is reported and does not gate the call. n=6, n=9, and the n=18 probe are the same count and are not part of that gate.

| n | role | B | n < 0.02 | frac < 0.02 | n < 0.05 | frac < 0.05 | min gap |
|--:|:-----|--:|---------:|------------:|---------:|------------:|--------:|
| 6 | train | 128 | 0 | 0.000 | 125 | 0.977 | 0.0201 |
| 6 | hold | 256 | 5 | 0.020 | 247 | 0.965 | 0.0157 |
| 9 | train | 128 | 0 | 0.000 | 128 | 1.000 | 0.0218 |
| 9 | hold | 256 | 4 | 0.016 | 256 | 1.000 | 0.0180 |
| 12 | transfer | 128 | 1 | 0.008 | 128 | 1.000 | 0.0192 |
| 15 | transfer | 64 | 0 | 0.000 | 64 | 1.000 | 0.0233 |
| 18 | probe | 16 | 0 | 0.000 | 16 | 1.000 | 0.0238 |

Train n=6 C_n Walsh mass in degrees 1–4 is 0.120 (degree 3 is 0.031, degree ≥5 is 4.690e-34).

## Transfer table

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The 95% CI resamples instances inside each seed and keeps every seed (4000 draws, seed 5136). Paired W/T/L counts instances with ε_method < ε_Fourier, exact ties, and losses, summed over seeds. μ=0 R is the PR #19 unsat seal. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. n=18 is the probe.

| method | L | n | role | B | Med ε | seed std | R | 95% CI | paired W/T/L | mark | μ=0 R | drop |
|:-------|--:|--:|:-----|--:|------:|---------:|--:|:-------|:-------------|:-----|------:|-----:|
| fourier_qaoa | 2 | 6 | hold | 256 | 0.0155 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 9 | hold | 256 | 0.0143 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 12 | transfer | 128 | 0.0130 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 15 | transfer | 64 | 0.0143 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 18 | probe | 16 | 0.0144 | 0.0001 | — | — | — |  | — | — |
| power_p | 2 | 6 | hold | 256 | 0.0097 | 0.0005 | 0.630 | 0.610–0.661 | 1280/0/0 | BEAT | 0.915 | +0.285 |
| power_p | 2 | 9 | hold | 256 | 0.0116 | 0.0001 | 0.811 | 0.793–0.823 | 1273/0/7 | BEAT | 0.961 | +0.150 |
| power_p | 2 | 12 | transfer | 128 | 0.0128 | 0.0002 | 0.986 | 0.959–1.007 | 411/0/229 | TIE | 1.000 | +0.014 |
| power_p | 2 | 15 | transfer | 64 | 0.0147 | 0.0002 | 1.027 | 1.016–1.068 | 47/0/273 | UNDER | 0.994 | -0.033 |
| power_p | 2 | 18 | probe | 16 | 0.0158 | 0.0002 | 1.103 | 1.066–1.167 | 0/0/80 | UNDER | — | — |
| walsh | 2 | 6 | hold | 256 | 0.0034 | 0.0001 | 0.221 | 0.216–0.229 | 1280/0/0 | BEAT | 0.670 | +0.449 |
| walsh | 2 | 9 | hold | 256 | 0.0035 | 0.0001 | 0.245 | 0.235–0.249 | 1280/0/0 | BEAT | 0.705 | +0.460 |
| walsh | 2 | 12 | transfer | 128 | 0.0037 | 0.0001 | 0.286 | 0.273–0.291 | 640/0/0 | BEAT | 0.761 | +0.475 |
| walsh | 2 | 15 | transfer | 64 | 0.0043 | 0.0002 | 0.301 | 0.291–0.312 | 320/0/0 | BEAT | 0.733 | +0.432 |
| walsh | 2 | 18 | probe | 16 | 0.0045 | 0.0002 | 0.314 | 0.291–0.334 | 80/0/0 | BEAT | — | — |
| fourier_qaoa | 16 | 6 | hold | 256 | 0.0029 | 0.0002 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 9 | hold | 256 | 0.0020 | 0.0002 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 12 | transfer | 128 | 0.0017 | 0.0002 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 15 | transfer | 64 | 0.0017 | 0.0002 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 18 | probe | 16 | 0.0017 | 0.0002 | — | — | — |  | — | — |
| power_p | 16 | 6 | hold | 256 | 0.0003 | 0.0002 | 0.106 | 0.095–0.115 | 1280/0/0 | BEAT | 0.978 | +0.872 |
| power_p | 16 | 9 | hold | 256 | 0.0002 | 0.0002 | 0.119 | 0.115–0.126 | 1280/0/0 | BEAT | 0.951 | +0.832 |
| power_p | 16 | 12 | transfer | 128 | 0.0002 | 0.0002 | 0.141 | 0.135–0.154 | 640/0/0 | BEAT | 0.990 | +0.849 |
| power_p | 16 | 15 | transfer | 64 | 0.0003 | 0.0002 | 0.161 | 0.151–0.170 | 320/0/0 | BEAT | 0.992 | +0.831 |
| power_p | 16 | 18 | probe | 16 | 0.0003 | 0.0002 | 0.175 | 0.163–0.202 | 80/0/0 | BEAT | — | — |
| walsh | 16 | 6 | hold | 256 | 0.0002 | 0.0002 | 0.085 | 0.076–0.091 | 1280/0/0 | BEAT | 0.925 | +0.840 |
| walsh | 16 | 9 | hold | 256 | 0.0002 | 0.0001 | 0.086 | 0.082–0.090 | 1280/0/0 | BEAT | 0.895 | +0.809 |
| walsh | 16 | 12 | transfer | 128 | 0.0002 | 0.0001 | 0.096 | 0.091–0.102 | 640/0/0 | BEAT | 0.940 | +0.844 |
| walsh | 16 | 15 | transfer | 64 | 0.0002 | 0.0001 | 0.106 | 0.100–0.117 | 320/0/0 | BEAT | 0.914 | +0.808 |
| walsh | 16 | 18 | probe | 16 | 0.0002 | 0.0001 | 0.120 | 0.109–0.132 | 80/0/0 | BEAT | — | — |

Per-seed median ε, per-seed R, and per-seed paired wins:

- `fourier_qaoa` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0155, 0.0154, 0.0155, 0.0154, 0.0155
- `fourier_qaoa` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0143, 0.0143, 0.0143, 0.0143, 0.0143
- `fourier_qaoa` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0130, 0.0130, 0.0130, 0.0130, 0.0131
- `fourier_qaoa` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0143, 0.0143, 0.0143, 0.0143, 0.0143
- `fourier_qaoa` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0144, 0.0143, 0.0144, 0.0144, 0.0145
- `power_p` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0104, 0.0095, 0.0092, 0.0104, 0.0097 R 0.676, 0.617, 0.598, 0.677, 0.630 (median of those R = 0.630); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0117, 0.0116, 0.0115, 0.0117, 0.0115 R 0.818, 0.812, 0.803, 0.816, 0.801 (median of those R = 0.812); paired wins 256/256, 249/256, 256/256, 256/256, 256/256
- `power_p` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0128, 0.0132, 0.0129, 0.0128, 0.0128 R 0.987, 1.016, 0.987, 0.982, 0.980 (median of those R = 0.987); paired wins 84/128, 67/128, 85/128, 85/128, 90/128
- `power_p` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0146, 0.0151, 0.0148, 0.0146, 0.0147 R 1.027, 1.061, 1.031, 1.021, 1.024 (median of those R = 1.027); paired wins 11/64, 5/64, 7/64, 13/64, 11/64
- `power_p` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0158, 0.0163, 0.0159, 0.0158, 0.0157 R 1.104, 1.136, 1.103, 1.099, 1.087 (median of those R = 1.103); paired wins 0/16, 0/16, 0/16, 0/16, 0/16
- `walsh` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0034, 0.0034, 0.0034, 0.0034, 0.0035 R 0.220, 0.223, 0.221, 0.220, 0.230 (median of those R = 0.221); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0033, 0.0035, 0.0035, 0.0033, 0.0035 R 0.233, 0.246, 0.245, 0.233, 0.245 (median of those R = 0.245); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0035, 0.0038, 0.0037, 0.0035, 0.0037 R 0.267, 0.289, 0.286, 0.267, 0.285 (median of those R = 0.285); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0040, 0.0043, 0.0043, 0.0040, 0.0043 R 0.280, 0.304, 0.302, 0.280, 0.300 (median of those R = 0.300); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0041, 0.0045, 0.0045, 0.0041, 0.0045 R 0.287, 0.316, 0.313, 0.287, 0.314 (median of those R = 0.313); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `fourier_qaoa` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0029, 0.0031, 0.0028, 0.0028, 0.0032
- `fourier_qaoa` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0020, 0.0022, 0.0019, 0.0020, 0.0023
- `fourier_qaoa` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0019, 0.0017, 0.0017, 0.0020
- `fourier_qaoa` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0019, 0.0017, 0.0017, 0.0021
- `fourier_qaoa` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0018, 0.0016, 0.0017, 0.0020
- `power_p` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0006, 0.0003, 0.0003, 0.0007 R 0.093, 0.200, 0.109, 0.107, 0.227 (median of those R = 0.109); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0005, 0.0002, 0.0002, 0.0006 R 0.100, 0.216, 0.121, 0.119, 0.254 (median of those R = 0.121); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0004, 0.0002, 0.0002, 0.0006 R 0.117, 0.229, 0.142, 0.141, 0.301 (median of those R = 0.142); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `power_p` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0005, 0.0003, 0.0003, 0.0007 R 0.135, 0.241, 0.161, 0.161, 0.335 (median of those R = 0.161); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `power_p` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0004, 0.0003, 0.0003, 0.0008 R 0.150, 0.244, 0.176, 0.175, 0.378 (median of those R = 0.176); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `walsh` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0005, 0.0002, 0.0002, 0.0005 R 0.076, 0.175, 0.088, 0.079, 0.159 (median of those R = 0.088); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0004, 0.0002, 0.0002, 0.0004 R 0.074, 0.194, 0.088, 0.077, 0.175 (median of those R = 0.088); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0004, 0.0002, 0.0001, 0.0004 R 0.090, 0.205, 0.098, 0.085, 0.197 (median of those R = 0.098); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0004, 0.0002, 0.0002, 0.0005 R 0.108, 0.214, 0.108, 0.092, 0.217 (median of those R = 0.108); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0004, 0.0002, 0.0002, 0.0005 R 0.122, 0.211, 0.122, 0.106, 0.234 (median of those R = 0.122); paired wins 16/16, 16/16, 16/16, 16/16, 16/16

Walsh L=16 n=12: R=0.096, instance CI 0.091–0.102, seed-resample CI 0.085–0.205, pooled paired wins 640/640 (Wilson 95% 0.994–1.000; pairs repeat instances across seeds). PR #28 point estimate 0.097.
Walsh L=16 n=15: R=0.106, instance CI 0.100–0.117, seed-resample CI 0.092–0.217, pooled paired wins 320/320 (Wilson 95% 0.988–1.000; pairs repeat instances across seeds). PR #28 point estimate 0.119.
Walsh L=16 n=18 probe: R=0.120, instance CI 0.109–0.132, pooled paired wins 80/80 (Wilson 95% 0.954–1.000). Exploratory.

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| n | role | γ | Med ε | Walsh L=16 Med ε |
|--:|:-----|--:|------:|------------------:|
| 6 | hold | 0.0000 | 0.0689 | 0.0002 |
| 9 | hold | 0.0000 | 0.0636 | 0.0002 |
| 12 | transfer | 0.0000 | 0.0605 | 0.0002 |
| 15 | transfer | 0.0000 | 0.0607 | 0.0002 |
| 18 | probe | 0.0000 | 0.0606 | 0.0002 |

## Rule

PASS if frozen L=16 Walsh at n=12 and at n=15 both have ratio-of-medians ≤ 0.98, the instance-bootstrap 95% interval lies inside that BEAT line, every seed beats its own Fourier, every seed wins a strict majority of instances, the n=12 drop versus PR #19's 0.940 is at least 0.02 on both the ratio of medians and the median per-seed ratio, and the n=12 and n=15 ground states have peak < 0.5 and median product gap ≥ 0.02. Round 7 also requires at least 95% of the n=12 instances and at least 95% of the n=15 instances to have product gap ≥ 0.02. A shortfall on that fraction is needs_edit and is a quantumness fail, even when the median gap and the Walsh ratios still look like a BEAT. The fraction of instances with product gap < 0.05 is reported and does not move the call. power_p does not gate PASS. n=18 does not gate PASS.

A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. needs_edit also covers a BEAT whose interval, seed list, or paired counts do not all sit inside the bar. KILL if a ground state at n=6, 9, or 12 is a bitstring; if n=12 loses to Fourier and is not below 0.940; or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05).

power_p control at n=12 L=16: R=0.141 (BEAT), instance CI 0.135–0.154, published μ=0 power_p R=0.990, drop +0.849.

Optimizer time summed over fits 508.6s. Branch `cursor/unsat-walsh-mu100-seal-3d9d`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

