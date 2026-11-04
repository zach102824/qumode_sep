# RESULTS — Round 8 seal, unsat λ=1.00 + μ=−1.00 Walsh freeze

**Verdict: needs_edit**

quantumness fail: n=12 product gap ≥ 0.02 on 65.6% of instances (44/128 below 0.02, frac < 0.02 = 0.344, frac < 0.05 = 1.000); need ≥ 95%; quantumness fail: n=15 product gap ≥ 0.02 on 60.9% of instances (25/64 below 0.02, frac < 0.02 = 0.391, frac < 0.05 = 1.000); need ≥ 95% n=18 probe L=16 Walsh R=0.108 Med ε=0.0001 on B=16 (peak 0.000, product gap 0.020); exploratory, not gated. PR #32 point estimates were n=12 R=0.076 and n=15 R=0.096.

## Setup

Same gate as the PR #30 seal: BinH U_c^X, Hamming-shell SNAP. The Hamiltonian is the PR #32 operating point, λ=1.00 and μ=−1.00. Intra-cavity XX is scored from the X histogram. Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, use those frozen angles. n=21 was not loaded. PR #32 froze this cell at 3 seeds (n=12 L=16 Walsh R=0.076 on B=64, median product gap 0.021, 19/64 instances under 0.02; n=15 R=0.096 on B=32, 13/32 under 0.02). PR #30 sealed λ=0.75 at this μ with n=12 R=0.096 and 127/128 instances at gap ≥ 0.02. This run keeps the PR #30 budget and asks whether λ=1.00 still BEATs and clears that 95% instance gap.

```text
H = diag(unsat) − 1.00 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −1.00
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 1.00 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from `beat_qaoa_12h/train_packs` (base 51e9). Selection holdout B=256 per n (PR #32 used 128). Sealed transfer B=128 at n=12 and B=64 at n=15 (PR #32 used 64 and 32). Seeds [0, 1, 2, 3, 4]. L=[2, 16]. L-BFGS rounds 18/48, max_iter 8/8, lr=0.35. That is the PR #30 optimizer budget on the larger selection holdout. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. Early stop is 8 rounds without a 10⁻⁶ improvement.

PR #32 point estimates, 3 seeds: n=12 L=16 Walsh R=0.076 on B=64, n=15 L=16 Walsh R=0.096 on B=32. μ=0 comparison remains the published PR #19 ratio of medians (n=12 L=16 Walsh R=0.940), not a retrain.

n=18 probe B=16. One Lanczos-plus-product ground state took 4.8s, under the 90s feasibility limit. The probe is reported and is not part of PASS.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|.

Worst |ΔE0| Lanczos vs dense = 5.15e-14. Histogram vs ⟨ψ|H|ψ⟩ = 1.24e-14. μ=0 histogram vs Pauli field = 5.33e-15.

| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |
|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|
| 6 | train | 128 | 0.037 | 0.100 | 0.026 | 0.064 | 1.439 | 1.0e-14 |
| 6 | hold | 256 | 0.037 | 0.098 | 0.025 | 0.062 | 1.437 | 1.0e-14 |
| 9 | train | 128 | 0.007 | 0.112 | 0.022 | 0.057 | 1.439 | 3.6e-14 |
| 9 | hold | 256 | 0.007 | 0.110 | 0.022 | 0.058 | 1.440 | 5.2e-14 |
| 12 | transfer | 128 | 0.001 | 0.115 | 0.021 | 0.055 | 1.435 | 5.3e-14 |
| 15 | transfer | 64 | 0.000 | 0.114 | 0.021 | 0.056 | 1.435 | 2.2e-13 |
| 18 | probe | 16 | 0.000 | 0.111 | 0.020 | 0.056 | 1.434 | 1.9e-12 |

Instance product-gap tails. A count is the number of formulas with (E_prod − E_0) / |E_0| strictly below the bar. Round 8 keeps the Round 7 instance gate and needs at least 95% of the n=12 instances and at least 95% of the n=15 instances at or above 0.02. The fraction below 0.05 is reported and does not gate the call. n=6, n=9, and the n=18 probe are the same count and are not part of that gate.

| n | role | B | n < 0.02 | frac < 0.02 | n < 0.05 | frac < 0.05 | min gap |
|--:|:-----|--:|---------:|------------:|---------:|------------:|--------:|
| 6 | train | 128 | 23 | 0.180 | 128 | 1.000 | 0.0149 |
| 6 | hold | 256 | 53 | 0.207 | 256 | 1.000 | 0.0116 |
| 9 | train | 128 | 24 | 0.188 | 128 | 1.000 | 0.0164 |
| 9 | hold | 256 | 75 | 0.293 | 256 | 1.000 | 0.0138 |
| 12 | transfer | 128 | 44 | 0.344 | 128 | 1.000 | 0.0147 |
| 15 | transfer | 64 | 25 | 0.391 | 64 | 1.000 | 0.0177 |
| 18 | probe | 16 | 8 | 0.500 | 16 | 1.000 | 0.0180 |

Train n=6 C_n Walsh mass in degrees 1–4 is 0.120 (degree 3 is 0.031, degree ≥5 is 4.690e-34).

## Transfer table

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The 95% CI resamples instances inside each seed and keeps every seed (4000 draws, seed 5136). Paired W/T/L counts instances with ε_method < ε_Fourier, exact ties, and losses, summed over seeds. μ=0 R is the PR #19 unsat seal. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. n=18 is the probe.

| method | L | n | role | B | Med ε | seed std | R | 95% CI | paired W/T/L | mark | μ=0 R | drop |
|:-------|--:|--:|:-----|--:|------:|---------:|--:|:-------|:-------------|:-----|------:|-----:|
| fourier_qaoa | 2 | 6 | hold | 256 | 0.0131 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 9 | hold | 256 | 0.0126 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 12 | transfer | 128 | 0.0114 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 15 | transfer | 64 | 0.0127 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 18 | probe | 16 | 0.0129 | 0.0001 | — | — | — |  | — | — |
| power_p | 2 | 6 | hold | 256 | 0.0080 | 0.0003 | 0.617 | 0.597–0.639 | 1280/0/0 | BEAT | 0.915 | +0.298 |
| power_p | 2 | 9 | hold | 256 | 0.0101 | 0.0000 | 0.801 | 0.791–0.815 | 1258/0/22 | BEAT | 0.961 | +0.160 |
| power_p | 2 | 12 | transfer | 128 | 0.0116 | 0.0001 | 1.013 | 0.966–1.019 | 380/0/260 | TIE | 1.000 | -0.013 |
| power_p | 2 | 15 | transfer | 64 | 0.0133 | 0.0001 | 1.044 | 1.020–1.073 | 49/0/271 | UNDER | 0.994 | -0.050 |
| power_p | 2 | 18 | probe | 16 | 0.0144 | 0.0002 | 1.115 | 1.068–1.169 | 0/0/80 | UNDER | — | — |
| walsh | 2 | 6 | hold | 256 | 0.0025 | 0.0000 | 0.195 | 0.185–0.198 | 1280/0/0 | BEAT | 0.670 | +0.475 |
| walsh | 2 | 9 | hold | 256 | 0.0028 | 0.0001 | 0.220 | 0.214–0.226 | 1280/0/0 | BEAT | 0.705 | +0.485 |
| walsh | 2 | 12 | transfer | 128 | 0.0032 | 0.0001 | 0.279 | 0.262–0.285 | 640/0/0 | BEAT | 0.761 | +0.482 |
| walsh | 2 | 15 | transfer | 64 | 0.0037 | 0.0002 | 0.291 | 0.281–0.304 | 320/0/0 | BEAT | 0.733 | +0.442 |
| walsh | 2 | 18 | probe | 16 | 0.0039 | 0.0002 | 0.306 | 0.288–0.331 | 80/0/0 | BEAT | — | — |
| fourier_qaoa | 16 | 6 | hold | 256 | 0.0019 | 0.0001 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 9 | hold | 256 | 0.0014 | 0.0001 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 12 | transfer | 128 | 0.0012 | 0.0001 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 15 | transfer | 64 | 0.0012 | 0.0001 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 18 | probe | 16 | 0.0012 | 0.0001 | — | — | — |  | — | — |
| power_p | 16 | 6 | hold | 256 | 0.0002 | 0.0002 | 0.091 | 0.085–0.099 | 1280/0/0 | BEAT | 0.978 | +0.887 |
| power_p | 16 | 9 | hold | 256 | 0.0001 | 0.0001 | 0.104 | 0.098–0.107 | 1280/0/0 | BEAT | 0.951 | +0.847 |
| power_p | 16 | 12 | transfer | 128 | 0.0001 | 0.0001 | 0.124 | 0.117–0.134 | 640/0/0 | BEAT | 0.990 | +0.866 |
| power_p | 16 | 15 | transfer | 64 | 0.0002 | 0.0001 | 0.137 | 0.129–0.147 | 320/0/0 | BEAT | 0.992 | +0.855 |
| power_p | 16 | 18 | probe | 16 | 0.0002 | 0.0001 | 0.153 | 0.140–0.171 | 80/0/0 | BEAT | — | — |
| walsh | 16 | 6 | hold | 256 | 0.0001 | 0.0001 | 0.064 | 0.060–0.072 | 1280/0/0 | BEAT | 0.925 | +0.861 |
| walsh | 16 | 9 | hold | 256 | 0.0001 | 0.0001 | 0.064 | 0.061–0.066 | 1280/0/0 | BEAT | 0.895 | +0.831 |
| walsh | 16 | 12 | transfer | 128 | 0.0001 | 0.0001 | 0.077 | 0.071–0.081 | 640/0/0 | BEAT | 0.940 | +0.863 |
| walsh | 16 | 15 | transfer | 64 | 0.0001 | 0.0001 | 0.089 | 0.084–0.096 | 320/0/0 | BEAT | 0.914 | +0.825 |
| walsh | 16 | 18 | probe | 16 | 0.0001 | 0.0001 | 0.108 | 0.087–0.111 | 80/0/0 | BEAT | — | — |

Per-seed median ε, per-seed R, and per-seed paired wins:

- `fourier_qaoa` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0131, 0.0130, 0.0131, 0.0131, 0.0130
- `fourier_qaoa` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0126, 0.0127, 0.0126, 0.0126, 0.0127
- `fourier_qaoa` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0114, 0.0115, 0.0114, 0.0114, 0.0115
- `fourier_qaoa` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0127, 0.0128, 0.0127, 0.0127, 0.0128
- `fourier_qaoa` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0129, 0.0130, 0.0129, 0.0128, 0.0129
- `power_p` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0079, 0.0085, 0.0080, 0.0080, 0.0084 R 0.605, 0.649, 0.610, 0.616, 0.648 (median of those R = 0.616); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0101, 0.0102, 0.0101, 0.0101, 0.0101 R 0.802, 0.801, 0.802, 0.797, 0.798 (median of those R = 0.801); paired wins 249/256, 252/256, 251/256, 254/256, 252/256
- `power_p` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0116, 0.0116, 0.0116, 0.0114, 0.0115 R 1.019, 1.010, 1.014, 1.002, 1.003 (median of those R = 1.010); paired wins 70/128, 76/128, 68/128, 87/128, 79/128
- `power_p` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0134, 0.0133, 0.0134, 0.0132, 0.0132 R 1.053, 1.041, 1.054, 1.037, 1.032 (median of those R = 1.041); paired wins 7/64, 11/64, 7/64, 11/64, 13/64
- `power_p` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0144, 0.0144, 0.0144, 0.0140, 0.0143 R 1.116, 1.112, 1.117, 1.093, 1.101 (median of those R = 1.112); paired wins 0/16, 0/16, 0/16, 0/16, 0/16
- `walsh` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0025, 0.0026, 0.0025, 0.0025, 0.0025 R 0.195, 0.196, 0.190, 0.195, 0.192 (median of those R = 0.195); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0028, 0.0027, 0.0027, 0.0028, 0.0028 R 0.225, 0.214, 0.212, 0.225, 0.219 (median of those R = 0.219); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0033, 0.0030, 0.0030, 0.0032, 0.0032 R 0.286, 0.261, 0.265, 0.285, 0.278 (median of those R = 0.278); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0038, 0.0035, 0.0035, 0.0038, 0.0037 R 0.300, 0.273, 0.274, 0.299, 0.290 (median of those R = 0.290); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0041, 0.0037, 0.0037, 0.0040, 0.0039 R 0.317, 0.286, 0.288, 0.316, 0.305 (median of those R = 0.305); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `fourier_qaoa` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0019, 0.0021, 0.0018, 0.0019, 0.0021
- `fourier_qaoa` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0014, 0.0015, 0.0013, 0.0013, 0.0016
- `fourier_qaoa` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0012, 0.0013, 0.0012, 0.0012, 0.0014
- `fourier_qaoa` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0012, 0.0014, 0.0012, 0.0012, 0.0015
- `fourier_qaoa` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0012, 0.0013, 0.0012, 0.0012, 0.0015
- `power_p` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0004, 0.0002, 0.0002, 0.0005 R 0.088, 0.200, 0.093, 0.094, 0.223 (median of those R = 0.094); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0003, 0.0001, 0.0001, 0.0004 R 0.101, 0.214, 0.107, 0.103, 0.251 (median of those R = 0.107); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0003, 0.0001, 0.0001, 0.0004 R 0.122, 0.225, 0.125, 0.113, 0.288 (median of those R = 0.125); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `power_p` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0003, 0.0002, 0.0001, 0.0005 R 0.137, 0.231, 0.137, 0.121, 0.312 (median of those R = 0.137); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `power_p` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0003, 0.0002, 0.0002, 0.0005 R 0.153, 0.234, 0.150, 0.129, 0.349 (median of those R = 0.153); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `walsh` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0003, 0.0001, 0.0001, 0.0003 R 0.062, 0.154, 0.065, 0.066, 0.138 (median of those R = 0.066); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0003, 0.0001, 0.0001, 0.0003 R 0.064, 0.170, 0.065, 0.064, 0.154 (median of those R = 0.065); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0002, 0.0001, 0.0001, 0.0002 R 0.077, 0.179, 0.069, 0.063, 0.171 (median of those R = 0.077); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0003, 0.0001, 0.0001, 0.0003 R 0.089, 0.188, 0.076, 0.068, 0.186 (median of those R = 0.089); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0002, 0.0001, 0.0001, 0.0003 R 0.108, 0.190, 0.089, 0.075, 0.204 (median of those R = 0.108); paired wins 16/16, 16/16, 16/16, 16/16, 16/16

Walsh L=16 n=12: R=0.077, instance CI 0.071–0.081, seed-resample CI 0.063–0.179, pooled paired wins 640/640 (Wilson 95% 0.994–1.000; pairs repeat instances across seeds). PR #32 point estimate 0.076.
Walsh L=16 n=15: R=0.089, instance CI 0.084–0.096, seed-resample CI 0.068–0.188, pooled paired wins 320/320 (Wilson 95% 0.988–1.000; pairs repeat instances across seeds). PR #32 point estimate 0.096.
Walsh L=16 n=18 probe: R=0.108, instance CI 0.087–0.111, pooled paired wins 80/80 (Wilson 95% 0.954–1.000). Exploratory.

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| n | role | γ | Med ε | Walsh L=16 Med ε |
|--:|:-----|--:|------:|------------------:|
| 6 | hold | 0.0000 | 0.0616 | 0.0001 |
| 9 | hold | 0.0000 | 0.0577 | 0.0001 |
| 12 | transfer | 0.0000 | 0.0552 | 0.0001 |
| 15 | transfer | 0.0000 | 0.0556 | 0.0001 |
| 18 | probe | 0.0000 | 0.0554 | 0.0001 |

## Rule

PASS if frozen L=16 Walsh at n=12 and at n=15 both have ratio-of-medians ≤ 0.98, the instance-bootstrap 95% interval lies inside that BEAT line, every seed beats its own Fourier, every seed wins a strict majority of instances, the n=12 drop versus PR #19's 0.940 is at least 0.02 on both the ratio of medians and the median per-seed ratio, and the n=12 and n=15 ground states have peak < 0.5 and median product gap ≥ 0.02. Round 8 keeps that gate and also requires at least 95% of the n=12 instances and at least 95% of the n=15 instances to have product gap ≥ 0.02. A shortfall on that fraction is needs_edit and is a quantumness fail, even when the median gap and the Walsh ratios still look like a BEAT. The fraction of instances with product gap < 0.05 is reported and does not move the call. power_p does not gate PASS. n=18 does not gate PASS.

A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. needs_edit also covers a BEAT whose interval, seed list, or paired counts do not all sit inside the bar. KILL if a ground state at n=6, 9, or 12 is a bitstring; if n=12 loses to Fourier and is not below 0.940; or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05).

power_p control at n=12 L=16: R=0.124 (BEAT), instance CI 0.117–0.134, published μ=0 power_p R=0.990, drop +0.866.

Optimizer time summed over fits 498.1s. Branch `cursor/unsat-walsh-lam1-seal-a846`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

