# RESULTS — Round 8 seal, unsat λ=0.75 + μ=−0.90 Walsh freeze

**Verdict: PASS**

frozen n=12 L=16 Walsh R=0.094 (instance 95% CI 0.088–0.099) and n=15 R=0.111 (CI 0.103–0.117) both stay BEAT. 5/5 and 5/5 seeds beat. Paired majorities 5/5 and 5/5. Drop vs μ=0 unsat Walsh 0.846. Instance product-gap tails: n=12 frac gap<0.02 = 0.000 (0/128), frac gap<0.05 = 1.000; n=15 frac gap<0.02 = 0.000 (0/64), frac gap<0.05 = 1.000. At least 95% of each sealed size has gap ≥ 0.02. n=18 probe L=16 Walsh R=0.127 Med ε=0.0002 on B=16 (peak 0.000, product gap 0.030); exploratory, not gated. PR #31 point estimates were n=12 R=0.092 and n=15 R=0.117.

## Setup

Same gate as the PR #30 seal: BinH U_c^X, Hamming-shell SNAP. The Hamiltonian is the PR #31 EDGE champ, μ=−0.90. Intra-cavity XX is scored from the X histogram. Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, use those frozen angles. n=21 was not loaded. PR #27 sealed μ=−0.50 at n=12 L=16 Walsh R=0.147 with product gap 0.064. PR #30 sealed μ=−1.00 at n=12 R=0.096 (1/128 instances under 0.02). PR #31 named μ=−0.90 the EDGE champ (n=12 R=0.092 on B=64, median product gap 0.032, 64/64 instances at or above 0.02). This run asks the larger sample to keep that BEAT and the 95% instance gap.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −0.90
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from `beat_qaoa_12h/train_packs` (base 51e9). Selection holdout B=256 per n (PR #31 used 128). Sealed transfer B=128 at n=12 and B=64 at n=15 (PR #31 used 64 and 32). Seeds [0, 1, 2, 3, 4]. L=[2, 16]. L-BFGS rounds 18/48, max_iter 8/8, lr=0.35. That is the PR #30 optimizer budget on a larger selection holdout. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. Early stop is 8 rounds without a 10⁻⁶ improvement.

PR #31 point estimates, 3 seeds: n=12 L=16 Walsh R=0.092 on B=64, n=15 L=16 Walsh R=0.117 on B=32. μ=0 comparison remains the published PR #19 ratio of medians (n=12 L=16 Walsh R=0.940), not a retrain.

n=18 probe B=16. One Lanczos-plus-product ground state took 3.7s, under the 90s feasibility limit. The probe is reported and is not part of PASS.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|.

Worst |ΔE0| Lanczos vs dense = 9.77e-15. Histogram vs ⟨ψ|H|ψ⟩ = 8.88e-15. μ=0 histogram vs Pauli field = 3.55e-15.

| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |
|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|
| 6 | train | 128 | 0.043 | 0.133 | 0.039 | 0.075 | 1.100 | 6.8e-15 |
| 6 | hold | 256 | 0.044 | 0.131 | 0.039 | 0.073 | 1.098 | 8.9e-15 |
| 9 | train | 128 | 0.008 | 0.149 | 0.034 | 0.067 | 1.099 | 3.2e-14 |
| 9 | hold | 256 | 0.008 | 0.144 | 0.033 | 0.068 | 1.100 | 3.9e-14 |
| 12 | transfer | 128 | 0.002 | 0.150 | 0.032 | 0.065 | 1.095 | 5.5e-14 |
| 15 | transfer | 64 | 0.000 | 0.150 | 0.032 | 0.065 | 1.094 | 1.4e-12 |
| 18 | probe | 16 | 0.000 | 0.147 | 0.030 | 0.066 | 1.094 | 8.1e-12 |

Instance product-gap tails. A count is the number of formulas with (E_prod − E_0) / |E_0| strictly below the bar. The Round 8 seal needs at least 95% of the n=12 instances and at least 95% of the n=15 instances at or above 0.02. The fraction below 0.05 is reported and does not gate the call. n=6, n=9, and the n=18 probe are the same count and are not part of that gate.

| n | role | B | n < 0.02 | frac < 0.02 | n < 0.05 | frac < 0.05 | min gap |
|--:|:-----|--:|---------:|------------:|---------:|------------:|--------:|
| 6 | train | 128 | 0 | 0.000 | 115 | 0.898 | 0.0225 |
| 6 | hold | 256 | 1 | 0.004 | 225 | 0.879 | 0.0180 |
| 9 | train | 128 | 0 | 0.000 | 128 | 1.000 | 0.0250 |
| 9 | hold | 256 | 0 | 0.000 | 255 | 0.996 | 0.0204 |
| 12 | transfer | 128 | 0 | 0.000 | 128 | 1.000 | 0.0219 |
| 15 | transfer | 64 | 0 | 0.000 | 64 | 1.000 | 0.0268 |
| 18 | probe | 16 | 0 | 0.000 | 16 | 1.000 | 0.0276 |

Train n=6 C_n Walsh mass in degrees 1–4 is 0.120 (degree 3 is 0.031, degree ≥5 is 4.690e-34).

## Transfer table

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The 95% CI resamples instances inside each seed and keeps every seed (4000 draws, seed 5136). Paired W/T/L counts instances with ε_method < ε_Fourier, exact ties, and losses, summed over seeds. μ=0 R is the PR #19 unsat seal. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. n=18 is the probe.

| method | L | n | role | B | Med ε | seed std | R | 95% CI | paired W/T/L | mark | μ=0 R | drop |
|:-------|--:|--:|:-----|--:|------:|---------:|--:|:-------|:-------------|:-----|------:|-----:|
| fourier_qaoa | 2 | 6 | hold | 256 | 0.0163 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 9 | hold | 256 | 0.0153 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 12 | transfer | 128 | 0.0141 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 15 | transfer | 64 | 0.0153 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 18 | probe | 16 | 0.0155 | 0.0000 | — | — | — |  | — | — |
| power_p | 2 | 6 | hold | 256 | 0.0115 | 0.0000 | 0.708 | 0.688–0.728 | 1280/0/0 | BEAT | 0.915 | +0.207 |
| power_p | 2 | 9 | hold | 256 | 0.0124 | 0.0000 | 0.807 | 0.795–0.819 | 1280/0/0 | BEAT | 0.961 | +0.154 |
| power_p | 2 | 12 | transfer | 128 | 0.0132 | 0.0001 | 0.932 | 0.907–0.952 | 582/0/58 | BEAT | 1.000 | +0.068 |
| power_p | 2 | 15 | transfer | 64 | 0.0150 | 0.0001 | 0.976 | 0.960–0.997 | 211/0/109 | BEAT | 0.994 | +0.018 |
| power_p | 2 | 18 | probe | 16 | 0.0160 | 0.0001 | 1.032 | 1.001–1.068 | 18/0/62 | UNDER | — | — |
| walsh | 2 | 6 | hold | 256 | 0.0040 | 0.0001 | 0.243 | 0.232–0.249 | 1280/0/0 | BEAT | 0.670 | +0.427 |
| walsh | 2 | 9 | hold | 256 | 0.0039 | 0.0001 | 0.257 | 0.248–0.263 | 1280/0/0 | BEAT | 0.705 | +0.448 |
| walsh | 2 | 12 | transfer | 128 | 0.0041 | 0.0002 | 0.291 | 0.280–0.302 | 640/0/0 | BEAT | 0.761 | +0.470 |
| walsh | 2 | 15 | transfer | 64 | 0.0048 | 0.0002 | 0.311 | 0.301–0.325 | 320/0/0 | BEAT | 0.733 | +0.422 |
| walsh | 2 | 18 | probe | 16 | 0.0050 | 0.0003 | 0.321 | 0.300–0.350 | 80/0/0 | BEAT | — | — |
| fourier_qaoa | 16 | 6 | hold | 256 | 0.0029 | 0.0003 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 9 | hold | 256 | 0.0020 | 0.0002 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 12 | transfer | 128 | 0.0017 | 0.0002 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 15 | transfer | 64 | 0.0017 | 0.0003 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 18 | probe | 16 | 0.0016 | 0.0002 | — | — | — |  | — | — |
| power_p | 16 | 6 | hold | 256 | 0.0003 | 0.0003 | 0.104 | 0.096–0.113 | 1280/0/0 | BEAT | 0.978 | +0.874 |
| power_p | 16 | 9 | hold | 256 | 0.0002 | 0.0002 | 0.116 | 0.110–0.120 | 1280/0/0 | BEAT | 0.951 | +0.835 |
| power_p | 16 | 12 | transfer | 128 | 0.0002 | 0.0002 | 0.136 | 0.129–0.147 | 640/0/0 | BEAT | 0.990 | +0.854 |
| power_p | 16 | 15 | transfer | 64 | 0.0003 | 0.0003 | 0.152 | 0.143–0.162 | 320/0/0 | BEAT | 0.992 | +0.840 |
| power_p | 16 | 18 | probe | 16 | 0.0003 | 0.0003 | 0.170 | 0.154–0.193 | 80/0/0 | BEAT | — | — |
| walsh | 16 | 6 | hold | 256 | 0.0002 | 0.0002 | 0.080 | 0.074–0.088 | 1280/0/0 | BEAT | 0.925 | +0.845 |
| walsh | 16 | 9 | hold | 256 | 0.0002 | 0.0002 | 0.079 | 0.075–0.083 | 1280/0/0 | BEAT | 0.895 | +0.816 |
| walsh | 16 | 12 | transfer | 128 | 0.0002 | 0.0002 | 0.094 | 0.088–0.099 | 640/0/0 | BEAT | 0.940 | +0.846 |
| walsh | 16 | 15 | transfer | 64 | 0.0002 | 0.0002 | 0.111 | 0.103–0.117 | 320/0/0 | BEAT | 0.914 | +0.803 |
| walsh | 16 | 18 | probe | 16 | 0.0002 | 0.0002 | 0.127 | 0.110–0.136 | 80/0/0 | BEAT | — | — |

Per-seed median ε, per-seed R, and per-seed paired wins:

- `fourier_qaoa` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0163, 0.0163, 0.0162, 0.0163, 0.0163
- `fourier_qaoa` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0153, 0.0153, 0.0153, 0.0153, 0.0153
- `fourier_qaoa` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0141, 0.0141, 0.0141, 0.0141, 0.0141
- `fourier_qaoa` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0153, 0.0153, 0.0154, 0.0153, 0.0153
- `fourier_qaoa` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0155, 0.0155, 0.0155, 0.0155, 0.0155
- `power_p` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0115, 0.0115, 0.0116, 0.0115, 0.0115 R 0.708, 0.708, 0.718, 0.708, 0.708 (median of those R = 0.708); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0124, 0.0124, 0.0124, 0.0124, 0.0124 R 0.807, 0.807, 0.805, 0.807, 0.807 (median of those R = 0.807); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0132, 0.0132, 0.0131, 0.0132, 0.0132 R 0.933, 0.933, 0.924, 0.932, 0.932 (median of those R = 0.932); paired wins 116/128, 116/128, 118/128, 116/128, 116/128
- `power_p` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0150, 0.0150, 0.0148, 0.0150, 0.0150 R 0.976, 0.976, 0.965, 0.976, 0.976 (median of those R = 0.976); paired wins 42/64, 42/64, 43/64, 42/64, 42/64
- `power_p` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0160, 0.0160, 0.0158, 0.0160, 0.0160 R 1.034, 1.032, 1.022, 1.032, 1.032 (median of those R = 1.032); paired wins 3/16, 3/16, 6/16, 3/16, 3/16
- `walsh` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0040, 0.0039, 0.0040, 0.0038, 0.0040 R 0.243, 0.240, 0.246, 0.235, 0.244 (median of those R = 0.243); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0039, 0.0038, 0.0041, 0.0038, 0.0040 R 0.257, 0.245, 0.264, 0.249, 0.261 (median of those R = 0.257); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0041, 0.0038, 0.0043, 0.0039, 0.0042 R 0.291, 0.271, 0.304, 0.279, 0.297 (median of those R = 0.291); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0048, 0.0044, 0.0050, 0.0045, 0.0049 R 0.311, 0.288, 0.323, 0.294, 0.317 (median of those R = 0.311); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0050, 0.0045, 0.0052, 0.0047, 0.0051 R 0.321, 0.291, 0.336, 0.301, 0.328 (median of those R = 0.321); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `fourier_qaoa` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0029, 0.0032, 0.0029, 0.0027, 0.0033
- `fourier_qaoa` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0020, 0.0023, 0.0020, 0.0019, 0.0024
- `fourier_qaoa` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0020, 0.0017, 0.0017, 0.0021
- `fourier_qaoa` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0020, 0.0017, 0.0017, 0.0022
- `fourier_qaoa` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0019, 0.0016, 0.0016, 0.0021
- `power_p` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0008, 0.0003, 0.0003, 0.0009 R 0.097, 0.241, 0.105, 0.105, 0.264 (median of those R = 0.105); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0006, 0.0002, 0.0002, 0.0007 R 0.107, 0.266, 0.116, 0.113, 0.301 (median of those R = 0.116); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0006, 0.0002, 0.0002, 0.0007 R 0.131, 0.283, 0.136, 0.132, 0.340 (median of those R = 0.136); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `power_p` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0006, 0.0003, 0.0002, 0.0008 R 0.150, 0.291, 0.152, 0.140, 0.374 (median of those R = 0.152); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `power_p` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0006, 0.0003, 0.0003, 0.0009 R 0.170, 0.297, 0.170, 0.157, 0.413 (median of those R = 0.170); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `walsh` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0006, 0.0002, 0.0002, 0.0006 R 0.072, 0.185, 0.080, 0.085, 0.176 (median of those R = 0.085); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0005, 0.0002, 0.0002, 0.0005 R 0.072, 0.205, 0.079, 0.078, 0.196 (median of those R = 0.079); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0001, 0.0004, 0.0002, 0.0001, 0.0005 R 0.091, 0.219, 0.094, 0.083, 0.216 (median of those R = 0.094); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0005, 0.0002, 0.0002, 0.0005 R 0.111, 0.227, 0.111, 0.089, 0.234 (median of those R = 0.111); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0004, 0.0002, 0.0002, 0.0005 R 0.130, 0.228, 0.126, 0.101, 0.254 (median of those R = 0.130); paired wins 16/16, 16/16, 16/16, 16/16, 16/16

Walsh L=16 n=12: R=0.094, instance CI 0.088–0.099, seed-resample CI 0.083–0.219, pooled paired wins 640/640 (Wilson 95% 0.994–1.000; pairs repeat instances across seeds). PR #31 point estimate 0.092.
Walsh L=16 n=15: R=0.111, instance CI 0.103–0.117, seed-resample CI 0.089–0.234, pooled paired wins 320/320 (Wilson 95% 0.988–1.000; pairs repeat instances across seeds). PR #31 point estimate 0.117.
Walsh L=16 n=18 probe: R=0.127, instance CI 0.110–0.136, pooled paired wins 80/80 (Wilson 95% 0.954–1.000). Exploratory.

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| n | role | γ | Med ε | Walsh L=16 Med ε |
|--:|:-----|--:|------:|------------------:|
| 6 | hold | 0.0000 | 0.0733 | 0.0002 |
| 9 | hold | 0.0000 | 0.0681 | 0.0002 |
| 12 | transfer | 0.0000 | 0.0649 | 0.0002 |
| 15 | transfer | 0.0000 | 0.0652 | 0.0002 |
| 18 | probe | 0.0000 | 0.0649 | 0.0002 |

## Rule

PASS if frozen L=16 Walsh at n=12 and at n=15 both have ratio-of-medians ≤ 0.98, the instance-bootstrap 95% interval lies inside that BEAT line, every seed beats its own Fourier, every seed wins a strict majority of instances, the n=12 drop versus PR #19's 0.940 is at least 0.02 on both the ratio of medians and the median per-seed ratio, and the n=12 and n=15 ground states have peak < 0.5 and median product gap ≥ 0.02. Round 8 also requires at least 95% of the n=12 instances and at least 95% of the n=15 instances to have product gap ≥ 0.02. A shortfall on that fraction is needs_edit and is a quantumness fail, even when the median gap and the Walsh ratios still look like a BEAT. The fraction of instances with product gap < 0.05 is reported and does not move the call. power_p does not gate PASS. n=18 does not gate PASS.

A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. needs_edit also covers a BEAT whose interval, seed list, or paired counts do not all sit inside the bar. KILL if a ground state at n=6, 9, or 12 is a bitstring; if n=12 loses to Fourier and is not below 0.940; or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05).

power_p control at n=12 L=16: R=0.136 (BEAT), instance CI 0.129–0.147, published μ=0 power_p R=0.990, drop +0.854.

Optimizer time summed over fits 451.1s. Branch `cursor/unsat-walsh-mu090-seal-7533`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

