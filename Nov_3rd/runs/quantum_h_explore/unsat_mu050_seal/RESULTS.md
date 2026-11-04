# RESULTS — Round 6 seal, unsat λ=0.75 + μ=−0.50 Walsh freeze

**Verdict: PASS**

frozen n=12 L=16 Walsh R=0.147 (instance 95% CI 0.141–0.157) and n=15 R=0.168 (CI 0.154–0.189) both stay BEAT. 5/5 and 5/5 seeds beat. Paired majorities 5/5 and 5/5. Drop vs μ=0 unsat Walsh 0.793. n=18 probe L=16 Walsh R=0.186 Med ε=0.0003 on B=16 (peak 0.000, product gap 0.061); exploratory, not gated. PR #26 point estimates were n=12 R=0.150 and n=15 R=0.176.

## Setup

Same gate as the PR #25 deepen: BinH U_c^X, Hamming-shell SNAP. The Hamiltonian is the PR #26 winner, μ=−0.50. Intra-cavity XX is scored from the X histogram. Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, use those frozen angles. n=21 was not loaded.

```text
H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j ,  μ = −0.50
U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max
⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩
ε = (⟨H⟩ − E_0) / n
```

Train B=128 per n from `beat_qaoa_12h/train_packs` (base 51e9). Selection holdout B=256 per n (PR #26 used 128). Sealed transfer B=128 at n=12 and B=64 at n=15 (PR #26 used 64 and 32). Seeds [0, 1, 2, 3, 4]. L=[2, 16]. L-BFGS rounds 18/48, max_iter 8/8, lr=0.35. That is the PR #25 optimizer budget on a larger selection holdout. The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. Early stop is 8 rounds without a 10⁻⁶ improvement.

PR #26 point estimates, 3 seeds: n=12 L=16 Walsh R=0.150 on B=64, n=15 L=16 Walsh R=0.176 on B=32. μ=0 comparison remains the published PR #19 ratio of medians (n=12 L=16 Walsh R=0.940), not a retrain.

n=18 probe B=16. One Lanczos-plus-product ground state took 5.3s, under the 90s feasibility limit. The probe is reported and is not part of PASS.

## Quantumness

E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|.

Worst |ΔE0| Lanczos vs dense = 8.88e-15. Histogram vs ⟨ψ|H|ψ⟩ = 6.22e-15. μ=0 histogram vs Pauli field = 3.55e-15.

| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |
|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|
| 6 | train | 128 | 0.064 | 0.217 | 0.075 | 0.103 | 0.728 | 1.3e-14 |
| 6 | hold | 256 | 0.064 | 0.219 | 0.075 | 0.100 | 0.725 | 1.3e-14 |
| 9 | train | 128 | 0.015 | 0.253 | 0.067 | 0.094 | 0.726 | 3.4e-14 |
| 9 | hold | 256 | 0.015 | 0.246 | 0.064 | 0.095 | 0.727 | 2.9e-14 |
| 12 | transfer | 128 | 0.003 | 0.262 | 0.064 | 0.090 | 0.720 | 1.6e-12 |
| 15 | transfer | 64 | 0.001 | 0.262 | 0.064 | 0.093 | 0.722 | 1.1e-10 |
| 18 | probe | 16 | 0.000 | 0.258 | 0.061 | 0.093 | 0.721 | 5.2e-10 |

Train n=6 C_n Walsh mass in degrees 1–4 is 0.120 (degree 3 is 0.031, degree ≥5 is 4.690e-34).

## Transfer table

R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). BEAT if R≤0.98, TIE if R≤1.02, else UNDER. The 95% CI resamples instances inside each seed and keeps every seed (4000 draws, seed 5136). Paired W/T/L counts instances with ε_method < ε_Fourier, exact ties, and losses, summed over seeds. μ=0 R is the PR #19 unsat seal. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. n=18 is the probe.

| method | L | n | role | B | Med ε | seed std | R | 95% CI | paired W/T/L | mark | μ=0 R | drop |
|:-------|--:|--:|:-----|--:|------:|---------:|--:|:-------|:-------------|:-----|------:|-----:|
| fourier_qaoa | 2 | 6 | hold | 256 | 0.0225 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 9 | hold | 256 | 0.0217 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 12 | transfer | 128 | 0.0205 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 15 | transfer | 64 | 0.0222 | 0.0000 | — | — | — |  | — | — |
| fourier_qaoa | 2 | 18 | probe | 16 | 0.0219 | 0.0000 | — | — | — |  | — | — |
| power_p | 2 | 6 | hold | 256 | 0.0173 | 0.0006 | 0.769 | 0.719–0.776 | 1280/0/0 | BEAT | 0.915 | +0.146 |
| power_p | 2 | 9 | hold | 256 | 0.0188 | 0.0000 | 0.866 | 0.856–0.882 | 1273/0/7 | BEAT | 0.961 | +0.095 |
| power_p | 2 | 12 | transfer | 128 | 0.0193 | 0.0003 | 0.942 | 0.932–0.981 | 489/0/151 | BEAT | 1.000 | +0.058 |
| power_p | 2 | 15 | transfer | 64 | 0.0218 | 0.0006 | 0.985 | 0.978–1.027 | 145/0/175 | TIE | 0.994 | +0.009 |
| power_p | 2 | 18 | probe | 16 | 0.0222 | 0.0007 | 1.012 | 0.999–1.105 | 21/0/59 | TIE | — | — |
| walsh | 2 | 6 | hold | 256 | 0.0074 | 0.0000 | 0.327 | 0.312–0.338 | 1280/0/0 | BEAT | 0.670 | +0.343 |
| walsh | 2 | 9 | hold | 256 | 0.0073 | 0.0002 | 0.338 | 0.327–0.348 | 1280/0/0 | BEAT | 0.705 | +0.367 |
| walsh | 2 | 12 | transfer | 128 | 0.0073 | 0.0004 | 0.355 | 0.343–0.375 | 640/0/0 | BEAT | 0.761 | +0.406 |
| walsh | 2 | 15 | transfer | 64 | 0.0083 | 0.0005 | 0.375 | 0.350–0.389 | 320/0/0 | BEAT | 0.733 | +0.358 |
| walsh | 2 | 18 | probe | 16 | 0.0082 | 0.0006 | 0.375 | 0.340–0.411 | 80/0/0 | BEAT | — | — |
| fourier_qaoa | 16 | 6 | hold | 256 | 0.0027 | 0.0005 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 9 | hold | 256 | 0.0019 | 0.0004 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 12 | transfer | 128 | 0.0017 | 0.0004 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 15 | transfer | 64 | 0.0017 | 0.0005 | — | — | — |  | — | — |
| fourier_qaoa | 16 | 18 | probe | 16 | 0.0016 | 0.0004 | — | — | — |  | — | — |
| power_p | 16 | 6 | hold | 256 | 0.0005 | 0.0005 | 0.174 | 0.156–0.188 | 1280/0/0 | BEAT | 0.978 | +0.804 |
| power_p | 16 | 9 | hold | 256 | 0.0004 | 0.0004 | 0.195 | 0.184–0.206 | 1280/0/0 | BEAT | 0.951 | +0.756 |
| power_p | 16 | 12 | transfer | 128 | 0.0004 | 0.0004 | 0.225 | 0.209–0.243 | 640/0/0 | BEAT | 0.990 | +0.765 |
| power_p | 16 | 15 | transfer | 64 | 0.0004 | 0.0005 | 0.242 | 0.220–0.263 | 320/0/0 | BEAT | 0.992 | +0.750 |
| power_p | 16 | 18 | probe | 16 | 0.0004 | 0.0005 | 0.274 | 0.239–0.308 | 80/0/0 | BEAT | — | — |
| walsh | 16 | 6 | hold | 256 | 0.0003 | 0.0004 | 0.125 | 0.116–0.136 | 1280/0/0 | BEAT | 0.925 | +0.800 |
| walsh | 16 | 9 | hold | 256 | 0.0003 | 0.0004 | 0.131 | 0.123–0.140 | 1280/0/0 | BEAT | 0.895 | +0.764 |
| walsh | 16 | 12 | transfer | 128 | 0.0002 | 0.0004 | 0.147 | 0.141–0.157 | 640/0/0 | BEAT | 0.940 | +0.793 |
| walsh | 16 | 15 | transfer | 64 | 0.0003 | 0.0004 | 0.168 | 0.154–0.189 | 320/0/0 | BEAT | 0.914 | +0.746 |
| walsh | 16 | 18 | probe | 16 | 0.0003 | 0.0004 | 0.186 | 0.170–0.230 | 80/0/0 | BEAT | — | — |

Per-seed median ε, per-seed R, and per-seed paired wins:

- `fourier_qaoa` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0225, 0.0225, 0.0225, 0.0225, 0.0225
- `fourier_qaoa` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0216, 0.0217, 0.0217, 0.0217, 0.0217
- `fourier_qaoa` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0205, 0.0205, 0.0205, 0.0205, 0.0205
- `fourier_qaoa` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0222, 0.0222, 0.0221, 0.0222, 0.0222
- `fourier_qaoa` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0219, 0.0219, 0.0219, 0.0219, 0.0219
- `power_p` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0173, 0.0173, 0.0160, 0.0163, 0.0173 R 0.770, 0.771, 0.714, 0.727, 0.769 (median of those R = 0.769); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0188, 0.0188, 0.0187, 0.0187, 0.0188 R 0.866, 0.867, 0.861, 0.864, 0.866 (median of those R = 0.866); paired wins 256/256, 256/256, 249/256, 256/256, 256/256
- `power_p` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0193, 0.0193, 0.0199, 0.0198, 0.0193 R 0.940, 0.942, 0.971, 0.963, 0.941 (median of those R = 0.942); paired wins 112/128, 108/128, 76/128, 82/128, 111/128
- `power_p` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0218, 0.0218, 0.0229, 0.0227, 0.0218 R 0.983, 0.985, 1.035, 1.024, 0.983 (median of those R = 0.985); paired wins 42/64, 41/64, 8/64, 12/64, 42/64
- `power_p` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0221, 0.0222, 0.0235, 0.0232, 0.0221 R 1.011, 1.012, 1.073, 1.060, 1.011 (median of those R = 1.012); paired wins 7/16, 7/16, 0/16, 0/16, 7/16
- `walsh` L=2 n=6 seeds [0, 1, 2, 3, 4]: 0.0074, 0.0074, 0.0073, 0.0073, 0.0074 R 0.327, 0.329, 0.326, 0.326, 0.331 (median of those R = 0.327); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=9 seeds [0, 1, 2, 3, 4]: 0.0073, 0.0075, 0.0071, 0.0071, 0.0076 R 0.339, 0.346, 0.326, 0.327, 0.350 (median of those R = 0.339); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=2 n=12 seeds [0, 1, 2, 3, 4]: 0.0073, 0.0076, 0.0069, 0.0070, 0.0078 R 0.355, 0.371, 0.335, 0.340, 0.382 (median of those R = 0.355); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=2 n=15 seeds [0, 1, 2, 3, 4]: 0.0083, 0.0087, 0.0077, 0.0078, 0.0089 R 0.375, 0.392, 0.346, 0.353, 0.403 (median of those R = 0.375); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=2 n=18 seeds [0, 1, 2, 3, 4]: 0.0082, 0.0086, 0.0075, 0.0077, 0.0089 R 0.375, 0.394, 0.343, 0.351, 0.407 (median of those R = 0.375); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `fourier_qaoa` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0026, 0.0029, 0.0027, 0.0025, 0.0036
- `fourier_qaoa` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0019, 0.0022, 0.0019, 0.0018, 0.0028
- `fourier_qaoa` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0019, 0.0017, 0.0016, 0.0026
- `fourier_qaoa` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0016, 0.0020, 0.0017, 0.0017, 0.0027
- `fourier_qaoa` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0015, 0.0020, 0.0016, 0.0016, 0.0026
- `power_p` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0004, 0.0010, 0.0004, 0.0005, 0.0014 R 0.137, 0.337, 0.134, 0.187, 0.388 (median of those R = 0.187); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0008, 0.0003, 0.0004, 0.0013 R 0.161, 0.365, 0.155, 0.206, 0.462 (median of those R = 0.206); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `power_p` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0008, 0.0003, 0.0004, 0.0013 R 0.195, 0.391, 0.179, 0.229, 0.511 (median of those R = 0.229); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `power_p` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0004, 0.0008, 0.0003, 0.0004, 0.0014 R 0.225, 0.390, 0.199, 0.245, 0.537 (median of those R = 0.245); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `power_p` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0004, 0.0008, 0.0004, 0.0004, 0.0015 R 0.264, 0.408, 0.224, 0.279, 0.591 (median of those R = 0.279); paired wins 16/16, 16/16, 16/16, 16/16, 16/16
- `walsh` L=16 n=6 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0007, 0.0003, 0.0003, 0.0011 R 0.111, 0.244, 0.108, 0.134, 0.310 (median of those R = 0.134); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=9 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0005, 0.0002, 0.0003, 0.0011 R 0.126, 0.237, 0.116, 0.138, 0.380 (median of those R = 0.138); paired wins 256/256, 256/256, 256/256, 256/256, 256/256
- `walsh` L=16 n=12 seeds [0, 1, 2, 3, 4]: 0.0002, 0.0005, 0.0002, 0.0002, 0.0011 R 0.150, 0.255, 0.131, 0.149, 0.419 (median of those R = 0.150); paired wins 128/128, 128/128, 128/128, 128/128, 128/128
- `walsh` L=16 n=15 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0005, 0.0002, 0.0003, 0.0012 R 0.174, 0.262, 0.146, 0.157, 0.451 (median of those R = 0.174); paired wins 64/64, 64/64, 64/64, 64/64, 64/64
- `walsh` L=16 n=18 seeds [0, 1, 2, 3, 4]: 0.0003, 0.0005, 0.0003, 0.0003, 0.0013 R 0.198, 0.270, 0.158, 0.174, 0.488 (median of those R = 0.198); paired wins 16/16, 16/16, 16/16, 16/16, 16/16

Walsh L=16 n=12: R=0.147, instance CI 0.141–0.157, seed-resample CI 0.131–0.419, pooled paired wins 640/640 (Wilson 95% 0.994–1.000; pairs repeat instances across seeds). PR #26 point estimate 0.150.
Walsh L=16 n=15: R=0.168, instance CI 0.154–0.189, seed-resample CI 0.146–0.451, pooled paired wins 320/320 (Wilson 95% 0.988–1.000; pairs repeat instances across seeds). PR #26 point estimate 0.176.
Walsh L=16 n=18 probe: R=0.186, instance CI 0.170–0.230, pooled paired wins 80/80 (Wilson 95% 0.954–1.000). Exploratory.

## One-parameter cost pulse

SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].

| n | role | γ | Med ε | Walsh L=16 Med ε |
|--:|:-----|--:|------:|------------------:|
| 6 | hold | 0.0000 | 0.1004 | 0.0003 |
| 9 | hold | 0.0000 | 0.0947 | 0.0003 |
| 12 | transfer | 0.0000 | 0.0899 | 0.0002 |
| 15 | transfer | 0.0000 | 0.0926 | 0.0003 |
| 18 | probe | 0.0000 | 0.0917 | 0.0003 |

## Rule

PASS if frozen L=16 Walsh at n=12 and at n=15 both have ratio-of-medians ≤ 0.98, the instance-bootstrap 95% interval lies inside that BEAT line, every seed beats its own Fourier, every seed wins a strict majority of instances, the n=12 drop versus PR #19's 0.940 is at least 0.02 on both the ratio of medians and the median per-seed ratio, and the n=12 and n=15 ground states have peak < 0.5 and product gap ≥ 0.02. power_p does not gate PASS. n=18 does not gate PASS.

A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. needs_edit also covers a BEAT whose interval, seed list, or paired counts do not all sit inside the bar. KILL if a ground state at n=6, 9, or 12 is a bitstring; if n=12 loses to Fourier and is not below 0.940; or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05).

power_p control at n=12 L=16: R=0.225 (BEAT), instance CI 0.209–0.243, published μ=0 power_p R=0.990, drop +0.765.

Optimizer time summed over fits 466.0s. Branch `cursor/unsat-walsh-mu050-seal-9cbd`. `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written.

