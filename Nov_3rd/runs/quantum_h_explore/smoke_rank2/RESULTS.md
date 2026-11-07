# RESULTS — Rank2 intra-cavity XX/XXX on Arm A

**Verdict: needs_edit**

## Setup

`Nov_3rd/runs/opus_ideas/` has no opus reply in this checkout. Rank 1 is Arm A, which won: PR #13 PASS (power_p, L=16) and PR #11 (Walsh-degree on the unsat diagonal). This is the rank-2 deformation that was waiting on that win.

Hamiltonian, Opus sign, raw coefficients:

\[
H = \sum_E J_{abc} Z_a Z_b Z_c - \lambda \sum_i X_i + \mu \sum_{\mathrm{cav}} \sum_{i<j} X_i X_j + \kappa \sum_{\mathrm{cav}} X_1 X_2 X_3
\]

Cross-cavity triples and `J=±1` are the same degree-capped extraction as PR #13 (4-SAT packs, first-occurrence signs, degree ≤ 3). μ = κ = 0 is Arm A, retrained here on the same instances and the same L-BFGS budget. Gate sequence is unchanged: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, BinH `W=H^{⊗3}`, `U_c^X = W exp(-iγ H_c) W` with `H_c = Σ J ZZZ` only. The field and the X products are not inside `U_c`.

Readout uses the Z histogram for ⟨H_c⟩ and the **per-cavity** 8-outcome X histogram (final W omitted) for ⟨X_i⟩, ⟨X_i X_j⟩ and ⟨X_1 X_2 X_3⟩. Those operators are X-strings on the three qubits of one cavity, so a joint X measurement of that cavity is enough. **needs YY: no.** A Y term was not added.

Smoke λ prefers **0.5** (PR #13 operating point). A point moves to its entropy-peak λ only when λ=0.5 fails the quantumness gate on the full train+holdout. Grid μ,κ ∈ [-0.25, 0.0, 0.25] × [-0.25, 0.0, 0.25].

## Quantumness

Gate, same as Arm A: median ground-state bitstring peak < 0.5 and median product-state gap (E_prod − E_0)/|E_0| ≥ 2%. Product states are XZ-plane, ⟨Z⟩=cos θ, ⟨X⟩=sin θ. Bitstring expectations of X, XX and XXX vanish, so the best bitstring energy is still min H_c.

| μ | κ | λ | n | med peak | med prod gap | med S (bits) | med χ | med E0/n | med (E₊−E0)/n | pass |
|--:|--:|--:|--:|---------:|-------------:|-------------:|------:|---------:|---------------:|:----:|
| +0.00 | +0.00 | 0.50 | 6 | 0.109 | 0.073 | 1.711 | 1.110 | -0.807 | 0.307 | yes |
| +0.00 | +0.00 | 0.50 | 9 | 0.087 | 0.034 | 1.940 | 1.559 | -0.876 | 0.376 | yes |
| -0.25 | +0.00 | 0.50 | 6 | 0.071 | 0.176 | 1.233 | 0.573 | -0.956 | 0.206 | yes |
| -0.25 | +0.00 | 0.50 | 9 | 0.044 | 0.103 | 1.511 | 1.835 | -0.973 | 0.223 | yes |
| +0.25 | +0.00 | 0.50 | 6 | 0.238 | 0.056 | 1.852 | 0.293 | -0.777 | 0.527 | yes |
| +0.25 | +0.00 | 0.50 | 9 | 0.114 | 0.026 | 1.959 | 0.515 | -0.857 | 0.607 | yes |
| +0.00 | -0.25 | 0.50 | 6 | 0.097 | 0.119 | 1.625 | 0.809 | -0.858 | 0.275 | yes |
| +0.00 | -0.25 | 0.50 | 9 | 0.071 | 0.054 | 1.905 | 1.987 | -0.898 | 0.314 | yes |
| +0.00 | +0.25 | 0.50 | 6 | 0.188 | 0.065 | 1.704 | 0.270 | -0.801 | 0.384 | yes |
| +0.00 | +0.25 | 0.50 | 9 | 0.099 | 0.030 | 1.923 | 0.751 | -0.872 | 0.455 | yes |
| -0.25 | -0.25 | 0.50 | 6 | 0.064 | 0.177 | 1.076 | 0.418 | -1.021 | 0.187 | yes |
| -0.25 | -0.25 | 0.50 | 9 | 0.031 | 0.145 | 1.256 | 1.271 | -1.024 | 0.191 | yes |
| -0.25 | +0.25 | 0.50 | 6 | 0.079 | 0.128 | 1.436 | 0.718 | -0.895 | 0.228 | yes |
| -0.25 | +0.25 | 0.50 | 9 | 0.060 | 0.070 | 1.768 | 1.955 | -0.934 | 0.267 | yes |
| +0.25 | -0.25 | 0.50 | 6 | 0.240 | 0.036 | 1.793 | 0.582 | -0.770 | 0.437 | yes |
| +0.25 | -0.25 | 0.50 | 9 | 0.115 | 0.023 | 1.950 | 0.788 | -0.854 | 0.521 | yes |
| +0.25 | +0.25 | 0.50 | 6 | 0.235 | 0.089 | 1.780 | 0.227 | -0.807 | 0.641 | yes |
| +0.25 | +0.25 | 0.50 | 9 | 0.124 | 0.039 | 1.908 | 0.430 | -0.874 | 0.707 | yes |

Entropy-peak λ on the eigenvalue scan (no product search), full train+holdout:

- μ=+0.00, κ=+0.00: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer
- μ=-0.25, κ=+0.00: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer
- μ=+0.25, κ=+0.00: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer
- μ=+0.00, κ=-0.25: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer
- μ=+0.00, κ=+0.25: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer
- μ=-0.25, κ=-0.25: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer
- μ=-0.25, κ=+0.25: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer
- μ=+0.25, κ=-0.25: n6=0.5, n9=0.25; trained λ=0.5; reason=prefer
- μ=+0.25, κ=+0.25: n6=0.25, n9=0.25; trained λ=0.5; reason=prefer

## Train

Fourier QAOA: Zhou sine modes, q=L, mixer φ_j=−β j. power_p: φ_k=c1 k+c2 k²+c3 k³ (4L parameters) warm-started from that seed's trained Fourier dual and polished with L-BFGS. Init is the Boulebnane–Montanaro point (β,γ)=(0.01,−0.01) plus N(0,0.01²), same seed stream as PR #13 (the stream does not depend on μ, κ). Checkpoint is the best holdout median ε, including the warm start. B_train=32, B_hold=48, seeds=[0, 1, 2], L=[2, 16].

| μ | κ | λ | n | L | Med ε QAOA | Med ε SNAP | R | vs base R | vs base ε_SNAP | verdict |
|--:|--:|--:|--:|--:|-----------:|-----------:|--:|----------:|---------------:|:--------|
| +0.00 | +0.00 | 0.50 | 6 | 2 | 0.0440 | 0.0440 | 1.000 |  |  | TIE |
| +0.00 | +0.00 | 0.50 | 6 | 16 | 0.0096 | 0.0086 | 0.889 |  |  | BEAT |
| +0.00 | +0.00 | 0.50 | 9 | 2 | 0.0970 | 0.0969 | 0.999 |  |  | TIE |
| +0.00 | +0.00 | 0.50 | 9 | 16 | 0.0127 | 0.0094 | 0.736 |  |  | BEAT |
| -0.25 | +0.00 | 0.50 | 6 | 2 | 0.0208 | 0.0208 | 1.000 | 0.000 | 0.0232 | TIE |
| -0.25 | +0.00 | 0.50 | 6 | 16 | 0.0044 | 0.0030 | 0.684 | 0.205 | 0.0055 | BEAT |
| -0.25 | +0.00 | 0.50 | 9 | 2 | 0.0398 | 0.0396 | 0.996 | 0.003 | 0.0573 | TIE |
| -0.25 | +0.00 | 0.50 | 9 | 16 | 0.0119 | 0.0059 | 0.498 | 0.238 | 0.0034 | BEAT |
| +0.25 | +0.00 | 0.50 | 6 | 2 | 0.1446 | 0.1433 | 0.991 | 0.009 | -0.0993 | TIE |
| +0.25 | +0.00 | 0.50 | 6 | 16 | 0.0813 | 0.0727 | 0.894 | -0.005 | -0.0641 | BEAT |
| +0.25 | +0.00 | 0.50 | 9 | 2 | 0.1879 | 0.1879 | 1.000 | -0.001 | -0.0910 | TIE |
| +0.25 | +0.00 | 0.50 | 9 | 16 | 0.0359 | 0.0331 | 0.922 | -0.186 | -0.0238 | BEAT |
| +0.00 | -0.25 | 0.50 | 6 | 2 | 0.0367 | 0.0367 | 1.000 | 0.000 | 0.0073 | TIE |
| +0.00 | -0.25 | 0.50 | 6 | 16 | 0.0083 | 0.0076 | 0.924 | -0.035 | 0.0009 | BEAT |
| +0.00 | -0.25 | 0.50 | 9 | 2 | 0.0769 | 0.0768 | 1.000 | -0.000 | 0.0200 | TIE |
| +0.00 | -0.25 | 0.50 | 9 | 16 | 0.0124 | 0.0091 | 0.737 | -0.001 | 0.0002 | BEAT |
| +0.00 | +0.25 | 0.50 | 6 | 2 | 0.0898 | 0.0898 | 1.000 | 0.000 | -0.0458 | TIE |
| +0.00 | +0.25 | 0.50 | 6 | 16 | 0.0442 | 0.0442 | 1.000 | -0.111 | -0.0357 | TIE |
| +0.00 | +0.25 | 0.50 | 9 | 2 | 0.1343 | 0.1343 | 1.000 | -0.001 | -0.0374 | TIE |
| +0.00 | +0.25 | 0.50 | 9 | 16 | 0.0216 | 0.0173 | 0.804 | -0.068 | -0.0080 | BEAT |
| -0.25 | -0.25 | 0.50 | 6 | 2 | 0.0169 | 0.0169 | 1.000 | 0.000 | 0.0271 | TIE |
| -0.25 | -0.25 | 0.50 | 6 | 16 | 0.0044 | 0.0027 | 0.602 | 0.287 | 0.0059 | BEAT |
| -0.25 | -0.25 | 0.50 | 9 | 2 | 0.0288 | 0.0283 | 0.982 | 0.017 | 0.0686 | TIE |
| -0.25 | -0.25 | 0.50 | 9 | 16 | 0.0134 | 0.0075 | 0.556 | 0.180 | 0.0019 | BEAT |
| -0.25 | +0.25 | 0.50 | 6 | 2 | 0.0267 | 0.0267 | 1.000 | 0.000 | 0.0173 | TIE |
| -0.25 | +0.25 | 0.50 | 6 | 16 | 0.0054 | 0.0040 | 0.730 | 0.159 | 0.0046 | BEAT |
| -0.25 | +0.25 | 0.50 | 9 | 2 | 0.0573 | 0.0572 | 0.999 | 0.000 | 0.0397 | TIE |
| -0.25 | +0.25 | 0.50 | 9 | 16 | 0.0095 | 0.0064 | 0.670 | 0.066 | 0.0030 | BEAT |
| +0.25 | -0.25 | 0.50 | 6 | 2 | 0.0845 | 0.0802 | 0.949 | 0.051 | -0.0362 | BEAT |
| +0.25 | -0.25 | 0.50 | 6 | 16 | 0.0287 | 0.0219 | 0.764 | 0.125 | -0.0133 | BEAT |
| +0.25 | -0.25 | 0.50 | 9 | 2 | 0.1541 | 0.1538 | 0.998 | 0.001 | -0.0570 | TIE |
| +0.25 | -0.25 | 0.50 | 9 | 16 | 0.0229 | 0.0209 | 0.914 | -0.178 | -0.0116 | BEAT |
| +0.25 | +0.25 | 0.50 | 6 | 2 | 0.2158 | 0.2142 | 0.993 | 0.007 | -0.1703 | TIE |
| +0.25 | +0.25 | 0.50 | 6 | 16 | 0.1427 | 0.1366 | 0.957 | -0.068 | -0.1280 | BEAT |
| +0.25 | +0.25 | 0.50 | 9 | 2 | 0.2326 | 0.2324 | 0.999 | 0.000 | -0.1355 | TIE |
| +0.25 | +0.25 | 0.50 | 9 | 16 | 0.0549 | 0.0503 | 0.917 | -0.180 | -0.0410 | BEAT |

ε = (⟨H⟩ − E_0)/n on holdout, median over instances, then median over seeds. R = Med ε_SNAP / Med ε_QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. vs base R is R(μ=κ=0) − R (positive means SNAP's ratio improved). vs base ε_SNAP is ε_SNAP(baseline) − ε_SNAP (positive means a smaller residual). Lower ε is better.

Per-seed holdout Med ε:

- μ=+0.00, κ=+0.00, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0096, 0.0084, 0.0102
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0086, 0.0084, 0.0095
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0439, 0.0440, 0.0440
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.0439, 0.0440, 0.0440
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0096, 0.0140, 0.0127
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0081, 0.0099, 0.0094
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0969, 0.0970, 0.0970
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.0969, 0.0969, 0.0969
- μ=-0.25, κ=+0.00, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0044, 0.0047, 0.0044
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0028, 0.0030, 0.0031
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0208, 0.0208, 0.0208
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.0208, 0.0208, 0.0208
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0144, 0.0119, 0.0112
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0072, 0.0059, 0.0059
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0398, 0.0430, 0.0398
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.0396, 0.0424, 0.0396
- μ=+0.25, κ=+0.00, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0812, 0.0813, 0.0814
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0709, 0.0728, 0.0727
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.1446, 0.1446, 0.1446
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.1433, 0.1433, 0.1432
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0359, 0.0350, 0.0370
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0341, 0.0330, 0.0331
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.1879, 0.1879, 0.1879
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.1878, 0.1879, 0.1879
- μ=+0.00, κ=-0.25, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0084, 0.0083, 0.0076
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0070, 0.0083, 0.0076
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0367, 0.0367, 0.0367
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.0367, 0.0367, 0.0367
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0125, 0.0124, 0.0123
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0095, 0.0091, 0.0091
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0769, 0.0769, 0.0769
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.0768, 0.0768, 0.0768
- μ=+0.00, κ=+0.25, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0442, 0.0329, 0.0474
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0442, 0.0307, 0.0466
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0898, 0.0898, 0.0898
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.0898, 0.0898, 0.0898
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0207, 0.0221, 0.0216
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0172, 0.0176, 0.0173
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.1343, 0.1343, 0.1343
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.1343, 0.1343, 0.1343
- μ=-0.25, κ=-0.25, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0044, 0.0049, 0.0044
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0024, 0.0027, 0.0032
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0169, 0.0169, 0.0168
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.0169, 0.0169, 0.0168
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0134, 0.0133, 0.0153
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0075, 0.0073, 0.0077
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0288, 0.0311, 0.0288
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.0282, 0.0303, 0.0283
- μ=-0.25, κ=+0.25, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0063, 0.0054, 0.0054
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0040, 0.0040, 0.0039
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0267, 0.0267, 0.0267
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.0267, 0.0267, 0.0267
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0095, 0.0095, 0.0099
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0064, 0.0063, 0.0065
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0573, 0.0573, 0.0573
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.0572, 0.0572, 0.0572
- μ=+0.25, κ=-0.25, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0291, 0.0249, 0.0287
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.0272, 0.0219, 0.0214
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.0845, 0.0845, 0.0845
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.0802, 0.0812, 0.0791
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0223, 0.0236, 0.0229
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0209, 0.0209, 0.0215
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.1541, 0.1541, 0.1541
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.1538, 0.1539, 0.1538
- μ=+0.25, κ=+0.25, λ=0.50
  - `n6_L16_fourier_qaoa` seeds [0, 1, 2]: 0.1430, 0.1427, 0.1414
  - `n6_L16_power_p` seeds [0, 1, 2]: 0.1352, 0.1395, 0.1366
  - `n6_L2_fourier_qaoa` seeds [0, 1, 2]: 0.2153, 0.2158, 0.2158
  - `n6_L2_power_p` seeds [0, 1, 2]: 0.2142, 0.2143, 0.2140
  - `n9_L16_fourier_qaoa` seeds [0, 1, 2]: 0.0565, 0.0539, 0.0549
  - `n9_L16_power_p` seeds [0, 1, 2]: 0.0503, 0.0514, 0.0481
  - `n9_L2_fourier_qaoa` seeds [0, 1, 2]: 0.2326, 0.2326, 0.2326
  - `n9_L2_power_p` seeds [0, 1, 2]: 0.2324, 0.2324, 0.2324

## Baseline vs PR #13

Same protocol at μ=κ=0, λ=0.5. PR #13 published L=16 medians (n6 QAOA/SNAP/R = 0.0096/0.0086/0.889, n9 = 0.0127/0.0098/0.770).

- n=6: this run QAOA 0.0096 (Δ 0.0000), SNAP 0.0086 (Δ -0.0000), R 0.889 (PR13 0.889).
- n=9: this run QAOA 0.0127 (Δ 0.0000), SNAP 0.0094 (Δ -0.0004), R 0.736 (PR13 0.770).

## Gain rule and kill checks

Pre-registered before the train finished. A nonzero point **gains clearly** when quantumness passes on both n, L=16 power_p BEATs Fourier QAOA on both n, and R falls versus the in-script μ=κ=0 baseline on both n by more than the combined seed-std of R and by at least 0.02. Arm A kill rules on that point (L=16 gap inside one seed std, advantage shrinks n6→n9, or both methods saturated with (E₊−E_0)/n ≥ 0.05) block PASS. A same-direction improvement that misses that bar, a drop in ε_SNAP without a clearer R, or a new L=2 BEAT, is **needs_edit**. Per-seed ΔR and Δε are reported as a paired diagnostic (SNAP is warm-started from that seed's Fourier run). They do not override the unpaired bar. **KILL** if no nonzero point improves R or ε_SNAP on both n, or every nonzero point fails quantumness, or scoring ⟨H⟩ needs a Y correlator. PASS is also withheld when the μ=κ=0 baseline itself misses the L=16 BEAT gate, because the comparison to Arm A is then not the published one.

- μ=-0.25, κ=+0.00: class=`soft` quantum=yes L16 BEAT both=yes better R both=yes clear unpaired R=no paired every seed=yes better ε_SNAP both=yes ArmA kills=[]
  - n=6 per-seed ΔR (base−new) [0.242, 0.358, 0.239], per-seed Δε_SNAP [0.0057, 0.0053, 0.0064]
  - n=9 per-seed ΔR (base−new) [0.341, 0.210, 0.213], per-seed Δε_SNAP [0.0009, 0.0040, 0.0035]
- μ=+0.25, κ=+0.00: class=`no_gain` quantum=yes L16 BEAT both=yes better R both=no clear unpaired R=no paired every seed=no better ε_SNAP both=no ArmA kills=['advantage_shrinks_n6_to_n9']
  - n=6 per-seed ΔR (base−new) [0.016, 0.105, 0.041], per-seed Δε_SNAP [-0.0624, -0.0644, -0.0632]
  - n=9 per-seed ΔR (base−new) [-0.110, -0.235, -0.160], per-seed Δε_SNAP [-0.0260, -0.0232, -0.0238]
- μ=+0.00, κ=-0.25: class=`soft` quantum=yes L16 BEAT both=yes better R both=no clear unpaired R=no paired every seed=no better ε_SNAP both=yes ArmA kills=[]
  - n=6 per-seed ΔR (base−new) [0.058, 0.000, -0.066], per-seed Δε_SNAP [0.0016, 0.0001, 0.0019]
  - n=9 per-seed ΔR (base−new) [0.081, -0.029, -0.004], per-seed Δε_SNAP [-0.0014, 0.0007, 0.0003]
- μ=+0.00, κ=+0.25: class=`no_gain` quantum=yes L16 BEAT both=no better R both=no clear unpaired R=no paired every seed=no better ε_SNAP both=no ArmA kills=[]
  - n=6 per-seed ΔR (base−new) [-0.111, 0.066, -0.048], per-seed Δε_SNAP [-0.0357, -0.0224, -0.0370]
  - n=9 per-seed ΔR (base−new) [0.010, -0.088, -0.068], per-seed Δε_SNAP [-0.0091, -0.0077, -0.0080]
- μ=-0.25, κ=-0.25: class=`soft` quantum=yes L16 BEAT both=yes better R both=yes clear unpaired R=no paired every seed=yes better ε_SNAP both=yes ArmA kills=[]
  - n=6 per-seed ΔR (base−new) [0.353, 0.454, 0.216], per-seed Δε_SNAP [0.0062, 0.0057, 0.0064]
  - n=9 per-seed ΔR (base−new) [0.283, 0.162, 0.228], per-seed Δε_SNAP [0.0006, 0.0026, 0.0016]
- μ=-0.25, κ=+0.25: class=`soft` quantum=yes L16 BEAT both=yes better R both=yes clear unpaired R=no paired every seed=yes better ε_SNAP both=yes ArmA kills=[]
  - n=6 per-seed ΔR (base−new) [0.260, 0.265, 0.207], per-seed Δε_SNAP [0.0046, 0.0044, 0.0056]
  - n=9 per-seed ΔR (base−new) [0.170, 0.040, 0.078], per-seed Δε_SNAP [0.0017, 0.0036, 0.0029]
- μ=+0.25, κ=-0.25: class=`no_gain` quantum=yes L16 BEAT both=yes better R both=no clear unpaired R=no paired every seed=no better ε_SNAP both=no ArmA kills=['advantage_shrinks_n6_to_n9']
  - n=6 per-seed ΔR (base−new) [-0.045, 0.121, 0.188], per-seed Δε_SNAP [-0.0187, -0.0135, -0.0119]
  - n=9 per-seed ΔR (base−new) [-0.102, -0.179, -0.202], per-seed Δε_SNAP [-0.0129, -0.0110, -0.0122]
- μ=+0.25, κ=+0.25: class=`no_gain` quantum=yes L16 BEAT both=yes better R both=no clear unpaired R=no paired every seed=no better ε_SNAP both=no ArmA kills=[]
  - n=6 per-seed ΔR (base−new) [-0.056, 0.022, -0.032], per-seed Δε_SNAP [-0.1266, -0.1312, -0.1271]
  - n=9 per-seed ΔR (base−new) [-0.051, -0.247, -0.141], per-seed Δε_SNAP [-0.0423, -0.0415, -0.0388]

Family verdict: **needs_edit**. Negative μ lowers L=16 ε_SNAP and R on both n (paired on every seed at m-0.25_k+0.00, m-0.25_k-0.25, m-0.25_k+0.25). The pre-registered unpaired R-std bar does not clear, because baseline n9 Fourier seeds already scatter. That is an edit, not a kill. Positive μ does not gain. The μ=κ=0 baseline does reproduce the L=16 BEAT.

## Self-check

Rank2 ⟨ψ|H|ψ⟩ matches the per-cavity X histogram (grad |Δ|=6.914580019667937e-11). needs_YY=no. At μ=κ=0 the dense matrix matches `dense_hamiltonian`, and the product-state search is at least as low as Arm A's. Fock twin vs `Nov3rdAnsatz` max |Δp| Z=2.06935301533262e-08, X=4.596927047906263e-07. QAOA-point vs ideal mixer max |Δp|=4.4421986575282624e-10.

## Notes

- H_c is pure Walsh degree 3, so a Walsh-degree γ head is a null knob on this diagonal. It was not trained. PR #11's Walsh win was on the unsat-count diagonal, which has several degrees.
- power_p has 4L parameters and Fourier QAOA has 2L. The comparison asks whether the c2, c3 shells beat the linear mixer they are warm-started from.
- Holdout checkpoint selection uses the holdout median residual, same rule as PR #13, applied to both methods.
- XX and XXX matrix elements are real flip operators. The statevector check equates the per-cavity X histogram to ⟨ψ|H|ψ⟩, so the score does not use a Y string.
- Quantumness, entropy peaks, and E0 use the full train+holdout (not a B_scan slice).
- n21 phase2 ladder scripts and qaoa_angles/ were not written.

Wall clock 466.6s. Branch `cursor/quantum-h-rank2-xx-03ef`. phase2 n21 ladder was not started and its scripts were not edited.
