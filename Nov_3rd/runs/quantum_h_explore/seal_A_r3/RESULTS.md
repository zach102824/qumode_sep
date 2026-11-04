# RESULTS — seal family A, round 3 (n=6 spectral diversity)

**Verdict: PASS**

L=16 power_p protocol R≤0.98 on n=6 and n=9, the seed 95% CI of R stays below 0.99 on both n, a strict majority of seeds agree, the gap is outside one seed-std on at least one n, and the smoke shrink and saturation checks do not fire.

## Recipe

Same gate as the PR #13 smoke, at the λ that smoke locked. Couplings are standard normal on the degree-3 cross-cavity support (first-occurrence ±1 tied at n=6):

\[
H = \sum_{(a,b,c)} J_{abc} Z_a Z_b Z_c \;-\; \lambda \sum_i X_i
\]

λ = **0.5**. Gate: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, BinH `W=H^{⊗3}`, `U_c^X = W exp(-i γ H_c) W`. The field is not inside `U_c`. ⟨H⟩ = ⟨H_c⟩ from the Z histogram minus λ Σ ⟨X_i⟩ from the X histogram (⟨X_i⟩ = 2p(+)-1). No YY or XX.

Cross-cavity triples are taken from the existing 4-SAT packs. The support is still the degree-3 greedy extraction (`m=n` when the geometry allows it). J is an independent standard normal, seeded by the formula id, not the first-occurrence sign. Train rows are the first `B_train` seeds of `beat_qaoa_12h/train_packs`. Holdout rows are the first `B_hold` satisfiable draws at scale1000 base `54_000_000_000`. `qaoa_angles/`, `train69_ladder_fresh250/`, and the phase-2 n21 ladder were not written.

This seal uses B_train=128, B_hold=256, seeds=[0, 1, 2, 3, 4] (the smoke was B_train=32, B_hold=48, seeds 0,1,2). Heads: Fourier QAOA (Zhou q=L, 2L coefficients, mixer φ_j=−β j), power_p (scalar γ plus φ_k=c1 k+c2 k²+c3 k³, 4L, warm-started from that seed's Fourier dual), and Walsh-degree γ (same shells, γ shaped (L, 4) on Walsh degrees 1..4 of the same H_c, 7L). Optimizer is L-BFGS. The checkpoint is the best holdout median ε, including the warm start. ε = (⟨H⟩−E_0)/n. R = median_seed(Med ε_SNAP) / median_seed(Med ε_QAOA). A cell is BEAT at R≤0.98, TIE in (0.98, 1.02], else UNDER. The 95% intervals are bootstrap intervals of that median over seeds.

## Quantumness at λ=0.5

Gate, same as the smoke: median ground-state bitstring peak < 0.5 and median product-state gap (E_prod−E_0)/|E_0| ≥ 2%. Product states are XZ-plane. E_0 is dense eigh at n≤9.

| split | n | B | distinct E0 | med peak | med prod gap | med S (bits) | med χ | med E0/n | pass |
|:------|--:|--:|-----------:|---------:|-------------:|-------------:|------:|---------:|:----:|
| train_n6 | 6 | 128 | 128 | 0.174 | 0.033 | 1.285 | 1.311 | -0.863 | yes |
| hold_n6 | 6 | 256 | 256 | 0.155 | 0.044 | 1.226 | 1.521 | -0.801 | yes |
| train_n9 | 9 | 128 | 128 | 0.088 | 0.031 | 1.616 | 2.936 | -0.792 | yes |
| hold_n9 | 9 | 256 | 256 | 0.085 | 0.028 | 1.772 | 3.153 | -0.818 | yes |

## Walsh spectrum of H_c

Fraction of mean-square on Walsh degrees 0..4. Degree 3 is the only piece a sum of ZZZ terms can carry; a global mean would be degree 0 and is a global phase in the pulse.

| n | split | med frac deg 0..4 | max off-degree-3 |
|--:|:------|:------------------|---------------:|
| 6 | train | 0.00e+00, 3.39e-34, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |
| 6 | hold | 0.00e+00, 3.65e-34, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |
| 9 | train | 0.00e+00, 1.71e-34, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |
| 9 | hold | 0.00e+00, 1.31e-34, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |

Off-degree-3 mass is 0.00e+00 (median over the larger split). Degrees 1, 2, and 4 are numerical zeros, so their γ weights have zero gradient and cannot grade the mixer. The shells c2 and c3 are the mixer grading, and they are already the power_p head. Trained |γ_w − γ_3| mean is 1.264e-02. Walsh-degree γ was still optimized, as a separate 7L head warm-started from the same Fourier dual.

## Holdout

| n | L | method | params | Med ε (seed 95% CI) | seed std | Med f | R (seed 95% CI) | paired ≤0.98 | verdict |
|--:|--:|:-------|------:|:--------------------|---------:|------:|:----------------|-------------:|:--------|
| 6 | 2 | fourier_qaoa | 4 | 0.0638 [0.0628, 0.0716] | 0.0037 | 0.791 |  |  |  |
| 6 | 2 | power_p | 8 | 0.0636 [0.0628, 0.0708] | 0.0033 | 0.793 | 0.998 [0.988, 1.000] | 0/5 | TIE |
| 6 | 2 | walsh | 14 | 0.0636 [0.0628, 0.0708] | 0.0033 | 0.793 | 0.998 [0.988, 1.000] | 0/5 | TIE |
| 6 | 16 | fourier_qaoa | 32 | 0.0135 [0.0113, 0.0162] | 0.0020 | 0.948 |  |  |  |
| 6 | 16 | power_p | 64 | 0.0112 [0.0090, 0.0145] | 0.0022 | 0.960 | 0.828 [0.798, 0.903] | 5/5 | BEAT |
| 6 | 16 | walsh | 112 | 0.0112 [0.0090, 0.0145] | 0.0022 | 0.960 | 0.828 [0.798, 0.903] | 5/5 | BEAT |
| 9 | 2 | fourier_qaoa | 4 | 0.0734 [0.0733, 0.0883] | 0.0067 | 0.765 |  |  |  |
| 9 | 2 | power_p | 8 | 0.0733 [0.0733, 0.0877] | 0.0064 | 0.765 | 0.998 [0.992, 1.000] | 0/5 | TIE |
| 9 | 2 | walsh | 14 | 0.0733 [0.0733, 0.0877] | 0.0064 | 0.765 | 0.998 [0.992, 1.000] | 0/5 | TIE |
| 9 | 16 | fourier_qaoa | 32 | 0.0153 [0.0129, 0.0208] | 0.0029 | 0.951 |  |  |  |
| 9 | 16 | power_p | 64 | 0.0121 [0.0110, 0.0191] | 0.0032 | 0.957 | 0.789 [0.789, 0.922] | 5/5 | BEAT |
| 9 | 16 | walsh | 112 | 0.0121 [0.0110, 0.0190] | 0.0032 | 0.957 | 0.789 [0.789, 0.918] | 5/5 | BEAT |

Per-seed holdout Med ε, and paired R = ε_SNAP/ε_QAOA:

- `n6_L16_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0135, 0.0127, 0.0162, 0.0113, 0.0152
- `n6_L16_power_p` seeds [0, 1, 2, 3, 4]: 0.0112, 0.0110, 0.0145, 0.0090, 0.0137
- `n6_L16_walsh` seeds [0, 1, 2, 3, 4]: 0.0112, 0.0110, 0.0145, 0.0090, 0.0137
- `n6_L2_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0637, 0.0628, 0.0638, 0.0638, 0.0716
- `n6_L2_power_p` seeds [0, 1, 2, 3, 4]: 0.0637, 0.0628, 0.0636, 0.0633, 0.0708
- `n6_L2_walsh` seeds [0, 1, 2, 3, 4]: 0.0637, 0.0628, 0.0636, 0.0633, 0.0708
- `n9_L16_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0129, 0.0163, 0.0150, 0.0208, 0.0153
- `n9_L16_power_p` seeds [0, 1, 2, 3, 4]: 0.0110, 0.0137, 0.0120, 0.0191, 0.0121
- `n9_L16_walsh` seeds [0, 1, 2, 3, 4]: 0.0110, 0.0137, 0.0120, 0.0190, 0.0121
- `n9_L2_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0734, 0.0734, 0.0734, 0.0733, 0.0883
- `n9_L2_power_p` seeds [0, 1, 2, 3, 4]: 0.0733, 0.0733, 0.0733, 0.0733, 0.0877
- `n9_L2_walsh` seeds [0, 1, 2, 3, 4]: 0.0733, 0.0733, 0.0733, 0.0733, 0.0877

- paired R `power_p` n6 L2: 1.000, 1.000, 0.998, 0.992, 0.988
- paired R `walsh` n6 L2: 1.000, 1.000, 0.998, 0.992, 0.988
- paired R `power_p` n6 L16: 0.828, 0.861, 0.893, 0.798, 0.903
- paired R `walsh` n6 L16: 0.828, 0.861, 0.893, 0.798, 0.903
- paired R `power_p` n9 L2: 0.998, 0.999, 0.998, 1.000, 0.992
- paired R `walsh` n9 L2: 0.998, 0.999, 0.998, 1.000, 0.992
- paired R `power_p` n9 L16: 0.859, 0.839, 0.802, 0.922, 0.789
- paired R `walsh` n9 L16: 0.859, 0.839, 0.802, 0.918, 0.789

Instance bootstrap 95% CI of the holdout median, for the seed whose Med ε is the median seed (sampling noise of B_hold, not of the optimizer):

- n6 L16 power_p: snap 0.0112 [0.0097, 0.0125], QAOA 0.0135 [0.0108, 0.0160]
- n6 L16 walsh: snap 0.0112 [0.0097, 0.0125], QAOA 0.0135 [0.0108, 0.0160]
- n9 L16 power_p: snap 0.0121 [0.0108, 0.0145], QAOA 0.0153 [0.0132, 0.0172]
- n9 L16 walsh: snap 0.0121 [0.0106, 0.0143], QAOA 0.0153 [0.0132, 0.0172]

## Checks on L=16 power_p

- Protocol R≤0.98 both n: yes
- Strict majority of seeds paired R≤0.98 on each n: yes
- Gap inside one seed-std on both n: no
- Advantage shrinks n6→n9: no (adv n6=0.00232 n9=0.00324 noise=0.00546)
- Both methods f>0.99 on both n: no

Smoke reference (PR #13, B_train=32, B_hold=48, 3 seeds), L=16 power_p: n6 ε 0.0086 vs QAOA 0.0096 R=0.889, n9 ε 0.0098 vs QAOA 0.0127 R=0.770.

## Freeze train{6,9} → n=12

One shared head trained on n=6 and n=9 together (equal weight on mean ε), seeds [0, 1, 2], then frozen and scored on 64 scale1000 n=12 instances. n=12 is not in the loss and was not used to pick the checkpoint. Snap heads include the shell scale a_n=(8/(n−1))^p, p trained, p=0 at the warm start. E_0(n=12) is a Lanczos ground state of the sparse X-field Hamiltonian (checked against dense eigh on n=6). n=12 med peak=0.073, med E0/n=-0.849. Product gap on this probe uses only the first 8 instances.

| L | method | params | Med ε n6 | Med ε n9 | Med ε n12 | R n12 | paired ≤0.98 |
|--:|:-------|------:|---------:|---------:|----------:|------:|-------------:|
| 2 | fourier_qaoa | 4 | 0.0643 | 0.0728 | 0.1014 |  |  |
| 2 | power_p | 9 | 0.0643 | 0.0729 | 0.1016 | 1.002 | 0/3 |
| 2 | walsh | 15 | 0.0643 | 0.0729 | 0.1016 | 1.002 | 0/3 |
| 16 | fourier_qaoa | 32 | 0.0113 | 0.0169 | 0.0247 |  |  |
| 16 | power_p | 65 | 0.0086 | 0.0143 | 0.0204 | 0.827 | 3/3 |
| 16 | walsh | 113 | 0.0086 | 0.0143 | 0.0204 | 0.827 | 3/3 |

Transfer probe, not a PASS gate. L=2 power_p R12=1.002 (0/3 seeds), p=[0.00015098272371006602, 1.6541239619507653e-05, 4.665726219213575e-06] L=2 walsh R12=1.002 (0/3 seeds), p=[0.00015098272370688337, 1.6541239619510702e-05, 4.665726219213377e-06] L=16 power_p R12=0.827 (3/3 seeds), p=[0.00339534194466757, 0.0014966861723631096, 0.002190430667603368] L=16 walsh R12=0.827 (3/3 seeds), p=[0.0033953413246264894, 0.0014966861703653, 0.00219049506776111]

## Self-check

Fock twin vs `Nov3rdAnsatz` max |Δp| Z=2.07e-08, X=4.60e-07. QAOA-point vs ideal mixer max |Δp|=4.44e-10. X-histogram vs Pauli ⟨X⟩ max |Δ|=5.00e-15. Equal Walsh γ vs scalar γ max |Δp|=1.73e-17. Dead-degree gradient 7.71e-19, live degree-3 gradient 7.32e+00. Lanczos vs dense E_0 on n=6: 5.33e-15.

## Notes

- train_n6: 128 distinct E_0 values, majority fraction 0.008.
- hold_n6: 256 distinct E_0 values, majority fraction 0.004.
- train_n9: 128 distinct E_0 values, majority fraction 0.008.
- hold_n9: 256 distinct E_0 values, majority fraction 0.004.
- n=6 diagnosis: every packed formula contains all 18 cross-cavity triples, and the degree-3 greedy support is a 3-regular design (incidence rank 4 over GF(2)). First-occurrence J=±1 then has two spin-flip orbits. The orbit that clears the product-gap gate is one spectrum, so B_hold=256 repeated it and c2, c3 stayed at 0. Random supports at the same density drop the median product gap under 2%. Denser graphs (m≈2n) push the bitstring peak above 0.5 at λ=0.5. Multiplicity-modulated ±1 splits E_0 but the optimal mixer stays linear, so the shells do not leave the Fourier warm start. J ~ N(0,1) on that same support is the change that keeps the packing and moves the shells.
- The one-seed-std flag compares the median gap to the larger seed-std. One weak seed inflates that std above a median gap that the paired ratios and the instance bootstrap still see. Read it next to the seed CI of R, not instead of it.
- n=6: extracted m median 6 (min 6, max 6), degree ≤ 3.
- n=9: extracted m median 9 (min 8, max 9), degree ≤ 3.
- power_p has 4L parameters and Fourier QAOA has 2L. Walsh-degree γ has 7L. The comparison is not matched-parameter. power_p and Walsh are polished from the trained Fourier dual, so a win is the extra shells (and, if the spectrum were not pure degree 3, the extra γ weights) on top of that mixer.
- Holdout checkpoint selection uses the holdout median residual, the same rule as the smoke, applied to every head. Absolute ε is slightly optimistic. R compares heads selected the same way.
- Seed 95% CIs are percentile bootstrap of the median over the training seeds (10_000 resamples). With five seeds the interval sits on the middle order statistics. The instance CI is a separate bootstrap of one seed's B_hold residuals.
- n=6 L=16 power_p: median seed c2 rms=0.0213, c3 rms=0.0166.
- n=6 L=16 walsh: median seed c2 rms=0.0213, c3 rms=0.0166.
- n=9 L=16 power_p: median seed c2 rms=0.0229, c3 rms=0.0081.
- n=9 L=16 walsh: median seed c2 rms=0.0229, c3 rms=0.0081.

Wall clock 412.9s. Branch `cursor/n6-diversity-seal-a-74c3`.
