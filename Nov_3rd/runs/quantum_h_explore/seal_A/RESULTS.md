# RESULTS — seal family A (PR #13 recipe)

**Verdict: needs_edit**

protocol R is not ≤0.98 on both n (n6=1.000, n9=0.768). paired R_s≤0.98 is not a strict majority on both n (n6 1/5, n9 4/5). the L=16 ε gap is inside one seed-std on both n.

## Recipe

Same Hamiltonian and gate as the PR #13 smoke, at the λ that smoke locked:

\[
H = \sum_{(a,b,c)} J_{abc} Z_a Z_b Z_c \;-\; \lambda \sum_i X_i
\]

λ = **0.5**. Gate: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, BinH `W=H^{⊗3}`, `U_c^X = W exp(-i γ H_c) W`. The field is not inside `U_c`. ⟨H⟩ = ⟨H_c⟩ from the Z histogram minus λ Σ ⟨X_i⟩ from the X histogram (⟨X_i⟩ = 2p(+)-1). No YY or XX.

Cross-cavity triples are taken from the existing 4-SAT packs. `J=±1` is the product of the three literal signs. Degree cap 3, so `m=n` when the geometry allows it. Train rows are the first `B_train` seeds of `beat_qaoa_12h/train_packs`. Holdout rows are the first `B_hold` satisfiable draws at scale1000 base `54_000_000_000`. `qaoa_angles/`, `train69_ladder_fresh250/`, and the phase-2 n21 ladder were not written.

This seal uses B_train=128, B_hold=256, seeds=[0, 1, 2, 3, 4] (the smoke was B_train=32, B_hold=48, seeds 0,1,2). Heads: Fourier QAOA (Zhou q=L, 2L coefficients, mixer φ_j=−β j), power_p (scalar γ plus φ_k=c1 k+c2 k²+c3 k³, 4L, warm-started from that seed's Fourier dual), and Walsh-degree γ (same shells, γ shaped (L, 4) on Walsh degrees 1..4 of the same H_c, 7L). Optimizer is L-BFGS. The checkpoint is the best holdout median ε, including the warm start. ε = (⟨H⟩−E_0)/n. R = median_seed(Med ε_SNAP) / median_seed(Med ε_QAOA). A cell is BEAT at R≤0.98, TIE in (0.98, 1.02], else UNDER. The 95% intervals are bootstrap intervals of that median over seeds.

## Quantumness at λ=0.5

Gate, same as the smoke: median ground-state bitstring peak < 0.5 and median product-state gap (E_prod−E_0)/|E_0| ≥ 2%. Product states are XZ-plane. E_0 is dense eigh at n≤9.

| split | n | B | distinct E0 | med peak | med prod gap | med S (bits) | med χ | med E0/n | pass |
|:------|--:|--:|-----------:|---------:|-------------:|-------------:|------:|---------:|:----:|
| train_n6 | 6 | 128 | 2 | 0.109 | 0.073 | 1.527 | 1.110 | -0.807 | yes |
| hold_n6 | 6 | 256 | 2 | 0.109 | 0.073 | 1.527 | 1.110 | -0.807 | yes |
| train_n9 | 9 | 128 | 40 | 0.087 | 0.035 | 1.978 | 1.559 | -0.877 | yes |
| hold_n9 | 9 | 256 | 65 | 0.087 | 0.035 | 1.932 | 1.550 | -0.877 | yes |

## Walsh spectrum of H_c

Fraction of mean-square on Walsh degrees 0..4. Degree 3 is the only piece a sum of ZZZ terms can carry; a global mean would be degree 0 and is a global phase in the pulse.

| n | split | med frac deg 0..4 | max off-degree-3 |
|--:|:------|:------------------|---------------:|
| 6 | train | 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |
| 6 | hold | 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |
| 9 | train | 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |
| 9 | hold | 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00 | 0.00e+00 |

Off-degree-3 mass is 0.00e+00 (median over the larger split). Degrees 1, 2, and 4 are numerical zeros, so their γ weights have zero gradient and cannot grade the mixer. The shells c2 and c3 are the mixer grading, and they are already the power_p head. Trained |γ_w − γ_3| mean is 4.075e-04. Walsh-degree γ was still optimized, as a separate 7L head warm-started from the same Fourier dual.

## Holdout

| n | L | method | params | Med ε (seed 95% CI) | seed std | Med f | R (seed 95% CI) | paired ≤0.98 | verdict |
|--:|--:|:-------|------:|:--------------------|---------:|------:|:----------------|-------------:|:--------|
| 6 | 2 | fourier_qaoa | 4 | 0.0432 [0.0432, 0.0555] | 0.0055 | 0.859 |  |  |  |
| 6 | 2 | power_p | 8 | 0.0431 [0.0431, 0.0554] | 0.0055 | 0.859 | 1.000 [0.999, 1.000] | 0/5 | TIE |
| 6 | 2 | walsh | 14 | 0.0431 [0.0431, 0.0554] | 0.0055 | 0.859 | 1.000 [0.999, 1.000] | 0/5 | TIE |
| 6 | 16 | fourier_qaoa | 32 | 0.0075 [0.0069, 0.0078] | 0.0004 | 0.976 |  |  |  |
| 6 | 16 | power_p | 64 | 0.0075 [0.0069, 0.0078] | 0.0004 | 0.976 | 1.000 [0.974, 1.000] | 1/5 | TIE |
| 6 | 16 | walsh | 112 | 0.0075 [0.0069, 0.0078] | 0.0004 | 0.976 | 1.000 [0.974, 1.000] | 1/5 | TIE |
| 9 | 2 | fourier_qaoa | 4 | 0.0987 [0.0987, 0.1130] | 0.0064 | 0.729 |  |  |  |
| 9 | 2 | power_p | 8 | 0.0986 [0.0986, 0.1127] | 0.0063 | 0.729 | 0.999 [0.997, 1.000] | 0/5 | TIE |
| 9 | 2 | walsh | 14 | 0.0986 [0.0986, 0.1127] | 0.0063 | 0.729 | 0.999 [0.997, 1.000] | 0/5 | TIE |
| 9 | 16 | fourier_qaoa | 32 | 0.0115 [0.0106, 0.0162] | 0.0023 | 0.968 |  |  |  |
| 9 | 16 | power_p | 64 | 0.0088 [0.0083, 0.0154] | 0.0030 | 0.973 | 0.768 [0.722, 1.000] | 4/5 | BEAT |
| 9 | 16 | walsh | 112 | 0.0088 [0.0083, 0.0154] | 0.0030 | 0.973 | 0.768 [0.722, 1.000] | 4/5 | BEAT |

Per-seed holdout Med ε, and paired R = ε_SNAP/ε_QAOA:

- `n6_L16_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0078, 0.0073, 0.0077, 0.0075, 0.0069
- `n6_L16_power_p` seeds [0, 1, 2, 3, 4]: 0.0078, 0.0073, 0.0075, 0.0075, 0.0069
- `n6_L16_walsh` seeds [0, 1, 2, 3, 4]: 0.0078, 0.0073, 0.0075, 0.0075, 0.0069
- `n6_L2_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0432, 0.0432, 0.0432, 0.0432, 0.0555
- `n6_L2_power_p` seeds [0, 1, 2, 3, 4]: 0.0431, 0.0431, 0.0431, 0.0431, 0.0554
- `n6_L2_walsh` seeds [0, 1, 2, 3, 4]: 0.0431, 0.0431, 0.0431, 0.0431, 0.0554
- `n9_L16_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0106, 0.0115, 0.0115, 0.0162, 0.0109
- `n9_L16_power_p` seeds [0, 1, 2, 3, 4]: 0.0088, 0.0083, 0.0085, 0.0154, 0.0109
- `n9_L16_walsh` seeds [0, 1, 2, 3, 4]: 0.0088, 0.0083, 0.0085, 0.0154, 0.0109
- `n9_L2_fourier_qaoa` seeds [0, 1, 2, 3, 4]: 0.0987, 0.0987, 0.0987, 0.0987, 0.1130
- `n9_L2_power_p` seeds [0, 1, 2, 3, 4]: 0.0986, 0.0986, 0.0986, 0.0986, 0.1127
- `n9_L2_walsh` seeds [0, 1, 2, 3, 4]: 0.0986, 0.0986, 0.0986, 0.0986, 0.1127

- paired R `power_p` n6 L2: 1.000, 1.000, 1.000, 1.000, 0.999
- paired R `walsh` n6 L2: 1.000, 1.000, 1.000, 1.000, 0.999
- paired R `power_p` n6 L16: 1.000, 1.000, 0.974, 1.000, 1.000
- paired R `walsh` n6 L16: 1.000, 1.000, 0.974, 1.000, 1.000
- paired R `power_p` n9 L2: 1.000, 0.999, 0.999, 0.999, 0.997
- paired R `walsh` n9 L2: 1.000, 0.999, 0.999, 0.999, 0.997
- paired R `power_p` n9 L16: 0.834, 0.722, 0.741, 0.949, 1.000
- paired R `walsh` n9 L16: 0.834, 0.722, 0.741, 0.949, 1.000

Instance bootstrap 95% CI of the holdout median, for the seed whose Med ε is the median seed (sampling noise of B_hold, not of the optimizer):

- n6 L16 power_p: snap 0.0075 [0.0075, 0.0075], QAOA 0.0075 [0.0075, 0.0075]
- n6 L16 walsh: snap 0.0075 [0.0075, 0.0075], QAOA 0.0075 [0.0075, 0.0075]
- n9 L16 power_p: snap 0.0088 [0.0083, 0.0095], QAOA 0.0115 [0.0111, 0.0120]
- n9 L16 walsh: snap 0.0088 [0.0082, 0.0095], QAOA 0.0115 [0.0111, 0.0120]

## Checks on L=16 power_p

- Protocol R≤0.98 both n: no
- Strict majority of seeds paired R≤0.98 on each n: no
- Gap inside one seed-std on both n: yes
- Advantage shrinks n6→n9: no (adv n6=0.00000 n9=0.00266 noise=0.00334)
- Both methods f>0.99 on both n: no

Smoke reference (PR #13, B_train=32, B_hold=48, 3 seeds), L=16 power_p: n6 ε 0.0086 vs QAOA 0.0096 R=0.889, n9 ε 0.0098 vs QAOA 0.0127 R=0.770.

## Freeze train{6,9} → n=12

One shared head trained on n=6 and n=9 together (equal weight on mean ε), seeds [0, 1, 2], then frozen and scored on 64 scale1000 n=12 instances. n=12 is not in the loss and was not used to pick the checkpoint. Snap heads include the shell scale a_n=(8/(n−1))^p, p trained, p=0 at the warm start. E_0(n=12) is a Lanczos ground state of the sparse X-field Hamiltonian (checked against dense eigh on n=6). n=12 med peak=0.080, med E0/n=-0.918. Product gap on this probe uses only the first 8 instances.

| L | method | params | Med ε n6 | Med ε n9 | Med ε n12 | R n12 | paired ≤0.98 |
|--:|:-------|------:|---------:|---------:|----------:|------:|-------------:|
| 2 | fourier_qaoa | 4 | 0.0430 | 0.0986 | 0.1377 |  |  |
| 2 | power_p | 9 | 0.0429 | 0.0986 | 0.1378 | 1.000 | 0/3 |
| 2 | walsh | 15 | 0.0429 | 0.0986 | 0.1378 | 1.000 | 0/3 |
| 16 | fourier_qaoa | 32 | 0.0045 | 0.0169 | 0.0290 |  |  |
| 16 | power_p | 65 | 0.0025 | 0.0133 | 0.0245 | 0.845 | 3/3 |
| 16 | walsh | 113 | 0.0025 | 0.0133 | 0.0245 | 0.845 | 3/3 |

Transfer probe, not a PASS gate. L=2 power_p R12=1.000 (0/3 seeds), p=[2.4358760388553715e-06, 2.408511902979578e-06, 2.3937943487528286e-06] L=2 walsh R12=1.000 (0/3 seeds), p=[2.435876038855295e-06, 2.4085119029796326e-06, 2.393794348752837e-06] L=16 power_p R12=0.845 (3/3 seeds), p=[0.006749350149759422, 0.002099317960090351, 0.009195766869439619] L=16 walsh R12=0.845 (3/3 seeds), p=[0.006749350155690507, 0.0020993179600641843, 0.009195792952243698]

## Self-check

Fock twin vs `Nov3rdAnsatz` max |Δp| Z=2.07e-08, X=4.60e-07. QAOA-point vs ideal mixer max |Δp|=4.44e-10. X-histogram vs Pauli ⟨X⟩ max |Δ|=5.00e-15. Equal Walsh γ vs scalar γ max |Δp|=1.73e-17. Dead-degree gradient 0.00e+00, live degree-3 gradient 7.86e+00. Lanczos vs dense E_0 on n=6: 8.88e-16.

## Notes

- train_n6: 2 distinct E_0 values, majority fraction 0.71. At n=6 the cross-cavity degree-3 extraction is geometrically almost unique, so a larger B repeats two spectra rather than drawing new Hamiltonians.
- hold_n6: 2 distinct E_0 values, majority fraction 0.75. At n=6 the cross-cavity degree-3 extraction is geometrically almost unique, so a larger B repeats two spectra rather than drawing new Hamiltonians.
- The one-seed-std flag compares the median gap to the larger seed-std. One weak seed inflates that std above a median gap that the paired ratios and the instance bootstrap still see. Read it next to the seed CI of R, not instead of it.
- n=6: extracted m median 6 (min 6, max 6), degree ≤ 3.
- n=9: extracted m median 9 (min 8, max 9), degree ≤ 3.
- power_p has 4L parameters and Fourier QAOA has 2L. Walsh-degree γ has 7L. The comparison is not matched-parameter. power_p and Walsh are polished from the trained Fourier dual, so a win is the extra shells (and, if the spectrum were not pure degree 3, the extra γ weights) on top of that mixer.
- Holdout checkpoint selection uses the holdout median residual, the same rule as the smoke, applied to every head. Absolute ε is slightly optimistic. R compares heads selected the same way.
- Seed 95% CIs are percentile bootstrap of the median over the training seeds (10_000 resamples). With five seeds the interval sits on the middle order statistics. The instance CI is a separate bootstrap of one seed's B_hold residuals.
- n=6 L=16 power_p: median seed c2 rms=0.0000, c3 rms=0.0000.
- n=6 L=16 walsh: median seed c2 rms=0.0000, c3 rms=0.0000.
- n=9 L=16 power_p: median seed c2 rms=0.0141, c3 rms=0.0069.
- n=9 L=16 walsh: median seed c2 rms=0.0141, c3 rms=0.0069.

Training pass 354s. This file was regenerated from those cached runs. Branch `cursor/quantum-h-seal-a-d784`.
