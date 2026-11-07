# RESULTS — family A round 2, freeze {6,9} → n=12

**Verdict: PASS**

L=16 power_p frozen on {6,9} has R=0.924 ≤ 0.98 at n=12 against same-depth Fourier QAOA. The n=12 ground state stays quantum (peak 0.083, product gap 0.022).

## What was frozen

Angles are trained on the union of n=6 and n=9 only (equal mean of the two per-n mean residuals). The checkpoint is the best holdout score on those same two sizes. n=12 and n=15 are not in the loss and not in the checkpoint. The same tensor is then applied at the larger n.

Fourier QAOA stores 2L Zhou sine coefficients. The schedule does not depend on n. power_p stores per-layer γ, c1, c2, c3 and one exponent p (4L+1 numbers). At a new n the shell is φ_k = c1 k + a_n c2 k² + a_n c3 k³ with a_n = (8/(n−1))^p. a_9 = 1 for every p, so p is identified by n=6 versus n=9 and then extrapolated. γ and c1 are copied unchanged. The warm start is the trained Fourier dual (c2=c3=0, p=0), which is the linear Hamming shell φ_k = −β k.

ZZZ train B=32 per n, holdout B=48 per n, n=12 B=32, n=15 B=16. Seeds [0, 1, 2]. L=[2, 16]. Optimizer L-BFGS, rounds 18/24 (L=2/L=16), max_iter 8/8.

Train seeds are `beat_qaoa_12h/train_packs` (base 51e9). Holdout and transfer seeds are the scale1000 satisfiable prefix (base 54e9). The n=12 train pack is not used.

## Hamiltonian and gate

Champ, PR #13, λ=0.5:

\[
H = \sum J_{abc} Z_a Z_b Z_c \;-\; \lambda \sum_i X_i
\]

J=±1 on cross-cavity triples taken from the 4-SAT clauses, degree capped at 3. Gate sequence, unchanged:

`|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, `W=H^{⊗3}`, `U_c^X = W exp(-i γ H_c) W`.

The field is not inside U_c. ⟨H⟩ = ⟨H_c⟩ from the Z histogram after the final W, minus λ Σ ⟨X_i⟩. ε = (⟨H⟩ − E_0)/n. E_0 is the Lanczos ground energy of the sparse Hamiltonian (hypercube plus diagonal). It matches dense `eigh` at n=6 and n=9.

Walsh from the PR #11 recipe is a second arm, not a second parameterisation of the champ. On this ZZZ diagonal the Walsh mass sits in degree 3, so γ_1, γ_2, γ_4 multiply zero and graded γ collapses to scalar γ. It is not trained on the champ. The Walsh arm uses the unsat count at λ=0.75 (the PR #11 quantum point). The pulse diagonal is C_n = (n/m) unsat − max and the score is the raw unsat count. A 4-local unsat function has Walsh degree at most 4, so tying the four weights reproduces the scalar pulse up to a global phase. Untying them is a real knob: the non-constant mass is spread across degrees 1–4.

## GRAPE of e^{−i H_c}

H_c is diagonal in the Z basis, and W = H^{⊗3} is an involution, so one cost layer with the SNAP phase set to 0 is exactly exp(−i γ H_c^X). Those layers all commute:

\[
\prod_{\ell=1}^{L} \mathrm{e}^{-i \gamma_\ell H_c^{X}} = \mathrm{e}^{-i (\sum_\ell \gamma_\ell) H_c^{X}}.
\]

GRAPE with a single control along H_c therefore has a one-parameter optimum, and the locked U_c^X already applies that pulse. It is not the whole answer: the ground state is entangled (bitstring peak well below 1/2), and the Hamming-shell SNAP does not commute with H_c^X. The c2/c3 shells are extra mixer terms on top of the QAOA point φ_k=−β k, which is exp(−i β X/2) up to a global phase.

Numerical check, trained cost-only angles (φ=0) versus one layer whose γ is the sum, on the n=12 ZZZ batch: max |Δ⟨H⟩| = 3.61e-13. The L-layer cost-only circuit did not leave the one-parameter group.

| L | seed | n | cost-only Med ε | Fourier Med ε | power_p Med ε |
|--:|----:|--:|----------------:|---------------:|--------------:|
| 2 | 0 | 6 | 0.3065 | 0.0435 | 0.0435 |
| 2 | 0 | 9 | 0.3756 | 0.0971 | 0.0971 |
| 2 | 0 | 12 | 0.4200 | 0.1425 | 0.1425 |
| 2 | 0 | 15 | 0.4309 | 0.1538 | 0.1538 |
| 16 | 0 | 6 | 0.3065 | 0.0040 | 0.0026 |
| 16 | 0 | 9 | 0.3756 | 0.0174 | 0.0149 |
| 16 | 0 | 12 | 0.4200 | 0.0297 | 0.0274 |
| 16 | 0 | 15 | 0.4309 | 0.0435 | 0.0393 |

Cost-only residuals stay far above the mixer circuits, which is the same statement: the GRAPE pulse of H_c is necessary and already present, and it does not prepare the ground state.

## Quantumness

ZZZ gate, same as PR #13: median bitstring peak < 0.5 and median product gap (E_prod − E_0)/|E_0| ≥ 2%. Product states are XZ-plane, ⟨Z⟩=cos θ, ⟨X⟩=sin θ. Unsat arm: peak < 0.5 (PR #11). Entropy is the cavity-0 cut, in bits.

| split | n | family | med peak | med prod gap | med S | pass | m |
|:------|--:|:-------|---------:|-------------:|------:|:----:|--:|
| train | 6 | zzz | 0.109 | 0.073 | 1.895 | yes | 6 |
| train | 6 | unsat | 0.205 | — | 0.474 | yes | 60 |
| hold | 6 | zzz | 0.109 | 0.073 | 1.527 | yes | 6 |
| hold | 6 | unsat | 0.217 | — | 0.354 | yes | 60 |
| train | 9 | zzz | 0.087 | 0.035 | 1.981 | yes | 9 |
| train | 9 | unsat | 0.092 | — | 0.504 | yes | 89 |
| hold | 9 | zzz | 0.084 | 0.033 | 1.929 | yes | 8 |
| hold | 9 | unsat | 0.082 | — | 0.509 | yes | 89 |
| transfer | 12 | zzz | 0.083 | 0.022 | 1.957 | yes | 12 |
| transfer | 12 | unsat | 0.038 | — | 0.578 | yes | 119 |
| transfer | 15 | zzz | 0.084 | 0.016 | 1.969 | no | 15 |
| transfer | 15 | unsat | 0.014 | — | 0.630 | yes | 149 |

Lanczos versus dense eigh, worst |ΔE0| on two train instances at n=6 and n=9 (both diagonals) = 7.11e-15.

## Walsh mass (train n=6)

ZZZ H_c: degree-3 fraction 1.000000, degrees 1–4 1.000000. Graded γ is a null knob on the champ. It was not trained there.

Unsat C_n, fraction of all Walsh energy in degrees 1–4 0.122 (the rest is degree 0, the mean), degree 3 0.031, degree ≥5 0.000. Equal weights reproduce scalar γ. Free weights are the PR #11 head.

## Transfer table

R = (median over seeds of median ε) of the method, divided by the same number for Fourier QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. Lower ε is better. n=6 and n=9 are the selection holdout. n=12 and n=15 are frozen.

| family | L | method | n | role | params | Med ε | seed std | IQR | Med ⟨H⟩/n | R | mark |
|:-------|--:|:-------|--:|:-----|------:|------:|---------:|----:|----------:|--:|:-----|
| unsat | 2 | fourier_qaoa | 6 | hold | 4 | 0.0606 | 0.0000 | 0.0263 | -0.2582 |  |  |
| unsat | 2 | walsh | 6 | hold | 15 | 0.0445 | 0.0024 | 0.0188 | -0.2760 | 0.735 | BEAT |
| unsat | 2 | fourier_qaoa | 9 | hold | 4 | 0.0582 | 0.0000 | 0.0198 | -0.2644 |  |  |
| unsat | 2 | walsh | 9 | hold | 15 | 0.0440 | 0.0024 | 0.0164 | -0.2790 | 0.756 | BEAT |
| unsat | 2 | fourier_qaoa | 12 | transfer | 4 | 0.0544 | 0.0000 | 0.0188 | -0.2588 |  |  |
| unsat | 2 | walsh | 12 | transfer | 15 | 0.0452 | 0.0016 | 0.0151 | -0.2660 | 0.831 | BEAT |
| unsat | 2 | fourier_qaoa | 15 | transfer | 4 | 0.0565 | 0.0000 | 0.0124 | -0.2623 |  |  |
| unsat | 2 | walsh | 15 | transfer | 15 | 0.0434 | 0.0031 | 0.0114 | -0.2740 | 0.768 | BEAT |
| unsat | 16 | fourier_qaoa | 6 | hold | 32 | 0.0068 | 0.0045 | 0.0039 | -0.3107 |  |  |
| unsat | 16 | walsh | 6 | hold | 113 | 0.0055 | 0.0034 | 0.0042 | -0.3120 | 0.807 | BEAT |
| unsat | 16 | fourier_qaoa | 9 | hold | 32 | 0.0062 | 0.0039 | 0.0029 | -0.3171 |  |  |
| unsat | 16 | walsh | 9 | hold | 113 | 0.0049 | 0.0034 | 0.0035 | -0.3183 | 0.794 | BEAT |
| unsat | 16 | fourier_qaoa | 12 | transfer | 32 | 0.0069 | 0.0042 | 0.0044 | -0.3048 |  |  |
| unsat | 16 | walsh | 12 | transfer | 113 | 0.0061 | 0.0034 | 0.0042 | -0.3057 | 0.886 | BEAT |
| unsat | 16 | fourier_qaoa | 15 | transfer | 32 | 0.0063 | 0.0033 | 0.0059 | -0.3154 |  |  |
| unsat | 16 | walsh | 15 | transfer | 113 | 0.0053 | 0.0028 | 0.0061 | -0.3162 | 0.848 | BEAT |
| zzz | 2 | fourier_qaoa | 6 | hold | 4 | 0.0435 | 0.0000 | 0.2248 | -0.7630 |  |  |
| zzz | 2 | power_p | 6 | hold | 9 | 0.0435 | 0.0000 | 0.2245 | -0.7631 | 1.000 | TIE |
| zzz | 2 | fourier_qaoa | 9 | hold | 4 | 0.0971 | 0.0000 | 0.1097 | -0.7635 |  |  |
| zzz | 2 | power_p | 9 | hold | 9 | 0.0971 | 0.0000 | 0.1095 | -0.7636 | 0.999 | TIE |
| zzz | 2 | fourier_qaoa | 12 | transfer | 4 | 0.1425 | 0.0000 | 0.1021 | -0.7781 |  |  |
| zzz | 2 | power_p | 12 | transfer | 9 | 0.1425 | 0.0000 | 0.1020 | -0.7780 | 1.000 | TIE |
| zzz | 2 | fourier_qaoa | 15 | transfer | 4 | 0.1538 | 0.0000 | 0.1070 | -0.7716 |  |  |
| zzz | 2 | power_p | 15 | transfer | 9 | 0.1538 | 0.0000 | 0.1070 | -0.7716 | 1.000 | TIE |
| zzz | 16 | fourier_qaoa | 6 | hold | 32 | 0.0040 | 0.0053 | 0.0060 | -0.8026 |  |  |
| zzz | 16 | power_p | 6 | hold | 65 | 0.0026 | 0.0061 | 0.0035 | -0.8040 | 0.654 | BEAT |
| zzz | 16 | fourier_qaoa | 9 | hold | 32 | 0.0174 | 0.0060 | 0.0164 | -0.8582 |  |  |
| zzz | 16 | power_p | 9 | hold | 65 | 0.0149 | 0.0077 | 0.0188 | -0.8605 | 0.853 | BEAT |
| zzz | 16 | fourier_qaoa | 12 | transfer | 32 | 0.0297 | 0.0139 | 0.0524 | -0.8918 |  |  |
| zzz | 16 | power_p | 12 | transfer | 65 | 0.0274 | 0.0154 | 0.0454 | -0.8934 | 0.924 | BEAT |
| zzz | 16 | fourier_qaoa | 15 | transfer | 32 | 0.0435 | 0.0150 | 0.0750 | -0.8621 |  |  |
| zzz | 16 | power_p | 15 | transfer | 65 | 0.0393 | 0.0177 | 0.0717 | -0.8660 | 0.905 | BEAT |

Per-seed median ε, then per-seed R where a baseline exists:

- `unsat` L=2 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0606, 0.0606, 0.0606
- `unsat` L=2 walsh n=6 seeds [0, 1, 2]: 0.0445, 0.0409, 0.0454 R 0.735, 0.675, 0.750 (median of those R = 0.735)
- `unsat` L=2 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0582, 0.0582, 0.0582
- `unsat` L=2 walsh n=9 seeds [0, 1, 2]: 0.0440, 0.0401, 0.0446 R 0.756, 0.689, 0.765 (median of those R = 0.756)
- `unsat` L=2 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0544, 0.0544, 0.0544
- `unsat` L=2 walsh n=12 seeds [0, 1, 2]: 0.0452, 0.0428, 0.0459 R 0.831, 0.787, 0.843 (median of those R = 0.831)
- `unsat` L=2 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0565, 0.0565, 0.0565
- `unsat` L=2 walsh n=15 seeds [0, 1, 2]: 0.0434, 0.0384, 0.0441 R 0.768, 0.680, 0.780 (median of those R = 0.768)
- `unsat` L=16 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0068, 0.0137, 0.0053
- `unsat` L=16 walsh n=6 seeds [0, 1, 2]: 0.0055, 0.0106, 0.0042 R 0.807, 0.776, 0.782 (median of those R = 0.782)
- `unsat` L=16 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0062, 0.0119, 0.0045
- `unsat` L=16 walsh n=9 seeds [0, 1, 2]: 0.0049, 0.0103, 0.0039 R 0.794, 0.863, 0.878 (median of those R = 0.863)
- `unsat` L=16 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0069, 0.0133, 0.0055
- `unsat` L=16 walsh n=12 seeds [0, 1, 2]: 0.0061, 0.0113, 0.0048 R 0.886, 0.844, 0.882 (median of those R = 0.882)
- `unsat` L=16 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0063, 0.0113, 0.0050
- `unsat` L=16 walsh n=15 seeds [0, 1, 2]: 0.0053, 0.0096, 0.0043 R 0.848, 0.847, 0.855 (median of those R = 0.848)
- `zzz` L=2 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0435, 0.0435, 0.0435
- `zzz` L=2 power_p n=6 seeds [0, 1, 2]: 0.0435, 0.0435, 0.0435 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- `zzz` L=2 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0971, 0.0971, 0.0971
- `zzz` L=2 power_p n=9 seeds [0, 1, 2]: 0.0971, 0.0971, 0.0971 R 0.999, 0.999, 0.999 (median of those R = 0.999)
- `zzz` L=2 fourier_qaoa n=12 seeds [0, 1, 2]: 0.1425, 0.1425, 0.1425
- `zzz` L=2 power_p n=12 seeds [0, 1, 2]: 0.1425, 0.1425, 0.1425 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- `zzz` L=2 fourier_qaoa n=15 seeds [0, 1, 2]: 0.1538, 0.1538, 0.1538
- `zzz` L=2 power_p n=15 seeds [0, 1, 2]: 0.1538, 0.1538, 0.1538 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- `zzz` L=16 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0040, 0.0131, 0.0039
- `zzz` L=16 power_p n=6 seeds [0, 1, 2]: 0.0026, 0.0131, 0.0026 R 0.646, 1.000, 0.662 (median of those R = 0.662)
- `zzz` L=16 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0174, 0.0276, 0.0171
- `zzz` L=16 power_p n=9 seeds [0, 1, 2]: 0.0149, 0.0276, 0.0139 R 0.853, 1.000, 0.811 (median of those R = 0.853)
- `zzz` L=16 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0297, 0.0536, 0.0295
- `zzz` L=16 power_p n=12 seeds [0, 1, 2]: 0.0274, 0.0536, 0.0264 R 0.924, 1.000, 0.893 (median of those R = 0.924)
- `zzz` L=16 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0435, 0.0695, 0.0435
- `zzz` L=16 power_p n=15 seeds [0, 1, 2]: 0.0393, 0.0695, 0.0385 R 0.905, 1.000, 0.885 (median of those R = 0.905)

## Kill / pass rule

PASS if L=16 power_p on ZZZ has transfer R≤0.98 at n=12, the n=12 GS still passes peak<0.5 and product gap≥2%, n=15 does not flip the advantage through R≥1.10, and the per-seed ratios agree. KILL if the n=12 GS is a bitstring, or the n=6→n=9 advantage flips and R(n=12)≥1.05 (or R≥1.15). Otherwise needs_edit. E0 is exact, so a Med ⟨H⟩ win is the same ordering as ε; the 0.98 bar is the residual ratio.

- n=12 classical GS: no
- n=12 product-gap soft fail: no
- advantage reverses at n=12: no
- n=15 flips a n=12 win: no
- seed aggregate disagrees with the 0.98 line: no

L=16 ZZZ power_p: n6 R=0.654 adv ε=0.0014 (BEAT); n9 R=0.853 adv ε=0.0026 (BEAT); n12 R=0.924 adv ε=0.0023 (BEAT); n15 R=0.905 adv ε=0.0041 (BEAT).

The residual ratio softens from n=6 to n=12 (0.654 → 0.853 → 0.924) and then holds at n=15 (0.905). The advantage stays positive, so this is not a reversal. Seed 1 never left the Fourier warm start, so its R is 1.000 at every n. Seeds 0 and 2 are the beat (n=12 R = 0.924 and 0.893). The median of those three ratios is 0.924, the same side of 0.98 as the ratio of medians. Instance IQR of ε at n=12 is about 0.05, larger than the median gap 0.002, so the win is a shift of the median inside overlapping clouds.

n=15 still beats, and the bitstring peak is 0.084, but the product gap is 0.016, under the 2% gate. The PASS bar is n=12, where that gap is 0.022. n=15 is the extra size, not a second gate.

L=2 ZZZ power_p at n=12: R=1.000 (TIE). The success bar is the L=16 champ depth. L=2 is reported, not required.

Unsat Walsh L=16 at n=12: R=0.886 (BEAT). This arm does not change the ZZZ verdict. It is the PR #11 head on a different diagonal.

## Notes

- power_p has 4L+1 parameters and Fourier QAOA has 2L. The comparison asks whether the c2/c3 shells, scaled by a_n, beat the linear mixer they contain.
- n=6 and n=9 medians are best-by-hold on the reported holdout, so they are slightly optimistic. n=12 and n=15 are not.
- Med ⟨H⟩/n and ε agree on which method is lower. They are not the same number: ε is the median of (⟨H⟩−E_0)/n, while the ⟨H⟩/n column is the median energy divided by n. The success bar is the residual ratio. At n=12 L=16, power_p is lower on both (⟨H⟩/n −0.8934 vs −0.8918).
- Degree cap 3: a few larger-n ZZZ instances stop short of m=n instead of raising the degree.
- `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n21 ladder were not written or signalled.

Self-check: Fock vs Nov3rdAnsatz max |Δp| Z=2.07e-08, X=4.60e-07; QAOA-point max |Δp|=4.44e-10; X-hist vs Pauli 5.00e-15.

Wall clock 115.8s. Branch `cursor/arm-a-transfer-r2-6b03`.
