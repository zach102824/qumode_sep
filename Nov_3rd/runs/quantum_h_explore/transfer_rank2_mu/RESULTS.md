# RESULTS — Rank2 μ=−0.25 freeze {6,9} → n=12

**Verdict: PASS**

Frozen n=12 R=0.838 ≤ 0.98 and clearly below the μ=0 transfer (in-run 0.929, published 0.924).

## Setup

Same joint freeze as PR #15 (family A transfer), on the Rank2 deformation of PR #16 with κ = 0.

\[
H = \sum J_{abc} Z_a Z_b Z_c - \lambda \sum_i X_i + \mu \sum_{\mathrm{cav}} \sum_{i<j} X_i X_j
\]

λ = 0.5, κ = 0, μ ∈ {−0.25, 0}. μ = 0 is Arm A, retrained in this run on the same instances and the same L-BFGS budget. The published μ=0 transfer is PR #15, n=12 L=16 R = 0.924.

Angles are trained on the union of n=6 and n=9 only (equal mean of the two per-n mean residuals ε). The checkpoint is the best holdout score on those two sizes. n=12 and n=15 are not in the loss and not in the checkpoint. Fourier QAOA stores 2L Zhou sine coefficients. power_p stores per-layer γ, c1, c2, c3 and one exponent p (4L+1 numbers), warm-started from that seed's trained Fourier dual (c2=c3=0, p=0). At a new n the shell is φ_k = c1 k + a_n c2 k² + a_n c3 k³ with a_n = (8/(n−1))^p. a_9 = 1 for every p.

Gate sequence, unchanged: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, `W=H^{⊗3}`, `U_c^X = W exp(−iγ H_c) W`, `H_c = Σ J ZZZ` only. The field and the intra-cavity XX are not inside U_c. ⟨H⟩ uses the Z histogram for ⟨H_c⟩ and the per-cavity 8-outcome X histogram for ⟨X_i⟩ and ⟨X_i X_j⟩. needs YY: no.

Train B=32 per n, holdout B=48 per n, n=12 B=32, n=15 B=16. Seeds [0, 1, 2]. L=[16, 2]. L-BFGS rounds 18/24 (L=2/L=16), max_iter 8/8, lr=0.35. Train packs are `beat_qaoa_12h/train_packs` (base 51e9). Holdout and transfer use the scale1000 satisfiable prefix (base 54e9).

Pre-registered bar, fixed in `decide_rank2_transfer` before this run was scored. PASS if L=16 power_p at μ=−0.25 has frozen n=12 R≤0.98 against same-depth Fourier QAOA, the n=12 ground state passes peak<0.5 and product gap≥2%, the ratio is clearly below the in-run μ=0 transfer, and the ratio is ≤ the published μ=0 transfer 0.924. Clearly means the ratio-of-medians drops by ≥0.02 and the median paired drop on seeds whose μ=0 per-seed R is below 0.995 is ≥0.02 and at least that drop's std. A seed stuck at the Fourier point does not count and does not veto. The in-run μ=0 control must itself beat QAOA and land within 0.01 of 0.924. n=15 flipping through R≥1.10 downgrades a pass to needs_edit. KILL if the n=12 ground state is a bitstring, or n=12 fails to beat QAOA (R>1.02, or R≥1.05) and does not improve on μ=0. Anything else, including a beat that is not clearly below μ=0, is needs_edit.

## Quantumness

Gate, same as Arm A: median bitstring peak < 0.5 and median product gap (E_prod − E_0)/|E_0| ≥ 2%. Product states are XZ-plane. Entropy is the cavity-0 cut, in bits. E0 is Lanczos on the sparse rank-2 Hamiltonian.

| μ | split | n | med peak | med prod gap | med S | med E0/n | pass | B |
|--:|:------|--:|---------:|-------------:|------:|---------:|:----:|--:|
| -0.25 | hold | 6 | 0.071 | 0.178 | 1.139 | -0.956 | yes | 48 |
| -0.25 | train | 6 | 0.071 | 0.176 | 1.326 | -0.956 | yes | 32 |
| -0.25 | hold | 9 | 0.044 | 0.099 | 1.477 | -0.973 | yes | 48 |
| -0.25 | train | 9 | 0.042 | 0.106 | 1.554 | -0.973 | yes | 32 |
| -0.25 | transfer | 12 | 0.035 | 0.069 | 1.726 | -0.992 | yes | 32 |
| -0.25 | transfer | 15 | 0.038 | 0.052 | 1.700 | -0.985 | yes | 16 |
| +0.00 | hold | 6 | 0.109 | 0.073 | 1.527 | -0.807 | yes | 48 |
| +0.00 | train | 6 | 0.109 | 0.073 | 1.895 | -0.807 | yes | 32 |
| +0.00 | hold | 9 | 0.084 | 0.032 | 1.929 | -0.876 | yes | 48 |
| +0.00 | train | 9 | 0.087 | 0.035 | 1.981 | -0.877 | yes | 32 |
| +0.00 | transfer | 12 | 0.083 | 0.022 | 1.957 | -0.920 | yes | 32 |
| +0.00 | transfer | 15 | 0.084 | 0.013 | 1.969 | -0.931 | no | 16 |

Lanczos versus dense eigh, worst |ΔE0| on two train instances at n=6 and n=9 = 1.07e-14. At μ=0 the same rows match Arm A's dense and sparse ground energies to that tolerance.

## Transfer

ε = (⟨H⟩ − E_0)/n. Med ε is the median over seeds of the per-seed median. R = Med ε_power / Med ε_QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. Lower ε is better. n=6 and n=9 are the selection holdout. n=12 and n=15 are frozen.

| μ | L | method | n | role | params | Med ε | seed std | IQR | Med ⟨H⟩/n | R | mark |
|--:|--:|:-------|--:|:-----|------:|------:|---------:|----:|----------:|--:|:-----|
| -0.25 | 2 | fourier_qaoa | 6 | hold | 4 | 0.0254 | 0.0026 | 0.1195 | -0.9301 |  |  |
| -0.25 | 2 | power_p | 6 | hold | 9 | 0.0254 | 0.0026 | 0.1195 | -0.9301 | 1.000 | TIE |
| -0.25 | 2 | fourier_qaoa | 9 | hold | 4 | 0.0432 | 0.0020 | 0.0578 | -0.9245 |  |  |
| -0.25 | 2 | power_p | 9 | hold | 9 | 0.0432 | 0.0020 | 0.0578 | -0.9245 | 1.000 | TIE |
| -0.25 | 2 | fourier_qaoa | 12 | transfer | 4 | 0.0605 | 0.0030 | 0.0606 | -0.9256 |  |  |
| -0.25 | 2 | power_p | 12 | transfer | 9 | 0.0605 | 0.0030 | 0.0606 | -0.9256 | 1.000 | TIE |
| -0.25 | 2 | fourier_qaoa | 15 | transfer | 4 | 0.0587 | 0.0045 | 0.0534 | -0.9177 |  |  |
| -0.25 | 2 | power_p | 15 | transfer | 9 | 0.0587 | 0.0045 | 0.0534 | -0.9177 | 1.000 | TIE |
| -0.25 | 16 | fourier_qaoa | 6 | hold | 32 | 0.0046 | 0.0013 | 0.0857 | -0.9509 |  |  |
| -0.25 | 16 | power_p | 6 | hold | 65 | 0.0034 | 0.0010 | 0.0705 | -0.9522 | 0.723 | BEAT |
| -0.25 | 16 | fourier_qaoa | 9 | hold | 32 | 0.0171 | 0.0015 | 0.0438 | -0.9524 |  |  |
| -0.25 | 16 | power_p | 9 | hold | 65 | 0.0136 | 0.0007 | 0.0383 | -0.9566 | 0.792 | BEAT |
| -0.25 | 16 | fourier_qaoa | 12 | transfer | 32 | 0.0310 | 0.0020 | 0.0605 | -0.9585 |  |  |
| -0.25 | 16 | power_p | 12 | transfer | 65 | 0.0260 | 0.0015 | 0.0590 | -0.9638 | 0.838 | BEAT |
| -0.25 | 16 | fourier_qaoa | 15 | transfer | 32 | 0.0326 | 0.0021 | 0.0522 | -0.9415 |  |  |
| -0.25 | 16 | power_p | 15 | transfer | 65 | 0.0284 | 0.0010 | 0.0521 | -0.9441 | 0.870 | BEAT |
| +0.00 | 2 | fourier_qaoa | 6 | hold | 4 | 0.0435 | 0.0000 | 0.2248 | -0.7630 |  |  |
| +0.00 | 2 | power_p | 6 | hold | 9 | 0.0435 | 0.0000 | 0.2245 | -0.7631 | 1.000 | TIE |
| +0.00 | 2 | fourier_qaoa | 9 | hold | 4 | 0.0971 | 0.0000 | 0.1097 | -0.7635 |  |  |
| +0.00 | 2 | power_p | 9 | hold | 9 | 0.0971 | 0.0000 | 0.1095 | -0.7636 | 0.999 | TIE |
| +0.00 | 2 | fourier_qaoa | 12 | transfer | 4 | 0.1425 | 0.0000 | 0.1021 | -0.7781 |  |  |
| +0.00 | 2 | power_p | 12 | transfer | 9 | 0.1425 | 0.0000 | 0.1020 | -0.7780 | 1.000 | TIE |
| +0.00 | 2 | fourier_qaoa | 15 | transfer | 4 | 0.1538 | 0.0000 | 0.1070 | -0.7716 |  |  |
| +0.00 | 2 | power_p | 15 | transfer | 9 | 0.1538 | 0.0000 | 0.1070 | -0.7716 | 1.000 | TIE |
| +0.00 | 16 | fourier_qaoa | 6 | hold | 32 | 0.0040 | 0.0053 | 0.0060 | -0.8026 |  |  |
| +0.00 | 16 | power_p | 6 | hold | 65 | 0.0026 | 0.0061 | 0.0034 | -0.8040 | 0.656 | BEAT |
| +0.00 | 16 | fourier_qaoa | 9 | hold | 32 | 0.0174 | 0.0060 | 0.0164 | -0.8582 |  |  |
| +0.00 | 16 | power_p | 9 | hold | 65 | 0.0149 | 0.0076 | 0.0188 | -0.8604 | 0.855 | BEAT |
| +0.00 | 16 | fourier_qaoa | 12 | transfer | 32 | 0.0297 | 0.0139 | 0.0524 | -0.8918 |  |  |
| +0.00 | 16 | power_p | 12 | transfer | 65 | 0.0276 | 0.0153 | 0.0456 | -0.8932 | 0.929 | BEAT |
| +0.00 | 16 | fourier_qaoa | 15 | transfer | 32 | 0.0435 | 0.0150 | 0.0750 | -0.8621 |  |  |
| +0.00 | 16 | power_p | 15 | transfer | 65 | 0.0395 | 0.0176 | 0.0715 | -0.8658 | 0.909 | BEAT |

Per-seed median ε, then per-seed R where a Fourier baseline exists:

- μ=-0.25 L=2 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0254, 0.0255, 0.0209
- μ=-0.25 L=2 power_p n=6 seeds [0, 1, 2]: 0.0254, 0.0255, 0.0209 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- μ=-0.25 L=2 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0432, 0.0432, 0.0398
- μ=-0.25 L=2 power_p n=9 seeds [0, 1, 2]: 0.0432, 0.0432, 0.0398 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- μ=-0.25 L=2 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0605, 0.0612, 0.0557
- μ=-0.25 L=2 power_p n=12 seeds [0, 1, 2]: 0.0605, 0.0612, 0.0557 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- μ=-0.25 L=2 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0587, 0.0595, 0.0514
- μ=-0.25 L=2 power_p n=15 seeds [0, 1, 2]: 0.0587, 0.0595, 0.0514 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- μ=-0.25 L=16 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0031, 0.0046, 0.0057
- μ=-0.25 L=16 power_p n=6 seeds [0, 1, 2]: 0.0022, 0.0034, 0.0041 R 0.705, 0.723, 0.714 (median of those R = 0.714)
- μ=-0.25 L=16 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0187, 0.0171, 0.0157
- μ=-0.25 L=16 power_p n=9 seeds [0, 1, 2]: 0.0136, 0.0137, 0.0124 R 0.726, 0.798, 0.790 (median of those R = 0.790)
- μ=-0.25 L=16 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0315, 0.0310, 0.0279
- μ=-0.25 L=16 power_p n=12 seeds [0, 1, 2]: 0.0260, 0.0260, 0.0234 R 0.824, 0.840, 0.840 (median of those R = 0.840)
- μ=-0.25 L=16 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0340, 0.0326, 0.0298
- μ=-0.25 L=16 power_p n=15 seeds [0, 1, 2]: 0.0285, 0.0284, 0.0268 R 0.839, 0.870, 0.899 (median of those R = 0.870)
- μ=+0.00 L=2 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0435, 0.0435, 0.0435
- μ=+0.00 L=2 power_p n=6 seeds [0, 1, 2]: 0.0435, 0.0435, 0.0435 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- μ=+0.00 L=2 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0971, 0.0971, 0.0971
- μ=+0.00 L=2 power_p n=9 seeds [0, 1, 2]: 0.0971, 0.0971, 0.0971 R 0.999, 0.999, 0.999 (median of those R = 0.999)
- μ=+0.00 L=2 fourier_qaoa n=12 seeds [0, 1, 2]: 0.1425, 0.1425, 0.1425
- μ=+0.00 L=2 power_p n=12 seeds [0, 1, 2]: 0.1425, 0.1425, 0.1425 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- μ=+0.00 L=2 fourier_qaoa n=15 seeds [0, 1, 2]: 0.1538, 0.1538, 0.1538
- μ=+0.00 L=2 power_p n=15 seeds [0, 1, 2]: 0.1538, 0.1538, 0.1538 R 1.000, 1.000, 1.000 (median of those R = 1.000)
- μ=+0.00 L=16 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0040, 0.0131, 0.0039
- μ=+0.00 L=16 power_p n=6 seeds [0, 1, 2]: 0.0026, 0.0131, 0.0026 R 0.649, 1.000, 0.664 (median of those R = 0.664)
- μ=+0.00 L=16 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0174, 0.0276, 0.0171
- μ=+0.00 L=16 power_p n=9 seeds [0, 1, 2]: 0.0149, 0.0276, 0.0141 R 0.855, 1.000, 0.820 (median of those R = 0.855)
- μ=+0.00 L=16 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0297, 0.0536, 0.0295
- μ=+0.00 L=16 power_p n=12 seeds [0, 1, 2]: 0.0276, 0.0536, 0.0266 R 0.929, 1.000, 0.900 (median of those R = 0.929)
- μ=+0.00 L=16 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0435, 0.0695, 0.0435
- μ=+0.00 L=16 power_p n=15 seeds [0, 1, 2]: 0.0395, 0.0695, 0.0385 R 0.909, 1.000, 0.886 (median of those R = 0.909)

## vs μ=0

Published PR #15 μ=0 n=12 L=16 R = 0.924. In-run μ=0 R = 0.929 (control match: yes, tolerance 0.01). μ=−0.25 R = 0.838. Drop versus in-run μ=0 = 0.091. Drop versus published = 0.086.

Paired per-seed ΔR (μ=0 − μ=−0.25) = 0.105, 0.160, 0.060. Active seeds (μ=0 per-seed R < 0.995): 2. Active median drop = 0.083, std = 0.031. Clear: yes.

PR #15 μ=0 L=16 ratio-of-medians, for the same freeze: n=6 R=0.654, n=9 R=0.853, n=12 R=0.924, n=15 R=0.905. Fourier n=12 seeds there were 0.0297, 0.0536, 0.0295 and power_p was 0.0274, 0.0536, 0.0264. This run's μ=0 Fourier n=12 seeds are 0.0297, 0.0536, 0.0295 and match those three values. power_p n=12 is 0.0276, 0.0536, 0.0266 (published 0.0274, 0.0536, 0.0264). Seed 1 is the warm start in both runs when its ratio is 1. The 0.924 → in-run control drift is inside the 0.01 window, and the μ=−0.25 drop on the ratio-of-medians is much larger than that drift. At μ=−0.25 every n=12 seed beats its own Fourier run (per-seed R 0.824, 0.840, 0.840), including the seed that stays at R=1.000 when μ=0.

n=15 μ=0 product gap here is 0.013 against PR #15's 0.016. Peak and entropy match PR #15, so the ground state is the same one; the product search used here reached a slightly lower product energy. The n=12 μ=0 gap is 0.022 (PR #15 reported 0.022). n=15 is not the pass gate. At μ=−0.25 the frozen n=15 L=16 ratio is 0.870 (BEAT).

## Rule

- beat QAOA (R≤0.98): yes
- n=12 quantum: yes
- clearly below in-run μ=0: yes
- ≤ published 0.924: yes
- μ=0 control reproduced: yes
- n=15 flips past 1.10: no

Verdict: **PASS**.

## Notes

- Cost-only layers still commute. On the n=12 μ=−0.25 batch, L cost-only layers versus one pulse with Γ=Σγ differ by |Δ⟨H⟩| = 5.68e-14. The XX term is outside U_c, scored from the X histogram, so GRAPE of e^{−i H_c} is the same one-parameter pulse as PR #15 and was not retrained.
- power_p has 4L+1 parameters and Fourier QAOA has 2L. The comparison asks whether the c2/c3 shells, scaled by a_n, beat the linear mixer they contain.
- n=6 and n=9 medians are best-by-hold on the reported holdout, so they are slightly optimistic. n=12 and n=15 are not.
- PR #16's n=9 L=16 R=0.498 was per-size training at μ=−0.25, not this joint freeze. The joint holdout here is softer (n=9 L=16 R=0.792, BEAT). The pass gate is the frozen n=12 ratio, not that in-distribution number.
- `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n21 ladder were not written. n=21 instances were not loaded.

Self-check: rank2 statevector 1e-08, μ=0 energy |Δ|=0.00e+00, grad |Δ|=0.00e+00. needs_YY=no.

Wall clock 95.2s. Branch `cursor/rank2-mu-transfer-0f9a`.
