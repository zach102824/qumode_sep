# RESULTS — Gaussian J × Rank2 μ=−0.25, freeze {6,9} → n=12

**Verdict: PASS**

Frozen n=12 R=0.642 ≤ 0.98 and clearly below both parents (gauss μ=0 0.827, ±1 μ=−0.25 0.838) and the in-run gauss control 0.883, which does not reproduce the PR #20 freeze 0.827.

## Setup

Seal A round 3 packing (PR #20) with the Rank-2 XX deformation (PR #18), κ = 0.

\[
H = \sum J_{abc} Z_a Z_b Z_c - \lambda \sum_i X_i + \mu \sum_{\mathrm{cav}} \sum_{i<j} X_i X_j
\]

J is an independent standard normal on the degree-3 cross-cavity support (salt 91), not the first-occurrence sign. λ = 0.5. μ ∈ {−0.25, 0}. μ = 0 is the Gaussian control, retrained on the same instances and the same L-BFGS budget. Published parents, frozen n=12 L=16 power_p: gauss μ=0 R = 0.827 (PR #20), ±1 μ=−0.25 R = 0.838 (PR #18).

Angles train on the union of n=6 and n=9 only. The checkpoint minimises the mean of the two holdout median residuals ε, including the warm start. That is the PR #20 freeze rule. n=12 and n=15 are not in the loss and not in the checkpoint. Fourier QAOA stores 2L Zhou sine coefficients. power_p stores per-layer γ, c1, c2, c3 and one exponent p (4L+1 numbers), warm-started from that seed's trained Fourier dual (c2=c3=0, p=0). At a new n the shell is φ_k = c1 k + a_n c2 k² + a_n c3 k³ with a_n = (8/(n−1))^p. a_9 = 1 for every p.

Gate sequence, unchanged: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, `W=H^{⊗3}`, `U_c^X = W exp(−iγ H_c) W`, `H_c = Σ J ZZZ` only. The field and the intra-cavity XX are not inside U_c. ⟨H⟩ uses the Z histogram for ⟨H_c⟩ and the per-cavity 8-outcome X histogram for ⟨X_i⟩ and ⟨X_i X_j⟩. needs YY: no.

Train B=128 per n, holdout B=256 per n, n=12 B=64, n=15 B=16. Seeds [0, 1, 2]. L=[16]. L-BFGS rounds 24 at L=16, max_iter 8, lr=0.35. Train packs are `beat_qaoa_12h/train_packs` (base 51e9). Holdout and transfer use the scale1000 satisfiable prefix (base 54e9).

Bar in `decide_gauss_xx`. PASS if L=16 power_p at μ=−0.25 has frozen n=12 R≤0.98 against same-depth Fourier QAOA, the n=12 ground state passes peak<0.5 and product gap≥2%, the ratio is clearly below both published parents (drop ≥ 0.02 versus 0.827 and versus 0.838), and the ratio is clearly below the in-run Gaussian μ=0 control. Clearly versus the control means the ratio-of-medians drops by ≥0.02 and the median paired drop on seeds whose control per-seed R is below 0.995 is ≥0.02 and at least that drop's std. The in-run control must itself beat QAOA. Whether it also lands within 0.02 of the PR #20 freeze is reported and does not veto a ratio that is already clearly under 0.827. n=15 flipping through R≥1.10 downgrades a pass to needs_edit. A beat of QAOA that is not clearly under both parents is not synergistic and is needs_edit. KILL if the n=12 ground state is a bitstring, or n=12 fails to beat QAOA (R>1.02) and does not improve on the Gaussian control. Anything else is needs_edit.

## Packing

n=6 train: J std 1.068, median |J| 0.748, distinct μ=0 E0 128/128, degree ≤ 3, cross-cavity yes.

## Quantumness

Gate: median bitstring peak < 0.5 and median product gap (E_prod − E_0)/|E_0| ≥ 2%. Product states are XZ-plane. Entropy is the cavity-0 cut, in bits. E0 is Lanczos on the sparse rank-2 Hamiltonian. At μ=0 that matrix is the Gaussian transverse-field Hamiltonian.

| μ | split | n | med peak | med prod gap | med S | med E0/n | pass | B |
|--:|:------|--:|---------:|-------------:|------:|---------:|:----:|--:|
| -0.25 | hold | 6 | 0.086 | 0.116 | 0.961 | -0.934 | yes | 256 |
| -0.25 | train | 6 | 0.103 | 0.115 | 1.025 | -0.979 | yes | 128 |
| -0.25 | hold | 9 | 0.031 | 0.096 | 1.204 | -0.933 | yes | 256 |
| -0.25 | train | 9 | 0.031 | 0.097 | 1.133 | -0.921 | yes | 128 |
| -0.25 | transfer | 12 | 0.015 | 0.083 | 1.297 | -0.954 | yes | 64 |
| -0.25 | transfer | 15 | 0.014 | 0.066 | 1.502 | -0.952 | yes | 16 |
| +0.00 | hold | 6 | 0.155 | 0.044 | 1.226 | -0.801 | yes | 256 |
| +0.00 | train | 6 | 0.174 | 0.033 | 1.285 | -0.863 | yes | 128 |
| +0.00 | hold | 9 | 0.085 | 0.028 | 1.772 | -0.818 | yes | 256 |
| +0.00 | train | 9 | 0.088 | 0.031 | 1.616 | -0.792 | yes | 128 |
| +0.00 | transfer | 12 | 0.073 | 0.019 | 1.825 | -0.849 | no | 64 |
| +0.00 | transfer | 15 | 0.090 | 0.012 | 1.929 | -0.879 | no | 16 |

Lanczos versus dense eigh, worst |ΔE0| on the audit rows = 6.22e-15.

## Transfer

ε = (⟨H⟩ − E_0)/n. Med ε is the median over seeds of the per-seed median. R = Med ε_power / Med ε_QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. Lower ε is better. n=6 and n=9 are the selection holdout. n=12 and n=15 are frozen.

| μ | L | method | n | role | params | Med ε | seed std | IQR | Med ⟨H⟩/n | R | mark |
|--:|--:|:-------|--:|:-----|------:|------:|---------:|----:|----------:|--:|:-----|
| -0.25 | 16 | fourier_qaoa | 6 | hold | 32 | 0.0087 | 0.0017 | 0.0181 | -0.9194 |  |  |
| -0.25 | 16 | power_p | 6 | hold | 65 | 0.0046 | 0.0007 | 0.0145 | -0.9226 | 0.532 | BEAT |
| -0.25 | 16 | fourier_qaoa | 9 | hold | 32 | 0.0107 | 0.0019 | 0.0218 | -0.9236 |  |  |
| -0.25 | 16 | power_p | 9 | hold | 65 | 0.0064 | 0.0012 | 0.0165 | -0.9271 | 0.600 | BEAT |
| -0.25 | 16 | fourier_qaoa | 12 | transfer | 32 | 0.0131 | 0.0024 | 0.0295 | -0.9326 |  |  |
| -0.25 | 16 | power_p | 12 | transfer | 65 | 0.0084 | 0.0013 | 0.0231 | -0.9377 | 0.642 | BEAT |
| -0.25 | 16 | fourier_qaoa | 15 | transfer | 32 | 0.0119 | 0.0018 | 0.0194 | -0.9384 |  |  |
| -0.25 | 16 | power_p | 15 | transfer | 65 | 0.0080 | 0.0011 | 0.0169 | -0.9427 | 0.669 | BEAT |
| +0.00 | 16 | fourier_qaoa | 6 | hold | 32 | 0.0113 | 0.0038 | 0.0201 | -0.7790 |  |  |
| +0.00 | 16 | power_p | 6 | hold | 65 | 0.0090 | 0.0039 | 0.0159 | -0.7832 | 0.794 | BEAT |
| +0.00 | 16 | fourier_qaoa | 9 | hold | 32 | 0.0169 | 0.0055 | 0.0268 | -0.8002 |  |  |
| +0.00 | 16 | power_p | 9 | hold | 65 | 0.0146 | 0.0045 | 0.0228 | -0.8030 | 0.865 | BEAT |
| +0.00 | 16 | fourier_qaoa | 12 | transfer | 32 | 0.0247 | 0.0111 | 0.0285 | -0.8324 |  |  |
| +0.00 | 16 | power_p | 12 | transfer | 65 | 0.0218 | 0.0079 | 0.0243 | -0.8326 | 0.883 | BEAT |
| +0.00 | 16 | fourier_qaoa | 15 | transfer | 32 | 0.0349 | 0.0124 | 0.0330 | -0.8324 |  |  |
| +0.00 | 16 | power_p | 15 | transfer | 65 | 0.0314 | 0.0104 | 0.0287 | -0.8341 | 0.900 | BEAT |

Per-seed median ε, then per-seed R where a Fourier baseline exists:

- μ=-0.25 L=16 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0093, 0.0087, 0.0061
- μ=-0.25 L=16 power_p n=6 seeds [0, 1, 2]: 0.0047, 0.0046, 0.0035 R 0.502, 0.532, 0.579 (median of those R = 0.532)
- μ=-0.25 L=16 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0116, 0.0107, 0.0080
- μ=-0.25 L=16 power_p n=9 seeds [0, 1, 2]: 0.0064, 0.0067, 0.0044 R 0.556, 0.621, 0.552 (median of those R = 0.556)
- μ=-0.25 L=16 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0143, 0.0131, 0.0096
- μ=-0.25 L=16 power_p n=12 seeds [0, 1, 2]: 0.0084, 0.0088, 0.0064 R 0.588, 0.669, 0.668 (median of those R = 0.668)
- μ=-0.25 L=16 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0126, 0.0119, 0.0093
- μ=-0.25 L=16 power_p n=15 seeds [0, 1, 2]: 0.0080, 0.0082, 0.0062 R 0.632, 0.686, 0.664 (median of those R = 0.664)
- μ=+0.00 L=16 fourier_qaoa n=6 seeds [0, 1, 2]: 0.0113, 0.0175, 0.0106
- μ=+0.00 L=16 power_p n=6 seeds [0, 1, 2]: 0.0090, 0.0155, 0.0086 R 0.794, 0.888, 0.813 (median of those R = 0.813)
- μ=+0.00 L=16 fourier_qaoa n=9 seeds [0, 1, 2]: 0.0169, 0.0258, 0.0159
- μ=+0.00 L=16 power_p n=9 seeds [0, 1, 2]: 0.0146, 0.0222, 0.0142 R 0.865, 0.857, 0.893 (median of those R = 0.865)
- μ=+0.00 L=16 fourier_qaoa n=12 seeds [0, 1, 2]: 0.0247, 0.0433, 0.0234
- μ=+0.00 L=16 power_p n=12 seeds [0, 1, 2]: 0.0218, 0.0347, 0.0203 R 0.883, 0.801, 0.866 (median of those R = 0.866)
- μ=+0.00 L=16 fourier_qaoa n=15 seeds [0, 1, 2]: 0.0349, 0.0555, 0.0333
- μ=+0.00 L=16 power_p n=15 seeds [0, 1, 2]: 0.0314, 0.0485, 0.0297 R 0.900, 0.875, 0.892 (median of those R = 0.892)

## vs both parents

Published gauss μ=0 n=12 L=16 R = 0.827. Published ±1 μ=−0.25 n=12 L=16 R = 0.838. In-run gauss μ=0 R = 0.883 (control match: no, tolerance 0.02). μ=−0.25 R = 0.642. Drop versus gauss parent = 0.185. Drop versus ±1 μ=−0.25 parent = 0.196. Drop versus in-run control = 0.241.

Paired per-seed ΔR (in-run μ=0 − μ=−0.25) = 0.295, 0.132, 0.198. Active seeds (μ=0 per-seed R < 0.995): 3. Active median drop = 0.198, std = 0.082. Clear versus the control: yes.

Synergistic (clearly under both published parents and under the in-run control): yes.

Hamming shells left the Fourier dual (c2 = c3 = 0, p = 0) when the rms is above 0:

- μ=-0.25 L=16 seed=0: c2 rms=0.0260, c3 rms=0.0142, p=-0.0002
- μ=-0.25 L=16 seed=1: c2 rms=0.0181, c3 rms=0.0112, p=-0.0001
- μ=-0.25 L=16 seed=2: c2 rms=0.0163, c3 rms=0.0104, p=0.0014
- μ=+0.00 L=16 seed=0: c2 rms=0.0179, c3 rms=0.0079, p=0.0020
- μ=+0.00 L=16 seed=1: c2 rms=0.0206, c3 rms=0.0049, p=0.0015
- μ=+0.00 L=16 seed=2: c2 rms=0.0183, c3 rms=0.0041, p=0.0022

## Rule

- beat QAOA (R≤0.98): yes
- n=12 quantum: yes
- clearly below gauss μ=0 parent 0.827: yes
- clearly below ±1 μ=−0.25 parent 0.838: yes
- clearly below in-run gauss control: yes
- gauss control reproduced: no
- gauss control beats QAOA: yes
- n=15 flips past 1.10: no (R=0.669, BEAT)

Verdict: **PASS**.

## Notes

- Cost-only layers still commute. On the n=12 μ=−0.25 batch, L cost-only layers versus one pulse with Γ=Σγ differ by |Δ⟨H⟩| = 6.04e-14. The XX term is outside U_c, scored from the X histogram, so the ZZZ pulse is the same one-parameter family as the Gaussian seal.
- power_p has 4L+1 parameters and Fourier QAOA has 2L. The comparison asks whether the c2/c3 shells, scaled by a_n, beat the linear mixer they contain.
- n=6 and n=9 medians are best-by-hold on the reported holdout, so they are slightly optimistic. n=12 and n=15 are not.
- The Gaussian parent (PR #20) froze the same packing at μ=0 with this checkpoint rule, B_train=128, B_hold=256, n=12 B=64, seeds 0–2, and reported R=0.827. The ±1 parent (PR #18) used J=±1, B_train=32, and a ratio-to-Fourier checkpoint, and reported R=0.838. Those two numbers are the parents. This run's μ=0 arm is the matched Gaussian control.
- Fourier at μ=0 reproduces that PR #20 freeze: n=12 per-seed medians are 0.0247, 0.0433, 0.0234 and the published median is 0.0247. power_p n=12 here is 0.0218, 0.0347, 0.0203 against PR #20's 0.0201, 0.0347, 0.0204. Seed 1 matches. Seed 0 is still descending at round 24 (hold score 0.01177 versus the published checkpoint 0.01120), which moves the ratio of medians from 0.827 to 0.883. The μ=−0.25 objective is the same circuit with the XX score, and its n=12 per-seed ratios are 0.588, 0.669, 0.668.
- At μ=0 the n=12 product gap on all 64 instances is 0.019, under the 2% line. PR #20 quoted that gap on the first 8 instances only. Peak 0.073 matches the published freeze. The pass gate is the μ=−0.25 ground state, which clears the line (peak 0.015, gap 0.083).
- `qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n21 ladder were not written. n=21 instances were not loaded.

Self-check: rank2 statevector 1e-08, μ=0 energy |Δ|=0.00e+00, grad |Δ|=0.00e+00. needs_YY=no.

L-BFGS fit time 172.6s across the twelve runs. Process wall 2.5s. Branch `cursor/gauss-rank2-mu-93a5`.
