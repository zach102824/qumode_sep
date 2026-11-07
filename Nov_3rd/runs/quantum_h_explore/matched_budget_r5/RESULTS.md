# RESULTS — Round 5 matched bakeoff of the μ=−0.25 champs

**Verdict: PASS**

Transfer ranking at L=16, lower frozen R first at n=12 then n=15: 1. C unsat λ=0.75 μ=-0.25 (walsh) R12=0.330 (BEAT) R15=0.355 (BEAT), 2. A gauss J μ=-0.25 (power_p) R12=0.700 (BEAT) R15=0.747 (BEAT), 3. B ±1 ZZZ μ=-0.25 (power_p) R12=0.701 (BEAT) R15=0.792 (BEAT). Winner: C unsat λ=0.75 μ=-0.25 via walsh.

## Protocol

Shared budget is PR #19 / PR #23, not the budget that produced the prior claims. Train B=128 per n, holdout B=128 per n, seeds [0, 1, 2]. L-BFGS rounds 18/48 (L=2/L=16), max_iter 8, lr=0.35. Early stop is off. Every head ran the full round count; a failed step restores θ and still counts. The checkpoint inside that budget minimizes the {6,9} holdout score (Fourier: mean of the two median residuals; champ: worst of the two ratios versus that seed's Fourier). n=12 and n=15 are scored after the freeze. Loss is the mean of the per-n mean residuals on n=6 and n=9. Champ heads warm-start from that seed's Fourier checkpoint (φ_k=−β k, c2=c3=0, p=0; Walsh copies γ onto degrees 1–4).

PR #21's gauss claim (n=12 R=0.642) used 24 L=16 rounds, B_hold=256, a residual checkpoint, and n=15 B=16. PR #23's unsat claim (n=12 R=0.330) used this round count with an 8-round early stop. PR #22's ±1 claim (n=12 R=0.701) is this protocol. Those three numbers are prior claims. This table retrains all three and does not reuse their angles.

Every arm has intra-cavity XX at μ=−0.25 and κ=0. A: H = Σ J_abc ZZZ − 0.5 Σ X + μ Σ_cav Σ_{i<j} XX, J~N(0,1) on the seal-A support (salt 91). B: the same with J=±1. C: H = unsat − 0.75 Σ X + μ Σ_cav Σ_{i<j} XX, and U_c^X pulses centered C_n = (n/m) unsat − max. Gate sequence is BinH U_c^X (W = H^{⊗3}) then the Hamming-shell SNAP. ⟨H_c⟩ is the Z histogram. ⟨X⟩ and ⟨XX⟩ are the per-cavity X histogram. needs YY: no.

Gate, unchanged: BEAT if R≤0.98, TIE if R≤1.02, else UNDER. R = median_seed(Med ε_head) / median_seed(Med ε_Fourier) on the same H. Quantum pass: median bitstring peak < 0.5 and median product gap (E_prod−E_0)/|E_0| ≥ 2%. Product states are XZ-plane. Ranking uses frozen L=16 R and does not drop a champ for the quantum note.

Reference checks: [{'family': 'gauss μ=0 diagonal', 'max_abs': 4.000078490662418e-09, 'ok': True}, {'family': 'rank2 μ=-0.25', 'max_abs': 3.497913070305003e-09, 'ok': True}, {'family': 'unsat μ=0 diagonal', 'max_abs': 4.131408548602167e-09, 'ok': True}]. XX |ΔE0| versus the μ=0 diagonal: gauss 0.947, unsat 1.191. μ=0 rank2 versus Fock |ΔE|=0.00e+00. Unsat μ=0 histogram versus Pauli |ΔE|=4.44e-15. Unsat μ=−0.25 histogram versus ⟨ψ|H|ψ⟩ |ΔE|=4.88e-15.

## Ranking table

n=6 and n=9 are the selection holdout, so those medians are slightly optimistic. n=12 and n=15 are not. Lower ε is better. Rank is filled only on the L=16 n=12 row.

| rank | champ | head | L | n | role | Med ε | seed std | Fourier Med ε | R | mark | med peak | med prod gap | quantum |
|-----:|:------|:-----|--:|--:|:-----|------:|---------:|--------------:|--:|:-----|---------:|-------------:|:--------|
|  | A gauss J μ=-0.25 | power_p | 2 | 6 | hold | 0.0242 | 0.0010 | 0.0243 | 0.998 | TIE | 0.082 | 0.116 | yes |
|  | A gauss J μ=-0.25 | power_p | 2 | 9 | hold | 0.0261 | 0.0033 | 0.0263 | 0.993 | TIE | 0.030 | 0.103 | yes |
|  | A gauss J μ=-0.25 | power_p | 2 | 12 | transfer | 0.0335 | 0.0030 | 0.0338 | 0.993 | TIE | 0.015 | 0.083 | yes |
|  | A gauss J μ=-0.25 | power_p | 2 | 15 | transfer | 0.0339 | 0.0052 | 0.0339 | 0.999 | TIE | 0.014 | 0.066 | yes |
|  | A gauss J μ=-0.25 | power_p | 16 | 6 | hold | 0.0027 | 0.0002 | 0.0044 | 0.610 | BEAT | 0.082 | 0.116 | yes |
|  | A gauss J μ=-0.25 | power_p | 16 | 9 | hold | 0.0036 | 0.0002 | 0.0060 | 0.588 | BEAT | 0.030 | 0.103 | yes |
| 2 | A gauss J μ=-0.25 | power_p | 16 | 12 | transfer | 0.0059 | 0.0007 | 0.0085 | 0.700 | BEAT | 0.015 | 0.083 | yes |
|  | A gauss J μ=-0.25 | power_p | 16 | 15 | transfer | 0.0060 | 0.0001 | 0.0080 | 0.747 | BEAT | 0.014 | 0.066 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 6 | hold | 0.0250 | 0.0025 | 0.0250 | 1.000 | TIE | 0.071 | 0.176 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 9 | hold | 0.0439 | 0.0023 | 0.0439 | 1.000 | TIE | 0.044 | 0.105 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 12 | transfer | 0.0587 | 0.0032 | 0.0587 | 1.000 | TIE | 0.035 | 0.072 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 15 | transfer | 0.0645 | 0.0039 | 0.0645 | 1.000 | TIE | 0.039 | 0.051 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 16 | 6 | hold | 0.0026 | 0.0010 | 0.0048 | 0.543 | BEAT | 0.071 | 0.176 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 16 | 9 | hold | 0.0099 | 0.0026 | 0.0136 | 0.730 | BEAT | 0.044 | 0.105 | yes |
| 3 | B ±1 ZZZ μ=-0.25 | power_p | 16 | 12 | transfer | 0.0162 | 0.0049 | 0.0231 | 0.701 | BEAT | 0.035 | 0.072 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 16 | 15 | transfer | 0.0226 | 0.0047 | 0.0286 | 0.792 | BEAT | 0.039 | 0.051 | yes |
|  | C unsat λ=0.75 μ=-0.25 | walsh | 2 | 6 | hold | 0.0150 | 0.0001 | 0.0322 | 0.466 | BEAT | 0.098 | 0.123 | yes |
|  | C unsat λ=0.75 μ=-0.25 | walsh | 2 | 9 | hold | 0.0146 | 0.0008 | 0.0320 | 0.456 | BEAT | 0.029 | 0.104 | yes |
|  | C unsat λ=0.75 μ=-0.25 | walsh | 2 | 12 | transfer | 0.0142 | 0.0010 | 0.0290 | 0.490 | BEAT | 0.008 | 0.112 | yes |
|  | C unsat λ=0.75 μ=-0.25 | walsh | 2 | 15 | transfer | 0.0160 | 0.0011 | 0.0332 | 0.483 | BEAT | 0.002 | 0.106 | yes |
|  | C unsat λ=0.75 μ=-0.25 | walsh | 16 | 6 | hold | 0.0005 | 0.0007 | 0.0021 | 0.251 | BEAT | 0.098 | 0.123 | yes |
|  | C unsat λ=0.75 μ=-0.25 | walsh | 16 | 9 | hold | 0.0005 | 0.0006 | 0.0017 | 0.278 | BEAT | 0.029 | 0.104 | yes |
| 1 | C unsat λ=0.75 μ=-0.25 | walsh | 16 | 12 | transfer | 0.0005 | 0.0006 | 0.0015 | 0.330 | BEAT | 0.008 | 0.112 | yes |
|  | C unsat λ=0.75 μ=-0.25 | walsh | 16 | 15 | transfer | 0.0006 | 0.0006 | 0.0016 | 0.355 | BEAT | 0.002 | 0.106 | yes |

Per-seed median ε (head) and per-seed R, L=16 transfer:

- A gauss J μ=-0.25 n=12 seeds [0, 1, 2]: ε 0.0059, 0.0064, 0.0049; R 0.678, 0.753, 0.691
- A gauss J μ=-0.25 n=15 seeds [0, 1, 2]: ε 0.0061, 0.0059, 0.0060; R 0.709, 0.734, 0.821
- B ±1 ZZZ μ=-0.25 n=12 seeds [0, 1, 2]: ε 0.0205, 0.0162, 0.0108; R 0.872, 0.701, 0.690
- B ±1 ZZZ μ=-0.25 n=15 seeds [0, 1, 2]: ε 0.0268, 0.0226, 0.0174; R 0.884, 0.792, 0.763
- C unsat λ=0.75 μ=-0.25 n=12 seeds [0, 1, 2]: ε 0.0005, 0.0015, 0.0005; R 0.330, 0.526, 0.328
- C unsat λ=0.75 μ=-0.25 n=15 seeds [0, 1, 2]: ε 0.0006, 0.0016, 0.0005; R 0.355, 0.587, 0.373

Prior n=12 claims, not used for the rank: gauss 0.642, unsat 0.330, rank2 0.701.

## Quantumness

Split medians (peak, product gap). The ranking table already carries the scored split.

- gauss train n=6: peak 0.103, gap 0.115, S 1.025, pass
- gauss hold n=6: peak 0.082, gap 0.116, S 0.964, pass
- gauss train n=9: peak 0.031, gap 0.097, S 1.133, pass
- gauss hold n=9: peak 0.030, gap 0.103, S 1.189, pass
- gauss transfer n=12: peak 0.015, gap 0.083, S 1.297, pass
- gauss transfer n=15: peak 0.014, gap 0.066, S 1.415, pass
- rank2 train n=6: peak 0.071, gap 0.176, S 1.139, pass
- rank2 hold n=6: peak 0.071, gap 0.176, S 1.139, pass
- rank2 train n=9: peak 0.042, gap 0.106, S 1.541, pass
- rank2 hold n=9: peak 0.044, gap 0.105, S 1.475, pass
- rank2 transfer n=12: peak 0.035, gap 0.072, S 1.739, pass
- rank2 transfer n=15: peak 0.039, gap 0.051, S 1.775, pass
- unsat train n=6: peak 0.095, gap 0.120, S 0.314, pass
- unsat hold n=6: peak 0.098, gap 0.123, S 0.321, pass
- unsat train n=9: peak 0.028, gap 0.110, S 0.380, pass
- unsat hold n=9: peak 0.029, gap 0.104, S 0.365, pass
- unsat transfer n=12: peak 0.008, gap 0.112, S 0.410, pass
- unsat transfer n=15: peak 0.002, gap 0.106, S 0.420, pass

Walsh mass on the first n=6 train pulse: gauss degree-3 fraction 1.000 (degrees 1–4 share 1.000), so the champ head is power_p. Unsat degrees 1–4 share 0.199 (degree 3 is 0.060), so the champ head is Walsh.

## Rule

PASS when every Fourier and champ fit ran the full round budget with early stop off and no LBFGS failure, the published diagonal checks matched, the XX term moved E0, the Z+X histogram matched ⟨ψ|H|ψ⟩, and the L=16 transfer ranking is filled for n=12 then n=15. needs_edit when that protocol broke. The quantum note is reported and does not change the rank.

Protocol status: full budget, no LBFGS failures, early stop off.

LBFGS failures across fits: 0. Wall clock 652.0s. n=21 was not loaded.

Branch `cursor/r5-champ-bakeoff-bd3c`.
