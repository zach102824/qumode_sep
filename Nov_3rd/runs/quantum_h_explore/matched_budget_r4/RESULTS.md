# RESULTS — matched-budget bakeoff of three champs

**Verdict: PASS**

Transfer ranking at L=16, lower R first at n=12 then n=15: 1. B ±1 ZZZ μ=-0.25 R12=0.701 (BEAT) R15=0.792 (BEAT), 2. C unsat λ=0.75 R12=0.888 (BEAT) R15=0.898 (BEAT), 3. A gauss ZZZ λ=0.5 R12=0.981 (TIE) R15=0.936 (BEAT). Winner: B ±1 ZZZ μ=-0.25 via power_p.

Gauss misses the product-gap gate on the frozen sets (n=12 gap 0.019, n=15 gap 0.012). The bitstring peak stays below 0.5, so the ground state is not a bitstring. Rank2 and unsat pass both gates (peak < 0.5 and product gap ≥ 2%) on n=12 and n=15. The ranking is by frozen R and does not drop gauss for that miss.

Unsat L=16 seeds 0 and 1 reproduce PR #19's per-seed n=12 R of 0.888 and 0.856. Seed 2, which that seal early-stopped at the Fourier warm start (R 0.996), ran all 48 rounds here and finished at R 0.929.

## Protocol

Taken from PR #19, not from PR #18 or PR #20. PR #19 is the seal that already freezes n=12 (B=64) and n=15 (B=32) and that moved L=16 from 24 rounds to 48 because Fourier's holdout score was still falling by a few 10⁻⁴ per round at 24. PR #18 used 18/24 rounds and B_train=32, B_hold=48. PR #20 used B_hold=256 and did not score n=15. A shorter budget would under-train the Fourier baseline.

Train B=128 per n, holdout B=128 per n, seeds [0, 1, 2]. L-BFGS rounds 18/48 (L=2/L=16), max_iter 8, lr=0.35. Early stop is off. Every head ran the full round count; a failed step restores θ and still counts. The checkpoint inside that budget minimizes the {6,9} holdout score (Fourier: mean of the two median residuals; champ: worst of the two ratios versus that seed's Fourier). n=12 and n=15 are scored after the freeze. Loss is the mean of the per-n mean residuals on n=6 and n=9. Champ heads warm-start from that seed's Fourier checkpoint (φ_k=−β k, c2=c3=0, p=0; Walsh copies γ onto degrees 1–4).

Published caches were not reused. PR #19 early-stopped Walsh L=16 seed 2 and power_p L=16 seeds 0 and 2 at round 8, and Fourier L=2 seed 1 at round 14, so those angles are not this budget. PR #18 and PR #20 used different rounds or batch sizes.

Gate, unchanged: BEAT if R≤0.98, TIE if R≤1.02, else UNDER. R = median_seed(Med ε_head) / median_seed(Med ε_Fourier) on the same H. Quantum pass: median bitstring peak < 0.5 and median product gap (E_prod−E_0)/|E_0| ≥ 2%. Product states are XZ-plane. Ranking uses frozen L=16 R and does not drop a champ for the quantum note.

Reference E0 on the first four published n=6 train rows matches gauss, rank2, and unsat to 4e-9. Rank2 at μ=0 matches the Fock twin, |ΔE|=0.

## Ranking table

n=6 and n=9 are the selection holdout, so those medians are slightly optimistic. n=12 and n=15 are not. Lower ε is better.

| rank | champ | head | L | n | role | Med ε | seed std | Fourier Med ε | R | mark | med peak | med prod gap | quantum |
|-----:|:------|:-----|--:|--:|:-----|------:|---------:|--------------:|--:|:-----|---------:|-------------:|:--------|
|  | A gauss ZZZ λ=0.5 | power_p | 2 | 6 | hold | 0.0620 | 0.0000 | 0.0624 | 0.994 | TIE | 0.150 | 0.045 | yes |
|  | A gauss ZZZ λ=0.5 | power_p | 2 | 9 | hold | 0.0677 | 0.0000 | 0.0680 | 0.996 | TIE | 0.075 | 0.036 | yes |
|  | A gauss ZZZ λ=0.5 | power_p | 2 | 12 | transfer | 0.1017 | 0.0000 | 0.1014 | 1.004 | TIE | 0.073 | 0.019 | no |
|  | A gauss ZZZ λ=0.5 | power_p | 2 | 15 | transfer | 0.1035 | 0.0000 | 0.1034 | 1.000 | TIE | 0.105 | 0.012 | no |
|  | A gauss ZZZ λ=0.5 | power_p | 16 | 6 | hold | 0.0058 | 0.0036 | 0.0066 | 0.876 | BEAT | 0.150 | 0.045 | yes |
|  | A gauss ZZZ λ=0.5 | power_p | 16 | 9 | hold | 0.0100 | 0.0039 | 0.0106 | 0.945 | BEAT | 0.075 | 0.036 | yes |
| 3 | A gauss ZZZ λ=0.5 | power_p | 16 | 12 | transfer | 0.0158 | 0.0093 | 0.0161 | 0.981 | TIE | 0.073 | 0.019 | no |
|  | A gauss ZZZ λ=0.5 | power_p | 16 | 15 | transfer | 0.0279 | 0.0090 | 0.0298 | 0.936 | BEAT | 0.105 | 0.012 | no |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 6 | hold | 0.0250 | 0.0025 | 0.0250 | 1.000 | TIE | 0.071 | 0.176 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 9 | hold | 0.0439 | 0.0023 | 0.0439 | 1.000 | TIE | 0.044 | 0.105 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 12 | transfer | 0.0587 | 0.0032 | 0.0587 | 1.000 | TIE | 0.035 | 0.072 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 2 | 15 | transfer | 0.0645 | 0.0039 | 0.0645 | 1.000 | TIE | 0.039 | 0.051 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 16 | 6 | hold | 0.0026 | 0.0010 | 0.0048 | 0.543 | BEAT | 0.071 | 0.176 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 16 | 9 | hold | 0.0099 | 0.0026 | 0.0136 | 0.730 | BEAT | 0.044 | 0.105 | yes |
| 1 | B ±1 ZZZ μ=-0.25 | power_p | 16 | 12 | transfer | 0.0162 | 0.0049 | 0.0231 | 0.701 | BEAT | 0.035 | 0.072 | yes |
|  | B ±1 ZZZ μ=-0.25 | power_p | 16 | 15 | transfer | 0.0226 | 0.0047 | 0.0286 | 0.792 | BEAT | 0.039 | 0.051 | yes |
|  | C unsat λ=0.75 | walsh | 2 | 6 | hold | 0.0398 | 0.0004 | 0.0594 | 0.670 | BEAT | 0.202 | 0.141 | yes |
|  | C unsat λ=0.75 | walsh | 2 | 9 | hold | 0.0404 | 0.0003 | 0.0574 | 0.705 | BEAT | 0.086 | 0.132 | yes |
|  | C unsat λ=0.75 | walsh | 2 | 12 | transfer | 0.0405 | 0.0003 | 0.0532 | 0.761 | BEAT | 0.038 | 0.138 | yes |
|  | C unsat λ=0.75 | walsh | 2 | 15 | transfer | 0.0437 | 0.0003 | 0.0596 | 0.733 | BEAT | 0.019 | 0.120 | yes |
|  | C unsat λ=0.75 | walsh | 16 | 6 | hold | 0.0020 | 0.0020 | 0.0025 | 0.795 | BEAT | 0.202 | 0.141 | yes |
|  | C unsat λ=0.75 | walsh | 16 | 9 | hold | 0.0018 | 0.0019 | 0.0021 | 0.863 | BEAT | 0.086 | 0.132 | yes |
| 2 | C unsat λ=0.75 | walsh | 16 | 12 | transfer | 0.0026 | 0.0023 | 0.0029 | 0.888 | BEAT | 0.038 | 0.138 | yes |
|  | C unsat λ=0.75 | walsh | 16 | 15 | transfer | 0.0027 | 0.0025 | 0.0030 | 0.898 | BEAT | 0.019 | 0.120 | yes |

Per-seed median ε (head) and per-seed R, L=16 transfer:

- A gauss ZZZ λ=0.5 n=12 seeds [0, 1, 2]: ε 0.0157, 0.0317, 0.0158; R 0.976, 0.998, 1.007
- A gauss ZZZ λ=0.5 n=15 seeds [0, 1, 2]: ε 0.0279, 0.0432, 0.0275; R 0.936, 0.997, 0.971
- B ±1 ZZZ μ=-0.25 n=12 seeds [0, 1, 2]: ε 0.0205, 0.0162, 0.0108; R 0.872, 0.701, 0.690
- B ±1 ZZZ μ=-0.25 n=15 seeds [0, 1, 2]: ε 0.0268, 0.0226, 0.0174; R 0.884, 0.792, 0.763
- C unsat λ=0.75 n=12 seeds [0, 1, 2]: ε 0.0026, 0.0065, 0.0025; R 0.888, 0.856, 0.929
- C unsat λ=0.75 n=15 seeds [0, 1, 2]: ε 0.0027, 0.0070, 0.0026; R 0.898, 0.882, 0.941

## Quantumness

Split medians (peak, product gap). The ranking table already carries the scored split. This list is not a second ranking.

- gauss train n=6: peak 0.174, gap 0.033, S 1.285, pass
- gauss hold n=6: peak 0.150, gap 0.045, S 1.308, pass
- gauss train n=9: peak 0.088, gap 0.031, S 1.616, pass
- gauss hold n=9: peak 0.075, gap 0.036, S 1.772, pass
- gauss transfer n=12: peak 0.073, gap 0.019, S 1.825, FAIL
- gauss transfer n=15: peak 0.105, gap 0.012, S 1.580, FAIL
- rank2 train n=6: peak 0.071, gap 0.176, S 1.139, pass
- rank2 hold n=6: peak 0.071, gap 0.176, S 1.139, pass
- rank2 train n=9: peak 0.042, gap 0.106, S 1.541, pass
- rank2 hold n=9: peak 0.044, gap 0.105, S 1.475, pass
- rank2 transfer n=12: peak 0.035, gap 0.072, S 1.739, pass
- rank2 transfer n=15: peak 0.039, gap 0.051, S 1.775, pass
- unsat train n=6: peak 0.206, gap 0.125, S 0.442, pass
- unsat hold n=6: peak 0.202, gap 0.141, S 0.418, pass
- unsat train n=9: peak 0.089, gap 0.130, S 0.513, pass
- unsat hold n=9: peak 0.086, gap 0.132, S 0.550, pass
- unsat transfer n=12: peak 0.038, gap 0.138, S 0.617, pass
- unsat transfer n=15: peak 0.019, gap 0.120, S 0.591, pass

Walsh mass on the first n=6 train pulse: gauss degree-3 fraction 1.000 (degrees 1–4 share 1.000), so graded γ is a null knob and the champ head is power_p. Unsat degrees 1–4 share 0.199 (degree 3 is 0.060), so the champ head is Walsh.

## Rule

PASS when every Fourier and champ fit ran the full round budget with no LBFGS failure, the published E0 check matched, and the L=16 transfer ranking is filled for n=12 then n=15. needs_edit when that protocol broke. KILL was not used.

Protocol status: full budget, no LBFGS failures.

LBFGS failures across fits: 0. Wall clock 674.0s. n=21 was not loaded.

Branch `cursor/matched-budget-bakeoff-368e`.
