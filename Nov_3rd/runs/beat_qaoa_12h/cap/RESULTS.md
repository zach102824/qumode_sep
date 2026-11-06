# CAP — clause-aware cavity compilation

Label: **clause-aware cavity compilation**. This is a variable relabeling.
U_c^X and Hamming-shell SNAP are the frozen champs; the circuit sees
f'(x) = f(π^{-1} x). Med is 1/p_succ with success on the original formula
(E'(x) = 0 iff un-permuted E is satisfied). Zero new parameters.
Identity π is exactly the published champ.

**Verdict: PASS**

PASS requires π_max or π_min Med strictly below the 8-random range
at both n=12 and n=15, for at least one of L=2 (Walsh seed1) and
L=16 (fresh250 power_p). KILL means that effect stays inside the
random range at n=12 and n=15 for both L. No retrain on KILL.

## Packing objective (mean Σ pair-weights inside cavities)

π_max / π_min are instance-adaptive. Identity and the 8 random
permutations are one π per n (PCG64 seed 20261104+n), shared
across the B instances. The objective separating is a check that the
search ran; it is not the Med gate.

| n | identity | π_max | π_min | random min | random max |
|---|----------|-------|-------|------------|------------|
| 6 | 144.12 | 150.19 | 137.92 | 143.75 | 144.19 |
| 9 | 133.38 | 152.24 | 116.21 | 133.11 | 133.98 |
| 12 | 129.87 | 158.98 | 102.69 | 129.47 | 130.22 |
| 15 | 127.67 | 166.75 | 92.04 | 127.14 | 128.23 |

## QAOA invariance

Gate: |ΔMed| vs identity < 1e-6, float64, same angles as the champ.
Every non-identity permutation passed. The largest absolute gap is below.

| champ | max |ΔMed| over 10 perms × CORE |
|-------|--------------------------------|
| walsh_L2 | 2.842e-14 |
| power_L16 | 2.665e-15 |

Smoke checks that cleared n12 and n15 together:

- walsh_L2 / pi_max

## SNAP Med (lower is better)

### L=2 Walsh-degree seed1 (frozen)

| perm | n6 | n9 | n12 | n15 |
|------|----|----|-----|-----|
| identity | 5.0231 | 11.9681 | 28.0352 | 72.7966 |
| pi_max | 4.9986 | 11.9898 | 27.7738 | 69.6601 |
| pi_min | 5.0230 | 12.1804 | 28.4289 | 73.1656 |
| random_0 | 5.0545 | 12.1062 | 28.1465 | 72.3462 |
| random_1 | 5.0044 | 11.9429 | 28.3087 | 72.6088 |
| random_2 | 4.9567 | 12.0086 | 28.2860 | 72.4660 |
| random_3 | 4.9567 | 12.0409 | 28.3544 | 73.2168 |
| random_4 | 4.9996 | 12.2550 | 27.9059 | 72.8555 |
| random_5 | 5.0367 | 12.1293 | 28.0177 | 71.8459 |
| random_6 | 4.9751 | 11.9943 | 28.3204 | 71.8418 |
| random_7 | 5.0545 | 12.1096 | 28.5794 | 72.0438 |

| perm | n12 vs random | n15 vs random |
|------|---------------|---------------|
| pi_max | 27.7738 [27.9059, 28.5794] BETTER | 69.6601 [71.8418, 73.2168] BETTER |
| pi_min | 28.4289 [27.9059, 28.5794] inside | 73.1656 [71.8418, 73.2168] inside |
| identity | 28.0352 [27.9059, 28.5794] inside | 72.7966 [71.8418, 73.2168] inside |

### L=16 fresh250 power_p (frozen)

| perm | n6 | n9 | n12 | n15 |
|------|----|----|-----|-----|
| identity | 1.1503 | 1.4183 | 1.8389 | 2.5791 |
| pi_max | 1.1508 | 1.4062 | 1.8039 | 2.5497 |
| pi_min | 1.1560 | 1.4286 | 1.8736 | 2.6068 |
| random_0 | 1.1499 | 1.4123 | 1.8223 | 2.6616 |
| random_1 | 1.1506 | 1.4210 | 1.8552 | 2.6144 |
| random_2 | 1.1502 | 1.4277 | 1.8311 | 2.5982 |
| random_3 | 1.1502 | 1.4170 | 1.8329 | 2.5814 |
| random_4 | 1.1556 | 1.4313 | 1.8523 | 2.5858 |
| random_5 | 1.1503 | 1.4171 | 1.8178 | 2.5637 |
| random_6 | 1.1562 | 1.4212 | 1.8351 | 2.5224 |
| random_7 | 1.1499 | 1.4253 | 1.8289 | 2.6425 |

| perm | n12 vs random | n15 vs random |
|------|---------------|---------------|
| pi_max | 1.8039 [1.8178, 1.8552] BETTER | 2.5497 [2.5224, 2.6616] inside |
| pi_min | 1.8736 [1.8178, 1.8552] worse-outside | 2.6068 [2.5224, 2.6616] inside |
| identity | 1.8389 [1.8178, 1.8552] inside | 2.5791 [2.5224, 2.6616] inside |

## Identity reproduction

Identity Med must match the published frozen-champ CORE eval.

- walsh_L2 SNAP n6: got 5.023059 published 5.023059 Δ=0.000e+00
- walsh_L2 SNAP n9: got 11.968106 published 11.968106 Δ=0.000e+00
- walsh_L2 SNAP n12: got 28.035188 published 28.035188 Δ=0.000e+00
- walsh_L2 SNAP n15: got 72.796650 published 72.796650 Δ=0.000e+00
- walsh_L2 QAOA float32 n6: got 5.325684 published 5.325684 Δ=0.000e+00
- walsh_L2 QAOA float32 n9: got 12.641587 published 12.641587 Δ=0.000e+00
- walsh_L2 QAOA float32 n12: got 29.606314 published 29.606314 Δ=0.000e+00
- walsh_L2 QAOA float32 n15: got 76.087836 published 76.087836 Δ=0.000e+00
- power_L16 SNAP n6: got 1.150322 published 1.150322 Δ=0.000e+00
- power_L16 SNAP n9: got 1.418302 published 1.418302 Δ=0.000e+00
- power_L16 SNAP n12: got 1.838893 published 1.838893 Δ=0.000e+00
- power_L16 SNAP n15: got 2.579069 published 2.579069 Δ=0.000e+00
- power_L16 QAOA float32 n6: got 1.233089 published 1.233089 Δ=0.000e+00
- power_L16 QAOA float32 n9: got 1.587915 published 1.587915 Δ=0.000e+00
- power_L16 QAOA float32 n12: got 2.085367 published 2.085367 Δ=0.000e+00
- power_L16 QAOA float32 n15: got 2.912676 published 2.912676 Δ=0.000e+00

# CAP retrain — packed train{6,9}

Clause-aware cavity compilation. Angles are retrained on f'(x) = f(π_max^{-1} x).
Reported Med uses that same relabeling at eval, so success is the original formula.
Identity columns apply the new angles with no relabeling.

## L=2 Walsh vs seed1 knee

Frozen Walsh seed1 (identity labeling, sealed QAOA) knee R = **0.9567** at n15.
Protocol match: max_train=100, Adam 400, LBFGS 8 rounds, w6=1, w9=2, seeds 0,1,2.

| seed | labeling | n6 R | n9 R | n12 R | n15 R | n15 Med | knee R |
|------|----------|------|------|-------|-------|---------|--------|
| seed0 | pi_max | 0.9095 | 0.9229 | 0.9604 | 0.9550 | 72.6671 | 0.9604 |
| seed0 | identity | 0.8906 | 0.9378 | 0.9765 | 0.9871 | 75.1079 | 0.9871 |
| seed1 | pi_max | 0.9354 | 0.9499 | 0.9386 | 0.9259 | 70.4461 | 0.9499 |
| seed1 | identity | 0.9344 | 0.9444 | 0.9530 | 0.9767 | 74.3186 | 0.9767 |
| seed2 | pi_max | 0.9362 | 0.9502 | 0.9407 | 0.9307 | 70.8187 | 0.9502 |
| seed2 | identity | 0.9365 | 0.9470 | 0.9541 | 0.9713 | 73.9007 | 0.9713 |

Beat seed1 means packed n15 R < 0.9567 (lower Med / same QAOA).

Frozen π_max with the published Walsh seed1 angles (no retrain) is the stronger point:
knee R **0.9484** (n9), n15 R **0.9155**, n15 Med **69.660**.
All three retrains are worse than that at n15. Seeds 1 and 2 still beat the
identity-labeling knee 0.9567; seed 0 does not (n12 R 0.9604).
Identity labeling of the retrained angles is worse at n15 (R 0.971–0.987),
so the improvement shows up only when π_max is applied at eval.

## L=16 power_p vs fresh250 champ and Fourier QAOA

Frozen power_p identity n15 Med = **2.5791**. Fourier QAOA Q=6 n15 Med = **2.3853**.
Protocol match: full train 250/250, Adam 600, LBFGS 12, fresh250 QAOA angles, seeds 0,1,2.

| seed | labeling | n15 Med | vs power_p 2.579 | vs Fourier 2.385 | n12 Med |
|------|----------|---------|-------------------|------------------|---------|
| seed0 | pi_max | 2.6008 | +0.0218 | +0.2155 | 1.8412 |
| seed0 | identity | 2.6672 | +0.0881 | +0.2818 | 1.8801 |
| seed1 | pi_max | 2.5426 | -0.0364 | +0.1573 | 1.8012 |
| seed1 | identity | 2.5808 | +0.0017 | +0.1955 | 1.8351 |
| seed2 | pi_max | 2.5596 | -0.0195 | +0.1742 | 1.8098 |
| seed2 | identity | 2.6072 | +0.0281 | +0.2219 | 1.8514 |
| frozen π_max | pi_max | 2.5497 | -0.0294 | +0.1643 | 1.8039 |

Frozen L=16 π_max did not clear the smoke (n15 Med 2.550 sits inside the
8-random range, whose minimum is 2.522). The best retrain, seed1 π_max n15
**2.543**, edges the identity champ and the frozen packing, and remains above
Fourier QAOA 2.385 on every seed.


Finished: 2026-11-04 08:52:42 +0000

