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

