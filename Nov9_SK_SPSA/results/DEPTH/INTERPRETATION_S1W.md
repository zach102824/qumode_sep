# DEPTH S1W — L1→L transfer does not recover a depth edge; no N18; GR unpatched

DEPTH C1 already rejected 4-start charged continuation. This stream asked whether **frozen S1JOINT** (the fair recipe that beat S4 but still lost energy to QAOA at L≥2) wastes Nov3’s L1 edge by starting L≥2 cold. Protocol: charge Q1=1024 to L=1, copy shared Uc and shell 0, extra shells exact 0, continue full-vector C0 on the remaining Q−1024. Q=1024·L. Seeds **10128–10135**. Energy selects. F diagnostic only.

**Both adopt gates fail at N15 L2. STOP. Do not run N18. Do not patch GR.**

| gate | N15 L2 mean ΔE | CI | WARM lower | pass |
|---|---|---|---|---|
| A WARM−COLD | +0.012 | [+0.0077, +0.017] | 0/8 | no |
| B WARM−QAOA | +0.030 | [−0.050, +0.085] | 1/8 | no |

Gate A’s CI lies entirely **above** 0: charging half of Q=2048 to L1 and continuing made Nov3 **worse** than cold S1JOINT at the same inclusive budget. Gate B does not reverse the QAOA energy comparison.

The rest of the matrix agrees. WARM never clears A or B at any of the four N×L cells. At N15 L3, WARM−COLD CI is also entirely above 0 (mean +0.058, 1/8). Cold Nov3 vs QAOA remains not an energy-win (N15 L2 COLD−QAOA Δ(E−E0)/N +0.00123, CI includes 0, 1/8).

Same charged-split cost as C1: L1 consumes 1024 queries that COLD spends at the evaluation depth (K=507 vs 1019 at L=2; 1019 vs 1531 at L=3). Lightly refining shared Uc did not make that trade worth it under S1.

96/96 fits ok. B* rejects 0. Charged 245760. Outside-cap 0. Frozen S1JOINT hashes matched. Prior streams unchanged. No ALT, no extra budget, no N18, no GR patch. Stop after this matrix.
