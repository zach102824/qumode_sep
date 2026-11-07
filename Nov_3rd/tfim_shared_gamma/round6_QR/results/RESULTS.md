# Round 6 QR — fixed-budget QAOA

This round is standard independent-angle QAOA only. No feature was changed. Comparisons use tau = 1e-09. A both-metric win requires lower energy and higher fidelity on the energy-selected states. A lower-energy and lower-fidelity QAOA state would be a tradeoff. These searches did not produce one.

Budgets were frozen before these starts. Bp = max(4·Rp, 53927) and the new allowance is Bp minus work already validated at that depth. The stride-4 family cap is the larger published arm inclusive, 53927. The constants 1556 and 28099 are already inside that ledger and were not added again. L5 fidelity is the full-precision value in the committed arm0 cell, 0.8898142677626181.

The best p=12 energy remains -8.325407549783, the stored optimum, after 118558 new evaluations. The allowance was 227074, so the shortfall is 108516. Against stride-4 L6 the energy delta is +0.120275 and the fidelity delta is -0.387652. Against KS L6 the energy delta is +0.004116 and the fidelity delta is -0.089198. QAOA is higher in energy and lower in fidelity in both comparisons. p=11, p=10, and p=9 likewise stayed on their stored optima, with shortfalls 95146, 138662, and 133076. p=6 met its allowance. p=7 is short by 415. The median of the new starts sits on the stored optimum at each depth, and the IQR stays large because the failed tail is included. No seal.

Manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. E0 = -8.6132024589. tau = 1e-09.
Wall 4419 s.

| p | E | error | F | var | new evals | allowed | shortfall | median | IQR | basin 1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | -8.325408 | 2.8779e-01 | 0.5147 | 0.421 | 118558 | 227074 | 108516 | -8.3254 | 7.306 | 555/912 |
| 11 | -8.298216 | 3.1499e-01 | 0.4849 | 0.455 | 127525 | 222671 | 95146 | -8.2982 | 7.311 | 663/1073 |
| 10 | -8.263750 | 3.4945e-01 | 0.4523 | 0.530 | 64704 | 203366 | 138662 | -8.2638 | 7.322 | 371/605 |
| 9 | -8.222874 | 3.9033e-01 | 0.4120 | 0.534 | 71598 | 204674 | 133076 | -8.2229 | 7.329 | 504/814 |
| 7 | -8.124283 | 4.8892e-01 | 0.3333 | 0.592 | 46511 | 46926 | 415 | -8.1243 | 7.357 | 385/617 |
| 6 | -8.056900 | 5.5630e-01 | 0.2947 | 0.682 | 46841 | 46841 | 0 | -8.0569 | 7.355 | 471/757 |

## Deltas against the frozen states

- p=12 vs KS_L6: dE +0.004116, dF -0.089198, dominates False, tradeoff False
- p=12 vs stride4_L5: dE +0.059748, dF -0.375096, dominates False, tradeoff False
- p=12 vs stride4_L6: dE +0.120275, dF -0.387652, dominates False, tradeoff False
- p=11 vs KS_L6: dE +0.031308, dF -0.119065, dominates False, tradeoff False
- p=11 vs stride4_L5: dE +0.086940, dF -0.404963, dominates False, tradeoff False
- p=11 vs stride4_L6: dE +0.147467, dF -0.417519, dominates False, tradeoff False
- p=10 vs KS_L6: dE +0.065774, dF -0.151609, dominates False, tradeoff False
- p=10 vs stride4_L5: dE +0.121406, dF -0.437508, dominates False, tradeoff False
- p=10 vs stride4_L6: dE +0.181932, dF -0.450063, dominates False, tradeoff False
- p=9 vs KS_L6: dE +0.106650, dF -0.191909, dominates False, tradeoff False
- p=9 vs stride4_L5: dE +0.162282, dF -0.477808, dominates False, tradeoff False
- p=9 vs stride4_L6: dE +0.222808, dF -0.490364, dominates False, tradeoff False
- p=7 vs KS_L6: dE +0.205241, dF -0.270630, dominates False, tradeoff False
- p=7 vs stride4_L5: dE +0.260873, dF -0.556528, dominates False, tradeoff False
- p=7 vs stride4_L6: dE +0.321400, dF -0.569084, dominates False, tradeoff False
- p=6 vs KS_L6: dE +0.272624, dF -0.309239, dominates False, tradeoff False
- p=6 vs stride4_L5: dE +0.328256, dF -0.595138, dominates False, tradeoff False
- p=6 vs stride4_L6: dE +0.388783, dF -0.607694, dominates False, tradeoff False
