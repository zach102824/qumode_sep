# Round 5 KS — packing-resolved full K

KS splits full K into cavity-intra Kin and cavity-inter Kout on the sublattice packing. G = Hc + thetaIn Kin/rIn + thetaOut Kout/rOut + thetaM Mhat, raw P = 3L+4. Kin + Kout = FK, the supports are disjoint, and rIn^2 + rOut^2 = rK^2. The nested map thetaIn = thetaK rIn/rK, thetaOut = thetaK rOut/rK reproduces the stored Rfull state to the replay tolerance. Parity of the reported states is +1, so the opposite-parity partner is not in these states.

Fresh, nested, and neighbor-transfer searches reach the same energy-selected state at L4, L5, and L6. The normalized coefficient distance from the nested line is 0.381 at L4, 0.132 at L5, and 0.153 at L6, above the frozen 0.001 threshold. Intra K is weighted more strongly than inter K. The branch screen does not retire the split: retire=False. L5 improves the stored Rfull energy by 0.00414, which is under 0.005, but the distance is not small. L6 improves it by 0.00804, above 0.005, with distance 0.153.

At L6, raw P = 22, the same count as QAOA p = 11. The energy-selected KS state is lower than p = 11 by 0.0313 in energy and higher by 0.119 in fidelity. It is lower than p = 12 by 0.00412 in energy and higher by 0.089 in fidelity. One hundred rescue starts at p = 8 through 12, on top of the fill batches, did not move those QAOA energies. The p = 12 energy remains the published 256-start value. Against the stored Rfull L6 state the same KS state is an energy lead of 0.00804 and a fidelity deficit of 0.00593. That is not both-metric dominance over stored Rfull. p = 11 is still 2445 evaluations short of the candidate inclusive count, and p = 9 through 4 are shorter. No seal.

The fresh-median robustness flag is false (False). The nested and transfer medians sit on the best energy. The fresh L6 median is about 0.30 higher because failed starts are included. L5 does not beat p = 12 on energy. L4 beats the stored L4 Rfull state on both energy and fidelity and does not beat p = 12 on energy.

Manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Packing `4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717`. Hc `eef806253f93036b128bc3aa2a7c06d35bbe60e7b9bcaf01cf920974b1da6a00`.
E0 = -8.6132024589. Full gap 1.613234e-03. Accessible gap 0.4337981341960315.
Optimizer time 429 s. Wall 1116 s.

| role | L | P | E | error | F | median | IQR | success | failed | evals | distance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KS_fresh | 1 | 7 | -7.77480 | 8.3840e-01 | 0.2550 | -7.0134 | 1.935 | 17 | 0 | 1137 | 1.876e-01 |
| KS_fresh | 2 | 10 | -8.07023 | 5.4297e-01 | 0.4260 | -7.5859 | 0.528 | 14 | 1 | 2085 | 5.429e-02 |
| KS_fresh | 3 | 13 | -8.22827 | 3.8493e-01 | 0.5648 | -7.8120 | 5.817 | 11 | 2 | 3844 | 5.175e-01 |
| KS_fresh | 4 | 16 | -8.25585 | 3.5736e-01 | 0.5724 | -0.8595 | 5.870 | 8 | 5 | 4334 | 3.813e-01 |
| KS_fresh | 5 | 19 | -8.29900 | 3.1420e-01 | 0.5953 | -0.9502 | 7.314 | 5 | 8 | 4446 | 1.320e-01 |
| KS_fresh | 6 | 22 | -8.32952 | 2.8368e-01 | 0.6039 | -8.0252 | 1.041 | 2 | 11 | 5075 | 1.528e-01 |
| KS_nested | 4 | 16 | -8.25585 | 3.5736e-01 | 0.5724 | -8.2558 | 0.000 | 7 | 18 | 9842 | 3.813e-01 |
| KS_nested | 5 | 19 | -8.29900 | 3.1420e-01 | 0.5953 | -8.2990 | 0.000 | 9 | 16 | 9660 | 1.320e-01 |
| KS_nested | 6 | 22 | -8.32952 | 2.8368e-01 | 0.6039 | -8.3295 | 0.000 | 7 | 18 | 9850 | 1.528e-01 |
| KS_transfer | 4 | 16 | -8.25585 | 3.5736e-01 | 0.5724 | -8.2558 | 0.094 | 11 | 14 | 9000 | 3.813e-01 |
| KS_transfer | 5 | 19 | -8.29900 | 3.1420e-01 | 0.5953 | -8.2990 | 0.000 | 7 | 16 | 9000 | 1.320e-01 |
| KS_transfer | 6 | 22 | -8.32952 | 2.8368e-01 | 0.6039 | -8.3295 | 0.000 | 7 | 17 | 9000 | 1.528e-01 |
| Rfull_fresh | 1 | 6 | -7.67878 | 9.3442e-01 | 0.2375 | -7.6788 | 0.700 | 16 | 1 | 809 |  |
| Rfull_fresh | 2 | 9 | -8.06734 | 5.4587e-01 | 0.4280 | -8.0524 | 4.445 | 15 | 0 | 1527 |  |
| Rfull_fresh | 3 | 12 | -8.11115 | 5.0206e-01 | 0.4331 | -7.8421 | 0.327 | 13 | 0 | 2906 |  |
| Rfull_fresh | 4 | 15 | -8.23694 | 3.7626e-01 | 0.6049 | -6.7878 | 7.321 | 6 | 7 | 4040 |  |
| Rfull_fresh | 5 | 18 | -8.25792 | 3.5529e-01 | 0.5379 | -0.9610 | 7.284 | 6 | 7 | 4179 |  |
| Rfull_fresh | 6 | 21 | -8.31911 | 2.9410e-01 | 0.6099 | -1.0380 | 7.293 | 4 | 9 | 4536 |  |

## QAOA after the fill and the 100-start rescue

- p=9: E -8.222874, F 0.4120, error 3.9033e-01, floor evals 6328, new evals 19395, short_5934
- p=10: E -8.263750, F 0.4523, error 3.4945e-01, floor evals 6319, new evals 29568, short_1972
- p=11: E -8.298216, F 0.4849, error 3.1499e-01, floor evals 6328, new evals 25941, short_2445
- p=12: E -8.325408, F 0.5147, error 2.8779e-01, floor evals 6328, new evals 34414, covered

## Both-metric cells

- KS_fresh L4 P=16: matched p=8 both True dE -0.0783129202700632, next both True dE -0.03297217258779561, p12 both False dE 0.06956120646649211, stored dE -0.09260672940549597
- KS_fresh L5 P=19: matched p=10 both True dE -0.035250309474140096, next both True dE -0.0007848970011110623, p12 both False dE 0.026407067886685454, stored dE -0.004139086314475904
- KS_fresh L6 P=22: matched p=11 both True dE -0.0313080900569318, next both True dE -0.004116125169135287, p12 both True dE -0.004116125169135287, stored dE -0.008040827904597592
- KS_nested L4 P=16: matched p=8 both True dE -0.0783129202733086, next both True dE -0.032972172591041016, p12 both False dE 0.0695612064632467, stored dE -0.09260672940874137
- KS_nested L5 P=19: matched p=10 both True dE -0.03525030947874086, next both True dE -0.0007848970057118265, p12 both False dE 0.02640706788208469, stored dE -0.004139086319076668
- KS_nested L6 P=22: matched p=11 both True dE -0.0313080900572178, next both True dE -0.004116125169421281, p12 both True dE -0.004116125169421281, stored dE -0.008040827904883585
- KS_transfer L4 P=16: matched p=8 both True dE -0.07831292027350045, next both True dE -0.03297217259123286, p12 both False dE 0.06956120646305486, stored dE -0.09260672940893322
- KS_transfer L5 P=19: matched p=10 both True dE -0.03525030947872132, next both True dE -0.0007848970056922866, p12 both False dE 0.02640706788210423, stored dE -0.004139086319057128
- KS_transfer L6 P=22: matched p=11 both True dE -0.031308090057457605, next both True dE -0.004116125169661089, p12 both True dE -0.004116125169661089, stored dE -0.008040827905123393
- Rfull_fresh L4 P=15: matched p=8 both True dE -0.059406092421758316, next both True dE -0.014065344739490726, p12 both False dE 0.088468034314797, stored dE None
- Rfull_fresh L5 P=18: matched p=10 both False dE 0.00583357171110066, next both False dE 0.040298984184129694, p12 both False dE 0.06749094907192621, stored dE None
- Rfull_fresh L6 P=21: matched p=11 both True dE -0.02089092288439076, next both False dE 0.006301042003405755, p12 both False dE 0.006301042003405755, stored dE None

## Incomplete
- qaoa p=8 short 13947 deferred
- qaoa p=7 short 13920 deferred
- qaoa p=6 short 13835 deferred
- qaoa p=5 short 15364 deferred
- qaoa p=4 short 16110 deferred
