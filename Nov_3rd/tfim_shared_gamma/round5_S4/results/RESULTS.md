# Round 5 S4 — stride-4 replay and transfer

The logical target is the frozen manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Stride-4 cavities are `[[0,4,8],[1,5,9],[2,6,10],[3,7,11]]`. Canonical packing hash `0f4330f4a9d4e84db3582beb8b607a2e1092918f7a28b68bb3cff4a1a514c979`. M and rM were recomputed for this packing. rM = 0.769011889337, rK = 2.712107384896.

Independent replay of D's energy-selected L6 state, commit `5ebca82a`, file sha256 `5e8c1a53dd016aa7c42b15e52014e305bf53966f84b122de8864dfde5b06e97d`: logical/packed overlap 1.055e-14 below 1, source overlap -3.109e-15 below 1, energy residual -5.151e-14, fidelity residual 5.107e-15.

Within-cavity bit reversal leaves this symmetric shell unchanged. Inverting the physical-to-logical basis map does change the state, and that direction was rejected. A π shift of one shell coefficient is a global phase. Negating gamma and the shells conjugates the state and does not make it the same ray.

## Arms

| arm | L | best E−E0 | F | var | median | IQR | marginal | basins | conjugate-only |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1.1230 | 0.1753 | 1.717 | 1.1230 | 1.1037 | 592 | 4 | 16 |
| 0 | 2 | 1.0449 | 0.1914 | 1.648 | 1.5523 | 0.8805 | 1384 | 8 | 9 |
| 0 | 3 | 0.6092 | 0.4280 | 1.142 | 1.4655 | 3.2203 | 2086 | 8 | 6 |
| 0 | 4 | 0.5077 | 0.6423 | 1.533 | 1.3662 | 7.3226 | 10532 | 14 | 30 |
| 0 | 5 | 0.2280 | 0.8898 | 0.973 | 1.0745 | 7.5164 | 18094 | 19 | 9 |
| 0 | 6 | 0.1675 | 0.9024 | 0.657 | 7.6187 | 7.2343 | 20961 | 19 | 9 |
| 1 | 4 | 0.4764 | 0.8179 | 2.644 | 0.5077 | 0.0000 | 9608 | 7 | 72 |
| 1 | 5 | 0.2280 | 0.8898 | 0.973 | 0.5100 | 0.2820 | 15390 | 7 | 12 |
| 1 | 6 | 0.1675 | 0.9024 | 0.657 | 0.1675 | 0.1288 | 26799 | 8 | 17 |
| 2 | 4 | 0.5077 | 0.6423 | 1.533 | 0.5077 | 0.1043 | 5992 | 6 | 77 |
| 2 | 5 | 0.2280 | 0.8898 | 0.973 | 0.2280 | 0.2820 | 8592 | 5 | 52 |
| 2 | 6 | 0.1675 | 0.9024 | 0.657 | 0.1675 | 0.0000 | 11244 | 4 | 0 |

Inclusive totals, source charged once: arm 0 = 53649, arm 1 = 53353 (includes 1556), arm 2 = 53927 (includes 28099).

## What the numbers say

The energy-selected L6 state is the same point in all three arms: error 0.1675, fidelity 0.9024, variance 0.657. A fresh forward in both the logical and packed simulators reproduces it, with gradient norm 5e-6. It is not D's stored control. That control remains error 0.2924 and fidelity 0.858, which has higher fidelity than QAOA p=12 and worse energy. The new state is better than that control on energy and fidelity.

Against the stored QAOA p=12 state (error 0.2878, fidelity 0.515, 89272 evaluations) the new state is lower in energy and higher in fidelity. That is a both-metric comparison of two energy-selected states. It is not a seal. The fresh arm's median is 2.55 on its first 24 starts and 7.62 after the preregistered random equalization starts. Nine of those 48 fresh starts beat p=12 on both metrics. The descending arm's L6 median is 0.1675, with 35 of 48 starts beating p=12 on both metrics. The cross-packing arm's L6 median is 0.1675 and its IQR is 0, with 21 of 24 starts beating p=12 on both metrics. Those two arms use stored schedules. They are not source-free discoveries.

At L5 the same split appears. Best error 0.2280 and fidelity 0.8898 in every arm. The fresh median is 1.07. The descending median is 0.510. The cross-packing median equals the best. Sublattice L5 best error is 0.3183. Sublattice L6 best error is 0.2917.

The retirement screen needs the descending median to sit more than 0.01 above the sublattice transfer median at both L5 and L6, and no stride-4 state to beat the stored sublattice energy. The L5 descending median is worse (0.510 versus 0.318). The L6 descending median is better (0.168 versus 0.292). The screen is not met, and stride-4 is not retired. High fidelity is not being traded against a worse energy at the new L6 point.

QAOA p=11 is the ceil match to P=21, not an equal parameter count. The existing p=9 through p=12 floor already has more evaluations than this round's inclusive totals (about 54k per arm). A new four-times rescue was not run. Until an independent replay and that rescue exist, the both-metric gap is a best-state result with a reproducible source-seeded median, not a sealed claim. Historical thresholds 0.390 and 0.315 are not the comparison used here.

Wall 434.8 s. Partial: False.
