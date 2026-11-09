# Historical B=0.1 maximin triples are partial

Protocol `dcf133bf` before any fit. The wall was raised in `40236eda` before any fidelity was computed. Running marker `86639669`. PREREG SHA-256 `698a8bdbb67be63988fb76fa32c3060a7e47556da2cd2fd6c814f86f3b75313b`.

This is the historical author field `B = 0.1`. It is not a critical-field result. `B_cri` stays withheld. It does not replace the ferromagnetic-sum result. That result stays partial, and its N=6 and N=9 passes were not packing-only.

## Rule

One partition at every N and every layer: maximize the worst triple's sum of `-J`, and keep the lexicographically smallest partition on a tie. Block means and popcount shells are rebuilt on those triples. `J@J` is unchanged. The circuit is the original shared-gamma KM: one gamma, one thetaK, one thetaM, three shell angles per layer, 21 parameters at L=6. QAOA has 12 parameters at L=6. Both use the round-4 start list. KM_mm does not get extra starts. Interleaved KM is a reference on the same instances and is not the claim.

## Verdict

Overall **partial**. N=6 and N=9 pass the frozen equal-depth rule. N=12 does not.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM_mm | 6 | 11 (11 / 0) | 0 | 1 | +0.287 [+0.111, +0.420] |
| KM_mm | 9 | 12 (10 / 2) | 0 | 0 | +0.309 [+0.166, +0.426] |
| KM_mm | 12 | 8 (8 / 0) | 0 | 4 | +0.131 [-0.007, +0.267] |

The N=9 fidelity-only wins are k=196 and k=198. N=12 has no loss. The four ties are k=191, k=195, k=199, and k=200. On each of those four the maximin energy is above the QAOA energy. The bootstrap interval for the N=12 median crosses 0, so N=12 does not pass. The selected L=6 states at N=12 are not evaluation-capped. The only capped selected L=6 state is N=6 k=194, and its fidelity is 1.

The same-window interleaved reference passes N=6 (11 joint, 0 losses, 1 tie, median +0.213 [+0.099, +0.411]), N=9 (10 wins, 8 joint and 2 fidelity-only, 0 losses, 2 ties, median +0.184 [+0.125, +0.250]), and N=12 (8 joint, 0 losses, 4 ties, median +0.143 [+0.005, +0.254]). Those pass bits are not a maximin gain. On this window the reference clears N=12 and the maximin packing does not. This is one fresh window. It is not a global confirmation of interleaved KM, and it is not a critical-field result.

## Energy against interleaved KM

Drop is interleaved energy minus maximin energy. The frozen descriptive rule is a median above 0.01, or at least four instances above 0.01. It is not a second win rule.

| N | median drop | instances above 0.01 |
|---|---|---|
| 6 | +0.0150 | 7 of 12 |
| 9 | +0.2796 | 9 of 12 |
| 12 | +0.1245 | 8 of 12 |

All three N meet that rule. The N=9 median is the large movement. The N=12 energy drop does not clear the fidelity interval. k=195 is worse by 0.349 and k=199 by 0.522.

## Equal parameter count

KM_mm at L=3 against QAOA at L=6 is 12 parameters and is not the verdict. N=6 has 6 wins (4 joint) and 5 losses, median +0.063 [-0.067, +0.226]. N=9 has 5 wins (1 joint) and 5 losses, median -0.002. N=12 has 2 wins (0 joint) and 5 losses, median -0.035 [-0.128, -0.016].

## Cost and replay

Optimizer wall 2429.3 s of 21600. Starts 43488. Logged energy evaluations 3688780. L=1 grid evaluations 32256. Capped starts 54. Independent Pauli replay of 432 states at each N passes. Maximum relative energy error is 3.8e-13, 3.2e-12, and 1.8e-13. Subspace mismatches are 0. k=202 and above stay unused. k=100 and k=101 stay unused.
