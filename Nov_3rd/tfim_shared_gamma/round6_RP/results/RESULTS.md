# Round 6 RP — independent stride-4 replay

Source commit `d9af7f2ab7e6482cfa1307c6b347e03676e46146`. Manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Packing `0f4330f4a9d4e84db3582beb8b607a2e1092918f7a28b68bb3cff4a1a514c979`.
E0 = -8.6132024589. rK = 2.712107384896. rM = 0.769011889337. τ = 1.000e-09.

L5 fidelity is RECOVERED from the committed cells. It is not missing.

| arm | L | k | direct E | F | Var | ΔE | ΔF | logical/physical | stored overlap | hash match | X+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 114 | -8.385155838742 | 0.889814267763 | 0.972934 | 6.395e-14 | -6.994e-15 | 0.99999999999998 | n/a | None | 1.000000 |
| 0 | 6 | 23 | -8.445682501758 | 0.902370146856 | 0.657101 | -1.599e-14 | 1.443e-15 | 0.99999999999999 | 1.00000000000000 | True | 1.000000 |
| 1 | 5 | 11 | -8.385155838742 | 0.889814305977 | 0.972935 | 1.137e-13 | -1.277e-14 | 0.99999999999999 | 1.00000000000000 | True | 1.000000 |
| 1 | 6 | 107 | -8.445682501760 | 0.902370135017 | 0.657100 | -1.332e-13 | 1.210e-14 | 0.99999999999999 | n/a | None | 1.000000 |
| 2 | 5 | 16 | -8.385155838742 | 0.889814341136 | 0.972934 | 2.238e-13 | -2.442e-14 | 0.99999999999999 | 1.00000000000000 | True | 1.000000 |
| 2 | 6 | 13 | -8.445682501760 | 0.902370179528 | 0.657100 | 2.842e-14 | -3.997e-15 | 0.99999999999999 | 1.00000000000000 | True | 1.000000 |

Family energy-selected L5 is arm 0: E -8.385155838742, F 0.889814267763, Var 0.972934.
Family energy-selected L6 is arm 2: E -8.445682501760, F 0.902370179528, Var 0.657100.

## Relabel

Joint swap of sites 0 and 5: ΔE -3.553e-15, ΔF 4.441e-16.
The same swap applied to the state only: ΔE 3.226e+00, ΔF -8.851e-01.
Joint within-cavity reversal: ΔE 1.776e-15, ΔF -3.331e-16.

## Costs

Cell marginals equal the sum of start `nfev`. No start is missing an energy. Optimizer `success=false` on 62 starts; those rows still have an energy and are not dropped.

Arm 0 inclusive 53649 (L1–L4 raw sums plus imported L5/L6).
Arm 1 inclusive 53353 = optimization 51797 + control nfev_sum 1556.
Arm 2 reconciled inclusive 52927 = optimization 25828 + raw source sum 27099.
The producer constant 28099 is 1000 above that raw sum and is not used.

QAOA floors at p=9..12 are 76799, 79751, 84980, 89272. Each is larger than these family totals.

## Both metrics

- arm0_L5: 16/48 starts beat QAOA p12 on both metrics at τ.
- arm0_L6: 9/48 starts beat QAOA p12 on both metrics at τ.
- arm1_L5: 17/48 starts beat QAOA p12 on both metrics at τ.
- arm1_L6: 35/48 starts beat QAOA p12 on both metrics at τ.
- arm2_L5: 17/24 starts beat QAOA p12 on both metrics at τ.
- arm2_L6: 21/24 starts beat QAOA p12 on both metrics at τ.

The three L5 energy-selected states are one ray (pairwise overlap 1). At L6, arms 0 and 2 are one ray and arm 1 is its conjugate.
Arm 0 L5 and arm 1 L6 are equalize winners. Their 24-row archives do not contain those vectors. Each matches another arm's archived ray at overlap 1, and the archived bytes match the producer state hash.
Where an archive row exists, the phase-aligned residual is about 1e-15, so the recomputed sha256 differs from the archived bytes. Normalized overlap is 1.

Fresh arm 0 medians stay far above the best energy. Nine of 48 fresh L6 starts beat p12 on both metrics. That is a best-state comparison with a weak fresh median. It is not a seal.

Direct H and H² on the L6 family state agree with the full spectrum sum to 1.243e-14. The lowest 20 eigenstates hold weight 0.965949 and do not account for the energy (accounted -8.279916 versus direct -8.445683).
X+ and R+ weights of that state are 1. The +/+ gap is 0.433798. The Eckart lower bound is 0.613830, and the fidelity 0.902370 lies above it.

Figures: `cdf_L5_L6.png`, `best_so_far.png`, `pareto.png`. Baseline: `baseline_v1.json`.
