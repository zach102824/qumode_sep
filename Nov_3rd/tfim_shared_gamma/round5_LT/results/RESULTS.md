# Round 5 LT — fresh ladder and leave-depth-out templates

Manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Packing `4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717`. Hc little-endian `d650ce798d622a3fe37ef3e7f86f2619f36eadf7acf0422aab821a539e216353`. E0 = -8.6132024589.

Preregistration `0a992ab286fb71123eb45153d251647a16c51baeaf5a84aa8f0582dbf5deee1e`, committed before the fitting run. Stored angles were not training seeds. Template L5 used fresh plain L3 and L4 only. Template L6 used fresh plain L3, L4, and L5 only. Plain L5 is a separate search from the L5 template. Wall time 95 s. The 75-minute cap was not reached.

Shell coefficients have a verified π/2 global-phase period on the fixed probe, and the gamma-plus-shell sign flip reproduces the conjugate. Gauge alignment of every training vector stayed on the original or conjugate ray with overlap 1. The raw template centers, before L-BFGS, have energy -3.340 at L5 and -2.803 at L6. The numbers below are after energy-only optimization.

| arm | L | E | error | F | Var(H) | median E | IQR | median F | screen | basin | starts | nfev | best tag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 1 | -7.678785 | 0.9344 | 0.2375 | 1.5700 | -7.3589 | 1.327 | 0.237 | 0/12 | 0/12 | 12/12 | 775 | restart_7 |
| plain | 2 | -8.067337 | 0.5459 | 0.4280 | 1.2302 | -8.0524 | 3.117 | 0.428 | 0/13 | 0/13 | 13/13 | 1268 | restart_1 |
| plain | 3 | -8.111146 | 0.5021 | 0.4331 | 0.7738 | -7.8175 | 7.116 | 0.372 | 0/13 | 0/13 | 13/13 | 2450 | linear |
| plain | 4 | -8.236940 | 0.3763 | 0.6049 | 0.7235 | -6.7525 | 7.271 | 0.053 | 0/13 | 0/13 | 13/13 | 4474 | restart_5 |
| plain | 5 | -8.294861 | 0.3183 | 0.6014 | 0.4524 | -0.9319 | 7.215 | 0.006 | 3/36 | 1/36 | 36/36 | 14284 | restart_1 |
| plain | 6 | -8.321483 | 0.2917 | 0.6098 | 0.3905 | -1.0467 | 7.143 | 0.003 | 3/44 | 2/44 | 44/44 | 17554 | restart_8 |
| template | 5 | -8.294861 | 0.3183 | 0.6014 | 0.4524 | -8.2225 | 0.162 | 0.550 | 12/25 | 7/25 | 25/25 | 13977 | s02_5 |
| template | 6 | -8.321483 | 0.2917 | 0.6098 | 0.3905 | -8.2970 | 0.099 | 0.574 | 12/25 | 1/25 | 25/25 | 17345 | s10_0 |

Plain L5 and L6 rows include the preregistered top-up. Before top-up, plain L5 was 13 starts, nfev 5624, median -0.971; plain L6 was 13 starts, nfev 4637, median -1.170. Template rows had no top-up. Every start finished.

## Posthoc rays

X-parity weight of each energy-selected state is 1. Each is a reflection eigenstate. The JSON field `P_even` is Hamming-weight parity, not this X parity.

- L5 plain: overlap with the stored ray 0.766, with the conjugate 1−2e-11. Energy-selected state is the conjugate.
- L5 template: overlap with the stored ray 1−7e-13. Same ray. The optimized center start has the same energy.
- L6 plain: overlap with the stored ray 0.791, with the conjugate 1−2e-12. Conjugate ray. Two of the 13 preregistered starts land on that energy.
- L6 template: overlap with the stored ray 1−2e-12. The optimized center start is only E=-8.236078 (F 0.549). The stored ray is the sigma=0.1 start `s10_0`, one of 25.

Deltas versus the stored energies are at most a few times 1e-11 in energy. Fidelity deltas sit at or under the 1e-8 dominance tolerance. These are the stored states, or their conjugates, not improvements on them.

## Equal-budget screen

Success means energy at or below the exact round-4 QAOA energy at p=ceil((3L+3)/2): -8.222874170728904 at L5 (p=9) and -8.298215584895452 at L6 (p=11), plus 1e-8. The preregistered screen uses the equalized sample.

| L | template hits | plain hits | gain | evals/success ratio | gain ≥ 0.25 | ratio ≤ 0.5 |
|---|---|---|---|---|---|---|
| 5 equalized | 12/25 | 3/36 | 0.397 | 0.245 | yes | yes |
| 6 equalized | 12/25 | 3/44 | 0.412 | 0.247 | yes | yes |
| 5 preregistered only | 12/25 | 2/13 | 0.326 | 0.414 | yes | yes |
| 6 preregistered only | 12/25 | 3/13 | 0.249 | 0.935 | no | no |

Marginal nfev was equalized by adding plain random starts (23 at L5, 31 at L6) up to the template's nfev. Shared ladder cost is not inside the ratio. No success count was zero. The L6 bars are met on the equalized sample and missed on the original 13-start plain sample. This screen is an optimizer diagnostic. It is not a seal.

Basin hits use the stored replay energy plus 1e-6 and are posthoc. At L6 the template median -8.297 is the screen neighborhood, and only 1/25 starts reach the stored energy.

## Both metrics against QAOA

Dominance requires energy lower by more than 1e-8 and fidelity higher by more than 1e-8 on the same energy-selected state. Plain and template bests agree with each other at the printed precision, so the deltas below cover both arms.

| comparison | dE | dF | both |
|---|---|---|---|
| L5 vs QAOA p5 | +0.3180 | +0.3430 | yes |
| L5 vs QAOA p6 | +0.2380 | +0.3067 | yes |
| L5 vs QAOA p9, ceil(P/2) | +0.0720 | +0.1894 | yes |
| L5 vs QAOA p10 | +0.0311 | +0.1491 | yes |
| L5 vs QAOA p12 | -0.003925 | +0.0867 | no |
| L6 vs QAOA p6 | +0.2646 | +0.3152 | yes |
| L6 vs QAOA p7 | +0.1972 | +0.2766 | yes |
| L6 vs QAOA p11, ceil(P/2) | +0.0233 | +0.1250 | yes |
| L6 vs QAOA p12 | -0.003925 | +0.0951 | no |

Positive dE means a lower energy than that QAOA state. p12 remains lower in energy by 0.003925 and lower in fidelity by about 0.09. That tradeoff is unchanged. Candidate run total is 72127 evaluations. The same-hash QAOA floors are 76799, 79751, 84980, and 89272 at p=9..12, all larger, so no new QAOA starts were added.

## Seal

The both-metric seal is not met. The energy-selected L5 and L6 states match a known ray rather than beating p12 on both energy and fidelity. Plain medians at L5 and L6 stay near -1. The template medians are tighter and clear the weaker screen, and at L6 only one template start reaches the stored energy.

Figures: `seed_cdf.png` (equalized samples), `best_so_far.png`, `pareto.png`. Every start is in `starts.csv` and `summary.json`.
