# Historical B=0.1 QAOA-depth diagnosis

Benchmark `B = 0.1`. This is the author's historical circuit field. It is not `B_cri`. The `|J|`-max confirmation, the Fiedler confirmation, the energy-only union, the quarter-pi restarts, the classical popcount initialization, and the repulsion escape stay as they were recorded.

## Idea

The circuit is the original interleaved shared-gamma KM. One gamma, one thetaK, one thetaM, three shell angles per layer, the same at every N. The round-4 start list is kept, including the nodata to K to KM nesting. QAOA uses that same list for two repeats. That pair is the historical equal-depth comparator.

After both repeats, sixteen more QAOA starts run at L=6 only. Eight are uniform in the parameter box. Eight add N(0, 0.75) to the energy-best historical L=6 vector and clip back to the box. The historical vector stays in the candidate set. Selection is the lowest true energy. This is not the round-3 arm that multiplied the L=1 grid. No packing changes, no π/4 or π/2 shell shift, no popcount shells, and no repulsion penalty.

## Protocol

Frozen in `bbaa6e06` before the fits. `PREREG.json` SHA-256 `1150031e970524aa0eee55a4acb6376bfefeaa78f4b10c3a851ac44b660d4ca6`. Instances k=118..129. k=100 and k=101 stay unused. Equal depth is KM at L=6 against historical QAOA at L=6. The deep column uses the same win rule and does not rewrite it. An N counts as under-optimized only if the median energy drop exceeds 0.01 or at least 4 of 12 instances drop by more than 0.01.

A size passes only when KM has more wins than losses and the bootstrap 95% interval for the median fidelity difference lies above 0. The historical comparison is confirmed only if N=6, 9, and 12 all pass. If the deep column reversed a historical pass, the round would not be called confirmed.

On development index k=4, before the confirmation jobs, the energy gradients matched finite differences, the sixteen deep vectors sat in the QAOA box, gamma was shared, and a π/2 shift of shell channel 0 remained a global phase.

## Result

Historical comparator: **partial**. Heavier QAOA search: **partial**. The extra starts did not lower the QAOA energy on any of the 36 instances, so the two columns are the same comparison. N=9 and N=12 are not under-optimized by the frozen rule.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM | 6 | 11 (11 / 0) | 0 | 1 | +0.143 [+0.065, +0.262] |
| KM | 9 | 7 (6 / 1) | 2 | 3 | +0.087 [−0.027, +0.131] |
| KM | 12 | 5 (5 / 0) | 0 | 7 | +0.045 [−0.026, +0.167] |

N=6 passes, and that pass is unchanged against the heavier search. N=9 has more wins than losses, but the interval includes 0. N=12 has more wins than losses, and the interval includes 0. The N=9 losses are:

- k118, joint, a time-rescaled start, F_sub 0.447 against 0.541, energy 0.290 higher.
- k119, fidelity only, a time-rescaled start, F_sub 0.083 against 0.205, energy 0.296 lower.

Every selected `qaoa_deep` label is `historical`. The closest extra start sat 0.089 above the historical energy (N=6 k120). None of the 576 extra starts entered the 0.01 window. Median energy drop is 0 at each N, and the count above 0.01 is 0. Mean L=6 F_sub, KM / QAOA: N=6 0.771 / 0.576; N=9 0.410 / 0.346; N=12 0.316 / 0.205.

Equal parameter count, KM at L=3 against QAOA at L=6, is not the verdict. Median dF is +0.027, −0.113, and −0.100. The N=6 interval includes 0. The N=9 and N=12 intervals lie below 0. The deep column matches these numbers because it selected the same QAOA states.

## Checks and cost

Independent replay of 300 states at each N, including the deep-selected vectors on the QAOA layout. Maximum relative energy error 4.6e-13. Maximum absolute F_sub error 4.4e-13. No subspace-count mismatch. All 36 jobs completed. Wall 1159 s of 4200. Starts 25488, of which 576 are the extra L=6 QAOA starts, and none of those 576 was selected. Logged energy evaluations 1962925. L=1 grid evaluations 18432. Capped starts 6, all KM at N=6, k=126, L=6.

`B_cri` stays withheld. This is not a global advantage. The heavier search is not an equal-search claim.
