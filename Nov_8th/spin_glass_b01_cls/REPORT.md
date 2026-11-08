# Historical B=0.1 classical popcount initialization

Benchmark `B = 0.1`. This is the author's historical circuit field. It is not `B_cri`. The `|J|`-max confirmation, the Fiedler confirmation, the energy-only union, and the quarter-pi restarts stay as they were recorded.

## Idea

The circuit is the original interleaved shared-gamma KM. One gamma, one thetaK, one thetaM, three shell angles per layer, the same at every N. The KM seeds are not the round-4 grid, not a packing union, and not a π/4 or π/2 hop of an optimized vector.

The shells come from the classical ZZ diagonal only. The smallest index that minimizes that diagonal is read as a bitstring. Each interleaved cavity contributes its popcount. One shared coefficient vector is the least-squares match of the shell weights to a 0/1 target on those popcounts. L=1 tries both signs of gamma, three scales, and both signs of that vector. Later depths keep the round-4 continuation and add two classical reseeds. QAOA keeps the round-4 start list. Equal search is not claimed.

## Protocol

Frozen in `c0dddb7e` before the fits. `PREREG.json` SHA-256 `45957ee313312515005432d069a670a11ad30ee1bc5e075fe01ebcdb53822d2c`. Instances k=92..99 and k=102..105. k=100 and k=101 stay unused. Two repeats. Equal depth is KM at L=6 against QAOA at L=6.

A size passes only when KM has more wins than losses and the bootstrap 95% interval for the median fidelity difference lies above 0. The round is confirmed only if N=6, 9, and 12 all pass.

On development index k=4, before the confirmation jobs, the classical shells changed the state. Overlap against the same gamma with zero shells was 0.708, 0.742, and 0.571 at N=6, 9, and 12. A π/2 shift of shell channel 0 remained a global phase.

## Result

Overall: **partial**. The classical seeds do not close N=9 or N=12.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM | 6 | 11 (11 / 0) | 0 | 1 | +0.253 [+0.127, +0.418] |
| KM | 9 | 7 (7 / 0) | 3 | 2 | +0.111 [−0.040, +0.245] |
| KM | 12 | 2 (1 / 1) | 2 | 8 | +0.002 [−0.025, +0.050] |

N=6 passes. N=9 has more wins than losses, but the interval includes 0. N=12 has two wins and two losses, and the interval includes 0. The losses are:

- N9 k94, joint, a positive classical reseed, F_sub 0.638 against 0.737, energy 0.261 higher.
- N9 k95, joint, F_sub 0.771 against 0.824, energy 0.243 higher.
- N9 k99, joint, F_sub 0.359 against 0.489, energy 0.495 higher.
- N12 k95, joint, F_sub 0.042 against 0.140, energy 0.081 higher.
- N12 k96, fidelity only, F_sub 0.000 against 0.249, energy 0.314 lower. Energy kept a state with no ground-space overlap.

The one fidelity-only win is N12 k105, F_sub 0.233 against 0.046, energy 0.326 higher. Mean L=6 F_sub, KM / QAOA: N=6 0.873 / 0.593; N=9 0.572 / 0.471; N=12 0.145 / 0.133. Every L=6 state traces to a classical L=1 seed. Six of the 36 selected labels are themselves a classical reseed.

Equal parameter count, KM at L=3 against QAOA at L=6, is not the verdict. Median dF is −0.004, −0.144, and −0.062. N=9 and N=12 have more losses than wins, and both intervals lie at or below 0.

## Checks and cost

Independent replay of 288 states at each N. Maximum relative energy error 8.5e-13. Maximum absolute F_sub error 4.6e-13. No subspace-count mismatch. All 72 jobs completed. Wall 447.4 s of 3600. Starts 12240, of which 1584 carry a classical label. Logged energy evaluations 925477. L=1 QAOA grid evaluations 4608. Capped starts 25, all KM at N=6 (13 at L=6, 11 at L=5, 1 at L=4).

`B_cri` stays withheld. This is not a global advantage.
