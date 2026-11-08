# Historical B=0.1 repulsion escape

Benchmark `B = 0.1`. This is the author's historical circuit field. It is not `B_cri`. The `|J|`-max confirmation, the Fiedler confirmation, the energy-only union, the quarter-pi restarts, and the classical popcount initialization stay as they were recorded.

## Idea

The circuit is the original interleaved shared-gamma KM. One gamma, one thetaK, one thetaM, three shell angles per layer, the same at every N. The round-4 start list is kept, including the nodata to K to KM nesting. QAOA keeps that list and does not get the extra runs. Equal search is not claimed.

After the standard starts at a depth, φ is the energy-best KM state. Three kicks copy its shells exactly: negate thetaK and thetaM, swap those two angles, or add N(0, 1) to the three head angles. Each kick minimizes energy plus |⟨φ|ψ⟩|² with weight 1, then a pure-energy polish starts from the penalized minimizer. The selected state is the lowest true energy. The penalty is not a selection value. No shell is shifted by π/4 or by π/2.

## Protocol

Frozen in `0db98373` before the fits. `PREREG.json` SHA-256 `93ed4230099ce65d77a3f14804aabe5fd0bcf0b036f38f098e3c821bb2e14907`. Instances k=106..117. k=100 and k=101 stay unused. Two repeats. Equal depth is KM at L=6 against QAOA at L=6.

A size passes only when KM has more wins than losses and the bootstrap 95% interval for the median fidelity difference lies above 0. The round is confirmed only if N=6, 9, and 12 all pass.

On development index k=4, before the confirmation jobs, the penalty gradient matched a finite difference and the three kicks changed the state. Overlaps against the unkicked state were below 0.24 at N=6, 9, and 12. A π/2 shift of shell channel 0 remained a global phase.

## Result

Overall: **partial**. The repulsion escape does not close N=9 or N=12.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM | 6 | 11 (11 / 0) | 1 | 0 | +0.134 [+0.014, +0.292] |
| KM | 9 | 7 (7 / 0) | 1 | 4 | +0.214 [−0.051, +0.363] |
| KM | 12 | 4 (3 / 1) | 1 | 7 | +0.016 [−0.039, +0.180] |

N=6 passes. N=9 has more wins than losses, but the interval includes 0. N=12 has more wins than losses, and the interval includes 0. The losses are:

- N6 k117, fidelity only, a nested K start, F_sub 0.013 against 0.558, energy 0.190 lower. The escape did not find a lower energy than this state.
- N9 k117, joint, and the selected state is itself the sign-flip polish, F_sub 0.155 against 0.277, energy 0.022 higher.
- N12 k117, fidelity only, F_sub 0.311 against 0.404, energy 0.142 lower. The escape did not replace it.

Five of the 36 L=6 states are an escape label. Two are N=6 joint wins (`polish_swap` on k106 and k107). One is the N=9 joint win on k115. One is the N=9 loss on k117. One is an N=12 tie on k108, F_sub 0.042 against 0.121. Mean L=6 F_sub, KM / QAOA: N=6 0.848 / 0.728; N=9 0.534 / 0.374; N=12 0.273 / 0.195.

Equal parameter count, KM at L=3 against QAOA at L=6, is not the verdict. Median dF is −0.016, −0.058, and −0.041. Each size has more losses than wins, and the N=9 and N=12 intervals lie below 0.

## Checks and cost

Independent replay of 288 states at each N. Maximum relative energy error 7.8e-13. Maximum absolute F_sub error 3.8e-13. No subspace-count mismatch. All 72 jobs completed. Wall 939 s of 4200. Starts 27504, of which 2592 are repulsion or polish. Logged energy evaluations 2142548. L=1 grid evaluations 18432. Capped starts 43, all at N=6.

`B_cri` stays withheld. This is not a global advantage.
