# Historical B=0.1: scale-normalized restarts are partial

Protocol `dbec6066` before any fit. PREREG SHA-256 `01a6658ef6f48bf62d17edcdce4a9e850a76471fe7410c57ef1db044b30c713f`. Fresh instances k=154..165, twelve per N. k=100 and k=101 were not used. The circuit is the original interleaved shared-gamma KM, with the round-4 starts and the nodata to K to KM nesting. After both repeats, twelve L=6 directions of radius 1 are drawn on the unit sphere in box half-width units and minimized on true energy from the continuation vector. The continuation vector stays a candidate. QAOA uses the round-4 list only. No packing change, no pi/4 or pi/2 shell shift, no popcount shells, no repulsion penalty, no extra QAOA starts, no fidelity post-selection, and no full-box uniform draw.

The verdict is the radius-inclusive column at equal depth. The continuation column is the equal-search baseline. Equal parameter count is not the verdict.

## Result

Both columns are **partial**. They match on every instance. Zero of the 432 radius starts has a lower energy than the continuation vector, so every selected label is `historical`. `deeper` is false at N=6, 9, and 12.

| N | continuation | radius | wins / losses / ties | median dF [95% CI] | deeper |
|---|---|---|---|---|---|
| 6 | pass | pass | 12 joint / 0 / 0 | +0.050 [+0.013, +0.083] | false |
| 9 | fail | fail | 6 joint / 2 / 4 | +0.110 [−0.051, +0.243] | false |
| 12 | fail | fail | 4 joint / 0 / 8 | −0.008 [−0.047, +0.178] | false |

N=6 passes on the continuation search: 12 joint wins and an interval above 0. The radius column does not create that pass. N=9 does not. The losses are k155, joint, F_sub 0.357 against 0.459, energy 0.642 higher, and k161, fidelity only, F_sub 0.056 against 0.204, energy 0.056 lower. N=12 has no losses, and its interval includes 0.

Four of the 432 directions return to the continuation energy within 1e-10: N9 k161, and N12 k158, k159, and k165. None goes below it. The nearest start that does not recover that energy sits 0.038 above at N=6, 8.07 above at N=9, and 1.29 above at N=12. No radius start hit the evaluation cap.

Equal parameter count, continuation KM at L=3 against QAOA at L=6, is not a pass. Median dF is −0.013, −0.095, and −0.104.

Independent replay of 300 states at each N: maximum relative energy error 6.5e-13, no subspace-count mismatches. Wall 1131.8 s of 4200. Starts 25344, of which 432 are the extra L=6 directions. Logged energy evaluations 2036775. L=1 grid evaluations 18432. Capped starts 0.

## Reading

On this sample the scale-normalized directions did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten. N=6 clears the historical comparator with that same continuation search, and a lower-energy radius state was not available. N=9 and N=12 do not clear it. This is one fresh window, not a global advantage. `B_cri` stays withheld.
