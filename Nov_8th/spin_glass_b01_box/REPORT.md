# Historical B=0.1: uniform L=6 KM starts are not confirmed

Protocol `9d7e54e0` before any fit. PREREG SHA-256 `8131d186053868438881ea1b5b5fbce06f7975e5c9104c178c5d8b5a88c14fb1`. Fresh instances k=142..153, twelve per N. k=100 and k=101 were not used. The circuit is the original interleaved shared-gamma KM, with the round-4 starts and the nodata to K to KM nesting. After both repeats, twelve L=6 vectors are drawn uniformly in the KM box and minimized on true energy. The continuation vector stays a candidate. QAOA uses the round-4 list only. No packing change, no pi/4 or pi/2 shell shift, no popcount shells, no repulsion penalty, no extra QAOA starts, and no fidelity post-selection.

The verdict is the box-inclusive column at equal depth. The continuation column is the equal-search baseline. Equal parameter count is not the verdict.

## Result

Both columns are **not confirmed**. They match on every instance. Zero of the 432 box starts has a lower energy than the continuation vector, so every selected label is `historical`.

| N | continuation | box | wins / losses / ties | median dF [95% CI] | deeper |
|---|---|---|---|---|---|
| 6 | fail | fail | 8 joint / 1 / 3 | +0.026 [−0.001, +0.234] | false |
| 9 | pass | pass | 8 joint, 1 fidelity-only / 1 / 2 | +0.161 [+0.052, +0.264] | false |
| 12 | fail | fail | 5 joint / 0 / 7 | +0.024 [−0.007, +0.229] | false |

N=6 has more wins than losses, and the interval for the median still includes 0, so the frozen pass rule fails. The loss is k152, joint, F_sub 0.676 against 0.787, energy 0.139 higher. N=12 has no losses, and its interval also includes 0. N=9 passes on this window with the continuation search. The box column does not create that pass. It does not revise the N=9 failures on earlier windows.

The best box start sits 0.275 above the continuation energy (N6 k148, `box_6`). The best at N=9 sits 6.04 above, and the best at N=12 sits 16.83 above. None of the 432 draws is within 0.1. No box start hit the evaluation cap.

Equal parameter count, continuation KM at L=3 against QAOA at L=6, is not a pass. Median dF is −0.047, −0.061, and −0.097.

Independent replay of 300 states at each N: maximum relative energy error 2.9e-13, no subspace-count mismatches. Wall 1090.7 s of 4200. Starts 25344, of which 432 are the extra L=6 draws. Logged energy evaluations 1994365. L=1 grid evaluations 18432. Capped starts 2, both continuation KM at N6 k148 and L=5. The selected L=6 state on that instance is `p1_time_rescaled` and was not capped.

## Reading

On this sample the uniform L=6 draws did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten. N=9 clears the historical comparator with that same continuation search, and a lower-energy box state was not available to keep or lose that clearance. N=6 and N=12 do not clear it. This is one fresh window, not a global advantage. `B_cri` stays withheld.
