# Round 6: extra initial points inside the original shared-gamma circuit

Protocol frozen at `282b9340` before this fit (`PREREG.json` sha256 `933cf6a85e88e64685ba2440b96d3ab8b90fe65353f400137e669c91b4b53352`). The circuit is unchanged: one gamma, one thetaK, one thetaM, three shell controls per layer, interleaved grouping, FK = J@J, FM = block means on that grouping. QAOA is the same equal-depth independent-layer mixer. The four new starts are initial values of those existing parameters. Fidelity was computed after the energy comparison and was not used to accept or reject a start.

## Gate

Development cells are round-4 instances N=9 and N=12, k=4,5,6,7, repeat 0, L=6. Confirmation on fresh k=44..51 was allowed only if at least 4 of these 8 cells lowered the stored KM energy by 0.001 or more. None did. Confirmation was not run. k=44..51 were not fit.

| N | k | dE of best new start | label | evaluations of that start |
|---|---|---|---|---|
| 9 | 4 | -1.9e-10 | couple | 390 |
| 9 | 5 | +2.2e-11 | qproj | 238 |
| 9 | 6 | +4.8e-11 | qproj | 118 |
| 9 | 7 | +0.0276 | qproj | 487 |
| 12 | 4 | +5.2e-12 | couple | 159 |
| 12 | 5 | +1.5e-10 | qproj | 84 |
| 12 | 6 | +8.7e-11 | qproj | 96 |
| 12 | 7 | -3.8e-11 | qproj | 184 |

The N=6 k=4 sanity cell, not part of the gate, was worse by 0.0024. No start hit the 2000-evaluation cap. The 0.15 and 0.40 perturbations either fell into the same energy or landed far above it (several units). N9 k7, the round-3 case where KM has ground-space weight 0 at a lower energy than QAOA, was not repaired: every new start is higher in energy than the stored KM point.

Fidelity of the energy-selected new state matches the stored KM fidelity wherever the energies match. It does not turn the N9 k7 loss into a win, because the new energy is worse and was not selected over the stored point.

Wall time for the nine cells was 44 s.

## What this says

On these cells the QAOA projection and the classical diagonal correlation land in the energy basin the standard search already found, or in a worse one. Round 3 had already shown that independent repeats agree at N=9 and N=12. Together, that is evidence that the missing equal-depth advantage at those sizes is not a missed initial point of this kind.

This is a local-optimum statement about this budget. It is not a proof that no initial point inside the same circuit can do better, and it is not a statement about a different circuit.
