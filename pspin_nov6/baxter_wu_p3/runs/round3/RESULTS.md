# Round 3 — matched Gamma grid on staggered Baxter–Wu strips

Same Hamiltonian as round 2, same normalization at every size. Triangle coefficients are −1 on up faces and +1 on down faces. They are not divided by N, and Gamma is not rescaled by a fitted Gamma*.

```
H = - sum_▲ ZZZ + sum_▼ ZZZ - Gamma * sum_i X_i
```

Up face `(x,y), (x+1,y), (x+1,y+1)`. Down face `(x,y), (x,y+1), (x+1,y+1)`. Index `x + Lx*y` is bit q (LSB = qubit 0). The target is the full-space ground state.

This file is the strip worker only (3×Ly). Width-4 tori are not in this run. The numeric success checks in the planning note are diagnostics. A ladder is reported whether or not it is monotone, whether or not it beats the anneal, and whether or not a certified cat bound exists.

## Verified

Periods, recomputed from each cost diagonal and checked on a random state. On 3×3, 3×4, 3×5, and 3×6 the cost eigenvalues are multiples of 8, so gamma has period π/4 ≈ 0.785398. A shift by half that period is not a global phase (overlaps 0.176, 0.089, 0.053, 0.033). Beta has period π and not π/2. Geometry hashes (sha256 of the canonical JSON) are in `periods.json`:

| torus | sha256 |
|---|---|
| 3×3 | `5b7f5074b41d…` full string in periods.json |
| 3×4 | `13a2dbf9a191…` |
| 3×5 | `dd2e895b2d95…` |
| 3×6 | `72597a804a74…` |

Ground-state residuals of the Lanczos vector are 1e-14 to 1e-11. Dense eigh matches Lanczos at 3×3 and 3×4 for Gamma 0.8 and 1.0 (overlap 1). `Emax + E0` remains consistent with the odd-body sign symmetry.

Gamma* on the scan `{0.4, 0.6, …, 2.0}` is the left endpoint 0.4 for every strip (gaps 0.188, 0.0016, 0.067, 0.00028). That is not an interior minimum. The gap keeps falling as Gamma decreases. `Gamma/Gamma*` in the CSV uses this endpoint value and is not a retuning of H. The matched fields below are the raw numbers 0.6, 0.8, 1.0, and 1.2.

L=1 energy search and L=1 fidelity search agree. `F1max − F1E` is at most about 0.03 on every cell, and the fidelity-maximizing point has worse energy. No cell is the false-hard pattern.

Energy selects the schedule. Fidelity is not forced to rise. The only non-monotone step is 3×4 at Gamma 0.7, L=5 → L=6: F goes from 0.8854 to 0.8848 while the energy decreases. That step is kept.

## Matched ladders (energy-selected F)

`F1max` is the best fidelity on the L=1 grid. `F1E` is the fidelity at the energy minimizer. Deep F is the last depth.

| torus | Gamma | F1max | F1E | deep F | depth | F1max ≤ ½ deep |
|---|---:|---:|---:|---:|---:|---|
| 3×3 N=9 | 0.6 | 0.837 | 0.836 | 1.000 | 4 | no |
| 3×3 | 0.8 | 0.874 | 0.870 | 1.000 | 4 | no |
| 3×3 | 1.0 | 0.904 | 0.901 | 1.000 | 4 | no |
| 3×3 | 1.2 | 0.932 | 0.926 | 1.000 | 4 | no |
| 3×4 N=12 | 0.6 | 0.192 | 0.164 | 0.876 | 6 | yes |
| 3×4 | 0.8 | 0.269 | 0.236 | 0.895 | 6 | yes |
| 3×4 | 1.0 | 0.402 | 0.368 | 0.922 | 6 | yes |
| 3×4 | 1.2 | 0.585 | 0.563 | 0.956 | 6 | no |
| 3×5 N=15 | 0.6 | 0.365 | 0.339 | 0.939 | 6 | yes |
| 3×5 | 0.8 | 0.453 | 0.426 | 0.976 | 6 | yes |
| 3×5 | 1.0 | 0.565 | 0.543 | 0.989 | 6 | no |
| 3×5 | 1.2 | 0.690 | 0.671 | 0.994 | 6 | no |
| 3×6 N=18 | 0.8 | 0.090 | 0.079 | 0.420 | 5 | yes |
| 3×6 | 1.0 | 0.196 | 0.182 | 0.665 | 5 | yes |

N=9 is an easy control: one layer is already above 0.8. The strips where L=1 is much smaller than the deep value are 3×4 at 0.6 and 0.8, 3×5 at 0.6 and 0.8, and 3×6 at 0.8 and 1.0. Two of those fields, 0.6 and 0.8, show that pattern on both non-÷3 strips. Only 0.8 and 1.0 were run at N=18. N=18 at 0.6 and 1.2 was not given its own ladder.

N=18, Gamma 0.8, depth series: 0.079, 0.224, 0.274, 0.274, 0.420. N=18, Gamma 1.0: 0.182, 0.342, 0.506, 0.581, 0.665. N=12, Gamma 0.6: 0.164, 0.410, 0.628, 0.756, 0.756, 0.876. N=15, Gamma 0.6: 0.339, 0.705, 0.888, 0.939, 0.939, 0.939.

Restarts: 20 random starts plus the Pagano seeds on N≤15. N=18 used 8 random starts. Several L=1 basin gaps are ~1e-13, so many starts land on the same energy. Some deeper levels still have a second basin 0.01–0.7 above the winner, and a few polish steps moved the energy by more than 1e-4 (`converged=false` in the CSV). Those rows are the less settled ones.

## Replay of the round-2 point, separate from the matched claim

3×4 at Gamma 0.7, same 20-restart search: F1max 0.225, F1E 0.194, F(L=6) 0.885. Round 2 at this point stopped at 0.770. The extra restarts and the warm start from Gamma 0.6 raised the deep fidelity. The last step is the tiny downward fidelity step above. 3×5 at 0.6 and 3×6 at 1.0 are already rows of the matched grid; they reproduce the round-2 fidelities (N=15 L1 0.339 and L5 0.939; N=18 L1 0.182 and L5 0.665).

## Angle transfer (no reoptimization)

3×4 angles evaluated on the other strips at the same raw Gamma:

| Gamma | L | F on 3×5 | vs 3×4 F | F on 3×6 | vs 3×4 F | N=18 reopt F at that L |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 6 | 0.887 | +0.010 | 0.581 | −0.295 | not run |
| 0.8 | 5 | 0.937 | +0.135 | 0.524 | −0.279 | 0.420 |
| 0.8 | 6 | 0.912 | +0.017 | 0.621 | −0.274 | not run |
| 1.0 | 5 | 0.960 | +0.111 | 0.630 | −0.220 | 0.665 |
| 1.0 | 6 | 0.946 | +0.024 | 0.713 | −0.209 | not run |
| 1.2 | 6 | 0.976 | +0.020 | 0.872 | −0.084 | not run |

Transfer onto 3×5 at L=6 stays within about 0.02 of the 3×4 fidelity. Transfer onto 3×6 does not. At Gamma 0.8, L=5, the transferred state on 3×6 has F=0.524 and a lower energy than the 8-restart reoptimization (F=0.420). A further energy polish of that transferred seed reached F=0.471 at a still lower energy. The blind N=18 search at Gamma 0.8 is therefore not the best energy found. At Gamma 1.0 the blind L=5 point (F=0.665) is better in energy and fidelity than the transferred seed and than an energy polish of that seed (F=0.589). The N=21 transfer test was not opened: the 3×6 numbers are not within 0.05 of the reoptimized fidelity at the low-L1 fields, and N=21 statevectors were not justified.

## Diagnostics that are not gates

Shared gamma is close at L=1 and lower at high L (N=12, Gamma 0.6, L=6: 0.517 vs 0.876; N=18, Gamma 1.0, L=5: 0.536 vs 0.665).

The matched anneal is below the energy-selected QAOA at the deep level of the low-L1 rows (N=18, Gamma 1.0, L=5: 0.357 vs 0.665; N=18, Gamma 0.8, L=5: 0.001 vs 0.420). That is a comparison with one Trotter shape, not a claim about every anneal.

Product overlap and the equal-positive classical cat are computed overlaps, not certified upper bounds. At Gamma 0.6 the ground state still has classical-manifold weight 0.92 (N=12) and 0.89 (N=15). At N=18, Gamma 0.8, that weight is 0.79; at Gamma 1.0 it is 0.61. A high fidelity there is fidelity to a state that still sits mostly on that manifold. Optimized product overlaps on those ground states are 0.03, 0.07, 0.01, and 0.05 respectively, so they are not single product states, but the product numbers are lower bounds from a local search.

Width 3 is a quasi-1D strip. Nothing here is a two-dimensional thermodynamic claim. F values are this worker's statevectors until the other worker replays them. Hashes of the ground states and of the QAOA states are in `gs_energies.csv` and `angles.json`.

## Unknown

Whether an independent implementation reproduces these fidelities to 0.01. Whether a longer N=18 search at Gamma 0.8 closes the gap to the transferred schedule. Whether Gamma 0.6 or 1.2 on 3×6 continues the same pattern. The 4×4 orientation. Any asymptotic scaling.
