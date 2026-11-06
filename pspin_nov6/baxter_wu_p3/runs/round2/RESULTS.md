# Round 2 — staggered Baxter–Wu triangles

This is not the round-1 sparse hypergraph and not a random-sign glass. The Hamiltonian is the triangular-lattice torus with opposite signs on the two triangle orientations:

```
H = - sum_{▲} Z_a Z_b Z_c + sum_{▼} Z_a Z_b Z_c - Gamma * sum_i X_i
```

`lambda = 1`. Up faces are `{(x,y),(x+1,y),(x+1,y+1)}` and down faces are `{(x,y),(x,y+1),(x+1,y+1)}`, indices mod `(Lx, Ly)`, site index `x + Lx*y`. Every listed geometry has `M = 2N`, no duplicate face, and no up face equal to a down face.

The fidelity target is the lowest eigenvector of this operator on the full `2^N` space. No sublattice sector was projected out.

## What was checked before QAOA

Gauge. Flipping a set of spins multiplies each triangle by ±1. Reaching the uniform ferromagnet (`-sum` of every face) from the staggered pattern is the GF(2) system `A x = b` with `b = 1` only on down faces. That system is unsolvable on 3×3, 3×4, 4×3, 3×5, 3×6, 6×3, 3×7, and 7×3. A direct enumeration of all `2^N` gauges on N = 9, 12, 18, and 21 found zero solutions. The classical minimum lies strictly above the unconstrained bound `-2N` (N=9: −8 vs −18, degeneracy 72; N=12: −16 vs −24, degeneracy 18; N=15: −16 vs −30, degeneracy 360; N=18: −24 vs −36, degeneracy 48; N=21: −24 vs −42, degeneracy 1512). The staggered texture was kept. It is not gauge-trivial.

Coloring. `color = (x+y) mod 3` is a proper 3-coloring, and the three double-sublattice products of X commute with every triangle, only when both periods are divisible by 3.

| torus | 3-coloring | Z2×Z2 | uniform-ferro classical degeneracy |
|---|---|---|---|
| 3×3 (N=9) | yes | yes | 4 |
| 3×4 and 4×3 (N=12) | no | no | 1 (only the all-+ state) |
| 3×5 (N=15) | no | no | — (staggered degeneracy 360) |
| 3×6 and 6×3 (N=18) | yes | yes | 4 |
| 3×7 and 7×3 (N=21) | no | no | 1 |

N=12 and N=21 do not have the symmetry the planning note assumed. On the tori where the symmetry exists, the full-space ground state is the +1 eigenvector of all three double flips (measured, not imposed): exactly +1 on 3×3 and on 3×6.

Periods. For `lambda = 1` every cost eigenvalue is a multiple of 8, so `gamma` has period `π/4`. A shift by `π/8` is not a global phase. `beta` has period `π` and not `π/2`. Both were checked on generic states, and a one- and two-layer circuit matches a dense `expm` at N=9. Lanczos matches a dense full-space diagonalization at N=9, Gamma = 1, overlap 1. `Emax = -E0` on every ground state that was computed (conjugation by `prod Y_i`).

Uniform-sign control, same faces, both orientations ferro. On 3×4 at Gamma = 0.8 the ground state has half-cut entropy 0.001 and best bitstring overlap 0.95. That is the dressed all-+ product. It was not used as a QAOA target.

`lambda = 0.5` is also frustrated (same sign obstruction, weaker down weight). It was surveyed and not promoted to the QAOA grid; the production runs are the ±1 texture.

## QAOA

Independent `(gamma_l, beta_l)`, initial `|+>`, objective = full energy. L=1 is a grid on `[0, π/4) × [0, π)` plus local energy minimization, and a separate fidelity maximization from the best fidelity nodes. Higher L uses the Pagano interpolation, the L=2 perturbed seed, zero-layer pads, and random starts. Shared-gamma is a side check. A duration-matched digitized anneal uses `beta` negative, as required by `H(s) = (1-s)(-sum X) + s H`.

No cell is false-hard in the round-1 p=4 sense. On every cell the best L=1 fidelity and the energy-selected L=1 fidelity differ by at most about 0.03, and the fidelity maximizer has worse energy. Energy is nested in L on every ladder. Principal-branch replay matches the optimized state.

### Depth ladders worth quoting

Half-cut is a spatial strip when one period is even (`y < Ly/2` on 3×4, 3×6; `x < Lx/2` is unavailable). On 3×3 and 3×5 both periods are odd, so the cut is the lowest `N//2` indices. Entropies below are that cut, in bits.

| geometry | Gamma | L1max | F(L=1) | deep F | F_product | S | weight on classical manifold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3×4, N=12 | 0.7 | 0.225 | 0.194 | 0.770 at L=6 | 0.047 | 3.36 | 0.88 |
| 3×5, N=15 | 0.6 | 0.357 | 0.339 | 0.951 at L=5 | 0.069 | 3.72 | 0.89 |
| 3×6, N=18 | 1.0 | 0.196 | 0.182 | 0.665 at L=5 | 0.051 | 4.00 | 0.61 |

N=12, Gamma 0.7, energy-selected F by depth: 0.047, 0.194, 0.440, 0.492, 0.648, 0.648, 0.770.

N=15, Gamma 0.6: 0.069, 0.339, 0.705, 0.766, 0.918, 0.951.

N=18, Gamma 1.0: 0.051, 0.182, 0.342, 0.506, 0.581, 0.665.

These three sizes each move well past the round-1 sparse result at N=12, which was about 0.222 → 0.293. The L=1 ceiling here is the grid/fidelity maximum, not an energy-selected point sitting under a hidden 0.9 fidelity.

Neighboring fields, same geometries:

- N=12, Gamma 0.9: L1max 0.325 → L6 0.815, F_product 0.105, S 3.24.
- N=12, Gamma 1.1: L1max 0.493 → L6 0.865. L=1 is no longer low.
- N=12, Gamma 1.3: L1max 0.680 → L6 0.970.
- N=15, Gamma 0.8: L1max 0.451 → L5 0.954. N=15, Gamma 1.0: L1max 0.560 → L5 0.979.
- N=18, Gamma 1.3: L1max 0.591 → L5 0.865. L=1 is not low.

N=9 (3×3) does not have a low L=1. At Gamma 0.5, 0.8, and 1.0, L1max is 0.811, 0.870, and 0.901, and F reaches 1.000 by L=4 to L=6. The ground state there is still far from a single bitstring (best bit overlap about 0.01, best product overlap 0.29–0.50) and is close to the equal positive superposition of the 72 classical ground states (overlap 0.82–0.95). Small N makes that state easy for one layer.

Shared gamma tracks L=1 and falls behind at high L. At N=12, Gamma 0.7, L=6 shared-gamma fidelity is 0.507 against independent 0.770. At N=18, Gamma 1.0, L=5 shared gamma is 0.536 against 0.665.

The matched anneal is usually below QAOA after L=1. At N=12, Gamma 0.7, L=6 the anneal fidelity is 0.002 against QAOA 0.770. The comparison is with that fixed Trotter shape, not with a reoptimized anneal.

## What this does not show

The quoted fields are different at each N. F at L=5 is 0.65 (N=12, Gamma 0.7, actually L=5 equals L=4 there), 0.95 (N=15, Gamma 0.6), and 0.67 (N=18, Gamma 1.0). That is not a 1/N collapse, and it is not evidence for or against `2^{-N/2}`. N=21 was not given a QAOA ladder: one full-space ground state there already takes a few minutes, and an N=18 ladder took about 16 minutes per field.

At the low-Gamma side of the window a large fraction of the ground-state weight still sits on the classical ground manifold, in the equal positive cat of those bitstrings. That cat is the full-space ground state, not a sector that was chosen. It is not a product state. The N=18, Gamma 1.0 point is the least cat-like of the three quoted ladders (manifold weight 0.61).

Source claims in the planning brief were not re-read from the papers. The gauge rank, the coloring, the periods, and the fidelities above were computed in this directory.
