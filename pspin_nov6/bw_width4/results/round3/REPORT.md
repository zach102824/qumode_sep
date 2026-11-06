# Round 3 — staggered Baxter–Wu, width 4

SQS and the pure four-body family are closed. This round is only the staggered triangular-torus model

```
H = - sum_▲ Z_a Z_b Z_c + sum_▼ Z_a Z_b Z_c - Gamma * sum_i X_i
▲(x,y) = {(x,y), (x+1,y), (x+1,y+1)}
▼(x,y) = {(x,y), (x,y+1), (x+1,y+1)}
index = x + Lx * y, qubit q = bit q
```

`λ = 1`. No random couplings. The triangle list matches the reference enumeration: the 3×3 faces agree with `pspin_nov6/baxter_wu_p3` on `cursor/sparse-p3-qaoa-round1-13b4`, and the cost diagonal agrees with an independent Kronecker product of `Z` (max error 0). Wall time 951 s. No speedup claim.

## 4×3 is the same model as 3×4

Both tori have 12 sites, 12 up faces, 12 down faces, no duplicate face, and no up face equal to a down face (`M = 24`). An affine map of the triangular lattice,

```
(x, y) -> (0, 0) + x*(0, -1) + y*(-1, 0)
```

sends the 4×3 signed faces onto the 3×4 signed faces with up staying up and down staying down. The cost spectra are identical. The full Hamiltonian at `Gamma = 1` has eigenvalue difference at most `4e-14` (`E0 = -17.842` on both). 4×3 is an orientation and implementation check, not a second system size.

Neither torus is 3-colorable. `color = (x+y) mod 3` has monochrome edges, and forcing colors around the faces conflicts. There are no sublattices A, B, C, so `m_A, m_B, m_C` are not reported. The 3-sublattice momentum `K = (b1+b2)/3`, with `a1=(1,0)`, `a2=(1/2, √3/2)`, is not on the finite-torus grid unless both periods are multiples of 3. It is not, on 4×3 or on 4×4. Structure factors below are on the momenta that do exist. They are not Bragg peaks at K.

`Gamma` has period `π/4`: every cost eigenvalue is a multiple of 8, and a shift by `π/8` is not a global phase. `beta` has period `π`. `Emax = -E0` on every ground state computed here (residual of `Emax+E0` at most `1e-13`).

Classical ground-state degeneracy of `H_C` is 18 on 4×3, the same count previously reported for 3×4, and 640 on 4×4. The uniform-ferromagnet value would be `-2N` (`-24` and `-32`). The minima are `-16` and lower than that bound is impossible for the staggered signs; the reference already showed the staggered pattern is not a gauge copy of the uniform ferromagnet on 3×4 and 4×3.

## Ground state

Full `2^N` space. Dense `eigh` at N=12, Lanczos at N=16. Residuals are about `1e-14`. The half cut is the sites with `x < 2` (8 of 16 sites on 4×4, 6 of 12 on 4×3). A second cut `y < 2` exists only on 4×4; `S_min` over those explicit cuts is in the CSV. Entropies below are the `x < 2` cut, in bits.

`Gamma*` on this four-point grid is the grid point with the smallest gap. That point is `Gamma = 0.6` for both tori, i.e. the edge of the window, so `Gamma/Gamma*` is not an interior finite-size critical estimate.

| torus | Gamma | Gamma/0.6 | E0 | gap | F_manifold | F_product search | S (x<2) | ⟨ZZZ⟩_▲ = −⟨ZZZ⟩_▼ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4×3 | 0.6 | 1 | -16.576 | 0.010 | 0.920 | 0.032 | 3.38 | 0.639 |
| 4×3 | 0.8 | 1.33 | -17.084 | 0.042 | 0.837 | 0.070 | 3.32 | 0.608 |
| 4×3 | 1.0 | 1.67 | -17.842 | 0.140 | 0.694 | 0.155 | 3.12 | 0.553 |
| 4×3 | 1.2 | 2 | -18.946 | 0.372 | 0.493 | 0.309 | 2.66 | 0.474 |
| 4×4 | 0.6 | 1 | -19.197 | 0.134 | 0.896 | 0.062 | 4.17 | 0.473 |
| 4×4 | 0.8 | 1.33 | -20.667 | 0.240 | 0.803 | 0.115 | 3.85 | 0.448 |
| 4×4 | 1.0 | 1.67 | -22.394 | 0.422 | 0.672 | 0.208 | 3.42 | 0.408 |
| 4×4 | 1.2 | 2 | -24.417 | 0.710 | 0.516 | 0.340 | 2.86 | 0.355 |

The product number is the best real nonnegative product state found by coordinate descent. It is not a certified upper bound on every product state. The manifold weight is the probability of the classical ground set of `H_C`. At `Gamma = 0.6` that weight is about 0.9, so the ground state is close to the uniform superposition of those bitstrings. That superposition was not imposed. No orbit-cat inequality was proved.

## QAOA

Independent `(γ_ℓ, β_ℓ)`, `|+⟩`, energy of the full `H`. Layer 1 acts first: cost, then `sum X`. L=1 is a grid on `[0, π/4) × [0, π)` plus local minimization, and a separate fidelity maximization. Higher L uses the Pagano L=2 offset, interpolation of the raw angles for L≥3, perturbations, a sign flip, random starts, and zero-pads at both ends. The unevolved zero-pad is kept, so the recorded energy does not increase with L. Fidelity is not forced to be monotone. Where the flag `fidelity_monotone` is false, the energy and the printed fidelity are unchanged from the previous depth: the search returned the shallower circuit. Shared-γ is a sidecar. The anneal is the midpoint Trotter of `(1-s)(-sum X) + s H`, with total angle weight matched to the raw QAOA duration. Replays of the saved angles reproduce energy and fidelity with error 0.

`F1E` is the energy-selected L=1 overlap. `F1max` is the fidelity-maximized L=1 overlap. Their difference is 0.02–0.03. None of these cells is the round-1 false-hard split.

### 4×3 (same physics as 3×4)

| Gamma | F(L=0) | F1E | F1max | L=2 | L=3 | L=4 | L=5 | L=6 | anneal F at L=6 | shared-γ F at L=6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.032 | 0.164 | 0.193 | 0.381 | 0.381 | 0.628 | 0.628 | 0.744 | 0.000 | 0.391 |
| 0.8 | 0.070 | 0.236 | 0.269 | 0.480 | 0.633 | 0.774 | 0.814 | 0.915 | 0.002 | 0.508 |
| 1.0 | 0.155 | 0.368 | 0.403 | 0.569 | 0.757 | 0.841 | 0.916 | 0.966 | 0.024 | 0.828 |
| 1.2 | 0.309 | 0.563 | 0.592 | 0.730 | 0.862 | 0.912 | 0.950 | 0.980 | 0.068 | 0.784 |

This sits next to the reference 3×4 ladders (Gamma 0.7: L1max 0.225, L=6 fidelity 0.770; Gamma 0.9: L1max 0.325, L=6 fidelity 0.815). The orientation agrees, and it is not a new N.

### 4×4

| Gamma | F(L=0) | F1E | F1max | L=2 | L=3 | L=4 | L=5 | anneal F at L=5 | shared-γ F at L=5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.062 | 0.302 | 0.328 | 0.526 | 0.765 | 0.906 | 0.968 | 0.014 | 0.838 |
| 0.8 | 0.115 | 0.385 | 0.413 | 0.588 | 0.800 | 0.924 | 0.973 | 0.002 | 0.863 |
| 1.0 | 0.208 | 0.508 | 0.533 | 0.708 | 0.721 | 0.892 | 0.892 | 0.020 | 0.789 |
| 1.2 | 0.340 | 0.654 | 0.673 | 0.787 | 0.787 | 0.906 | 0.951 | 0.227 | 0.947 |

Width 4 does not fail. At Gamma 0.6 the L=1 ceiling is 0.33 and the energy-selected state reaches 0.97 by L=5, with the triangle expectations moving from 0 at L=0 to `⟨ZZZ⟩_▲ = 0.462 = -⟨ZZZ⟩_▼` against a ground-state value 0.473. The same pattern holds at 0.8. At 1.0 and 1.2, L=1 is already above 0.5, so those fields are not a low-L=1 test; the depth still improves the energy. Calling this “quasi-1D only” would be wrong: the strip works and the square works. It is also not a universal claim about larger widths.

## Errors versus depth

`r_L = (E-E0)/gap`. The spectral bound `F ≥ 1-r_L` is vacuous whenever `r_L > 1`, which is most of the ladder, because the gap is smaller than the residual energy. It only becomes informative at the right-hand end of the higher-Gamma curves (4×3, Gamma 1.2, L=6: `r_L = 0.28`; 4×4, Gamma 1.2, L=5: `r_L = 0.39`). Overlap is the direct number everywhere else.

Triangle and correlator errors do shrink from L=1 to the last depth:

| point | energy error L=1 → last | |Δ⟨ZZZ⟩_▲| L=1 → last | connected-ZZ error L=1 → last |
|---|---|---|---|
| 4×3, 0.6 | 5.39 → 1.56 | 0.404 → 0.118 | 0.205 → 0.039 |
| 4×3, 1.0 | 2.78 → 0.189 | 0.347 → 0.027 | 0.190 → 0.005 |
| 4×4, 0.6 | 4.28 → 0.227 | 0.237 → 0.011 | 0.107 → 0.005 |
| 4×4, 1.0 | 2.31 → 0.565 | 0.201 → 0.055 | 0.096 → 0.023 |

The matched anneal stays far below the energy-QAOA fidelity on these curves (typically under 0.03, and 0.23 only for 4×4 at Gamma 1.2, L=5, against QAOA 0.95). That is one Trotter schedule with its duration fixed to the QAOA angle weight, not an optimized anneal and not a runtime comparison.

Shared-γ tracks the independent schedule at L=1 and is lower at the deepest point except 4×4, Gamma 1.2, where it is close (0.947 vs 0.951).

## What was not done

No angle jitter, no `J` disorder, and no `λ ∈ {0.75, -1}` ladder. The reference uniform-sign control on 3×4 was already a dressed product state; repeating it was not needed to read the staggered grid. No p=4 probe. No certified product-state or orbit-cat upper bound. Fidelity was not monotone at the two plateaus noted above, and that is not a failure of the energy objective.
