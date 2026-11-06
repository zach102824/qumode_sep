# Round 4 terminal report

Snapshots: A `fbaac8375a9159678c0ece3e54bde60bbf62d5f1` (PR #71, no round-4 directory), C `b8d2d35e96315a9083ac1dc164b3be987c8ed187`. Hamiltonian rebuilt here. No N=21. No new family. A later N=18 commit is not in this file.

```text
H = -sum_up ZZZ + sum_down ZZZ - Gamma * sum X
```

Gamma period is π/4 on 3×4, 4×3, 3×5, 3×6, and 4×4 (cost steps of 8). π/8 is not a period. Full-space Krylov. Residuals about 1e-14. No symmetry sector was used.

## PASS/FAIL at Γ=0.8

| N | F_deep | F1max | (a) F_deep ≥ 2 F1max | (b) F_deep ≥ 0.4 | (c) \|ΔF\| ≤ 1e-6 | line |
| --- | ---: | ---: | --- | --- | --- | --- |
| 12 | 0.894930 | 0.269498 | pass | pass | pass (ΔF ≈ 9e-16) | **PASS** |
| 15 | 0.975507 | 0.459717 | pass | pass | pass (ΔF ≈ 6e-15) | **PASS** |
| 16 | 0.973364 | 0.412965 | pass | pass | pass (ΔF ≈ 4e-15) | **PASS** |
| 18 | 0.419771 blind | 0.095229 | pass | pass | pass (ΔF ≈ 4e-15) | **PASS** |

Schmidt upper bound versus that deep fidelity: N=12 is 0.150 against 0.895. N=18 is 0.055 against 0.420. Both are far below.

N=18 **PASS** uses the blind energy-selected state. Two other verified states are not the same number: transferred F=0.523587 at energy −23.484869, and the lower-energy polish F=0.470995 at −23.596105. The blind state is −23.412114. Lowest energy is not highest fidelity.

## Replay

All |ΔF| below are inside 1e-10. Energies match to about 1e-13 or better on the same state.

| schedule | published F | replay F | energy |
| --- | ---: | ---: | ---: |
| 3×4 Γ=0.8 L=6 | 0.894930 | 0.894930 | −16.512339 |
| 3×5 Γ=0.8 L=6 | 0.975507 | 0.975507 | −19.774608 |
| 4×4 Γ=0.8 L=5 | 0.973364 | 0.973364 | −20.485661 |
| 3×6 Γ=0.8 L=5 blind | 0.419771 | 0.419771 | −23.412114 |
| 3×6 transferred 3×4 L=5 angles | 0.523587 | 0.523587 | −23.484869 |
| 3×6 polish of that transfer | 0.470995 | 0.470995 | −23.596105 |
| 3×4 Γ=0.6 L=6 (A) | 0.876461 | 0.876461 | −15.806964 |
| 4×3 Γ=0.6 L=6 (C) | 0.744375 | 0.744375 | −15.010816 |

3×4 and 4×3 at Γ=0.6 have the same cost histogram and the same E0=−16.5754689916, gap 0.010150. The 0.744 and 0.876 circuits are both exact. They are different optimizer results. The lower energy is 0.876. Neither replaces the other.

## L1 grids

Best-found after a period grid, a halved local step, and Nelder–Mead, until the change dropped below 1e-4. Not a global certificate.

| torus | F1max | published refined value |
| --- | ---: | ---: |
| 3×4 | 0.269498 | 0.269498 |
| 3×5 | 0.459717 | 0.459717 |
| 3×6 | 0.095229 | 0.095229 |
| 4×4 | 0.412965 | 0.412965 |

A 32×16 grid on 4×4 stayed at 0.355599. A 64×32 grid plus refinement reached 0.412965. The ratio for N=16 uses 0.412965. No grid maximum exceeded the published refined value by more than 0.02. Heatmaps are `figures/L1_heatmap_3x4.png`, `3x5`, `4x4`, and `3x6`. Those images are the raw grids. The table is the refined best-found.

## Manifold diagnostics at Γ=0.8

w = ‖P_M ψ‖². F_flat = |⟨u|ψ⟩|². Equality holds only if the projection is flat. Full space. Phase spread on M is 0 after the global sign choice. No negative amplitude on M above 1e-10.

| torus | m | orbits | w | F_flat | w−F_flat | F_flat/w | PR | PR/m | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3×4 | 18 | 2 | 0.836557 | 0.834532 | 2.025e-3 | 0.9976 | 17.822 | 0.990 | 0.0493 |
| 3×5 | 360 | 24 | 0.796015 | 0.787966 | 8.048e-3 | 0.9899 | 344.615 | 0.957 | 0.1011 |
| 3×6 | 48 | 4 | 0.786101 | 0.786101 | 1.1e-16 | 1 | 48.000 | 1 | 0 |
| 4×4 | 640 | 46 | 0.802757 | 0.801522 | 1.235e-3 | 0.9985 | 636.021 | 0.994 | 0.0393 |

N=18 at Γ=0.8 is actually flat (PR=m, CV=0). That is not a double print: w and F_flat were computed separately and differ by 1e-16. N=12 and N=15 are not flat. Heaviest single torus-orbit weights are 0.518, 0.050, 0.295, and 0.023. High w is not one orbit.

The same split on the earlier anchors: 3×4 Γ=0.7 differs by 2.81e-3, 3×5 Γ=0.6 by 9.83e-3, and 3×6 Γ=1.0 by 0 (PR=48 on m=48).

## Observables at Γ=0.8

| torus | E0 | gap | residual | ⟨ZZZ⟩_▲ per face | ⟨X⟩ per site | entropy cut | S (bits) | Schmidt upper |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 3×4 | −17.084068 | 0.042215 | 2.0e-14 | 0.6088 | 0.2576 | y<2, 6 sites | 3.318 | 0.150 |
| 4×3 | −17.084068 | 0.042215 | 2.7e-14 | 0.6088 | 0.2576 | x<2, 6 sites | 3.318 | 0.150 |
| 3×5 | −19.860289 | 0.227480 | 8.6e-14 | 0.4748 | 0.4681 | y<2, 6 sites | 3.141 | 0.197 |
| 3×6 | −25.559302 | 0.025644 | 3.5e-14 | 0.6145 | 0.2388 | y<3, 9 sites | 4.202 | 0.055 |
| 4×4 | −20.667026 | 0.240137 | 8.2e-14 | 0.4470 | 0.4972 | x<2, 8 sites | 3.852 | 0.196 |

⟨ZZZ⟩_▼ = −⟨ZZZ⟩_▲ on each of these states. The Schmidt number is the minimum largest eigenvalue of the reduced state over those cuts and 16 random balanced cuts. It upper-bounds a fully factorized product. It is far below F_deep on N=12 and N=18.

## Stop

N=12 **PASS**. N=15 **PASS**. N=16 **PASS**. N=18 **PASS**, with the three fidelities kept separate. No scaling and no speedup.
