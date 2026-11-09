# Critical-field scan: no single B_cri adopted

Protocol commit `2c3a0aae`, prereg SHA-256 `0226715302ca7d0e30c4bee8182f05c367a93432c5d06fff99d376b3fb988629`. The scan ran after that commit. Wall time 20.1 s. No circuit was optimized.

## Definition that was frozen

The target operator is the published spin-glass Hamiltonian

```
H = sum_{i<j} J_ij Z_i Z_j + B sum_i X_i,
```

with `J_ij ~ Normal(0, 1)` and no Kac factor (He, Kemper, Freericks, arXiv:2408.03251). `B` is the coefficient in that operator. It is not a QAOA mixer angle.

The matching published critical-field analysis is the transverse-field Sherrington-Kirkpatrick model (Mukherjee, Rajak, Chakrabarti, Phys. Rev. E 92, 042107 (2015), arXiv:1412.2973). Their density at `J = 1` is normal with variance `1/N`. Author variance 1 maps onto that normalization by `B = Γ √N`. Their infinite-volume T=0 extrapolation is `Γ_c = 1.62 ± 0.05`. Their finite-size estimator is a crossing of the disorder-averaged Binder cumulant. The frozen rule adopts one `Γ` only when the `N=6` with `9` crossing and the `N=9` with `12` crossing each exist once inside `[0.40, 2.40]` and differ by at most 0.15.

The author Table 1 `B_c` is the field of the minimum same-symmetry gap on one matrix. That quantity was not an adoption input. Antiferromagnetic critical fields were not used. Historical circuit runs stay at `B = 0.1`.

## What was computed

Ensemble `SeedSequence([20261170, N, k])`, `k = 0..63`, `N = 6, 9, 12`. `Γ` grid `0.20, 0.25, …, 2.80`. Binder `g` from `Q_2` and `Q_4` on the lowest eigenvector, as in their equation (5). Bootstrap 400 resamples, seed `20261171`.

Startup checks: Walsh moments agree with a direct pair sum to 0, and the variance map `B = Γ √N` reproduces the normalized spectrum to 0 on an `N = 4` sample.

`Q_1` on the grid is at most `2.8×10^{-12}`. The lowest vector is a spin-reflection eigenstate on this grid. Full diagonalization of `N = 12` ground energies agrees with the Lanczos energy to about `10^{-13}` at `B = 0.1` and at `Γ = 0.2, 0.76, 0.9, 1.6` (instances `k = 0, 2, 4, 5`).

## Crossings

| pair | Γ in the window | bootstrap median | bootstrap 2.5% | bootstrap 97.5% | missing fraction |
|---|---|---|---|---|---|
| 6 with 9 | 0.76124 | 0.76173 | 0.53478 | 0.92667 | 0 |
| 9 with 12 | 0.93756 | 0.95013 | 0.61571 | 1.13505 | 0.0125 |

Each pair has one in-window crossing. The point estimates differ by 0.17632, which is above the frozen tolerance 0.15. The bootstrap intervals overlap, and the degeneracy flag did not trip. The frozen rule still withholds a single `Γ`. No `B_cri(N)` is declared.

The same rule's numbers, if someone were to use one crossing anyway, would be:

| prescription | Γ | B(N=6) | B(N=9) | B(N=12) |
|---|---|---|---|---|
| 6–9 crossing, not adopted | 0.76124 | 1.86464 | 2.28371 | 2.63700 |
| 9–12 crossing, not adopted | 0.93756 | 2.29654 | 2.81267 | 3.24779 |
| infinite-volume anchor, not the finite-N estimator | 1.62 ± 0.05 | 3.968 ± 0.122 | 4.860 ± 0.150 | 5.612 ± 0.173 |
| historical author ramp | — | 0.1 | 0.1 | 0.1 |

These fields differ by more than the tolerance that was frozen. The benchmark would change under the choice. The source gate is this table.

At `B = 0.1` the median `Q_2` is 0.986, 0.992, and 0.996 at `N = 6, 9, 12`. That field is `Γ ≈ 0.041, 0.033, 0.029`, below the grid and on the ordered side of both crossings. It remains the historical circuit setting.

## Diagnostics that were not allowed to choose the field

The largest one-step drop of `Q_2` sits at `Γ ≈ 0.28` to `0.33`, on the first steps of a grid that starts at 0.20. That location is a grid-edge feature of the drop, not a second adopted critical point.

On `k = 0..7`, the minimum same-parity gap along this grid is at the low-`Γ` edge for 7 of 8 samples at `N = 6`, 6 of 8 at `N = 9`, and the median `Γ` at `N = 12` is 0.325. The corresponding author-unit fields at `Γ = 0.2` are `B = 0.490, 0.600, 0.693`. Several minima are on the boundary, so the path minimum can sit lower, toward the historical ramp. This is the Landau-Zener bottleneck. It is not the Binder crossing.

Lanczos with two Ritz values does not return the tunneling partner at `B = 0.1`. Full diagonalization for `N = 12`, `k = 0..7`, gives ordinary gaps from `1.1×10^{-14}` to `2.2×10^{-13}`. The Lanczos median gap stored in `scan/summary.json` for that field is withdrawn. Ground energies on the checked states still match.

## What this blocks

Circuit sweeps at a critical field are not started. Shared gamma stays mandatory (`layout_gate.py` passed: a `2+4L` per-layer vector is rejected, and one forward pass uses the same `D` and the same three shell weights at every layer). Per-layer `KM_gl` stays out of scope. AF jobs were not touched.
