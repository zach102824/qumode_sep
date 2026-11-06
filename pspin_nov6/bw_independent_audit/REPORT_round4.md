# Round 4 — Gamma 0.8 replay

Circulant stays closed. No new family. This file checks schedules already stored on two other branches, on a Hamiltonian built in this directory.

Snapshots, not later commits:

- A `cursor/sparse-p3-qaoa-round1-13b4` at `fbaac8375a9159678c0ece3e54bde60bbf62d5f1` (2026-11-06 17:58:42 UTC). No `runs/round4` directory was on that commit.
- C `cursor/pure-p4-ferro-af-7dc4` at `b8d2d35e96315a9083ac1dc164b3be987c8ed187` (2026-11-06 17:09:28 UTC).

If A publishes a later N=18 search, that commit is not in these numbers.

```text
H = -sum_up ZZZ + sum_down ZZZ - Gamma * sum X
up (x,y),(x+1,y),(x+1,y+1); down (x,y),(x,y+1),(x+1,y+1); index x+Lx*y
```

Every cost eigenvalue on 3×4, 4×3, 3×5, 3×6, and 4×4 is a multiple of 8. Gamma period is π/4. A shift of π/8 is not a global phase (half-period relative spread 2). The 3×4 and 4×3 classical histograms are identical. Full-space Krylov. Residuals are about 1e-14. No amplitude on the ground state is below −1e-8. The solver did not use a symmetry sector.

## Replay

Circuit: cost then mixer, stored layer 0 first. Pass means |ΔF| ≤ 1e-6 and, where the published energy is for the same state, |ΔE| ≤ 1e-6. Every row below is inside 1e-10 in F. The largest |ΔF| is about 7e-15.

| schedule | published F | replay F | replay energy |
| --- | ---: | ---: | ---: |
| 3×4 Γ=0.8 L=1 | 0.236036 | 0.236036 | −13.079740 |
| 3×4 Γ=0.8 L=6 | 0.894930 | 0.894930 | −16.512339 |
| 3×5 Γ=0.8 L=1 | 0.426269 | 0.426269 | −16.349675 |
| 3×5 Γ=0.8 L=6 | 0.975507 | 0.975507 | −19.774608 |
| 4×4 Γ=0.8 L=1 | 0.385416 | 0.385416 | −17.439654 |
| 4×4 Γ=0.8 L=5 | 0.973364 | 0.973364 | −20.485661 |
| 3×6 Γ=0.8 L=5 blind | 0.419771 | 0.419771 | −23.412114 |
| 3×6 Γ=0.8 L=5, 3×4 angles transferred | 0.523587 | 0.523587 | −23.484869 |
| 3×6 Γ=0.8 L=5, energy polish of that transfer | 0.470995 | 0.470995 | −23.596105 |

Intermediate depths on those ladders replay the same way. 4×3 at Γ=0.8 matches 3×4 to the same tolerance; it is an orientation, not another N.

### N=18 at Γ=0.8, three different states

These are not one fidelity.

| state | F | energy | relative to the other two |
| --- | ---: | ---: | --- |
| blind 8-restart search | 0.419771 | −23.412114 | highest energy, lowest F |
| raw transfer of the 3×4 L=5 angles | 0.523587 | −23.484869 | highest F |
| energy polish of that transfer | 0.470995 | −23.596105 | lowest energy |

Lower energy is not higher fidelity. The polish is below the raw transfer in F and below the blind search in energy. A later search can still move this point. This snapshot does not pick one of the three as the only N=18 result.

### N=12 at Γ=0.6, two optimizers

Same cost spectrum and the same E0 = −16.5754689916, gap 0.01015, on 3×4 and on 4×3.

| search | orientation | L=6 F | L=6 energy |
| --- | --- | ---: | ---: |
| A, later | 3×4 | 0.876461 | −15.806964 |
| C | 4×3 | 0.744375 | −15.010816 |

Both circuits replay exactly. They are different points found by different searches. The 0.876 state has the lower energy. The 0.744 state is not a bad replay of the 0.876 state.

## L=1 maximum, best-found

One verified γ period, then a local grid with the step halved, then Nelder–Mead, until the fidelity moved by less than 1e-4. That is not a proof of the global maximum.

| torus | this best-found F1max | published refined F1max | note |
| --- | ---: | ---: | --- |
| 3×4 Γ=0.8 | 0.269498 | 0.269498 | stable after one refinement (next change 3e-11) |
| 3×5 Γ=0.8 | 0.459717 | 0.459717 | matches A's CSV, not the prose 0.453, which is the unrefined grid |
| 3×6 Γ=0.8 | 0.095229 | 0.095229 | prose 0.090 is the unrefined grid |
| 4×4 Γ=0.8 | 0.412965 | 0.412965 | a 32×16 grid stuck in a second basin at 0.3556; a 64×32 grid plus refinement reached C's value. The second basin is real. |

No best-found F1max exceeds the published refined value by more than 0.02.

Using the replayed deep fidelity and these F1max values: 0.895 / 0.269, 0.976 / 0.460, 0.973 / 0.413, and 0.420 / 0.095 are all greater than 2, and each deep fidelity is at least 0.4. For N=18 the blind 0.420, the transfer 0.524, and the polish 0.471 each clear those two numeric bars. They remain three states.

## Manifold weight is not flatness

Let M be the classical ground set of the diagonal, m = |M|, w = ‖P_M |GS⟩‖², and u the flat equal-phase vector on M. Cauchy–Schwarz gives |⟨u|GS⟩|² ≤ w, with equality if and only if P_M|GS⟩ is proportional to u. The participation ratio is (Σ|a|²)²/Σ|a|⁴ for the normalized projection a = P_M ψ / √w. It equals m when that projection is flat.

Round 3 printed pairs that look equal after two digits. They are not equal except where the projection really is flat.

| state | m | orbits | w | F_flat | w − F_flat | PR | CV of \|a\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3×4 Γ=0.7 | 18 | 2 | 0.884284 | 0.881472 | 2.81e-3 | 17.77 | 0.056 |
| 3×5 Γ=0.6 | 360 | 24 | 0.893672 | 0.883846 | 9.83e-3 | 343.3 | 0.105 |
| 3×6 Γ=1.0 | 48 | 4 | 0.607019 | 0.607019 | 0 | 48 | 0 |
| 3×4 Γ=0.8 | 18 | 2 | 0.836557 | 0.834532 | 2.02e-3 | 17.82 | 0.049 |
| 3×5 Γ=0.8 | 360 | 24 | 0.796015 | 0.787966 | 8.05e-3 | 344.6 | 0.101 |
| 3×6 Γ=0.8 | 48 | 4 | 0.786101 | 0.786101 | 1e-16 | 48 | 0 |
| 4×4 Γ=0.8 | 640 | 46 | 0.802757 | 0.801522 | 1.23e-3 | 636.0 | 0.039 |

Phases on M are 0 after the global sign choice; no negative amplitude on M above 1e-10. N=18 at both fields is flat on M. N=12 and N=15 are not. High w does not mean the cat is flat. One torus orbit never holds the whole manifold weight: the heaviest orbit is 0.52 of the 3×4 state, 0.05 of the 3×5 state, 0.29 of the 3×6 state, and 0.02 of the 4×4 state.

## Product bound and cuts

The largest eigenvalue of a reduced density matrix upper-bounds overlap with any state that factors across that cut, hence with any fully factorized product. An optimized product overlap would only be a lower bound and is not used.

At Γ=0.8 the minimum of that eigenvalue over the named cuts and 16 random balanced cuts is 0.150 (3×4), 0.197 (3×5), 0.055 (3×6), and 0.196 (4×4). Each is well below the replayed deep fidelity on that torus.

Entropies, in bits, at Γ=0.8:

| torus | cut | sites | S |
| --- | --- | ---: | ---: |
| 3×4 | y < 2 | 6 | 3.318 |
| 4×3 | x < 2 | 6 | 3.318 |
| 3×5 | y < 2 | 6 | 3.141 |
| 3×6 | y < 3 | 9 | 4.202 |
| 4×4 | x < 2 and y < 2 | 8 | 3.852 |

Per-face ⟨ZZZ⟩_▲ = −⟨ZZZ⟩_▼ at Γ=0.8 is 0.609 on 3×4, 0.475 on 3×5, 0.614 on 3×6, and 0.447 on 4×4. Those are expectations, not an order-parameter scaling.

## Result of this batch

Every stored angle that was replayed passes the 1e-6 test, and the fidelities pass 1e-10. The four sizes at Γ=0.8 meet F_deep ≥ 0.4 and F_deep ≥ 2 F1max on the numbers above, with the N=18 caveat that three verified states sit at 0.420, 0.524, and 0.471. The Schmidt bound is far below those deep fidelities on N=12 and N=18. No scaling, no phase-transition location, and no speedup is inferred.
