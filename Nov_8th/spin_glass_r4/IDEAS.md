# Round 4 ideas, from the round-3 results

No new Opus consultation. The choice below uses only the finished round-3 run (`ad9e6cde`).

## What round 3 established

- At N=6 the retained shared-gamma KM family beats equal-depth QAOA on fresh instances (18/22 joint wins, median fidelity difference +0.150). At N=9 and N=12 it does not: median differences +0.018 and -0.015, and both confidence intervals include 0.
- This is not an optimizer-budget effect at N=9 or N=12. All three independent repeats reached the same best energy in every cell, and QAOA with three times the starts found a lower energy in 0 of 132 cells at those sizes.
- Several losses have KM at a *lower* energy than QAOA and a *worse* fidelity (N6 k12, N9 k7, N9 k20, N12 k13, N12 k17). On N9 k7 the KM energy is 1.06 below QAOA while the ground-space weight is 0. Energy-only selection is doing what it should; the energy minimum of this ansatz is not the high-fidelity state. Choosing a higher-energy state because its fidelity is better would leak the ground state into selection, so it is not an allowed fix.
- The box is slack at N=9 and N=12 (median diagonal coefficient under 5% of the box, maximum under 8%). Widening it is not a remedy for the sizes that failed. It does bind in a few N=6 cells, which already pass.
- The retained KM solutions are not a fixed-shape rescaling. The linear coefficients (a, b, c) sit in the box, but b/a reaches 55 and c/a reaches 85 in cases where a is near 0 and the feature coefficient is not (N9 k7: a = 0.001, c = 0.093). A per-layer family that forces the feature part to be theta times gamma, with theta bounded, would exclude the retained family's own solutions.

## Two changes this justifies

1. **KM_pack, a coupling-informed change of the retained family.** The shells and the M feature are built on the interleaved triples [a, a+N/3, a+2N/3]. For iid couplings that grouping is an arbitrary labeling. On the 24 round-3 instances the partition into triples that maximizes the sum of |J| inside triples differs from the interleaved one on 71 of 72 instances, and it captures a median 40% more absolute coupling. KM_pack keeps the shared-gamma circuit, the linear coefficients, the box and the optimizer, and changes only that partition. K_pack and nodata_pack are its nested-seed providers, so it gets the same nesting KM gets. It is not a new gate set and not a redefinition of depth. QAOA has no triples, so its circuit is unchanged.
2. **KM_gl, a new family, labeled as such.** Each layer has its own coefficient a_l on Hc; the feature coefficients b and c stay shared, one value for every layer, in the same box. Setting all a_l equal reproduces KM exactly, including the solutions where b or c dominates a. This is the per-layer schedule round 3's depth curves motivate: at N=9 and N=12, QAOA's fidelity keeps rising with depth while KM's advantage closes. It is reported beside KM, never as KM. It has 2+4L parameters against KM's 3+3L and QAOA's 2L.

## What this round will not do

- No transfer of angles between instances. Every instance is fit on its own.
- No use of the ground state, eigenvectors or fidelity in the features, the packing score, the starts or the selection.
- QAOA keeps the round-3 standard budget. Round 3 already showed the tripled budget does not move it, so dropping that control does not weaken QAOA.
- k=28..43 are reserved for a later confirmation round and are not fit here. k=4..15 is the development set, including the round-3 losses.

## Advancement rule

Stated numerically in `PREREG.json`. A variant advances only if, on this fixed set at L=6, its paired fidelity advantage over QAOA improves on KM's at N=9 or N=12 without giving up the N=6 result and without fewer joint (fidelity and energy) wins. If neither variant advances, that is the result, and the next round has to try something else.
