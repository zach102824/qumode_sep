# Round 3 — independent Baxter–Wu audit

Circulant ring stays rejected. This directory does not continue it and does not use the other branch's builder or optimizer. Published term lists, the QAOA table, and stored angles were read only as comparison data.

**Stored-angle replay agrees.** On every stored schedule that was rerun, the fidelity difference is at most about 2e-14 and the energy difference is at most about 3e-13. That is well inside a 0.01 tolerance. There is no convention mismatch to reconcile: cost then mixer, with the stored list applied from layer 0, is the matching order. Usefulness claims are not frozen.

The narrative L1 maxima 0.357 (3×5) and 0.196 (3×6) are coarser than the refined values in that branch's own CSV. An independent period grid plus Nelder–Mead matches the CSV column `l1_fidelity_max_F`: 0.2255, 0.3693, and 0.2059. Those are best-found on a stated grid, not a proof that a narrower peak is absent.

## Hamiltonian

```text
H = -sum_up Z_a Z_b Z_c + sum_down Z_a Z_b Z_c - Gamma * sum_i X_i
```

Up face at (x, y): {(x,y), (x+1,y), (x+1,y+1)}. Down face: {(x,y), (x,y+1), (x+1,y+1)}. Site index x + Lx y, periodic. Z|0⟩ = +1.

The sorted up/down lists reproduce the published ZZZ terms exactly (SHA-256 match on 3×3, 3×4, 3×5, and 3×6). No up face equals a down face. Field terms in the published list are −X on each site. Classical spectra:

| torus | E_min | degeneracy | E_max of the diagonal | gamma period |
| --- | ---: | ---: | ---: | ---: |
| 3×3 | −8 | 72 | 8 | π/4 |
| 3×4 | −16 | 18 | 16 | π/4 |
| 3×5 | −16 | 360 | 16 | π/4 |
| 3×6 | −24 | 48 | 24 | π/4 |

The period is the gcd of the diagonal: every cost eigenvalue differs by multiples of 8, so γ → γ+π/4 is a global phase, and γ → γ+π/8 is not. Beta period π was not re-derived as a new theorem; the mixer is exp(−i β X) on each qubit, whose single-qubit period is π, and a π/2 shift is not a global phase on a generic state in the checks already done for this generator. The replay used the published angles inside that convention.

## Symmetries that actually exist

Translations by one site in x and in y permute up faces among themselves and down faces among themselves. They are symmetries on every torus checked, including 4×3.

A 3-face-coloring is forced by any one triangle, because each adjacent triangle shares an edge. Trying all 6 proper colorings of the first triangle:

- 3×3 and 3×6: every trial extends. Color (x+y) mod 3 has no monochrome edge. Flipping two color classes hits two sites on every face, so those X products commute with every ZZZ. Z2×Z2 is a symmetry only here.
- 3×4, 4×3, and 3×5: every trial conflicts, and the formula has 3 monochrome wrapped edges. There is no 3-sublattice coloring. Z2×Z2 was not used.

The ground state is the lowest eigenvector of the full 2^N operator. It is nonnegative (no amplitude below −1e-8) and invariant under both translations (deviation below 1e-12). Momentum zero is a consequence of positivity plus translation symmetry, not a sector that was projected out. On 3×4 and 3×5 there is no coloring sector to have cherry-picked.

A corrected dense diagonalization at N=9, Gamma=1 agrees with Krylov: E0 = −12.4945205434313, gap 0.749023, difference 2.5e-14. An earlier block-fill of the X matrix in the audit script was wrong and was fixed; the log line with gap 504 is that bug, not a disagreement of the Krylov ground state. Krylov residuals on the anchors are about 1e-14. The smallest anchor gap is 0.021 at 3×4, Gamma=0.7, still many orders above the residual, and the vector is nonnegative. No unresolved near-degeneracy showed up in these solves.

## Anchor replay

Cut for the entropy: 3×4 uses rows y < 2, sites {0,1,2,3,4,5}. 3×6 uses y < 3, sites {0,…,8}. 3×5 has odd height, so the cut is the lowest 7 indices, not a spatial half. Entropies are that cut, in bits.

| point | E0 | gap | S | published deep F | replay ΔF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3×4, Γ=0.7, L=6 | −16.80388241 | 0.0214 | 3.361 | 0.769535 | −3e-15 |
| 3×5, Γ=0.6, L=5 | −18.58370666 | 0.1260 | 3.721 | 0.951360 | +3e-15 |
| 3×6, Γ=1.0, L=5 | −26.61075418 | 0.1006 | 4.002 | 0.665195 | −2e-14 |

Every intermediate depth on those three ladders replays the same way. Their E0, gap, entropy, manifold weight, and flat-cat overlap match this build to the digits they printed.

Independent L=1, best-found, Nelder–Mead on a grid of one γ period by one β period (72×36, 48×24, and 36×18):

| point | this F1max | their CSV F1max | F at best energy |
| --- | ---: | ---: | ---: |
| 3×4, Γ=0.7 | 0.2255 | 0.2255 | 0.1944 |
| 3×5, Γ=0.6 | 0.3693 | 0.3693 | 0.3388 |
| 3×6, Γ=1.0 | 0.2059 | 0.2059 | 0.1819 |

Energy-selected L=1 fidelity stays within 0.03 of F1max. These rows are not the false-hard pattern where a fidelity search finds a much higher overlap than energy minimization.

## What the ground state is

The largest Schmidt eigenvalue across the tested cuts upper-bounds the overlap with any state that factors across that cut, and therefore with any fully factorized product. The minimum over the strip (or lowest-index) cut plus 30 random balanced cuts (12 at N=18) is:

| point | product upper | manifold weight | flat cat of all classical minima | best one torus orbit |
| --- | ---: | ---: | ---: | ---: |
| 3×4, Γ=0.7 | 0.124 | 0.884 | 0.881 | 0.542 (2 orbits, 18 states) |
| 3×5, Γ=0.6 | 0.140 | 0.894 | 0.884 | 0.057 (24 orbits, 360 states) |
| 3×6, Γ=1.0 | 0.106 | 0.607 | 0.607 | 0.228 (4 orbits, 48 states) |

A fully factorized product cannot carry the deep overlaps 0.77, 0.95, and 0.67. A single torus orbit of one bitstring cannot either, except that on 3×4 one orbit is already half the state and the two orbits together are the cat. The flat superposition of the whole classical ground manifold does account for the state: its overlap equals the manifold weight. At these fields the ground state is that entangled cat, lightly dressed. High classical weight here is not a single-orbit construction. It is also not a claim that the state is far from the classical minima.

A product overlap found by optimizing θ is only a lower bound and is not reported as one. The Schmidt number above is the upper bound that was actually computed.

## Other fields, after the replay

Same build, L=1 only except where their angles already existed and were replayed.

| point | gap | S | F1max best-found | F at best energy | manifold weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3×4, Γ=0.8 | 0.042 | 3.318 | 0.270 | 0.236 | 0.837 |
| 3×4, Γ=1.0 | 0.140 | 3.116 | 0.403 | 0.369 | 0.694 |
| 3×5, Γ=0.8 | 0.227 | 3.409 | 0.460 | 0.426 | 0.796 |
| 3×5, Γ=1.0 | 0.402 | 2.989 | 0.575 | 0.543 | 0.663 |
| 3×6, Γ=0.8 | 0.026 | 4.202 | 0.095 | 0.079 | 0.786 |

Stored deeper schedules at 3×5, Γ=0.8 and Γ=1.0, replay with the same 1e-14 fidelity agreement, including F(L=5) = 0.954 and 0.979. No deep search was run at 3×4 Γ=0.8 or 1.0, or at 3×6 Γ=0.8. Their own table at 3×6, Γ=1.3 has F(L=4)=0.8669 and F(L=5)=0.8649 while the energy decreases. That schedule was not replayed here. It is already a counterexample, in their file, to reading energy minimization as monotone fidelity.

## What is not claimed

No new geometry and no new grid. No anneal comparison, no structure factor, and no seal. The three anchors are reproducible on this Hamiltonian. The state those circuits approach is the entangled cat of the classical ground manifold, with a proved product upper bound near 0.11–0.14 and a single-orbit overlap that does not replace that cat.
