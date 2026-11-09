# Historical B=0.1 round: scale-normalized restarts around the continuation vector

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, the QAOA-depth diagnosis, the energy-legal selection diagnostic, and the uniform L=6 box initialization stay as recorded. This round does not change the packing, does not move a shell by π/4 or π/2, does not build popcount shells, does not repel from a trapped state, does not add QAOA starts, does not re-rank states by fidelity, and does not draw another vector from the full parameter box.

## Why this change

The uniform box round minimized 432 L=6 draws. None beat the continuation energy. The closest sat 0.275 above at N=6, 6.04 above at N=9, and 16.83 above at N=12. A second space-filling design would be the same probe.

The continuation that does reach those energies perturbs by 0.1 and 0.3 in raw radians while the depth grows. The head box is `[-2π/s_J, 2π/s_J]` and the shell box is `[-2π, 2π]`. At N=9 and N=12 the head half-width is about a third of the shell half-width, so a raw kick of 0.3 is a much larger fraction of the head box than of the shell box. Round 6's 0.15 and 0.40 kicks had the same raw geometry and were centered on a QAOA projection, not on the continuation minimum.

## What is fit

Fresh instances k=154..165. The circuit is the original interleaved shared-gamma KM. The round-4 starts stay, including the nodata to K to KM nesting, two repeats. QAOA uses that same list and receives nothing extra.

After both repeats, twelve KM starts are drawn at L=6 only. Each direction is uniform on the unit sphere in the 21 coordinates after dividing by that coordinate's box half-width. The start is the continuation vector plus that direction. The continuation vector stays in the candidate set, so the selected energy is at most the continuation energy. Fidelity is not used. The same radius, 1, is used at every N.

Equal depth is the comparison. The continuation column is equal search. The radius column is not, and that is disclosed. Equal parameter count is reported and is not the claim.
