# Historical B=0.1 round: continue the L=1 states the energy beam drops

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, the QAOA-depth diagnosis, the energy-legal selection diagnostic, the uniform L=6 box, and the scale-normalized L=6 restarts stay as recorded. This round does not change the packing, does not move a shell by π/4 or π/2, does not build popcount shells, does not repel from a trapped state, does not add QAOA starts, does not re-rank states by fidelity, and does not draw another vector from the full parameter box or from a sphere around the finished L=6 vector.

## Why this change

Kicks around the finished continuation vector are closed. None of 432 uniform L=6 draws and none of 432 scale-normalized directions of radius 1 beat that energy. Round 3 already multiplied the same L=1 distribution and, at N=9 and N=12, the repeats landed on one energy.

A spherical rank-1 head was measured on the development instance k=4 and was not adopted. Least squares of `v v^T` onto K-hat and M-hat, with `v` the top eigenvector of J, gives `(thetaK, thetaM)` inside the existing random L=1 box (about +0.14 to +0.52) and a pair correlation of only 0.27, 0.27, and 0.52 at N=6, 9, and 12. That point is not a new region.

The same development instance, one repeat, shows what the energy beam drops. After the usual twelve L=1 optimizations, `distinct_parents` keeps three. The discarded states that are not duplicates sit at half-width distance 2.11, 1.95, and 1.73 at N=6, 1.31 at N=9, and 1.66, 1.51, and 1.08 at N=12. Those states have never been continued. The threshold 0.5 is below every one of those distances and above the duplicate copies at distance 0. It is frozen here. It is not refit on k=166..177.

## What is fit

Fresh instances k=166..177. The circuit is the original interleaved shared-gamma KM. The round-4 starts stay, including the nodata to K to KM nesting, two repeats. QAOA uses that same list and receives nothing extra.

After the L=1 KM optimizations of a repeat, keep up to three optimized states that the beam of three did not keep and whose half-width distance to the nearest kept state is at least 0.5. Each kept state is continued on its own from L=2 through L=6 by Zhou interpolation, time rescaling, and the recorded raw perturbations 0.1 and 0.3. A lineage is not put back into the energy beam, so a worse L=1 energy cannot drop it, and it does not change the continuation column. The continuation vector stays a candidate. Fidelity is not used. The same separation and the same count are used at every N.

Equal depth is the comparison. The continuation column is equal search. The diverse column is not, and that is disclosed. Equal parameter count is reported and is not the claim.
