# Historical B=0.1 round: do uniform L=6 starts find a lower KM energy?

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, the QAOA-depth diagnosis, and the energy-legal selection diagnostic stay as recorded. This round does not change the packing, does not move a shell by π/4 or π/2, does not build popcount shells, does not repel from a trapped state, does not add QAOA starts, and does not re-rank states by fidelity.

## Why this change

The QAOA-depth diagnosis found that extra full-depth QAOA starts lowered no historical energy. The energy-legal diagnostic then found that post-selection inside the joint-energy window does not change the pass bits. The N=9 and N=12 miss is the KM energy minimum the continuation search returns. That continuation grows depth by depth and perturbs the head by 0.1 and 0.3. Round 6's QAOA projection and classical-diagonal head, including 0.15 and 0.40 kicks of that projection, stayed in the same basin on development cells and were not confirmed. Neither of those is a draw from the KM parameter box.

## What is fit

Fresh instances k=142..153. The circuit is the original interleaved shared-gamma KM. The round-4 starts stay, including the nodata to K to KM nesting, two repeats. QAOA uses that same list and receives nothing extra.

After both repeats, twelve KM starts are drawn at L=6 only. Gamma, thetaK, and thetaM are uniform in the linear box. Each shell angle is uniform in [-2π, 2π]. Each start is minimized on true energy. The continuation's L=6 vector stays in the candidate set, so the selected energy is at most the continuation energy. Fidelity is not used. The same twelve draws are used at every N.

Equal depth is the comparison. The continuation column is equal search. The box column is not, and that is disclosed. Equal parameter count is reported and is not the claim.
