# Historical B=0.1 round: ferromagnetic triples

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The recorded initialization probes stay closed. Uniform L=6 draws, scale-normalized spheres, and the diverse L=1 lineages did not beat the continuation energy. On the diverse window the selected L=6 states are stationary: projected gradients are about 1e-4, none of the N=9 or N=12 winners hit the evaluation cap, and none sits on the box bound. Another random start, kick, or lineage would be the same probe. The N=6 passes on those windows are the continuation search. They are not a new gain.

## Why this change

The circuit's shell is a popcount phase. It depends on how many of the three cavity sites are flipped, not on which site. Under `H = sum J_ij Z_i Z_j`, a negative coupling is ferromagnetic: the bond energy is lower when the two Z eigenvalues agree. A cavity of three negative couplings has its classical minima at the two fully aligned states, popcount 0 and popcount 3, which are exactly the configurations a popcount shell can name. The |J|-max rule groups large couplings of either sign, so a cavity can be antiferromagnetic and its minimum can be a popcount-1 state that depends on which site is flipped. The Fiedler rule orders sites by an eigenvector of the |J| Laplacian. Neither formula is this one.

On the development instances k=0..4 the ferromagnetic partition matched |J|-max once at N=6 and never at N=9 or N=12, and it matched the Fiedler partition never. That check used no circuit energy. The formula is frozen here and is not refit on k=178..189.

## What is fit

Fresh instances k=178..189. One rule at every N: triples maximizing the sum of `-J` on their edges, the lexicographically smallest partition on ties. The same triples at every layer. Block means and popcount shells are rebuilt on those triples. `J@J` is unchanged. Interleaved KM is fit on the same instances as a reference and is not the claim. QAOA is the equal-depth comparator and does not use the triples. No union with another packing, no per-N choice, no added starts, and no fidelity in the packing or the selection.
