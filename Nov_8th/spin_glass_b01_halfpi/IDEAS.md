# Historical B=0.1 round: half-pi shell restarts

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The `|J|`-max confirmation, the Fiedler confirmation, and the energy-only union stay as recorded. This round does not assign one packing to one N.

## Why this change

The union on k=68..79 passed only at N=6. Its losses were fidelity-only: the lower-energy circuit was already below QAOA in energy and still had the worse ground-space overlap. Another draw from the same Gaussian perturbations does not leave that basin. Round 6 already showed that a QAOA projection and a Pearson coupling of the classical diagonals do not leave it either.

The shell weights on each cavity popcount are the odd integers `(3, 1, -1, -3)`, `(3, -1, -1, 3)`, and `(1, -1, 1, -1)`. Adding π to any one shell channel therefore multiplies every X-basis amplitude by the same phase. That start is not a new search. Adding π/2 changes the relative phase between popcounts by π. The rule frozen here is that nontrivial hop, applied to the energy-best standard point of each shared-gamma KM packing, at every depth, with the same four displacements at N=6, 9, and 12.

## What is fit

Fresh instances k=80..91. Both packings are fit. The claimed state is still the lower energy of the two, and equal energy keeps the `|J|`-max state. QAOA keeps the round-4 start list and two repeats. Equal depth is the claim. Equal parameter count is reported and is not the claim. Per-layer gamma is not fit.
