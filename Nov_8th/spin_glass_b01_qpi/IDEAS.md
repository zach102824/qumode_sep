# Historical B=0.1 round: quarter-pi shell restarts

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The `|J|`-max confirmation, the Fiedler confirmation, and the energy-only union stay as recorded. This round does not assign one packing to one N.

## Why this change

The union on k=68..79 passed only at N=6. Its losses were fidelity-only: the lower-energy circuit was already below QAOA in energy and still had the worse ground-space overlap. Round 6 already showed that a QAOA projection and a Pearson coupling of the classical diagonals stay in that basin. Gaussian perturbations of size 0.1 and 0.3 do not make a fixed shell-sector hop.

A first version of this hop added π/2 to one shell channel. That protocol was committed and then stopped before any confirmation instance. On development index k=4, at N=6, 9, and 12, a π/2 shift of any shell channel left the prepared state unchanged. The pair features are flip-symmetric, so the X-basis support stays in the even sector, and on that sector every shell weight changes in steps of 4. A coefficient shift of π/2 is then a global phase. A shift of π/4 is a relative sign. Those overlaps were computed before this protocol was frozen. No confirmation energy was minimized, and no fidelity entered the choice.

## What is fit

Fresh instances k=80..91. Both packings are fit. After the standard starts at each depth, KM on each packing gets four more L-BFGS starts from its energy-best standard vector: add π/4 to shell channel 0, the same for channel 1, the same for channel 2, and negate the shared gamma while adding π/4 to channel 0. The claimed state is the lower energy of the two packings. Equal energy keeps the `|J|`-max state. QAOA keeps the round-4 start list and two repeats. Equal depth is the claim. Equal parameter count is reported and is not the claim. Per-layer gamma is not fit.
