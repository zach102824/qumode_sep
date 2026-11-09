# Historical B=0.1 round: classical popcount initialization

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The `|J|`-max confirmation, the Fiedler confirmation, the energy-only union, and the quarter-pi restarts stay as recorded. This round does not take the lower energy of two packings, and it does not shift an optimized shell by π/4 or by π/2.

## Why this change

Those rounds kept the standard search. Energy selection then kept states that were already below QAOA in energy and still had poor ground-space overlap. Adding another local hop made one of those states worse: on k=80..91, N9 k90 selected a quarter-pi start with F_sub 0.004. A shift of π/2 does not change the state at all.

At B=0.1 the instances sit on the ordered side of the withheld Binder crossings. The new seeds are built from the classical ZZ bitstring only: the smallest index that minimizes the ZZ diagonal, then the cavity popcounts of that bitstring on the original interleaved triples. Three shell coefficients are the least-squares match to a 0/1 target on those popcounts. They are not a displacement of a previously optimized vector. The standard L=1 grid, the random mixer starts, and the quarter-pi restarts are not used for KM.

## What is fit

Fresh instances k=92..99 and k=102..105. k=100 and k=101 stay unused. One shared-gamma KM circuit, interleaved packing, the same rule at every N. QAOA keeps the round-4 start list and two repeats. Equal depth is the claim. Equal parameter count is reported and is not the claim. Per-layer gamma is not fit.
