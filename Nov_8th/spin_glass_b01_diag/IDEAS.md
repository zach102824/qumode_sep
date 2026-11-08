# Historical B=0.1 round: is the QAOA comparator thin at N=9 and N=12?

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The `|J|`-max confirmation, the Fiedler confirmation, the energy-only union, the quarter-pi restarts, the classical popcount initialization, and the repulsion escape stay as recorded. This round does not change the KM packing, does not add π/4 or π/2 to a shell, does not build popcount shells, and does not repel from a trapped state.

## Why this change

Those six slices missed N=9, N=12, or both. Several losses are fidelity-only: KM already has the lower energy, and energy-only selection keeps a poor overlap. That pattern does not say whether the QAOA state itself is a weak local minimum. Round 3 did multiply the L=1 QAOA starts by three, on k=4..27, and that arm did not lower N=9 or N=12 energy. It did not draw new points in the full L=6 box, and it did not kick the L=6 vector by a large noise. The later slices also gave KM extra starts and left QAOA on repeats 0 and 1.

## What is fit

Fresh instances k=118..129. KM is the original interleaved shared-gamma circuit with the round-4 start list and the nodata to K to KM nesting, two repeats, nothing else. QAOA uses that same list for repeats 0 and 1. That pair is the historical equal-depth comparator.

After both repeats, sixteen more QAOA starts are run at L=6 only. Eight are uniform in the parameter box. Eight add N(0, 0.75) to the energy-best standard L=6 vector. The standard vector stays in the candidate set. Selection is the lowest true energy. Fidelity is not used. The same rule is used at every N.

Equal depth is the comparison. The historical comparator is KM against standard QAOA. The deep column asks whether that comparator was thin. Equal parameter count is reported and is not the claim. Equal search is not claimed for the deep column.
