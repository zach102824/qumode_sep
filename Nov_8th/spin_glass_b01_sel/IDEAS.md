# Historical B=0.1 round: is energy-only selection discarding an energy-legal state?

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The `|J|`-max confirmation, the Fiedler confirmation, the energy-only union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, and the QAOA-depth diagnosis stay as recorded. This round does not change the packing, does not add π/4 or π/2 to a shell, does not build popcount shells, does not repel from a trapped state, and does not add QAOA starts.

## Why this question

The QAOA-depth diagnosis on k=118..129 found that sixteen extra L=6 QAOA starts lowered no historical energy. N=9 and N=12 are not an under-searched comparator under that probe. Several recorded losses are fidelity-only: the energy-selected KM state has the lower energy and the worse subspace overlap. Energy-only selection keeps that state. The open question is whether the same standard search already contains another KM state that is still energy-legal against QAOA and has a higher subspace overlap.

A joint win requires the KM energy to be at most the QAOA energy plus 1e-6. Any start above that line cannot repair a joint comparison. The diagnostic pool is exactly that line. A wider pool would rank states that are not energy-legal, and that ranking is not this claim.

## What is fit

Fresh instances k=130..141. The circuit is the original interleaved shared-gamma KM. The round-4 starts stay, including the nodata to K to KM nesting, two repeats. QAOA uses that same list. Nothing else is optimized.

Selection for the claim is the lowest true energy. Fidelity is computed after that choice. The diagnostic column then looks only at L=6 KM starts whose true energy is at most the energy-only QAOA energy plus 1e-6, and keeps the one with the largest subspace overlap. Ties go to the lower energy, then the lower repeat, then the label. If that pool is empty, the diagnostic state is the energy-only state. The diagnostic column is not the verdict.

Equal depth and equal search are the comparison. Equal parameter count is reported and is not the claim. The highest subspace overlap in the whole L=6 KM list is reported with its energy gap and has no pass rule.
