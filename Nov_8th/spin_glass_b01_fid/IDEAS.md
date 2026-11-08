# Historical B=0.1 round: Fiedler cavity partition

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld. No new field rule is introduced.

The recorded `|J|`-max packing confirmation on k=44..55 is left as it stands. This round does not refit that rule, and it does not pick a packing according to which N won last time.

## Why this change

The diagonal feature `K` is built from the full coupling matrix and does not use the cavity assignment. The block-mean feature `M`, and the three-site shell, do. The `|J|`-max rule scores each triple by its three edges and never looks at the rest of the graph. The rule frozen here is one global function of `J` at every N: the sites are ordered by the Fiedler vector of the Laplacian of `|J|`, and the cavities are contiguous triples in that order. The same partition is used at every layer. `K` and `M` are then the original pair-square and block-mean constructions.

That is spectral clustering of the classical coupling graph. It does not use a quantum ground state, an eigenvector of `H`, or a fidelity.

## What is fit

Fresh instances k=56..67. Interleaved shared-gamma KM is the reference. The claimed family is the same shared-gamma circuit on the Fiedler partition. Equal-depth QAOA is the comparator. Its circuit has no triples. Per-layer gamma is not fit, and the round-6 projected initialization is not repeated.
