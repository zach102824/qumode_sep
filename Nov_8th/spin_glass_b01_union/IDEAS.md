# Historical B=0.1 round: energy-only union of two packings

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld.

The `|J|`-max confirmation on k=44..55 and the Fiedler confirmation on k=56..67 stay as recorded. This round does not replace either verdict, and it does not assign one packing to one N and the other packing to another N.

## Why this change

Each of those packings is the same shared-gamma circuit. Each one, under the same pass rule, missed a different size: `|J|`-max missed N=9, and Fiedler missed N=12. The legal selection rule is energy. The rule frozen here is therefore one rule at every N: fit both circuits with the same optimizer, then keep the state with the lower energy. The two searches do not read each other, and the choice does not use a ground state or a fidelity.

If the lower-energy circuit is the one with the worse fidelity, that instance counts against the claim.

## What is fit

Fresh instances k=68..79. The claimed state is the energy-selected one of KM on the `|J|`-max triples and KM on the Fiedler triples. Equal-depth QAOA, two repeats, is the comparator. Two further QAOA repeats are a secondary equal-search report and are not the verdict. Per-layer gamma is not fit.
