# Historical B=0.1 round: maximin ferromagnetic triples

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld. The ferromagnetic-sum round stays partial. Its N=6 and N=9 passes are not a packing-only gain, because interleaved KM on that same window also passed those sizes.

## Why this change

The recorded initialization probes stay closed. Another random start is not a new idea. `|J|`-max, Fiedler, their energy-only union, quarter-pi restarts, classical popcount seeds, and the repulsion escape are closed too.

The shell is three phases of the popcount, and those phases are shared. The even shell weight is `+3` on popcount 0 and 3 and `-1` on popcount 1 and 2, and the weights are added across cavities. A negative coupling is ferromagnetic: the bond energy is lower when the two Z eigenvalues agree, so a cavity of three negative couplings has its classical minima at popcount 0 and 3. The sum rule maximizes the total of `-J` inside the triples. It can buy that total by leaving one cavity weak. At N=12 there are four cavities on one shell, so the weak cavity is phased by a shell that the strong cavities set.

The maximin rule maximizes the worst cavity's sum of `-J`. Ties keep the lexicographically smallest sorted partition. It is not the sum, not `|J|`, not a spectral order, and not a union. On the development instances k=0..4, which are outside this window, the partition differs from the ferromagnetic sum, from `|J|`-max, from Fiedler, and from the interleaved triples at every N=12 instance. At N=12 k=0 the worst triple moves from 1.290 under the sum to 1.888 under the maximin, and the sum of `-J` falls from 11.930 to 10.832. That check used classical `J` only. The formula is frozen here and is not refit on k=190..201.

## What is fit

Fresh instances k=190..201. One rule at every N and every layer. Block means and popcount shells are rebuilt on those triples. `J@J` is unchanged. Interleaved KM is fit on the same instances as a reference and is not the claim. QAOA is the equal-depth comparator and does not use the triples. No union, no per-N choice, no added starts, and no fidelity in the packing or the selection.
