# Window-controlled confirmation of the frozen interleaved circuit

## What this round is

The circuit does not change. The packing does not change. The start list does not change. The Hamiltonian does not change: it is the published full spin glass, `sum_{i<j} J_ij Z_i Z_j + 0.1 sum_i X_i`.

The question is whether the original shared-gamma KM, with the interleaved triples `[a, a+N/3, a+2N/3]`, clears equal-depth QAOA on a fresh sample when a single block of 12 instances is not allowed to carry the claim.

## Why another packing or another initialization is not the next fit

Ferromagnetic-sum triples and maximin triples both lowered the KM energy relative to interleaved, including at N=12, and neither cleared the N=12 fidelity interval. On the maximin window, k=190..201, the unchanged interleaved reference did clear N=12. On the ferromagnetic window, k=178..189, that same reference did not. The N=12 pass bit of the frozen circuit moves with the instance window. A third packing, or another random start around the same continuation, would add another energy comparison on another noisy window. It would not tell us whether the equal-depth result is stable.

Earlier rounds already spent the admissible variants that stay inside the Nov 3 circuit: `|J|`-max packing, Fiedler packing, their energy-only union, quarter-pi shell restarts, a classical popcount head, a repulsion penalty, a deeper QAOA control, energy-legal selection as a diagnostic, uniform box starts, scale-normalized radius starts, and diverse L=1 lineages. Those stay as recorded. This round does not refit them and does not rebrand their N=6 results.

## The control

k=202..225 is split before any fit into block A (202..213) and block B (214..225). The pool of 24 is the comparison sample. A size clears only when the pool meets the existing win rule and both blocks agree in the pre-registered sense: median fidelity difference positive, and wins at least losses. A block whose interval happens to clear zero is printed and is not a result. Earlier windows are not mixed into the bootstrap. That mix would be favorable batch selection after the fact.

## What is still out of scope

No per-layer gamma, no extra shell angles, no per-N packing, no ground-state or fidelity information in the starts or the selection, no field other than the historical 0.1, and no change to the published Hamiltonian. `B_cri` stays withheld.
