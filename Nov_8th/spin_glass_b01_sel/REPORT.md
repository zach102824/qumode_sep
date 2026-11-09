# Historical B=0.1: energy-legal selection diagnostic is partial

Protocol `9c3ebcff` before any fit. PREREG SHA-256 `f5dc9032705c77387f816bc98a02a3c154265000e221c4e0feff13193050c29a`. Fresh instances k=130..141, twelve per N. k=100 and k=101 were not used. The circuit is the original interleaved shared-gamma KM, with the round-4 starts and the nodata to K to KM nesting. QAOA uses that same list. No packing change, no pi/4 or pi/2 shell shift, no popcount shells, no repulsion penalty, and no extra QAOA starts.

The claim is energy-only selection against equal-depth QAOA at L=6. The diagnostic column keeps the highest subspace overlap among L=6 KM starts whose energy is at most the QAOA energy plus 1e-6. If that pool is empty, the diagnostic state is the energy-only state. The diagnostic column is not the verdict. The highest overlap in the whole L=6 list is reported with its energy gap and has no pass rule.

## Result

Both columns are **partial**. N=6 passes and N=9 and N=12 do not. `selection_limited` is false at every N: the diagnostic column does not turn a failing size into a pass.

| N | energy-only | diagnostic | wins / losses / ties | median dF [95% CI] |
|---|---|---|---|---|
| 6 | pass | pass | 11 joint / 0 / 1 | +0.202 [+0.043, +0.385] |
| 9 | fail | fail | 3 joint / 4 / 5 | −0.048 [−0.072, +0.021] |
| 12 | fail | fail | 4 joint / 0 / 8 | −0.007 [−0.018, +0.109] |

The two columns have the same win, loss, and tie counts. Median subspace-overlap gain of the diagnostic state is below 1e-6 at each N. Six instances trip the frozen "differs" flag. Two of them move the overlap by a visible amount, and both were already joint wins: N6 k137, overlap 0.868 to 0.888 at 0.010 higher energy, and N9 k136, overlap 0.423 to 0.543 at 0.009 higher energy. The other four move the overlap by at most 0.0003. None of the N=9 losses change.

The N=9 losses are k132, fidelity only, F_sub 0.008 against 0.178, energy 0.714 lower; k133, joint, F_sub 0.239 against 0.320, energy 0.552 higher, and the energy-legal pool is empty; k134, fidelity only, F_sub 0.163 against 0.241, energy 0.059 lower; and k141, joint, F_sub 0.340 against 0.407, energy 0.726 higher, pool empty. The N=6 tie is k134, F_sub 0.447 against 0.473, energy 0.343 higher, pool empty.

Mean L=6 F_sub, KM / QAOA: N=6 0.889 / 0.618; N=9 0.265 / 0.267; N=12 0.291 / 0.245.

The unrestricted maximum, which is not a comparator, has median overlap gain at most about 1e-6 and median energy gap below 1e-9. The largest energy gap is N9 k132, where the overlap rises from 0.008 to 0.040 at an energy 12.6 higher. That state is not energy-legal against QAOA.

Equal parameter count, KM at L=3 against QAOA at L=6, is not a pass. Median dF is −0.054, −0.065, and −0.093.

Independent replay of 312 states at each N: maximum relative energy error 5.0e-13, no subspace-count mismatches. Wall 1046 s of 4200. Starts 24912. Logged energy evaluations 1985526. L=1 grid evaluations 18432. Capped starts 34, all KM at N=6, nine at L=5 and twenty-five at L=6. The energy-selected L=6 state is itself capped on N6 k135, both repeats; the selected label is `p1_perturb_0.1` and its subspace overlap is 1.000.

## Reading

On this sample the miss at N=9 and N=12 is not an energy-legal state that energy-only selection threw away. The starts that remain energy-legal against QAOA sit on the energy minimum's overlap. States with a meaningfully higher overlap, where any exist, are above the QAOA energy and cannot become joint wins. This is not a proof that no parameter vector inside the same circuit does better, and it is not a global advantage. `B_cri` stays withheld.
