# Historical B=0.1: diverse L=1 lineages are partial

Protocol `1545e6a2` before any fit. PREREG SHA-256 `82ac2ac410aea9512bc5e5e865f0e25f49432d142616bdf85bf012faab80cbe6`. Fresh instances k=166..177, twelve per N. k=100 and k=101 were not used. The circuit is the original interleaved shared-gamma KM, with the round-4 starts and the nodata to K to KM nesting. After those L=1 optimizations, up to three states that the energy beam did not keep, and that sit at least 0.5 box half-widths from it, are continued on their own through L=6. The continuation vector stays a candidate. QAOA uses the round-4 list only. No packing change, no pi/4 or pi/2 shell shift, no popcount shells, no repulsion penalty, no extra QAOA starts, no fidelity post-selection, no full-box uniform draw, and no scale-normalized sphere around the finished L=6 vector.

The verdict is the diverse-inclusive column at equal depth. The continuation column is the equal-search baseline. Equal parameter count is not the verdict.

## Result

Both columns are **partial**. They match on every instance. Zero of the 848 L=6 lineage states has a lower energy than the continuation vector, so every selected label is `historical`. `deeper` is false at N=6, 9, and 12.

| N | continuation | diverse | wins / losses / ties | median dF [95% CI] | deeper |
|---|---|---|---|---|---|
| 6 | pass | pass | 9 joint / 0 / 3 | +0.206 [+0.028, +0.368] | false |
| 9 | fail | fail | 8 (7 joint, 1 fidelity-only) / 2 / 2 | +0.137 [-0.034, +0.173] | false |
| 12 | fail | fail | 4 (3 joint, 1 fidelity-only) / 1 / 7 | +0.015 [-0.067, +0.126] | false |

N=6 passes on the continuation search: 9 joint wins and an interval above 0. The diverse column does not create that pass. N=9 does not. The losses are k171, fidelity only, F_sub 0.084 against 0.254, energy 0.009 lower, and k176, joint, F_sub 0.198 against 0.286, energy 0.193 higher. N=12 does not. The loss is k172, fidelity only, F_sub 0.154 against 0.232, energy 0.218 lower.

Four L=6 lineage states at N=9 and eleven at N=12 return to the continuation energy within 1e-8. None goes below it. The nearest start that does not recover that energy sits 0.040 above at N=6, 0.161 above at N=9, and 1.122 above at N=12. Two hundred twelve lineages were opened. No lineage start hit the evaluation cap. Twelve continuation starts at N=6 did, and the selected L=6 state at N=6 k=170 is one of them.

Equal parameter count, continuation KM at L=3 against QAOA at L=6, is not a pass. Median dF is -0.018, -0.154, and -0.103.

Independent replay of 300 states at each N: maximum relative energy error 5.1e-13, no subspace-count mismatches. Wall 1308.3 s of 4200. Starts 29152, of which 4240 are lineage starts. Logged energy evaluations 2466200. L=1 grid evaluations 18432. Capped starts 12.

## Reading

On this sample the diverse lineages did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten. N=6 clears the historical comparator with that same continuation search, and a lower-energy lineage state was not available. N=9 and N=12 do not clear it. This is one fresh window, not a global advantage. `B_cri` stays withheld.
