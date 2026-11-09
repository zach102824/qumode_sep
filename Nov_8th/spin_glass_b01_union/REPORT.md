# Historical B=0.1 energy-only union

Benchmark `B = 0.1`. This is the author's historical circuit field. It is not `B_cri`. The `|J|`-max confirmation and the Fiedler confirmation stay as they were recorded.

## Idea

Both circuits are the Nov 3 shared-gamma family: one gamma, one thetaK, one thetaM, and three shell angles per layer. One uses the `|J|`-max triples. The other uses the Fiedler triples. The rule is the same at every N. Each circuit is fit on its own. The claimed state is whichever of the two energy-selected states has the lower energy. Equal energy keeps the `|J|`-max state. Fidelity is not used to choose.

## Protocol

Frozen in `d0570819` before the fits. `PREREG.json` SHA-256 `7959adc7afb94a190ff1e3e03c3c2ed0fada3aefb4c6987441b0e1f737934f98`. Instances k=68..79. Two repeats for each packing. QAOA repeats 0 and 1 are the comparator. QAOA repeats 2 and 3 are only the equal-search report.

A size passes only when the union has more wins than losses against equal-depth QAOA at L=6 and the bootstrap 95% interval for the median fidelity difference lies above 0. The round is confirmed only if N=6, 9, and 12 all pass.

## Result

Overall: **partial**. The union does not close the hole.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM_union | 6 | 11 (11 / 0) | 0 | 1 | +0.141 [+0.045, +0.426] |
| KM_union | 9 | 8 (7 / 1) | 1 | 3 | +0.077 [−0.021, +0.198] |
| KM_union | 12 | 8 (8 / 0) | 1 | 3 | +0.188 [−0.043, +0.364] |

N=6 passes. N=9 and N=12 do not, because each interval includes 0. The losses are both fidelity-only and both keep a lower energy than QAOA: N9 k78 (F_sub 0.413 against 0.495, energy 0.054 lower; the other packing had F_sub 0.432) and N12 k78 (F_sub 0.006 against 0.136, energy 0.370 lower; the other packing had F_sub 0.009). Energy-only selection keeps both.

Of 36 instances the lower energy was the `|J|`-max circuit 20 times and the Fiedler circuit 16 times. Mean L=6 F_sub, union / `|J|`-max / Fiedler / QAOA: N=6 0.834 / 0.832 / 0.751 / 0.606; N=9 0.567 / 0.577 / 0.551 / 0.458; N=12 0.504 / 0.459 / 0.439 / 0.312.

On this sample the `|J|`-max component at N=9 has median dF +0.076 [+0.039, +0.198], which would meet the interval bar. That column is not the claim. The recorded `|J|`-max confirmation, on k=44..55, did not pass N=9. This sample does not reopen it.

The four-repeat QAOA comparison, not the verdict, lowered one QAOA energy (N6 k68, by about 0.004) and left the L=6 win counts and medians unchanged at the reported precision. Equal parameter count, union at L=3 against QAOA at L=6, has median dF +0.036, −0.148, and −0.092, with more losses than wins at N=9 and N=12.

## Cost

Each component has 3+3L parameters, 21 at L=6. The union keeps one of those vectors. QAOA has 2L parameters, 12 at L=6.

Optimizer wall 2077 s of 3600. Starts 49824. Logged energy evaluations 3951901. L=1 grid evaluations 36864. Eleven starts reached 2000 evaluations, all at N=6 (5 `|J|`-max, 6 Fiedler).

Independent replay of 1728 KM_pack, KM_fid, and QAOA states rebuilt both partitions and the ground space without the runner. The largest relative energy error is 9.0e-13 and the largest F_sub error is 7.6e-13.

These are local optima under this budget, on these 12 instances at each N.
