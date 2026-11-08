# Historical B=0.1 packing round

Benchmark `B = 0.1`. This is the author's historical circuit field. It is not `B_cri`. The Binder source gate remains a withhold: the two crossings differ by 0.176 and no new field rule was introduced.

## Idea

The circuit is the Nov 3 shared-gamma family. One gamma, one thetaK, and one thetaM are used at every layer, with three shell angles per layer. The only change from the interleaved reference is the cavity partition: the triples that maximize the sum of `|J|` on their three edges, ties broken by the lexicographically smallest sorted partition. `K` and `M` are rebuilt on that partition. QAOA has no triples, so its circuit is the equal-depth independent-layer ansatz.

Per-layer gamma was not fit. The round-6 projected initialization was not repeated.

## Protocol

Frozen in `dd4beeb8` before the fits. `PREREG.json` SHA-256 `d147eb571151c819cff450bb0788db20c438570f98f45821414252d3c4b0b5a1`. Instances k=44..55, which earlier rounds did not fit. Two repeats. L-BFGS-B, analytic gradient, 2000 evaluations, the round-4 boxes and continuation. Selection is lowest energy. `F_sub` is computed afterwards, tolerance 1e-6.

A size passes only when KM_pack has more wins than losses against equal-depth QAOA at L=6 and the bootstrap 95% interval for the median fidelity difference lies above 0, on all 12 instances. The round is confirmed only if N=6, 9, and 12 all pass. It is partial if N=6 passes and another size does not.

## Result

Overall: **partial**.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM_pack | 6 | 12 (12 / 0) | 0 | 0 | +0.160 [+0.082, +0.428] |
| KM_pack | 9 | 7 (7 / 0) | 2 | 3 | +0.102 [−0.027, +0.210] |
| KM_pack | 12 | 7 (6 / 1) | 0 | 5 | +0.098 [+0.009, +0.242] |

N=6 and N=12 pass. N=9 does not, because the interval includes 0. The two losses are k=46, a joint loss (F_sub 0.292 against 0.409, energy 0.498 higher), and k=48, a fidelity-only loss (F_sub 0.018 against 0.369, energy 0.280 lower). Energy-only selection keeps k=48.

Mean L=6 F_sub is 0.943, 0.543, and 0.435 for KM_pack and 0.684, 0.472, and 0.313 for QAOA at N=6, 9, and 12.

The interleaved KM reference, fit on the same instances and not the claimed improvement, has median differences +0.103 [+0.051, +0.201], +0.223 [+0.030, +0.296], and +0.072 [−0.030, +0.198]. Its losses are N6 k47 (F_sub 0.419 against 0.555, energy 0.213 higher), N9 k52 (0.321 against 0.429, energy 0.069 higher), and N12 k44 (0.026 against 0.296, energy 0.554 higher). On this sample the packing change improved the N=6 and N=12 comparisons and made the N=9 comparison worse. That is a description of the reference column. It does not move the frozen claim onto interleaved KM, and it does not confirm interleaved KM.

## Parameters and cost

KM and KM_pack have 3+3L parameters, 21 at L=6. QAOA has 2L parameters, 12 at L=6.

The secondary comparison is KM_pack at L=3 against QAOA at L=6, both 12 parameters. It is not part of the verdict. Median fidelity differences are −0.017 [−0.101, +0.210], −0.096 [−0.147, −0.074], and −0.095 [−0.176, −0.044], with more losses than wins at each N.

Optimizer wall 1805 s of 4800. Starts 43488. Logged energy evaluations 3588872. L=1 grid evaluations 32256. Twenty-two starts reached 2000 evaluations, all at N=6: 19 interleaved KM and 3 KM_pack. QAOA reached none.

Independent replay of the KM, KM_pack, and QAOA states, 1296 states, rebuilt the couplings, the packing, and the ground space without the runner. The largest relative energy error is 1.0e-12 and the largest F_sub error is 2.5e-13.

These are local optima under this budget. The comparison is these 12 instances at each N. It is not a statement about every spin glass.
