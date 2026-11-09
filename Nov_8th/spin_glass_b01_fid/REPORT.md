# Historical B=0.1 Fiedler round

Benchmark `B = 0.1`. This is the author's historical circuit field. It is not `B_cri`. The Binder source gate remains a withhold. The `|J|`-max confirmation on k=44..55 is a separate recorded result and was not refit.

## Idea

The circuit is the Nov 3 shared-gamma family: one gamma, one thetaK, one thetaM, and three shell angles per layer. The cavity partition is one function of `J` at every N. Sites are ordered by the Fiedler vector of the Laplacian of `|J|`, and the cavities are the contiguous triples of that order. `K` and `M` are the original constructions. The quantum ground state is not used.

This is not a choice between the interleaved partition and the `|J|`-max partition, and it is not chosen separately at each N.

## Protocol

Frozen in `11506a25` before the fits. `PREREG.json` SHA-256 `602256e9eb4ca6c9959f402d98671de2892fe743c7c59eeae340e4d999ec5d36`. Instances k=56..67. Two repeats. The optimizer, boxes, and continuation are the round-4 procedure. Selection is lowest energy. `F_sub` is computed afterwards at tolerance 1e-6.

A size passes only when KM_fid has more wins than losses against equal-depth QAOA at L=6 and the bootstrap 95% interval for the median fidelity difference lies above 0. The round is confirmed only if N=6, 9, and 12 all pass.

## Result

Overall: **partial**.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM_fid | 6 | 12 (11 / 1) | 0 | 0 | +0.278 [+0.135, +0.406] |
| KM_fid | 9 | 8 (8 / 0) | 0 | 4 | +0.092 [+0.042, +0.218] |
| KM_fid | 12 | 7 (6 / 1) | 1 | 4 | +0.083 [−0.029, +0.200] |

N=6 and N=9 pass. N=12 does not, because the interval includes 0. The N=12 loss is k=61, a fidelity-only loss: F_sub 0.025 against 0.117, with energy 0.813 lower. Energy-only selection keeps it. The N=6 fidelity-only win is k=65, energy 0.070 higher. The N=12 fidelity-only win is k=60, energy 0.034 higher.

Mean L=6 F_sub is 0.874, 0.584, and 0.349 for KM_fid and 0.589, 0.405, and 0.261 for QAOA at N=6, 9, and 12.

The interleaved KM reference on the same instances, not the claimed family, has median differences +0.238 [+0.112, +0.369], +0.059 [−0.062, +0.239], and +0.083 [−0.002, +0.208]. Its losses are N9 k57 (F_sub 0.132 against 0.262, energy 0.014 lower) and N9 k59 (F_sub 0.001 against 0.176, energy 0.283 lower). Mean L=6 F_sub for interleaved KM is 0.827, 0.494, and 0.384. On this sample the Fiedler partition cleared the frozen N=9 bar and the interleaved reference did not. Neither cleared N=12. That description does not authorize using one packing at N=9 and another at N=12.

## Parameters and cost

KM and KM_fid have 3+3L parameters, 21 at L=6. QAOA has 2L parameters, 12 at L=6.

The secondary comparison is KM_fid at L=3 against QAOA at L=6, both 12 parameters. It is not part of the verdict. Median fidelity differences are −0.026 [−0.085, +0.031], −0.044 [−0.129, +0.048], and −0.066 [−0.106, −0.035], with more losses than wins at each N.

Optimizer wall 1981 s of 2700. Starts 43488. Logged energy evaluations 3532559. L=1 grid evaluations 32256. No start reached the 2000-evaluation cap.

Independent replay of the KM, KM_fid, and QAOA states, 1296 states, rebuilt the couplings, the Fiedler partition, and the ground space without the runner. The largest relative energy error is 9.9e-13 and the largest F_sub error is 6.9e-13.

These are local optima under this budget, on these 12 instances at each N.
