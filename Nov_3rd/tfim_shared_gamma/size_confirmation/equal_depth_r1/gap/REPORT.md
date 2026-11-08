# N=9 stationary gap

Shared-gamma L=6 has 21 parameters. QAOA p=10 has 20, p=11 has 22, p=12 has 24. No p equals 21.
The six-start equal-depth recipe is unchanged.

Rare state error 0.232689, gradient 1.226e-05, Hessian min 0.0904, negative eigenvalues 0.
Reliable state error 0.322260, gradient 1.764e-05, Hessian min 0.2195, negative eigenvalues 0.
Dense energy matches the logged propagation to 8.882e-16 and 0.000e+00.
Fidelity between the rare state and p=11: 0.930341. Reliable vs p=11: 0.911638.

| family | L | P | best error | best F | beats p=11 on both | evals |
|---|---|---|---|---|---|---|
| layer_gamma | 5 | 22 | 0.1766 | 0.6943 | False | 1237 |
| layer_gamma | 6 | 26 | 0.1465 | 0.7339 | True | 1565 |
| layer_theta | 4 | 21 | 0.2770 | 0.5692 | False | 1158 |
| layer_theta | 6 | 31 | 0.1982 | 0.6340 | False | 1814 |
| drop_K | 6 | 20 | 0.3104 | 0.6002 | False | 1623 |
| drop_M | 6 | 20 | 0.3484 | 0.6112 | False | 1664 |
| contiguous | 6 | 21 | 0.3054 | 0.7839 | False | 1522 |

p=11 reference error 0.170333, fidelity 0.706696. A pilot that does not beat both is not a replacement for six-start.

Both shared-gamma states are local minima. The Hessians are positive definite. Random-direction second differences match the quadratic forms. A short reoptimization along the softest mode returns to the same energy. The rare state and the p=11 state have fidelity 0.930 and the same parity. Their ground-state weights are 0.633 and 0.707, and both put the rest of that weight on the same-parity level 0.453 above the ground state. They are not in different parity sectors. The rare-state variance is 0.301, close to the p=11 variance 0.310. The reliable state’s variance is 0.804.

Per-layer gamma at L=6 has 26 parameters, four more than p=11. Its held-out best is error 0.146486 and fidelity 0.733888. A dense-matrix replay matches that energy and the hash. Twelve of 16 chains beat p=11 on both metrics, Wilson [0.505, 0.898]. Fifteen of 16 beat equal-depth p=6. The parameter-matched layer-gamma point, L=5 with 22 parameters, does not: error 0.1766 and fidelity 0.6943. Free theta per layer, even with 31 parameters, stays at error 0.1982. Dropping K or M is worse. Contiguous packing raises fidelity to 0.784 and leaves the energy at 0.305.

| question | evidence | reading |
|---|---|---|
| optimizer or budget inside one gamma | gradients near 1e-5, positive-definite Hessians, soft-mode steps return | the 0.233 point is stationary |
| one-gamma constraint | per-layer gamma at L=6 passes p=11; the 22-parameter L=5 version does not | a labeled change of the gamma constraint can move the energy; equal parameter count did not |
| packing | contiguous energy 0.305 | not the source of the matched gap |
| K or M feature | dropping either worsens the energy | both features are used; neither removal closes the gap |
| expressivity | this search did not prove a limit | no impossibility claim |

The six-start recipe stays the equal-depth recommendation. The one next experiment, if matched-parameter work continues, is a 48-chain held-out of per-layer gamma at L=6, kept as a different family. Shared-gamma matched-parameter search can stop: its known points are local minima in the same sector as p=11. N=15 and N=18 were not run.
