# Spin-glass round 1

Protocol commit `0d8627d9`. Hamiltonian, seeds, and budgets were frozen before the fits.
Discovery k = 0,1,2,3 and held-out k = 100,101 finished at N = 6, 9, and 12, L = 1 through 6.
101887 optimizer evaluations, 1872 starts. 709 starts stopped on the 80-evaluation ceiling.
Validation k = 50 was not fit. No instance was dropped. A local minimum is not a global one.

The gradient check on N = 6, k = 0 has max absolute error 3.0e-9. The ground-state residual is 1.5e-14. Replaying the same vector returns the same energy.

## Fidelity

The logged F is the squared overlap with the single lowest eigenvector. The ordinary gap at B = 0.1 is 10⁻¹⁴ to 10⁻⁶, and the ground space is twofold on 16 of the 18 instances. `F_sub` is the squared overlap with every eigenvector within 10⁻⁶ of E₀. It was computed after selection and was not used to select. At N = 6, F and F_sub agree. At N = 9 they do not: the logged F is numerically zero while `F_sub` is not.

L = 6, discovery. Each cell is one instance, in order k = 0,1,2,3.

| N | family | F | F_sub |
|---|---|---|---|
| 6 | KM | 0.464, 0.665, 0.821, 0.985 | same |
| 6 | K | 0.991, 0.683, 0.387, 0.935 | same |
| 6 | nodata | 0.777, 0.493, 0.455, 0.583 | same |
| 6 | QAOA | 0.970, 0.510, 0.516, 0.950 | same |
| 9 | KM | ~0, ~0, ~0, ~0 | 0.024, 0.672, 0.440, 0.534 |
| 9 | K | ~0 | 0.885, 0.696, 0.269, 0.508 |
| 9 | QAOA | ~0 | 0.747, 0.405, 0.338, 0.493 |
| 12 | KM | 0.357, 0.009, 0.036, 0.000 | 0.714, 0.018, 0.073, 0.001 |
| 12 | K | 0.359, 0.012, 0.046, 0.168 | 0.718, 0.024, 0.093, 0.620 |
| 12 | QAOA | 0.282, 0.000, 0.072, 0.118 | 0.564, 0.001, 0.144, 0.436 |

On the logged single-vector F, discovery KM is ahead of equal-depth QAOA on both energy and fidelity in 18/24 N = 6 cells, 6/24 N = 9 cells, and 13/24 N = 12 cells. The N = 9 count is a comparison of two zeros and is not a fidelity result. N = 6 k = 0 at L = 6 is the other way: QAOA F = 0.970, E = −10.757 against KM F = 0.464, E = −6.174. K, not KM, has F = 0.991 there.

Held-out L = 6, k = 100 then 101. K was not run on held-out.

| N | KM F / F_sub | QAOA F / F_sub | nodata F | data F |
|---|---|---|---|---|
| 6 | 0.978, 0.926 | 0.699, 0.550 | 0.558, 0.509 | 0.357, 0.274 |
| 9 | 0 / 0.260, 0.016 | 0 / 0.225, 0.202 | ~0 | ~0 |
| 12 | 0.037 / 0.714, 0.181 / 0.360 | 0.023 / 0.442, 0.144 / 0.288 | 0.017, 0.106 | ~0 |

The data family freezes the mean discovery KM thetas and refits only gamma and the shell. Its pretraining is the discovery KM search: 7209, 7553, and 7252 evaluations at N = 6, 9, and 12. On held-out it is worse than fitting KM per instance. The transferred thetas do not carry the per-instance optimum.

## Gaps

Spin-reflection parity, rounded to 5 decimals, as in the author spin-glass notebook. N = 6 uses the author step 10⁻⁴. N = 9 uses step 0.01. N = 12 uses nine fields. A minimum on the first or last field is a boundary.

At B = 0.1 the ordinary gap is the splitting inside the twofold ground space. The coupled gap is the gap to the next state of the same rounded parity, and it is order 1 where the parity of the numerical eigenvectors is stable. On four N = 12 instances the round-to-5 rule finds no partner at B = 0.1 because the two ground vectors are mixed by the eigensolver; those coupled values are missing, not zero. The path minimum on the coarse N = 12 grid is then at a neighboring field. None of these gaps was used to drop an instance. Four points per N do not support a correlation claim.

Full angles and per-start logs are in `selected.json` and `starts.jsonl`.
