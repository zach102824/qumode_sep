# Critical field for the spin-glass benchmark

This note is the source definition. The numerical adoption is in
`PREREG_SCAN.json` and is applied only after that file is hashed. Circuit
fits are not part of this definition. Historical runs at `B = 0.1` stay
historical.

## Hamiltonian that is already audited

He, Kemper, and Freericks, arXiv:2408.03251, equation for the unmodulated
operator. Open pair sum, `ħ = 1`:

```
H = sum_{i<j} J_ij Z_i Z_j + B sum_i X_i
```

`J_ij` are i.i.d. normal with mean 0 and variance 1. There is no Kac factor.
`Z|0⟩ = +1`. The author schedule ends at `B(t_f) = 0.1`. That number is the
final ramp value in the paper, not a critical field. Previous circuit
comparisons used it and remain the `B = 0.1` benchmark.

## What is not the critical field

The paper's Table 1 labels `B_c` as the field where the same-symmetry gap
`Δ(B)` is smallest. Appendix B defines `Δ(B)` as the Landau-Zener gap to the
first excited state of the same symmetry, and the figure caption calls the
vertical line at that minimum "the critical magnetic field". On the
antiferromagnet this minimum sits near `B ≈ 0.38` to `0.42`. On the two
printed spin-glass matrices it sits at `0.215` and at the scan endpoint
`0.1`.

That minimum is an annealing bottleneck on one matrix. It is not adopted
here. The ordinary gap `E1 − E0` is not adopted either. Antiferromagnetic
critical fields are not imported. Finite `N` has no true singularity, so a
single minimum of a gap on one sample is not turned into a thermodynamic
critical point.

## The published prescription that matches this ensemble

The interactions are all-to-all Gaussian, which is the transverse-field
Sherrington-Kirkpatrick model, not the long-range antiferromagnet. Mukherjee,
Rajak, and Chakrabarti, Phys. Rev. E 92, 042107 (2015), arXiv:1412.2973,
study that model at zero temperature by exact diagonalization. Their printed
density at `J = 1` is

```
ρ(J_ij) = (N / (2π))^{1/2} exp(−N J_ij^2 / 2),
```

so the variance is `1/N`. A sentence in the same paper says the variance is
`J/√N`. At `J = 1` that sentence is `N^{−1/2}`, which disagrees with the
density. The density is the one used here. It is the standard SK
normalization, and it is the normalization in which they quote

```
Γ_c(T = 0) = 1.62 ± 0.05
```

as the infinite-volume extrapolation of Binder crossings. Their operational
finite-size estimator is the crossing of the disorder-averaged Binder
cumulant between two sizes.

Author couplings have variance 1, not `1/N`. Scaling `J` by `√N` scales the
whole operator by `√N` when the transverse field is scaled the same way, and
the eigenstates match:

```
B = Γ √N.
```

`Γ` is the SK field. `B` is the coefficient in the author Hamiltonian. These
are not QAOA mixer angles. A mixer angle `β` multiplies `sum X` inside the
ansatz. `B` multiplies `sum X` in the scored energy `⟨ZZ⟩ + B ⟨sum X⟩`.

## Observable

On the lowest eigenvector,

```
Q_k = N^{−k} sum_{i1...ik} ⟨Z_{i1} ... Z_{ik}⟩^2,
g = (1/2) [ 3 − mean(Q_4 / Q_2^2) ].
```

The mean is over disorder. This is their equation (5) and their T=0 Binder
formula. `Q_1` is recorded only as a residual: a spin-reflection eigenstate
has `⟨Z_i⟩ = 0`, so `Q_1` is not an order parameter in a parity sector.

The pseudocritical field at these sizes is the crossing of `g` between
`N = 6` and `9`, and between `N = 9` and `12`, if that pair of crossings
exists, is unique inside the pre-registered window, and agrees to the
pre-registered tolerance. The working benchmark is then one `Γ` for every
instance and `B_cri(N) = Γ √N`. Instance-by-instance gap minima are saved
and do not move this choice. The infinite-volume `1.62 ± 0.05` is an anchor,
not a substitute for the finite-size crossing.

If the pre-registered rule fails, no `B_cri` is invented and no circuit is
run at a guessed field.

## Ensemble

Seeds `SeedSequence([20261170, N, k])`, `k = 0..63`, `N ∈ {6, 9, 12}`. Same
`J` law as the circuit instances, disjoint indices from `SeedSequence([20261115, N, k])`.
The ground state is used only to evaluate this field. It is not a feature,
a start, a packing rule, or a training target.

The numerical outcome of this frozen rule is in `RESULTS.md`. The rule
itself is the hashed `PREREG_SCAN.json` and was not edited after the scan.
