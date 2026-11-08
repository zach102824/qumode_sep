# Uc candidates, after the source gate

Not executed. The shared-gamma circuit stays the circuit. A later sweep, if
the parent accepts the Hamiltonian in `FREEZE_PROPOSAL.md`, compares these
generators. None of them is a ground-state oracle.

## Circuit that stays

Logical form, energy only:

```
|+⟩  →  [ exp(-i γ G)  then  shell_ℓ ]^L
```

`γ` and the feature thetas are shared by every layer. Each layer has its own
three shell controls, shared by the cavities. Raw parameter count `P = 3L+3`
when `G` has two feature thetas (`θ_K`, `θ_M`), and `P = 3L+2` or `P = 3L+1`
when an ablation drops a feature. The physical form is the same operator with
`W = H^{⊗N}` moved through the mixer, and no initial `W`.

`K_jk = sum_{i ≠ j,k} J_ij J_ik` and `M` is the block mean of `J` on the
interleaved triples `[a, a+N/3, a+2N/3]`. Both use the full-pair RMS,
including the zero slots, exactly as in the frozen antiferromagnetic code.
Those two features are functions of `J` only.

## Generators

1. `Hc` only. Shared `γ`, shell, no `K` or `M`.
2. `Hc + θ_K Khat`. Coupling-only. `θ_K` is fit per test instance, energy only.
3. `Hc + θ_M Mhat`. Coupling-only.
4. `Hc + θ_K Khat + θ_M Mhat`. Coupling-only. This is the current family.
5. Data-learned map, discovery and validation only. Inputs are the sorted
   upper-triangle entries of `J` and the two RMS scales. Output is a frozen
   `(θ_K, θ_M)` used on every test instance, or a frozen linear map from those
   inputs to the two thetas. The fit minimizes energy on discovery instances.
   Validation decides whether the map is kept. Test instances do not enter
   the fit and do not enter that decision. No eigenvector, ground-state
   energy, or fidelity is an input.

The data-learned map costs a pretraining budget that is reported separately
from the test budget. It is not free capacity. If the validation energy is
worse than generator 4 fit per instance, the map is not promoted.

Shell schedules may be initialized by the Pagano ascending rule already
checked against the supplement: linear into depth 3, natural cubic after that.
That initialization is not the trained object.

## QAOA ledger, separate from the generators

Equal-depth primary comparison: QAOA depth `p = L`, independent `γ_ℓ` and
`β_ℓ`, initial state `|+⟩`, order `exp(-i γ H_c)` then `exp(-i β sum X)`.
`H_c` is the ZZ part only. The transverse field stays in the measured
Hamiltonian. Fidelity and `η = (E - E_max)/(E_0 - E_max)` are computed on the
energy-selected vector and are not training targets. `η` is not fidelity.

A second ledger, not the primary one, is parameter-count matched (`p = ceil(P/2)`)
and a hardware-style count of two-qubit layers. Those numbers are not
substituted for the equal-depth table.

Initializations, same budget for every instance: the ascending endpoint
schedule, a Fourier initialization, and random restarts. The inclusive
evaluation cap is frozen before the runs and is the same order as the
candidate's inclusive evaluations at that depth. A start that hits the cap
is partial and stays in the distribution. Best, median, and the failed tail
are all reported. A local minimum is not called the global minimum.

## Hardness and the gap

On the discovery block only, two rankings are stored before any candidate
test run:

- minimum coupled gap along the author field path, with the ordinary gap
  beside it, and a flag if the minimum is on the path boundary;
- QAOA energy error at one frozen small budget, with no candidate state used.

The test comparison is over the whole test block. A later sentence may also
report the discovery-defined hard subset, but that subset is fixed from
discovery data. It is not redefined after the candidate results.

The correlation of either gap with the energy error is reported with `N` and
the disorder RMS in the same table. A correlation is not a cause. A small
ordinary gap can be a symmetry partner the drive does not reach; the
antiferromagnetic reproduction is the example.

## Cost

N=12 dense diagonalization is 4096 and is enough for the gap diagnostic.
No N=15 or N=18. No new optimization is included in this slice.
