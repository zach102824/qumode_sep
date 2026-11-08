# Proposed N=6, 9, 12 spin-glass target

Not run. This is the equation to show before any candidate-versus-QAOA sweep.
It is the paper's spin-glass Hamiltonian, not a new transverse field glued
onto a different model. It is also not the frozen antiferromagnetic manifest
`14f40daa…51da9`, which stays untouched.

## Target at one field

Open chain, `ħ = 1`, sites `0 … N-1`, `N ∈ {6, 9, 12}`:

```
H = sum_{0 ≤ i < j < N} J_ij Z_i Z_j + 0.1 sum_{i=0}^{N-1} X_i
```

`J_ij` for `i < j` are i.i.d. `Normal(0, 1)`. The diagonal and the lower
triangle are zero. There is no Kac rescaling and no extra disorder on the
field. The coefficient `0.1` is `B_f` in
`data/fig_5/8-site-MTE-optim_spin_glass-gpu.ipynb`. `Z|0⟩ = +1`, `Z|1⟩ = -1`,
and site `i` is bit `i` with site 0 the least significant bit, matching the
existing simulator. The author notebooks pack the highest bit first; the
spectrum does not depend on that packing once `J` is the same.

The two N=8 matrices in the author notebook are reproduction instances only.
They are not part of this N=6/9/12 list, and they are not redrawn.

## Draws, if a later slice is authorized

One `numpy` `SeedSequence` with entropy `20261115`. For instance index `k`
the sequence is `[20261115, N, k]`, and the upper triangle is filled in order
`(i, j)` with `i < j` by one `normal(0, 1)` draw each.

| role | k | used for |
|---|---|---|
| discovery | 0 … 15 | coupled-gap diagnostic and QAOA-only ranking |
| validation | 50 … 57 | freeze the data-learned map |
| test | 100 … 115 | the only instances that can support a candidate claim |

No instance is dropped after seeing a candidate energy. The test block is
not used to choose features, thetas, or which instances look hard.

## Coupled gap as a diagnostic, not as the target

On each discovery instance, the later slice would rerun the author spin-glass
rule: full diagonalization of `H_A + B H_B` for `B` from 0.1 to 10 in steps
of `10^{-4}`, and the gap from the ground state to the first excited state
with the same spin-reflection parity after `round(·, 5)`. The reported
coupled gap is the minimum on that path. If the minimum is at `B = 0.1` or
at the last grid point, it is marked boundary. The ordinary gap `E_1 - E_0`
is stored beside it and is not substituted for it.

That diagnostic does not change `H`. Ranking discovery instances by this
minimum, or by a QAOA-only energy at a budget frozen before the candidate
runs, is the allowed hardness rule. Ranking by the candidate's own fidelity
is not.

## What is not being claimed

The paper's Table 1 is an 8-site modulated-time-evolution energy error, not
a QAOA-versus-shared-gamma comparison at N=6, 9, or 12. A small coupled gap
is not, by itself, evidence that the shared-gamma method will beat QAOA.
