# Source contract for this comparison

The Hamiltonian is the one verified against He, Kemper, and Freericks,
arXiv:2408.03251, tag `2.0` commit `f4c329fa69dcf98b7be32374c145584209e21f0e`.

Open chain, `ħ = 1`:

```
H = sum_{i<j} J_ij Z_i Z_j + 0.1 sum_i X_i
```

`J_ij` for `i < j` are i.i.d. normal with mean 0 and variance 1. The matrix is
stored symmetric with a zero diagonal. There is no Kac factor. `0.1` is the
`B_f` in `data/fig_5/8-site-MTE-optim_spin_glass-gpu.ipynb`. Site `i` is bit
`i`, least significant bit is site 0, `Z|0⟩ = +1`.

The author antiferromagnetic code uses `J_ij = 1/|i-j|`, not this ensemble.
Those N=8 matrices are not redrawn and are not targets here.

The coupled gap is the minimum, over a field path, of the gap from the ground
state of `H_A + B H_B` to the first excited state with the same spin-reflection
parity after rounding the parity expectation to 5 decimal digits. That is the
rule in `spin_glass_def_parity_energy_sprectum.ipynb`. Spatial inversion is
not a symmetry of a generic `J` and is not part of that rule. The ordinary gap
is `E_1 - E_0` of the full spectrum at the same field. The path in the author
notebook is `B` from 0.1 to 10 in steps of `10^{-4}`. This slice uses that
grid at N=6. At N=9 the frozen grid step is 0.01. At N=12 the frozen grid is
the nine fields 0.1, 0.3, 0.5, 0.8, 1.2, 2, 4, 7, 10. A minimum on the first
or last grid point is marked as a boundary. These coarser grids are not the
author step, and they are not used to drop instances.

Packing for the shell, and only for the shell, is the interleaved triples
`[a, a+N/3, a+2N/3]`. The Hamiltonian is not permuted.
