# Coupled-gap source pin

This slice reads the paper and runs the author code. It does not start a new
spin-glass circuit comparison.

## Paper

- arXiv: https://arxiv.org/abs/2408.03251 and https://arxiv.org/pdf/2408.03251
- HTML used for the equations below: https://arxiv.org/html/2408.03251
  (the rendering dated 24 August 2026 in that page). An earlier HTML of v2
  uses the title "Modulated time evolution for efficient variational ground
  state preparation". The text quoted here is the later title.
- Title: Hidden local adiabatic ramp in the modulated time evolution and the
  quantum approximate optimization algorithm.
- Authors: Zekun He (Georgetown), A. F. Kemper (North Carolina State),
  J. K. Freericks (Georgetown).
- Journal landing page: https://doi.org/10.1063/5.0293153
- Reference [55] in that HTML is only "Z. He, A. F. Kemper, and J. K. Freericks,
  Dataset (2024)." The public archive that carries the code is
  https://github.com/zach102824/Modulated-time-evolution-MTE-and-QAOA
  and Zenodo https://doi.org/10.5281/zenodo.14212811 (version 2.0, 24 November 2024).

## Code pin

- Tag `2.0`: `f4c329fa69dcf98b7be32374c145584209e21f0e`
- `main` at the time of this run: `6494aa1a1009530f5d9758a96fd0266def6255d9`
- `git diff --stat 2.0 HEAD` changes only `README.md`. The notebooks used below
  are the tag-2.0 files.
- Local run: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1.
  The gap notebooks do not need the GPU/JAX stack described in the README.
  That stack is for the 12-site optimizers, which this slice does not run.

Files that define the gap:

- `data/fig_5/def_parity_energy_sprectum_sg.ipynb`
  Even-even orbit basis for the antiferromagnetic chain. The plotted gap is
  `vals[1] - vals[0]` inside that reduced matrix.
- `data/fig_5/spin_glass_def_parity_energy_sprectum.ipynb`
  Full 256-dimensional diagonalization. The gap kept is the first excited
  state whose spin-reflection parity, rounded to 5 digits, equals the ground
  state's. Both `J` matrices in that notebook are commented out. The same
  arrays are in `data/fig_5/8-site-MTE-optim_spin_glass-gpu.ipynb`, where the
  matrix labeled "critical energy gap 0.001" is the active one and the matrix
  labeled "0.009" is commented. The gap notebook's comment on the second
  matrix says `t_f = 493.01095833740817`, `E_c = 0.008849508983285403`,
  `B_c = 0.215`.

## Hamiltonian in the paper

Equation (3), open chain, `ħ = 1`:

```
H_0(t) = sum_{i<j} J_{ij} σ^z_i σ^z_j + B(t) sum_i σ^x_i
```

Equation (1) multiplies the whole operator by a modulation `λ̃(t)`. The final
energy in Appendix A is the expectation of the unmodulated `H_0(t_f)`.

The code sets the antiferromagnetic couplings to `J_ij = 1/|i-j|`
(`J_matrix_alpha = 1`). Appendix B writes `J_ij = 1/|i-j|^{-1}`. That LaTeX
is not what the notebook executes, and the notebook is what reproduces the
table. Spin-glass couplings are independent draws `J_ij ~ Normal(0, 1)` on
`i < j`. The two instances used for Table 1 are the literal matrices above,
not a seeded redraw. The generator in
`data/fig_5/spin_glass_differnt_j_ij.ipynb` calls `np.random.normal` with no
seed, and the pickle save is commented out.

In the spin-glass MTE notebook the field schedule used as the initial guess
runs from `B_0 = 10` to `B_f = 0.1`. The Hamiltonian whose energy is minimized
at the end of that schedule is `H_A + 0.1 H_B`.

## What "coupled" means in the source

Appendix B:

> The Landau-Zener problem tells us that the diabatic excitation depends on
> the instantaneous energy gap between the ground state and the first excited
> state of the same symmetry. We call this energy gap `Δ(B)`.

The same appendix names two symmetries for the transverse-field Ising model:
spin reflection (`σ^x → σ^x`, `σ^y → -σ^y`, `σ^z → -σ^z`) and spatial
inversion `i → N+1-i`. It says the 8-site and 12-site antiferromagnetic ground
states are even-even, and that this reduces the matrix from 256 to 72. The
executed notebook builds a basis of dimension 72.

The spin-glass gap notebook does not use spatial inversion. A generic `J` is
not mirror-symmetric, so that symmetry is absent. Its selection rule is
spin-reflection parity only, with `round(parity, 5)` equality.

`Δ` in Table 1 and Figure 3 is the minimum of that symmetry-sector gap along
the field path, not `E_1 - E_0` of the full spectrum, and not the gap of one
fixed target Hamiltonian. The path in the executed loops is `B` from 0.1 to 2
(antiferromagnet, step `10^{-4}`) or from 0.1 to 10 (spin glass, step
`10^{-4}`). A minimum that sits on the endpoint of the scan is path-boundary
dependent.

The ordinary spectral gap and this coupled gap are different numbers. At the
antiferromagnetic minimum they differ by more than an order of magnitude.
The cross-check is in `reproduction.json`.
