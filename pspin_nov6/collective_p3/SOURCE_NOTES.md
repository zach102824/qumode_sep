# Sources actually opened

Opus family notes were treated as an unverified hypothesis. Glass families were not run. Citations below are only statements checked against a text that was opened in this session, or derivations done here.

## Pagano et al., arXiv:1906.02700

Opened: https://arxiv.org/html/1906.02700 (the HTML rendering of the paper, including the supplement).

Checked and used:

- Eq. (2): independent angles, `|β,γ> = Π_k exp(-i β_k H_B/J0) exp(-i γ_k H_A/J0) |ψ0>`, objective `<H>`.
- Eq. (3): `η = (E(β,γ) - E_max) / (E_gs - E_max)`.
- Supplement bootstrap: L=1 from an analytic/grid seed; L=2 seed `(β, β-0.2)`, `(γ, γ+0.2)` then local optimization; higher depth by linear (depth 3) or cubic (depth > 3) interpolation in `s_i=(i-1)/(q-2)`, resampled at spacing `1/(q-1)`, then local optimization.

Not copied from that paper: the numerical value 0.2 is in their J0-normalized units. It was applied here to `(γ, β)` in the normalization below, and also at 0.05, 0.1, and 0.4. Their smooth-angle and N-transfer observations are about the long-range TFIM. N-transfer was not used; each N was optimized from its own L=1 grid.

## Wauters, Mbeng, Santoro, arXiv:2003.07419

Opened: https://ar5iv.labs.arxiv.org/html/2003.07419

Checked and used:

- Their Eq. (1) is this Hamiltonian at p=3, with transverse field `h = Γ`.
- Their QAOA applies the cost layer before the mixer, layer 1 closest to `|+>`.
- They factor the evolution as `exp(-i γ H_z) exp(-i β H_x)` with the super-extensive `H_z = -(Σ Z)^p` and `H_x = -Σ X`, and they restrict those angles to a period of π. The conversion to the normalization used here is `γ = γ_W N^{2}`, `β = -β_W`.
- They state that for h=0 and odd N, `(β, γ) = (π/4, π/4)` in that normalization prepares the classical ground state exactly. For every even N in this run the minimal `γ_W` period is π/4, so that particular `γ_W` is a null cost layer. This was checked from the spectrum, not taken from their appendix.
- They report that good minima at `P ≪ N` are initialization-dependent, and that perfect fidelity in their runs required `P` proportional to N (they write `P*_N = N+1` for odd p). This run stops at L=5 and does not test that threshold.

Not used as a result of this run: their exponential-annealing-gap statement. They attribute it to Jörg et al. (2010). That paper was not opened. On the grid computed here the many-body gap is O(1) down to 0.455 at N=32.

## Derived here

- Cubic identity `(Σ Z)^3 = 6 Σ_{i<j<k} Z_i Z_j Z_k + (3N-2) Σ Z_i`, exact on every basis state for N=6,8,10,12.
- Mean-field first-order point of `e = -cos^3 θ - Γ sin θ` is `Γ_c = 3√3/4 ≈ 1.299`, with spinodal `Γ=3/2`. This is infinite-N mean field. Finite-N gaps were computed by diagonalization.
- Global `Π Y_i` sends H to -H, so `E_max = -E_0`.
- Global `Π X_i` exchanges the sign of the cubic term and preserves QAOA fidelity.
