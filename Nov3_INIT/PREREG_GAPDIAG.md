# Bounded spectral-gap diagnostic: N=6 and N=12 quantum SK, entropy-peak B* vs g=1.5J

One bounded diagnostic authorized by Zach. **Not** a campaign restart. No SPSA, no ansatz, no circuit, no optimization, no quantum-query cost. All other experiments stay stopped. Stop after this batch.

## Hamiltonian and normalization (existing repo convention, unchanged)

\[ H(g) = \sum_{i<j} J_{ij} Z_i Z_j - g \sum_i X_i \]

`make_J(seed, N)` (`Nov9_SK_SPSA/experiments/FT/instance.py`): \(J_{ij} = \texttt{default\_rng(seed).normal}(0,\ 1/\sqrt N)\), upper triangle in row-major order. So \(J_{ij}\sim\mathcal N(0, J^2/N)\) with ensemble scale **J = 1** (standard SK normalization; the `1/sqrt(N)` is the standard deviation, not the variance). The fields are dimensionless in units of J:

- **g = B\*** (per instance): the existing *entropy-peak* field. Called "entropy-peak", **not** proven critical, **not** assumed to be the hardest point.
- **g = 1.5 J = 1.5** (fixed, same for every instance).

Diagonal \(\sum J_{ij} Z_i Z_j\) via `pair_diag(J, z_table(N))`. \(Z_q = 1-2\,\mathrm{bit}_q(x)\).

## B\* recipe (frozen, identical to `Nov3_INIT/bstar.py::maximize`, entropy part only)

Grid `linspace(0.05, 3.0, 60)`; ground state by `eigsh(k=1, SA, tol=1e-12)`; half-chain von Neumann entropy (nats) on the fixed balanced cut, qubits `0..N/2-1` vs the rest (`reshape(2^(N-N/2), 2^(N/2))`). Argmax must be interior; then `minimize_scalar(bounded, bracket=[grid[k-1], grid[k+1]], xatol=1e-6)`. Selection uses **entropy only**: `maximize`'s gap/parity contract is *not* applied as a selector, so no instance is admitted or rejected on gap. These are fresh seeds, so no cached B\* applies. B\* is computed for these 16 instances only, and its classical wall time is reported as preprocessing (not quantum queries). If an argmax lands on the grid edge, that instance is reported as "no interior B\*" for the B\* field and is **not** replaced.

## Panel (frozen before any gap computation)

- N=6: seeds **10904–10911** (8)
- N=12: seeds **10912–10919** (8)

Each instance is paired across the two fields (same J). The seeds were collision-checked across the worktree, all 132 unique git ref trees (filenames and seed-shaped tokens), and agent stores. The only hits were "local-later 10904+ unused" reservation notes. These seeds don't overlap 10888–10903 (PR130), the other stream's 10872–87, validation 11000–11031, or GR/final 20000–20999.

## Symmetry

\(P=\prod_i X_i\) maps \(|x\rangle\to|x\oplus 1^N\rangle\). It leaves \(Z_iZ_j\) invariant and commutes with each \(X_i\), so \([H,P]=0\) for every J and g. This is the only symmetry assumed. For generic random \(J_{ij}\) there is **no** total-spin (SU(2)) conservation, no \(\sum_i X_i\) or \(\sum_i Z_i\) conservation, and no site-permutation symmetry, because disorder breaks permutation invariance. Accidental disorder-specific symmetries aren't assumed. If one exists, it would show up only as unexpected zero couplings.

Driving protocol: we assume a **transverse-field ramp** in g, so \(\partial_g H = V = -\sum_i X_i\), which commutes with P. A different drive could break P and couple sectors, for example a longitudinal field \(\sum h_i Z_i\), a \(Y\)-term, or a catalyst that's odd under P. In that case the sector-resolved conclusions below would not carry over.

## Spectral method

- **N=6**: full dense `numpy.linalg.eigh` (dim 64). Parity of each eigenvector is \(\langle n|P|n\rangle\), resolved by diagonalizing P inside degenerate clusters (|ΔE| < 1e-9). Audit: dense parity-block eigh agrees.
- **N=12**: parity-block sparse. The representatives are \(x\) with top bit 0, and \(|x,\pm\rangle=(|x\rangle\pm|\bar x\rangle)/\sqrt2\) (dim 2048 each). \(H_\pm = \mathrm{diag}(H_c) - g[\sum_{q<N-1}\mathrm{flip}_q \pm \mathrm{flip}_{\text{mask}\setminus\text{top}}]\). We run `eigsh(SA, tol=1e-12)` with k ladder **8 → 16 → 32 → 64** per sector. k goes up only when no coupled state (below) is found in the + sector at the primary threshold. Audits: dense eigh of each 2048 block (eigenvalue agreement), full-space sparse \(H\) (dim 4096) `eigsh k=12` union agreement, a sparse commutator \(\|HP-PH\|_F\), and eigenpair residuals \(\|Hv-Ev\|\) in full space after embedding.

## Reported quantities (per instance, per field)

- Lowest 8 full-space eigenvalues with parities. E0, E1, full gap \(E_1-E_0\). Ground parity.
- Per sector: \(E_0^\pm, E_1^\pm\). \(\Delta_+=E_1^+-E_0^+\) (the symmetry-allowed excitation reachable from \(|+\rangle^{\otimes N}\), which lies in P=+). \(\Delta_-=E_1^- - E_0^-\). Cross-sector splitting \(E_0^- - E_0^+\).
- Couplings \(c_n = |\langle n|V|0\rangle|\) with \(V=-\sum X_i\), for all computed low eigenstates of both sectors. Opposite-sector \(c_n\) are computed numerically in **full space** (embedded vectors, V applied in the computational basis), not set to zero by construction, and should come out ~0 (round-off). **Coupled gap** \(\Delta_c(\tau)\) is \(E_n-E_0\) for the lowest \(n\ge1\) with \(c_n > \tau\). It's reported for τ ∈ {1e-10, 1e-8, 1e-6, 1e-4}, and **τ = 1e-8 is primary**. Same parity is necessary but not sufficient for \(c_n\ne0\). Also reported: \(c_n/(E_n-E_0)^2\) for the first coupled state (local adiabatic ratio, descriptive only).
- Means and ranges per (N, field). No statistical gate. Descriptive only.

## Explicitly not claimed

- No minimum gap over g is claimed. Only two fields are probed, with no scan.
- The entropy peak is **not** assumed to be critical or the hardest point.
- The minimum sector gap alone is **not** taken as adiabatic hardness. Matrix elements along the whole path and the ramp schedule matter.
- The full gap \(E_1-E_0\) is distinguished from the symmetry-allowed coupled gap. When \(E_1\) is P-odd, the full gap is irrelevant to an ideal P-preserving transverse ramp.

## Cost

Zero quantum energy queries (no variational evaluations). Classical preprocessing (B\* grid+refine, 16 instances) and the spectral wall time are reported. The historical SK query ledger is unchanged at 3252512.
