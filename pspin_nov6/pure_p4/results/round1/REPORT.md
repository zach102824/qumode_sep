# Family C round 1 — pure distinct-index 4-body ferro vs antiferro

This run is **not** the random p-spin glass. Every quadruple has the same coupling. The attached Pagano note (arXiv:1906.02700) was used only for the angle-search protocol and for the shape of the target plot. Depth is called \(L\) here so it is not confused with interaction order \(p=4\).

Wall time 326 s (numpy 2.5.4, scipy 1.18.1). About \(7.7\times 10^6\) statevector evaluations, all inside the symmetric irrep after a full-space ground-state check. No asymptotic or speedup claim is made.

## Hamiltonian (one convention everywhere)

\[
H_C = s\, J\, a_N \sum_{i<j<k<l} Z_i Z_j Z_k Z_l,
\qquad
H = H_C - \Gamma \sum_i X_i,
\]

with \(J=1\), \(s=-1\) (ferro) or \(s=+1\) (AF), and

\[
a_N = \frac{N}{\binom{N}{4}} = \frac{24}{(N-1)(N-2)(N-3)}.
\]

Qubit 0 is the MSB of the statevector. \(Z|0\rangle=+|0\rangle\). This \(a_N\) is extensive **for the fully polarized configurations**: their cost is exactly \(s N\). The same number is the polarized cost of the collective control

\[
H_C^{\mathrm{ctrl}} = s\, N\, m_z^4 = s\, M^4 / N^3, \qquad M=\sum_i Z_i,\quad m_z=M/N.
\]

The control is not the family. It exists to show the repeated-index terms that \(m_z^4\) contains and that the distinct sum does not.

Finite-\(N\) fact that has to stay visible: the AF minimum of \(a_N e_4\) is \(O(1)\) and shrinks, not extensive. Examples (classical, \(\Gamma=0\)):

| \(N\) | ferro \(M^\star\), \(E\) | pure AF \(M^\star\), \(E\), degeneracy | collective AF minimum |
|---:|---|---|---|
| 6 | \(\pm 6\), \(-6\) | \(\pm 4\), \(-2\), 12 | \(M=0\), \(E=0\) |
| 8 | \(\pm 8\), \(-8\) | \(\pm 4\), \(-1.143\), 56 | \(M=0\), \(E=0\) |
| 10 | \(\pm 10\), \(-10\) | \(\pm 4\) and \(\pm 6\), \(-0.667\), 330 | \(M=0\), \(E=0\) |
| 12 | \(\pm 12\), \(-12\) | \(\pm 6\), \(-0.655\), 440 | \(M=0\), \(E=0\) |

So the same \(a_N\) makes ferro a deep extensive well and AF a shallow well whose classical manifold is an intermediate magnetization shell, not the fully polarized cat and not the \(M=0\) shell of \(+N m_z^4\).

## Repeated-index identity

On commuting involutions \(Z_i^2=I\),

\[
M^4 = N + 4(M^2-N) + 3N(N-1) + 6(N-2)(M^2-N) + 24\, e_4,
\]

\[
e_4 = \big[M^4 - (6N-8)M^2 + 3N(N-2)\big]/24.
\]

The \((2,1,1)\) multiplicity is 6, not 12: the twelve strings with pattern \(\{a,a,b,c\}\) are shared by the labels \((b,c)\) and \((c,b)\). The identity is an operator equation, so it holds for odd and even \(N\), and every basis state of a given \(M\) has the same \(e_4\). Checked by enumerating every quadruple against the closed diagonal for \(N=4..8\), both signs (max abs error \(<10^{-9}\)). Term lists: `results/validation/terms_N6.txt` (15 tuples) and `terms_N7.txt` (35 tuples).

Consequence: \(a_N e_4 - N m_z^4\) is a polynomial in \(M\) that vanishes at \(|M|=N\) and is nonzero in between. It is a finite-\(N\) renormalization of the four-body piece plus an \(M^2\) (two-body) piece plus a constant. Numerics are in `lower_body_delta.json`.

## Parity, reachability, and the symmetric block

\(H_C\) and \(\sum X\) both commute with every permutation and with \(P=\prod_i X_i\). The QAOA circuit starts at \(|+\rangle^{\otimes N}\) (eigenvalue \(+1\) of \(P\), and fully symmetric), applies a diagonal phase that depends only on weight, then \(\exp(-i\beta\sum X)\). It stays in the even-parity Dicke subspace, dimension \(\lfloor N/2\rfloor+1\) inside the symmetric irrep of dimension \(N+1\).

That does **not** imply the full-space ground state is in that block. The check was numerical, on the actual sparse or dense Hamiltonian:

- Pure model, both signs, \(\Gamma\in\{0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.00,1.10,1.25,1.50,2.00\}\).
- \(N=6,7,8,9,10,11,12\): 168/168 points have the even-sector energy equal to the full-space \(E_0\) (discrepancies \(\sim 10^{-14}\)).
- \(N=14,16\): 48/48.
- \(N=20\) at \(\Gamma\in\{0.05,1.00,1.10\}\), both signs: 6/6.

Odd \(N\) is the same code path: the flip parity basis has no self-paired weight. At the featured \(\Gamma\) values the odd-sector gap is positive (for example ferro \(N=7\), \(\Gamma=1.1\), \(E_0^{\mathrm{odd}}-E_0^{\mathrm{even}}=0.265\); AF \(N=11\), \(\Gamma=0.05\), gap \(0.0046\)). At ferro \(\Gamma=0.05\) the even/odd cat splitting is numerically unresolved (\(\sim 10^{-11}\) or smaller), i.e. the GHZ doublet. QAOA only prepares the even cat.

Dicke evolution matches a full \(2^N\) statevector on random angles for \(N=5,6,7,8\) to \(10^{-15}\). Larger-\(N\) QAOA below uses the Dicke representation only after the full-space energy match above. This is a finite grid, not a theorem for every \(\Gamma\) or every \(N\).

## Angle periods and the circuit

Time order: layer \(k=1..L\) acts first. Inside a layer, \(\exp(-i\gamma_k H_C)\) then \(\exp(-i\beta_k \sum_i X_i)\). Angles multiply those generators directly (no extra \(1/J_0\)).

- Each \(\beta\) has projective period \(\pi\): \(\exp(-i(\beta+\pi)\sum X)=(-1)^N \exp(-i\beta\sum X)\). Period \(\pi/2\) is not a symmetry. Checked on the Dicke matrix of \(\sum X\).
- \(\gamma\) period is \(T_\gamma=2\pi/(a_N g)\) with \(g=\gcd\) of the distinct \(e_4\) values. Empirical check: no proper divisor of \(T_\gamma\) closes all diagonal phases for the sizes tested.

| \(N\) | \(g(e_4)\) | \(T_\gamma\) |
|---:|---:|---:|
| 6 | 4 | 3.927 |
| 8 | 2 | 27.489 |
| 10 | 8 | 16.493 |
| 12 | 2 | 129.591 |
| 16 |  | 357.356 |
| 20 |  | 761.051 |

Raw optimizer output is what the bootstrap sees. Folding into one period and a global sign flip \((\beta,\gamma)\to(-\beta,-\gamma)\) is reporting-only. Schedules are not smoothed. At ferro \(N=12\), \(\Gamma=1.1\), layers \(L=1..4\) sit near \(T_\gamma\) (canonical \(\gamma\) of order \(0.2\)–\(2\)), while the \(L=5\) winner is a different branch with raw \(\gamma\) of order \(1\)–\(6\). See `figures/angles_ferro_N8.png` for the same phenomenon at \(N=8\), where \(T_\gamma\approx 27.5\).

Circuit duration is reported two ways: \(\sum(|\beta_k|+|\gamma_k|)\) on the raw angles, and the same sum after folding. The raw sum is often \(\sim L\, T_\gamma\) when the solver sits one period away from a small canonical \(\gamma\). The canonical sum is the meaningful generator-time.

## Search

Primary objective: minimize \(\langle H\rangle\). Not fidelity.

- \(L=1\): grid over one full \(\gamma\) period (spacing \(0.06\), capped at 2400 points) and 36 \(\beta\) values in \([0,\pi)\), then L-BFGS-B from separated basins, the zero seed, a small seed, and random seeds. Nelder–Mead polish on the best two basins.
- \(L=2\): Pagano offset \((\beta^\star,\beta^\star-0.2)\), \((\gamma^\star,\gamma^\star+0.2)\), plus perturbations, sign flip, zero-pad, random.
- \(L\ge 3\): natural cubic spline (linear when only two previous nodes) of the **raw** angles vs \(s\in[0,1]\), resampled at \(L\) nodes, then the same local search. Natural cubic boundary conditions are not a global polynomial; a line is reproduced, a quadratic is not.
- The unevolved zero-pad of depth \(L-1\) is always kept. Best energy never increased with \(L\) (0 violations in 108 pure sequences).
- Shared \(\gamma\) (one \(\gamma\), independent \(\beta_k\)) is a sidecar.
- Separate \(L=1\) **fidelity** maximization, not used as the energy result.
- Digitized anneal baseline, not a \(2L\)-parameter optimization: midpoint Trotter of \(H(s)=(1-s)(-\sum X)+s H\), \(\Delta t=T/L\), \(T\) on \(\{0.25,0.5,1,2,4,8,12\}\), plus one \(T\) matched to a duration. Matching the raw duration is a poor resource match when raw \(\gamma\sim T_\gamma\); `anneal_canonical_match.json` repeats the match on the folded duration.

\(\eta=(E-E_{\max})/(E_0-E_{\max})\) uses the full-space edges. On this normalization \(E_{\max}-E_0\) is \(O(N)\) because it runs from the AF side of the spectrum to the ferro side, so \(\eta\) sits near 1 for any state whose energy is within \(O(1)\) of the polarized well. High \(\eta\) with small \(F\) is expected and is not success. Ferro \(N=20\), \(\Gamma=1.1\), \(L=1\): \(F=0.016\), \(\eta\approx 0.97\).

Half-cut entropy is the von Neumann entropy of qubits \([0, N/2)\), MSB order, in bits. For a symmetric state every subsystem of that size matches; the code path for a generic full-space vector does not assume that. Product overlap is the best real permutation-symmetric product state. Cat overlap is the even GHZ \((|0\cdots 0\rangle+|1\cdots 1\rangle)/\sqrt{2}\).

\(\Gamma\) list was fixed before reading any QAOA fidelity. The panel \(\Gamma\) is the maximizer of a ground-state-only score (low \(F_0\), not a cat, not a product, resolved gap), averaged on \(N=6,8,10,12\).

## What the ground state is doing

**Ferro, small \(\Gamma\).** Even GS is the GHZ cat. \(N=12\), \(\Gamma=0.05\): \(F_{\mathrm{cat}}=1\), half-cut entropy \(=1\) bit, symmetric-product overlap \(=0.50\), \(F_0=2^{-(N-1)}\) scale (\(0.001\)). Rejected.

**Ferro, \(\Gamma=1.1\) (top GS score; positive on 3/4 sizes, mean pulled negative by \(N=6\)).** Still cat-admixed, not a paramagnet.

| \(N\) | \(F_0\) | \(F_{\mathrm{cat}}\) | \(S_{1/2}\) (bits) | product \(F\) | \(m_2\) | \(C_4\) | full gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.623 | 0.544 | 0.98 | 0.62 | 0.625 | 0.514 | 0.366 |
| 8 | 0.436 | 0.589 | 1.15 | 0.45 | 0.697 | 0.582 | 0.194 |
| 10 | 0.278 | 0.629 | 1.23 | 0.44 | 0.765 | 0.656 | 0.102 |
| 12 | 0.159 | 0.656 | 1.23 | 0.46 | 0.821 | 0.718 | 0.051 |
| 16 | 0.041 | 0.656 | 1.13 | 0.49 | 0.881 | 0.785 | 0.011 |
| 20 | 0.010 | 0.617 | 1.07 | 0.49 | 0.898 | 0.806 | 0.0023 |

\(m_2=\langle M^2\rangle/N^2\) stays large. This is a partially melted cat, not a volume-law paramagnet and not a product state (best computational-basis population \(\approx 0.3\)).

**AF, \(\Gamma=0.05\).** Not a cat (\(F_{\mathrm{cat}}=0\)). Classical weight is the intermediate shell. Entropy is higher than the cat at small \(N\) and falls as the shell becomes a larger fraction of the \(|+\rangle\) binomial.

| \(N\) | \(F_0\) | \(S_{1/2}\) | product \(F\) | \(m_2\) | \(C_4\) | full gap | \(E_0\) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.255 | 1.98 | 0.27 | 0.441 | \(-0.330\) | \(1.8\times 10^{-4}\) | \(-2.018\) |
| 8 | 0.392 | 1.84 | 0.39 | 0.251 | \(-0.135\) | \(1.6\times 10^{-3}\) | \(-1.211\) |
| 12 | 0.524 | 1.58 | 0.52 | 0.200 | \(-0.043\) | \(5.6\times 10^{-3}\) | \(-0.933\) |
| 20 | 0.744 | 0.97 | 0.74 | 0.104 | \(-0.010\) | \(0.021\) | \(-1.111\) |

By \(N=20\), \(\Gamma=0.05\) is already paramagnet-side (\(F_0=0.74\)). The shallow AF well does not stay ordered under an \(O(1)\) field as \(N\) grows. Pushing \(\Gamma\) down toward \(0\) raises the classical degeneracy (hundreds to \(10^4\) bitstrings) and the \(|+\rangle\) overlap of the even shell remains \(O(1)\) because \(\binom{N}{n_\downarrow}/2^{N-1}\) is not exponentially small on that shell.

**Collective control.** Ferro GS overlap with the pure GS is \(0.97\)–\(1.00\) on the scanned window: both want \(|M|=N\), and the lower-body remainder is a small shift of a cat. AF overlap at \(\Gamma=0.05\) is \(0.021\) (\(N=6\)), \(0.107\) (\(N=8\)), \(0.212\) (\(N=12\)), \(0.354\) (\(N=16\)). Energies differ with them (\(N=6\): pure \(-2.018\) vs control \(-0.218\)). At \(\Gamma\gtrsim 1\) both are paramagnets and the overlap returns to \(\sim 1\). Control QAOA at \(N=8\) is easy-\(L=1\) at every \(\Gamma\) tried (AF \(\Gamma=0.05\): \(F_{L=1}=0.93\); ferro \(\Gamma=1.1\): \(F_{L=1}=0.98\)).

## QAOA fidelities (energy objective)

\(F\) is the squared overlap with the even ground subspace. On every row below that subspace is the full-space ground state, so \(F_{\mathrm{global}}=F_{\mathrm{reach}}\).

### Ferro, \(\Gamma=1.1\)

| \(N\) | \(L=0\) | \(L=1\) | \(L=2\) | \(L=3\) | \(L=4\) | \(L=5\) | \(\eta(L=5)\) | \(L=5\) winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 0.623 | 0.974 | 0.971 | 1.000 | 1.000 | 1.000 | 1.000 | pagano interp |
| 8 | 0.436 | 0.539 | 0.548 | 0.983 | 1.000 | 1.000 | 1.000 | pagano interp |
| 10 | 0.278 | 0.341 | 0.436 | 0.552 | 0.951 | 0.938 | 0.996 | sign-flip polish |
| 12 | 0.159 | 0.207 | 0.353 | 0.619 | 0.813 | 0.982 | 0.999 | random wide |
| 14 | 0.083 | 0.115 | 0.153 | 0.153 | 0.828 | 0.828 | 0.987 | zero-pad (no gain) |
| 16 | 0.041 | 0.061 | 0.094 | 0.147 | 0.256 | 0.630 | 0.983 | perturb |
| 20 | 0.010 | 0.016 | 0.022 | 0.050 | 0.125 | 0.125 | 0.975 | zero-pad (no gain) |

### AF, \(\Gamma=0.05\)

| \(N\) | \(L=0\) | \(L=1\) | \(L=2\) | \(L=3\) | \(L=4\) | \(L=5\) | \(\eta(L=5)\) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.255 | 0.711 | 0.981 | 1.000 | 1.000 | 1.000 | 1 |
| 8 | 0.392 | 0.805 | 0.908 | 0.992 | 1.000 | 1.000 | 1 |
| 12 | 0.524 | 0.670 | 0.873 | 0.983 | 0.998 | 1.000 | 1 |
| 20 | 0.744 | 0.808 | 0.861 | 0.917 | 0.971 | 0.997 | 1 |

AF \(L=1\) is already high and \(L=5\) is saturated. That is the easy regime.

## Rejected regimes (all 96 pure points with \(N\in\{6,8,10,12\}\))

A point can carry more than one label. **Zero** points were left unrejected.

| label | count (of 96) | rule |
|---|---:|---|
| easy \(L=1\) | 74 | energy-optimal or fidelity-max \(F(L=1)\ge 0.70\) |
| trivial paramagnet | 54 | \(F_0>0.5\) |
| near product | 35 | symmetric-product overlap \(>0.80\) and \(S_{1/2}<0.35\) |
| GHZ cat | 28 | \(F_{\mathrm{cat}}>0.85\) |
| borderline \(L=1\) | 22 | \(0.45\le F(L=1)<0.70\), and not already easy |
| unresolved gap | 8 | full gap \(<10^{-4}\) (ferro cat doublet) |

The Pagano-shaped gate used as a screen, not as a fit: \(F(L=1)\le 0.35\), fidelity-max \(L=1\) also \(\le 0.35\), \(F(L=5)-F(L=1)\ge 0.08\), \(F(L=5)<0.85\), and none of the labels above. Nothing passed.

The least-bad ferro curve is \(\Gamma=1.1\), \(N=16\): \(F\) goes \(0.041, 0.061, 0.094, 0.147, 0.256, 0.630\). \(L=1\) is genuinely low and \(F\) grows, but \(L=5\) is already \(0.63\), the GS still has \(F_{\mathrm{cat}}=0.66\) and \(m_2=0.88\), and the \(L=1\) fidelity maximizer reaches \(F=0.71\) at a much worse energy (\(-13.48\) vs energy-opt \(-17.63\); \(E_0=-18.61\)). \(N=20\) at the same \(\Gamma\) stalls: \(L=5\) did not beat \(L=4\) (\(F=0.125\), winner tag `zero_pad`). The \(L=1\) grid spacing there is \(T_\gamma/2400\approx 0.32\), coarser than \(2\pi/O(N)\), so that stall is not a proof that \(F\) saturates.

## Energy optima are not monotone fidelity

Best energy is monotone by the zero-pad carry. \(F\) of that state dropped in 6 pure sequences. Cleanest example, ferro \(N=10\), \(\Gamma=1.1\):

| \(L\) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| \(F\) | 0.278 | 0.341 | 0.436 | 0.552 | 0.951 | **0.938** |
| \(E\) |  | \(-11.091\) | \(-11.136\) | \(-11.266\) | \(-11.549\) | **\(-11.668\)** |

\(L=5\) is lower in energy and lower in \(F\) than \(L=4\).

Separate \(L=1\) fidelity maximization, same Hamiltonian, disagrees with the energy optimum on the ferro crossover (median gap over all \(N\le 12\) points is only \(0.005\); the tail is the crossover):

| point | \(F\) energy-opt | \(E\) energy-opt | \(F\) fidelity-max | \(E\) fidelity-max |
|---|---:|---:|---:|---:|
| ferro \(N=12\), \(\Gamma=1.1\) | 0.207 | \(-13.259\) | 0.904 | \(-12.132\) |
| ferro \(N=10\), \(\Gamma=1.1\) | 0.341 | \(-11.091\) | 0.845 | \(-10.054\) |
| ferro \(N=20\), \(\Gamma=1.1\) | 0.016 | \(-22.016\) | 0.766 | \(-16.841\) |

A high-overlap state exists at \(L=1\) and is not the energy minimum. Reporting only energy-selected \(F\) would hide that; treating the fidelity maximizer as the QAOA result would cherry-pick a worse-energy branch.

Shared-\(\gamma\) is not the same curve. Ferro \(N=8\), \(\Gamma=1.1\): independent \(F(L=3)=0.983\), shared-\(\gamma\) \(F(L=3)=0.53\), then shared \(F(L=5)=0.96\). It can flatten or reorder the depth trend. It is not used in the tables above.

## Anneal baseline

Best-of-grid linear Trotter (one parameter \(T\), same \(L\)) stays far from the energy-QAOA fidelity on the ferro crossover. Ferro \(N=12\), \(\Gamma=1.1\): anneal \(F\) is \(0.19, 0.21, 0.22, 0.24, 0.25\) across \(L=1..5\), while energy-QAOA goes \(0.21\to 0.98\). Canonical-duration matching does not close that gap (ferro \(N=12\), \(L=5\): anneal \(F=0.044\) vs QAOA \(0.98\)). AF is easier for both; the anneal still trails (AF \(N=12\), \(\Gamma=0.05\), \(L=5\): anneal best-grid \(F=0.64\), QAOA \(F=1\)).

This is a fixed linear schedule, not an optimized QAOA. It is not a runtime comparison and not a speedup.

## Why this family does not reproduce the target plot

The target shape is \(F\) growing with \(L=0..5\), falling with \(N\), absolute values often below \(0.4\), and a genuinely small \(L=1\). Two structural facts block it here.

1. Full-space ED on this grid puts the GS in the even Dicke block. The circuit is confined to a space of dimension \(N+1\). Getting \(F\to 1\) by \(L\sim 5\) at \(N\le 12\), or by \(L=5\) at \(N=16\) already at \(F=0.63\), is control in a small symmetric subspace. It is not evidence about \(2^N\)-hard QAOA.
2. The extensive normalization that makes the ferro polarized cost \(\pm N\) makes the AF well shallow, so the AF GS at \(\Gamma\ge 0.05\) is either easy from \(|+\rangle\) or already paramagnetic. The ferro GS in the only window that is not a pure cat and not a pure paramagnet is still a melted cat (\(F_{\mathrm{cat}}\sim 0.6\), \(m_2\gtrsim 0.8\)).

## Next direction

Stay off a random glass unless that family is explicitly reopened. The check that would change the conclusion is a **geometric** distinct-index 4-body term (plaquettes or a sparse pattern), same \(s=\pm 1\), same transverse field, so permutation symmetry is gone and the GS has to be shown again in the full space. Reusing this all-to-all \(a_N\) there is fine; inventing a second AF-only normalization would be a different experiment and should be labeled as such.

If the all-to-all ferro crossover is revisited, the useful measurement is the fidelity-selected branch versus the energy-selected branch at \(L=1\), on a finer \(\gamma\) grid than \(T_\gamma/2400\) once \(T_\gamma\gtrsim 10^2\) (\(N\ge 16\)). Do not read \(\eta\approx 1\) as overlap.

## Files

- `spectrum.csv` — full-H and even/odd edges, gap, \(F_0\), cat, entropy, moments, classical manifold.
- `qaoa.csv` — methods `init`, `qaoa_energy`, `shared_gamma`, `qaoa_fidelity_L1`, `anneal_bestT`, `anneal_matched`.
- `angles.json` — raw and canonical angles, winner tag, \(L=1\) grid minima.
- `pure_vs_collective_gs.json`, `lower_body_delta.json`, `anneal_canonical_match.json`, `gamma_selection.json`, `run_meta.json`.
- `figures/` — \(F\) vs \(1/N\), \(F\) vs \(L/N\), all-\(\Gamma\) multiples, regimes, baselines, angle branches, pure vs control.
- `../validation/` — identity, term lists, periods, Dicke vs full statevector.
