# Gupta et al. Fig. 2 (L=12, Γ=1): bounded exact-diagonalization reproduction

Paper: Gupta et al., "Ground state preparation of random all-to-all Hamiltonians using ADAPT-VQE", arXiv:2606.18339v1. Links: [abs](https://arxiv.org/abs/2606.18339), [pdf](https://arxiv.org/pdf/2606.18339v1). The analysis uses Sec. "Quantum Sherrington-Kirkpatrick model" (TeX lines 138–142) and Fig. 2 (`sk.pdf`).
Preregistration: `Nov3_INIT/PREREG_GUPTA_FIG2.md`, committed before any computation. Code: `Nov3_INIT/gupta_fig2.py`.
Raw data: `instances.csv` (30 rows), `summary.json`, `fig2_L12_digitized.json` (Fig. 2 vector extraction), `posthoc_variance_probe.json`.

Scope: Fig. 2 exact diagonalization only. ADAPT was not implemented, no field scan was run, SYK was not touched, and no SPSA was used. Quantum energy queries: 0. Classical cost: 142 s wall, peak RSS 1.0 GB. The run stopped after this batch.

## Verdict (preregistered rule, primary = paper-literal convention)

| quantity | ours (n=30) mean ± SEM, t-95% CI | Fig. 2 L=12 (30 extracted points) | ours − Fig. 2, Welch 95% CI | KS p | verdict |
|---|---|---|---|---|---|
| \(E_0/L\) | −1.11991 ± 0.00398 [−1.1281, −1.1118]; SD 0.0218; range [−1.1649, −1.0851] | −1.05729 ± 0.00216; SD 0.0118; range [−1.0859, −1.0357] | −0.0626 [−0.0717, −0.0535] | 1e-15 | **TENSION** |
| full gap \(E_1-E_0\) | 0.5313 ± 0.0379 [0.454, 0.609]; range [0.153, 1.026] | 0.8886 ± 0.0292; range [0.490, 1.191] | −0.357 [−0.453, −0.261] | 2.5e-7 | **TENSION** |
| \(\Delta_+\) vs Fig. 2 gap | 1.7306 ± 0.0322 [1.665, 1.796] | 0.8886 ± 0.0292 | +0.842 [+0.755, +0.929] | 2e-17 | **TENSION** |
| gap definition | — | — | — | — | **not resolved** under the primary convention (neither the full gap nor \(\Delta_+\) agrees) |

Under the convention literally stated in the paper, our fresh 30-instance ED does **not** reproduce Fig. 2 at L=12. The energy per site is lower by about 0.063 (≈ 14 combined SEMs) and the full gap is smaller by about 0.36. The difference is far outside 30-sample noise. The authors' seeds, raw data and generator code are not available, so this is reported as **statistical tension under stated assumptions**, not as evidence that the figure is wrong. The most likely cause is a convention difference (see below).

## Primary assumptions (frozen before compute)

- \(H=-\sum_{i<j}J_{ij}Z_iZ_j-\Gamma\sum_iX_i\) with Pauli operators, a single sum over i<j, Γ=+1 and no longitudinal field.
- \(J_{ij}\sim\mathcal N(0,\mathcal J/N)\) with \(\mathcal J=1\) and N=L=12. In the code: `make_J(seed,12)` gives `normal(0, 1/sqrt(12))` on the upper triangle, and the paper's \(-J\) sign is applied literally.
- Seeds 10920–10949 are fresh. They were collision-checked against the worktree, 138 trees from 164 fetched refs, and the agent stores.
- The realized \(\mathrm{var}(J_{ij})\cdot L\) averaged over instances is close to 1; per-instance values are in `instances.csv` (`J_var_times_L`).

Secondary convention probes (preregistered, on the same instances), both further from the figure:
- (S1) Spin-1/2 operators: \(E_0/L=-0.514\) and full gap 0.608. Not compatible with the figure's energy.
- (S2) Double sum \(i\ne j\): \(E_0/L=-1.511\) and full gap 0.054. Not compatible.

## Numerics and independent checks (all 30 instances)

| check | worst |
|---|---|
| A: Kronecker-built dense 4096² with full `eigh`, residual \(\|Hv-Ev\|\) over the lowest 64 | 9.5e-14 |
| B: bitwise parity-block sparse `eigsh` (k=8 per sector), residual | 1.0e-11 |
| A vs B, for \(E_0\), full gap and \(\Delta_+\) | 9.9e-14 |
| B full-space `eigsh` k=8 vs block union | 1.6e-13 |
| \(\|[H,P]\|_F\), A dense and B sparse | 0 and 0 |
| second-order perturbation theory (an analytic cross-check, not exact at Γ=1): \(E/L\approx-\Gamma-\sum_{i<j}J_{ij}^2/(4\Gamma L)\) | gives −1.109 against ED −1.120 |

## Parity and coupling structure (primary convention; \(P=\prod_iX_i\); drive \(\sum_iX_i\))

- The ground state is P=+ in 30/30 instances, and the first excited state is P=− in 30/30. So the full gap \(E_1-E_0\) equals the cross-sector split \(E_0^--E_0^+\). A P-preserving transverse-field process cannot reach that level: the largest opposite-sector coupling \(|\langle n_-|\sum X|0\rangle|\) is 2e-11, i.e. round-off.
- Same-sector (even) gap \(\Delta_+=E_1^+-E_0^+\): 1.731 ± 0.032, range [1.467, 2.251].
- \(\Delta_-\): 0.640 ± 0.068.
- Coupling \(c_1=|\langle1_+|\sum X|0\rangle|\): 0.98 ± 0.11, with a minimum of 0.128. So the first even excited state is coupled (above τ=1e-8) in 30/30 instances, and the coupled gap equals \(\Delta_+\).
- If Fig. 2's Δ is the full spectrum \(E_1-E_0\), which is what the caption says, it measures the parity-odd splitting, not the gap relevant to a symmetric transverse ramp. Same parity is necessary but not sufficient for coupling. Here it was verified numerically, not assumed.

## Post-hoc exploratory probe (NOT preregistered; does not change the verdict)

A second-order perturbation estimate suggested the figure's offset per site, \(-0.057\), matches couplings with **half** the stated variance, \(\mathrm{var}(J_{ij})=1/(2L)\). The same 30 instances were therefore rescaled by \(1/\sqrt2\):

| quantity | var 1/(2L), ours | Fig. 2 | Welch 95% CI (ours − Fig. 2) | KS p |
|---|---|---|---|---|
| \(E_0/L\) | −1.05713 ± 0.00155 (SD 0.0085) | −1.05729 ± 0.00216 (SD 0.0118) | +0.0002 [−0.0052, +0.0055] | 0.96 |
| full gap \(E_1-E_0\) | 0.9046 ± 0.0309 [0.529, 1.272] | 0.8886 ± 0.0292 [0.490, 1.191] | +0.016 [−0.069, +0.101] | 0.96 |
| \(\Delta_+\) | 2.216 | 0.8886 | +1.33 [+1.24, +1.42] | 2e-17 |

With \(\mathrm{var}=1/(2L)\), both Fig. 2 means and the gap scatter are reproduced, and the figure's Δ matches the **full** gap \(E_1-E_0\) (parity-odd), not \(\Delta_+\). Our E/L SD (0.0085) is somewhat smaller than the figure's (0.0118). With n=30 per side, that is modest evidence only and was not pursued further.

Hypotheses that would produce an effective variance of 1/(2L); these are **unverified** and not an accusation:
- symmetrizing a full Gaussian matrix as \(J=(A+A^\top)/2\) with \(A_{ij}\sim\mathcal N(0,1/L)\);
- reading the undefined "N" in "variance \(\mathcal J/N\)" as \(2L\), which matches the paper's SYK convention where N counts Majoranas and \(n=N/2\) counts qubits;
- an equivalent factor elsewhere in the code.

This would also shift where Γ=1 sits relative to the quoted \(\Gamma_c/\mathcal J=1.5\), if that \(\Gamma_c\) assumes variance 1/L. Confirming any of this needs the authors' code or data.

## Bottom line

- **Primary (paper-literal) convention: DISAGREE / TENSION** for both \(E_{GS}/L\) and Δ at L=12. The gap definition is not resolved under the primary convention.
- **Post hoc:** an effective coupling variance of 1/(2L) reproduces Fig. 2 at L=12 within 30-sample uncertainty, with Δ as the full, parity-odd gap. This is a convention hypothesis only. The verdict stays as preregistered.
- At L=12 the relevant symmetric-sector gap under the literal convention is \(\Delta_+\approx1.73\), not the plotted ≈0.89.
