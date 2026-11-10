# Bounded spectral-gap diagnostic: N=6 / N=12 quantum SK, entropy-peak B\* vs g=1.5J

Preregistration: `Nov3_INIT/PREREG_GAPDIAG.md` (committed before any gap was computed). Code: `Nov3_INIT/gapdiag.py`. Raw data: `instances.csv` (one row per instance×field), `levels.csv` (every computed level with sector and coupling), `summary.json`, `PANEL_FROZEN.json` (seeds + B\* frozen before spectra).

One bounded diagnostic. No SPSA, no ansatz, no optimization, 0 quantum energy queries (historical SK ledger unchanged, 3252512). All other experiments stayed stopped. Nothing continues after this batch.

## Definitions

- \(H(g)=\sum_{i<j}J_{ij}Z_iZ_j-g\sum_iX_i\), \(J_{ij}\sim\mathcal N(0,J^2/N)\) with **J = 1** (`make_J`: `normal(0, 1/sqrt(N))`, i.e. std \(1/\sqrt N\)). g is dimensionless (g/J). Empirical \(\sqrt N\cdot\mathrm{rms}(J_{ij})\) per instance is in `PANEL_FROZEN.json`.
- **Entropy-peak B\***: the field maximizing the half-cut ground-state von Neumann entropy. The cut is qubits `0..N/2−1`. The grid is `linspace(0.05,3,60)`, followed by a bounded refine with xatol 1e-6. This is the same recipe as `bstar.maximize`, but without its gap contract, so selection never looked at a gap. All 16 instances had an interior peak; none were excluded or replaced. B\* is **not** claimed to be critical or hardest.
- **g = 1.5J = 1.5**, the same for all instances.
- Seeds: N=6 `10904–10911`, N=12 `10912–10919`. These are fresh. The collision check covered the worktree, 132 git ref trees, and agent stores, and found only "10904+ unused" reservation notes. Each instance is paired across the two fields.
- \(P=\prod_iX_i\). \(\Delta_+=E_1^+-E_0^+\) is the gap reachable from \(|+\rangle^{\otimes N}\) (P=+). \(\Delta_-=E_1^--E_0^-\). Cross split \(=E_0^--E_0^+\). Full gap \(=E_1-E_0\) over the whole space.
- Drive: transverse ramp, \(V=\partial_gH=-\sum_iX_i\). Couplings are \(c_n=\|\Pi_n V|0\rangle\|\) (cluster-projected; no degenerate clusters occurred at tol 1e-9), computed in full computational space. The **coupled gap** \(\Delta_c(\tau)\) is \(E_n-E_0\) for the lowest same-sector \(n\ge1\) with \(c_n>\tau\). The primary threshold is τ=1e-8, with τ ∈ {1e-10, 1e-8, 1e-6, 1e-4} as a sensitivity check.

## Numerics and audits

| check | worst value |
|---|---|
| \(\|HP-PH\|_F\) (dense N=6, sparse N=12) | **0** exactly |
| ground-state \(\langle P\rangle\) | ≥ 0.9999999999999964 (all 32 cells P=+) |
| ground-state min amplitude (Perron–Frobenius positivity) | 4.1e-6 > 0 |
| eigenpair residual \(\|Hv-Ev\|\), full space | N=6 1.5e-14; N=12 1.5e-11 |
| N=6 dense full vs dense parity-block eigenvalues | 1.3e-14 |
| N=12 eigsh parity-block vs dense block eigh (2048) | 2.8e-13 |
| N=12 full-space eigsh k=12 (dim 4096) vs dense block union | 2.9e-13 |
| N=12 lowest-8 kept vs dense union | 2.3e-13 |
| embedded-vector parity error \(\|Pv\mp v\|_\infty\) | 0 |
| opposite-sector couplings \(\|\langle n_{-}\|V\|0_+\rangle\|\) | N=12 ≤ 1.6e-15; N=6 ≤ 4.0e-11 (worst case s10911 at B\*, where the cross split is 0.004 and dense-eigh round-off is amplified by the small split) |
| smallest same-sector coupling among all computed P=+ excited levels | 3.3e-5 (a high level, \(E-E_0\approx18.5\)). No same-sector zeros ⇒ no sign of an accidental extra symmetry among computed levels |
| τ-sensitivity of the coupled gap | 0/32 cells change across τ=1e-10…1e-4 |
| N=12 eigsh k | k=8 per sector sufficed in all 16 cells (ladder 8→16→32→64 not needed). N=6 dense: all 32 per sector |

An intermediate audit (full eigsh k=12 vs the union of the k=8 sector lists) first showed 0.547 at N=12 g=1.5. The cause was truncation, not a numerical error: there, more than 8 of the lowest 12 levels are P-odd. The audit now compares against the dense block spectra, and it was rerun on the same frozen panel.

Classical cost: B\* preprocessing for 16 instances took 5.6 s wall; spectra and audits took 19.2 s; peak RSS was 145 MiB.

## Per (N, field): mean [range]

| N, field | g | full gap \(E_1-E_0\) | \(\Delta_+\) | \(\Delta_-\) | cross split | coupled gap (τ=1e-8) | \(c_1=\|\langle1_+\|V\|0\rangle\|\) | \(c_1/\Delta_+^2\) |
|---|---|---|---|---|---|---|---|---|
| N=6 B\* | 0.307 [0.158, 0.429] | 0.0254 [0.0040, 0.0406] | 0.647 [0.0849, 1.177] | 0.599 [0.0768, 1.129] | = full gap | 0.647 [0.0849, 1.177] | 0.964 [0.356, 1.566] | 8.3 [1.13, 49.8] |
| N=6 g=1.5 | 1.5 | 1.799 [1.614, 2.098] | 4.296 [3.990, 4.757] | 0.709 [0.502, 1.184] | = full gap | 4.296 [3.990, 4.757] | 0.355 [0.040, 0.713] | 0.019 [0.0024, 0.0375] |
| N=12 B\* | 0.534 [0.223, 0.747] | 0.0563 [3.57e-5, 0.102] | 0.548 [0.103, 1.040] | 0.409 [0.0868, 0.857] | = full gap | 0.548 [0.103, 1.040] | 0.751 [0.336, 1.853] | 7.96 [0.474, 31.4] |
| N=12 g=1.5 | 1.5 | 1.472 [1.297, 1.639] | 3.478 [3.201, 3.691] | 0.454 [0.119, 0.709] | = full gap | 3.478 [3.201, 3.691] | 0.463 [0.0125, 0.851] | 0.040 [0.0009, 0.075] |

Counts (each 8/8 in every (N, field) cell): ground state in P=+, \(E_1\) is P-odd, coupled state found, coupled state = first P=+ excited state.

## What this does and does not show

1. **Full gap ≠ relevant gap.** In all 32 cells, \(E_1\) is P-odd, so the full gap is the cross-sector splitting \(E_0^--E_0^+\). Under an ideal P-preserving transverse ramp started from \(|+\rangle^{\otimes N}\), that level is **forbidden**: the computed \(\langle n_-|V|0_+\rangle\) are at round-off. At the entropy peak the full gap can be very small (N=12 s10913: 3.6e-5; N=6 s10911: 4.0e-3), while the symmetry-allowed \(\Delta_+\) stays 0.17 and 0.085. So the full gap is 1–4 orders of magnitude smaller than the relevant gap there.
2. **Coupled gap.** At both fields, in every instance, the first P=+ excited state already has a clearly nonzero coupling (min 0.0125). The coupled gap therefore equals \(\Delta_+\). This holds at all thresholds τ=1e-10…1e-4. Same parity was necessary; nonzero \(c_1\) is what makes it sufficient here. That's a measured result for these 32 cells, not a general guarantee.
3. **Entropy peak vs g=1.5.** At B\* the relevant \(\Delta_+\) is small (N=12 mean 0.55, min 0.10) and the local adiabatic ratio \(c_1/\Delta_+^2\) is large (mean ≈ 8, up to 31–50). At g=1.5 (deep paramagnet), \(\Delta_+\approx3.5\)–4.3 and the ratio is ≈ 0.02–0.04. That is a comparison of **two points only**. It is **not** a minimum gap along the path, and it does not establish that B\* is where the minimum sits or that B\* is the hardest point. A g-scan would be required, and it was not run.
4. **Adiabatic hardness** depends on the minimum of the *coupled* gap along the whole ramp, the matrix elements along that path, and the schedule. It is not given by any single sector gap at one field. The minimum sector gap alone is not reported as hardness.
5. **Protocol dependence.** These conclusions assume a P-preserving transverse ramp. Perturbations that break P would couple the sectors and make the tiny cross split physically relevant. Examples are longitudinal fields \(\sum h_iZ_i\), odd catalysts, noise, or control errors that aren't global-X symmetric.

## Spin-symmetry answer

For generic random \(J_{ij}\), **total spin is not conserved**. \(\sum_{i<j}J_{ij}Z_iZ_j\) with unequal couplings doesn't commute with \(S^2\) or with collective \(S^x=\tfrac12\sum X_i\). The disorder also breaks site-permutation symmetry, so there's no Dicke or symmetric-subspace reduction. The only symmetry guaranteed for every instance and field is the global \(\mathbb Z_2\) spin flip \(P=\prod_iX_i\), verified numerically here as \([H,P]=0\) exactly. Any further accidental, disorder-specific symmetry (for example, special coupling patterns) was **not** assumed. No same-sector zero couplings were seen among the computed levels, so there's no evidence of one in this panel.

## Per-instance table

| N | seed | field | g | E0 | full gap (E1 parity) | Δ+ | Δ− | split | coupled gap | c₁ | c₁/Δ+² | max opp. c | max resid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 10904 | Bstar | 0.309069 | -3.558487 | 0.01324 (-) | 0.8999 | 0.8883 | 0.01324 | 0.8999 | 1.523 | 1.88 | 5.4e-13 | 6.2e-15 |
| 6 | 10904 | g1.5J | 1.500000 | -9.398421 | 1.858 (-) | 4.256 | 0.5022 | 1.858 | 4.256 | 0.4295 | 0.0237 | 1.2e-14 | 1.2e-14 |
| 6 | 10905 | Bstar | 0.429109 | -3.985136 | 0.03192 (-) | 1.177 | 1.129 | 0.03192 | 1.177 | 1.566 | 1.13 | 4.3e-14 | 5.2e-15 |
| 6 | 10905 | g1.5J | 1.500000 | -9.287572 | 1.737 (-) | 4.195 | 0.8637 | 1.737 | 4.195 | 0.66 | 0.0375 | 3.4e-15 | 1.5e-14 |
| 6 | 10906 | Bstar | 0.344523 | -3.499858 | 0.01616 (-) | 1.124 | 1.111 | 0.01616 | 1.124 | 1.461 | 1.16 | 1.4e-13 | 4.2e-15 |
| 6 | 10906 | g1.5J | 1.500000 | -9.236477 | 1.792 (-) | 4.43 | 1.184 | 1.792 | 4.43 | 0.713 | 0.0363 | 1.4e-15 | 1.2e-14 |
| 6 | 10907 | Bstar | 0.335995 | -4.045107 | 0.04062 (-) | 0.4164 | 0.3159 | 0.04062 | 0.4164 | 0.3898 | 2.25 | 1.5e-13 | 5.6e-15 |
| 6 | 10907 | g1.5J | 1.500000 | -9.667513 | 1.69 (-) | 4.076 | 0.5039 | 1.69 | 4.076 | 0.1367 | 0.00823 | 5.8e-15 | 1.5e-14 |
| 6 | 10908 | Bstar | 0.273442 | -3.813761 | 0.02708 (-) | 0.2939 | 0.231 | 0.02708 | 0.2939 | 0.3558 | 4.12 | 1.9e-12 | 6.2e-15 |
| 6 | 10908 | g1.5J | 1.500000 | -9.707673 | 1.614 (-) | 3.99 | 0.5694 | 1.614 | 3.99 | 0.115 | 0.00722 | 2.9e-14 | 1.3e-14 |
| 6 | 10909 | Bstar | 0.311252 | -3.205757 | 0.03447 (-) | 0.5324 | 0.4764 | 0.03447 | 0.5324 | 0.999 | 3.52 | 2.7e-14 | 5.4e-15 |
| 6 | 10909 | g1.5J | 1.500000 | -9.325009 | 1.905 (-) | 4.532 | 0.7066 | 1.905 | 4.532 | 0.334 | 0.0163 | 1.6e-15 | 1.2e-14 |
| 6 | 10910 | Bstar | 0.293460 | -2.799698 | 0.036 (-) | 0.645 | 0.5648 | 0.036 | 0.645 | 1.056 | 2.54 | 2.5e-14 | 5.6e-15 |
| 6 | 10910 | g1.5J | 1.500000 | -9.238583 | 2.098 (-) | 4.757 | 0.6003 | 2.098 | 4.757 | 0.4092 | 0.0181 | 3.4e-15 | 1.3e-14 |
| 6 | 10911 | Bstar | 0.157945 | -3.245799 | 0.004001 (-) | 0.0849 | 0.07679 | 0.004001 | 0.0849 | 0.3591 | 49.8 | 4.0e-11 | 5.3e-15 |
| 6 | 10911 | g1.5J | 1.500000 | -9.511373 | 1.7 (-) | 4.133 | 0.7405 | 1.7 | 4.133 | 0.04034 | 0.00236 | 5.0e-15 | 1.3e-14 |
| 12 | 10912 | Bstar | 0.558582 | -9.515803 | 0.1024 (-) | 0.4332 | 0.2216 | 0.1024 | 0.4332 | 1.02 | 5.43 | 4.4e-16 | 7.5e-12 |
| 12 | 10912 | g1.5J | 1.500000 | -19.162509 | 1.639 (-) | 3.519 | 0.1194 | 1.639 | 3.519 | 0.02342 | 0.00189 | 8.9e-16 | 1.5e-11 |
| 12 | 10913 | Bstar | 0.223268 | -7.539812 | 3.572e-05 (-) | 0.1695 | 0.1694 | 3.572e-05 | 0.1695 | 0.5332 | 18.6 | 4.4e-16 | 5.4e-12 |
| 12 | 10913 | g1.5J | 1.500000 | -18.885355 | 1.297 (-) | 3.374 | 0.7091 | 1.297 | 3.374 | 0.8507 | 0.0747 | 1.3e-15 | 6.9e-12 |
| 12 | 10914 | Bstar | 0.635680 | -9.713379 | 0.06986 (-) | 0.8666 | 0.6184 | 0.06986 | 0.8666 | 0.3561 | 0.474 | 8.9e-16 | 5.6e-12 |
| 12 | 10914 | g1.5J | 1.500000 | -18.738169 | 1.482 (-) | 3.567 | 0.4516 | 1.482 | 3.567 | 0.6949 | 0.0546 | 6.7e-16 | 1.2e-11 |
| 12 | 10915 | Bstar | 0.617133 | -9.822365 | 0.05685 (-) | 0.7404 | 0.5592 | 0.05685 | 0.7404 | 0.36 | 0.657 | 8.9e-16 | 5.0e-12 |
| 12 | 10915 | g1.5J | 1.500000 | -18.885084 | 1.425 (-) | 3.529 | 0.4673 | 1.425 | 3.529 | 0.5815 | 0.0467 | 8.9e-16 | 8.1e-12 |
| 12 | 10916 | Bstar | 0.335991 | -7.445720 | 0.009371 (-) | 0.1035 | 0.08676 | 0.009371 | 0.1035 | 0.3361 | 31.4 | 1.3e-15 | 6.1e-12 |
| 12 | 10916 | g1.5J | 1.500000 | -18.931393 | 1.565 (-) | 3.691 | 0.4988 | 1.565 | 3.691 | 0.0125 | 0.000917 | 4.4e-16 | 8.1e-12 |
| 12 | 10917 | Bstar | 0.555052 | -9.422206 | 0.0665 (-) | 0.3546 | 0.2042 | 0.0665 | 0.3546 | 0.3689 | 2.93 | 8.9e-16 | 5.2e-12 |
| 12 | 10917 | g1.5J | 1.500000 | -19.019035 | 1.485 (-) | 3.352 | 0.2921 | 1.485 | 3.352 | 0.2645 | 0.0235 | 4.4e-16 | 1.4e-11 |
| 12 | 10918 | Bstar | 0.600329 | -9.503781 | 0.07042 (-) | 0.6799 | 0.5571 | 0.07042 | 0.6799 | 1.18 | 2.55 | 8.9e-16 | 3.5e-12 |
| 12 | 10918 | g1.5J | 1.500000 | -18.890754 | 1.536 (-) | 3.592 | 0.5434 | 1.536 | 3.592 | 0.5121 | 0.0397 | 1.6e-15 | 1.9e-12 |
| 12 | 10919 | Bstar | 0.746510 | -11.259255 | 0.07476 (-) | 1.04 | 0.8573 | 0.07476 | 1.04 | 1.853 | 1.71 | 8.9e-16 | 6.0e-12 |
| 12 | 10919 | g1.5J | 1.500000 | -18.965886 | 1.351 (-) | 3.201 | 0.5472 | 1.351 | 3.201 | 0.7615 | 0.0743 | 8.9e-16 | 8.7e-12 |

## Lowest 8 levels (E, parity)

- N=6 s=10904 Bstar: -3.558487+, -3.545249-, -2.658548+, -2.656918-, -2.411696-, -2.336891+, -2.233039+, -2.225719-
- N=6 s=10904 g1.5J: -9.398421+, -7.540064-, -7.037872-, -6.375246-, -6.258323-, -5.436429-, -5.142112+, -5.013925-
- N=6 s=10905 Bstar: -3.985136+, -3.953218-, -2.824650-, -2.808550+, -2.280817+, -2.233958-, -2.079944-, -1.906998+
- N=6 s=10905 g1.5J: -9.287572+, -7.550553-, -6.686887-, -6.072360-, -5.740768-, -5.663762-, -5.336214-, -5.092524+
- N=6 s=10906 Bstar: -3.499858+, -3.483701-, -2.376331+, -2.373194-, -2.276775-, -2.247709+, -1.996075-, -1.976187+
- N=6 s=10906 g1.5J: -9.236477+, -7.444565-, -6.260217-, -6.234144-, -6.001266-, -5.646924-, -5.278235-, -4.806927+
- N=6 s=10907 Bstar: -4.045107+, -4.004490-, -3.688593-, -3.628722+, -3.118366+, -3.055970-, -3.021185-, -2.920621+
- N=6 s=10907 g1.5J: -9.667513+, -7.977186-, -7.473268-, -7.003951-, -6.158939-, -5.591152+, -5.356936+, -5.189127-
- N=6 s=10908 Bstar: -3.813761+, -3.786683-, -3.555693-, -3.519825+, -3.212442-, -3.205016+, -2.721985+, -2.673972-
- N=6 s=10908 g1.5J: -9.707673+, -8.094090-, -7.524733-, -6.753078-, -6.245217-, -5.838287-, -5.718069+, -5.182411+
- N=6 s=10909 Bstar: -3.205757+, -3.171284-, -2.694888-, -2.673407+, -2.479265-, -2.400740+, -2.039572+, -1.972418-
- N=6 s=10909 g1.5J: -9.325009+, -7.419737-, -6.713146-, -6.668891-, -5.934069-, -5.424936-, -5.171991-, -4.793404+
- N=6 s=10910 Bstar: -2.799698+, -2.763695-, -2.198928-, -2.154683+, -1.898700+, -1.866005-, -1.796053-, -1.706099+
- N=6 s=10910 g1.5J: -9.238583+, -7.140757-, -6.540411-, -6.301798-, -6.211042-, -5.911457-, -4.911966-, -4.481355+
- N=6 s=10911 Bstar: -3.245799+, -3.241798-, -3.165011-, -3.160897+, -2.858237-, -2.857612+, -2.421109-, -2.420967+
- N=6 s=10911 g1.5J: -9.511373+, -7.811761-, -7.071290-, -6.727824-, -6.168283-, -5.452037-, -5.378210+, -4.968933+
- N=12 s=10912 Bstar: -9.515803+, -9.413426-, -9.191819-, -9.082612+, -8.990401-, -8.833080+, -8.761688+, -8.614335-
- N=12 s=10912 g1.5J: -19.162509+, -17.523844-, -17.404424-, -17.156618-, -16.604157-, -16.562145-, -16.030892-, -15.812409-
- N=12 s=10913 Bstar: -7.539812+, -7.539777-, -7.370356-, -7.370289+, -7.040333-, -7.040313+, -6.946483+, -6.946314-
- N=12 s=10913 g1.5J: -18.885355+, -17.588572-, -16.879492-, -16.477240-, -16.152523-, -15.853917-, -15.705925-, -15.570553-
- N=12 s=10914 Bstar: -9.713379+, -9.643517-, -9.025078-, -8.846816+, -8.671556+, -8.565853-, -8.383578+, -8.370554-
- N=12 s=10914 g1.5J: -18.738169+, -17.256519-, -16.804932-, -16.285019-, -15.972708-, -15.859901-, -15.506050-, -15.289105-
- N=12 s=10915 Bstar: -9.822365+, -9.765512-, -9.206307-, -9.081991+, -8.780860+, -8.747055-, -8.544650+, -8.537344-
- N=12 s=10915 g1.5J: -18.885084+, -17.460386-, -16.993109-, -16.433724-, -16.202516-, -16.015498-, -15.694175-, -15.547335-
- N=12 s=10916 Bstar: -7.445720+, -7.436349-, -7.349587-, -7.342240+, -7.182538-, -7.177085+, -6.923076-, -6.907544+
- N=12 s=10916 g1.5J: -18.931393+, -17.366657-, -16.867888-, -16.710342-, -16.644872-, -16.047931-, -15.889411-, -15.458868-
- N=12 s=10917 Bstar: -9.422206+, -9.355703-, -9.151519-, -9.067600+, -8.588273-, -8.509287+, -8.435626+, -8.337862-
- N=12 s=10917 g1.5J: -19.019035+, -17.533859-, -17.241785-, -16.751437-, -16.362793-, -16.066882-, -16.014036-, -15.666831+
- N=12 s=10918 Bstar: -9.503781+, -9.433363-, -8.876266-, -8.823927+, -8.625198-, -8.452392+, -8.330536+, -8.319900-
- N=12 s=10918 g1.5J: -18.890754+, -17.354571-, -16.811199-, -16.727567-, -16.416796-, -16.089450-, -15.938587-, -15.402287-
- N=12 s=10919 Bstar: -11.259255+, -11.184495-, -10.327190-, -10.219215+, -9.908496+, -9.833154-, -9.609109-, -9.451099+
- N=12 s=10919 g1.5J: -18.965886+, -17.614440-, -17.067239-, -16.728143-, -16.074320-, -16.000165-, -15.844256-, -15.764518+
