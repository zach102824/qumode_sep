# Round 2 — SQS four-body signs, not a random glass

Isolated from round 1. The all-to-all pure p=4 model is not reused. Signs are deterministic patterns on one Steiner quadruple system, not draws from a spin-glass ensemble. Wall time of `run_round2.py`: 131 s. No covering-radius optimality is claimed except the exhaustive N=8 certificate. No scaling theorem from these sizes.

## Hamiltonian and the two field scales

\[
H_C = -\sum_{b\in\mathcal{B}} J_b Z_{b_1}Z_{b_2}Z_{b_3}Z_{b_4},
\qquad
H = H_C - \Gamma_X \sum_i X_i,
\qquad
\Gamma_X = r\,\Gamma,
\]

\[
r = (N-1)(N-2)/6
\]

is the number of blocks through a point. \(J_b=\pm 1\). \(\Gamma\) is the brief's per-degree field. \(\Gamma_X\) is the coefficient of each \(X\). Round-1 \(\Gamma\) was a \(\Gamma_X\), so \(\Gamma=0.1\) here is not round-1 \(\Gamma=0.1\).

| \(N\) | \(r\) | example \(\Gamma\) | \(\Gamma_X=r\Gamma\) |
|---:|---:|---:|---:|
| 8 | 7 | 0.05 | 0.35 |
| 10 | 12 | 0.10 | 1.20 |
| 14 | 26 | — | — |
| 16 | 35 | — | — |

Qubit 0 is MSB. A configuration \(y\in\mathbb{F}_2^N\) has \(Z_i=(-1)^{y_i}\). With \(J_b=(-1)^{\varepsilon_b}\),

\[
E(y) = -b + 2\,u(y), \qquad u(y)=\mathrm{wt}(By+\varepsilon).
\]

\(u_{\min}\) is the frustration. It was computed by enumerating all \(2^N\) configurations for every pattern kept below, including N=14 and N=16. That certifies the frustration of the chosen signs. It does not certify that a larger frustration is impossible, except at N=8.

Gauge: \(\varepsilon\) and \(\varepsilon+By_0\) are conjugated by an \(X\)-string. That fixes \(|+\rangle\) and \(\sum X\). Reports use the gauge in which \(y=0\) is a classical minimizer (`signs_gauge_fixed.json`).

## Designs, checked by triple count

Every 3-subset sits in exactly one block (`verify_sqs`). Hanani's existence theorem was not used as a proof of these lists.

| \(N\) | design | \(b\) | status |
|---:|---|---:|---|
| 8 | AG(3,2) 2-flats of \(\mathbb{F}_2^3\) | 14 | exact-cover search returned the same list. \(\lvert\mathrm{Aut}\rvert=1344=\lvert\mathrm{AGL}(3,2)\rvert\), exhaustive backtrack |
| 10 | smallest-uncovered-triple exact cover on points 0..9 | 30 | \(\lvert\mathrm{Aut}\rvert=1440\), exhaustive. Isomorphism uniqueness is **not** certified |
| 14 | same exact-cover rule on points 0..13 | 91 | reproducible. Which of the four SQS(14) types this is was **not** identified. Aut group not exhaustively computed |
| 16 | AG(4,2) 2-flats of \(\mathbb{F}_2^4\) | 140 | triple count passed. Aut group not exhaustively computed |

Block lists: `blocks_N{8,10,14,16}.txt`.

Incidence rank of \(B\) over \(\mathbb{F}_2\): N=8 rank 4 (code size 16, 1024 cosets); N=10 rank 9 (code size 512); N=14 rank 13; N=16 rank 11 (code size 2048). The all-ones vector is in the kernel because every block has length 4. So the classical ground set of an unfrustrated signing always contains a global flip pair, and on N=8 and N=16 it is a larger linear code (\(\lvert\ker\rvert=16\) and \(32\)).

## Signs

Candidate family, all deterministic: uniform \(J=+1\); point-star (minus on blocks through a point); single-block flip; pair-star (minus on blocks through a fixed pair); and, on the affine geometries, minus on flats that do not contain 0.

**Point-stars are not frustrated.** After gauge fixing, \(u_{\min}=0\) and every \(J_b=+1\). Flipping every block through one point is a codeword of \(\mathrm{Im}(B)\), i.e. an \(X\)-string relabeling of the uniform ferromagnet. Checked on N=8 and N=10.

**Smallest positive frustration in the family** is a single-block flip: \(u_{\min}=1\), classical degeneracy equal to \(\lvert\ker B\rvert\). N=8 stabilizer of the gauge-fixed signs is 96 (exhaustive). N=10 stabilizer is 48 (exhaustive). N=14 and N=16 stabilizers were not exhaustively computed; the frustration and degeneracy were.

**Pair-star** is the next structured step: N=8 \(u_{\min}=3\), stab 48, deg 16; N=10 \(u_{\min}=4\), stab 32, deg 2; N=14 \(u_{\min}=6\), deg 2; N=16 \(u_{\min}=7\), deg 32.

**Covering radius, N=8 only.** Every one of the \(2^{14}\) sign vectors was enumerated. \(R=6\), and exactly one coset attains it. That coset is the affine off-origin signing (stabilizer 192, degeneracy 112, mean pairwise spin agreement 0.493). This is a certificate, not a heuristic. It was **not** repeated at N=10 (\(2^{21}\) cosets) or beyond, and those patterns are not called optimal.

The QAOA target rule in code minimizes \(u_{\min}\) first (small frustration), then maximizes the exact stabilizer, then breaks ties lexicographically. That selects a single-block flip (`block_8` on N=8, `block_24` on N=10). The runs below also keep uniform, `block_0` (same coset type as every other single flip on N=8: same \(u\), degeneracy, and stabilizer), pair-star, and the N=8 covering-radius signing. `block_0` and `block_8` are the same physics up to the design automorphism; the deep curves use `block_0` so the sign file matches the name.

## What the classical landscape is

Small-frustration signings are **not** a glass ensemble.

- N=10 and N=14, uniform / one-block / pair-star: exactly two classical minima, and they are global flips (mean agreement 0). This is a ferromagnet with a few unhappy blocks. The quantum ground state at small \(\Gamma\) is the even cat of that pair (\(F_{\mathrm{manifold}}=F_{\mathrm{polarized\,cat}}\), half-cut entropy 1 bit).
- N=8 and N=16, same signings: the minima are the kernel (or a coset), size 16 or 32, mean agreement about 0.47–0.48. That is a linear code, with a large design automorphism group on N=8 (1344). The small-\(\Gamma\) ground state has \(F_{\mathrm{manifold}}\approx 1\) and \(F_{\mathrm{polarized\,cat}}\approx 0.12\), so it is the code superposition, not the two-point GHZ.
- Affine off-origin, N=8: 112 minima out of 256, mean agreement 0.493, but stabilizer 192. The overlap between minima looks disordered; the symmetry does not. Call it symmetric high frustration with a glass-like classical overlap, not an antiferromagnet and not a random ensemble. N=16, same formula, not certified maximal: \(u_{\min}=60\), degeneracy 896, agreement 0.499. Same qualitative warning. It was not promoted to QAOA: the N=8 version is an easy paramagnet at every scanned \(\Gamma\) (\(F_0\ge 0.79\), \(F_{1,\max}\ge 0.97\)).

## Symmetry of the quantum ground state

\(P=\prod_i X_i\) commutes with \(H\) for every sign pattern (four \(Z\)'s per block). QAOA from \(\lvert+\rangle\) stays in \(P=+1\). The brief's instruction to target the stabilizer-trivial irrep was not followed. The target is the full-space ground state.

On every N=8 and N=10 screen point, dense `eigh` puts that ground state in \(P=+1\) (even weight 1 on the returned vector). The lowest \(P=-1\) energy is strictly higher once \(\Gamma\) is large enough to split the doublet. At the smallest \(\Gamma\) the even/odd gap is unresolved (multiplicity 2, gap printed as 0), the same GHZ doublet as round-1 ferro. Example: N=10 uniform, \(\Gamma=0.05\), multiplicity 2; at \(\Gamma=0.10\), multiplicity 1, gap still \(<10^{-3}\) but the even vector is the one returned and \(F_{\mathrm{cat}}=0.975\). Residuals on the dense solves are \(\sim 10^{-14}\).

N=14 and N=16 were not diagonalized. There is no claim that their ground state sits in a particular irrep.

## Angle domain

\(H_C\) eigenvalues are integers of the form \(-b+2u\). The gcd of spectral gaps is computed per signing.

| pattern | \(T_\gamma\) |
|---|---|
| N=8, all four signings used | \(\pi\) (gcd 2) |
| N=10 uniform | \(\pi/4\) (gcd 8) |
| N=10 block and pair | \(\pi/2\) (gcd 4) |

\(\beta\) is searched on \([0,\pi)\), the projective period of \(\sum X\). The suggestion \(\beta\in[0,\pi/2)\) was not assumed. Raw schedules are what the bootstrap sees. Folding into one period plus a global sign flip is reporting-only.

N=10 uniform, \(\Gamma=0.10\), energy-optimal \(\gamma\) stays \(O(0.1)\) through L=4 (\(F\le 0.071\)). At L=5 the winner tag is `last_offset_polish` and \(F\) jumps to 0.904, with a visibly different raw schedule (leading \(|\gamma|\) from \(\sim 0.09\) to \(\sim 0.33\)). That jump is kept, not smoothed. Canonical and raw values are both in `deep_angles.json`.

## L=1 energy versus L=1 fidelity max

This is the round-1 failure mode (energy-opt \(F=0.207\) against fidelity-max \(F=0.904\)). Here the two L=1 numbers are written on every screen row.

N=8, ordered \(\Gamma\): they agree. Uniform \(\Gamma=0.05\): \(F_{1E}=0.573\), \(F_{1\max}=0.575\). Pair \(\Gamma=0.02\): \(0.321\) vs \(0.352\). Not a false-hard split. They are also not low for long: see the depth table.

N=10 uniform, \(\Gamma=0.10\), **is** the false-hard pattern on a cat: \(F_{1E}=0.038\), \(F_{1\max}=0.454\), \(F_{\mathrm{cat}}=0.975\), entropy 1 bit. Energy-opt L=1 looks hard because it missed the branch that fidelity maximization finds. The state being prepared is the GHZ cat.

N=10 block, \(\Gamma=0.10\): \(F_{1E}=0.035\), \(F_{1\max}=0.235\), \(F_{\mathrm{cat}}=0.970\). The split is smaller. The target is still a cat.

## Depth curves (energy objective, independent angles)

Pagano bootstrap: L=1 grid plus local search and the zero seed; L=2 offset \((\beta,\beta-0.2)\), \((\gamma,\gamma+0.2)\); L≥3 natural-cubic resample of the raw angles; perturbations, sign flip, random, and a zero-pad carry. Shared-\(\gamma\) is only the L=6 sidecar in `deep.csv`.

| point | \(\Gamma_X\) | L=0 | L=1 | L=2 | L=3 | L=4 | L=5 | L=6 | \(F_{1\max}\) | what the GS is |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N=8 uniform \(\Gamma=0.05\) | 0.35 | 0.095 | 0.573 | 1.000 | 1 | 1 | 1 | 1 | 0.575 | code manifold, \(F_{\mathrm{man}}=0.995\) |
| N=8 block \(\Gamma=0.05\) | 0.35 | 0.104 | 0.449 | 0.835 | 0.996 | 1 | 1 | 1 | 0.458 | same, 16 minima |
| N=8 pair \(\Gamma=0.02\) | 0.14 | 0.097 | 0.321 | 0.555 | 0.822 | 1 | 1 | 1 | 0.352 | same |
| N=8 pair \(\Gamma=0.05\) | 0.35 | 0.183 | 0.458 | 0.686 | 0.900 | 1 | 1 | 1 | 0.481 | same |
| N=10 uniform \(\Gamma=0.10\) | 1.20 | 0.006 | 0.038 | 0.038 | 0.071 | 0.071 | 0.904 | 0.982 | 0.454 | polarized cat, \(F_{\mathrm{cat}}=0.975\) |
| N=10 block \(\Gamma=0.10\) | 1.20 | 0.006 | 0.035 | 0.040 | 0.042 | 0.048 | 0.073 | 0.095 | 0.235 | cat, \(F_{\mathrm{cat}}=0.970\) |
| N=10 pair \(\Gamma=0.10\) | 1.20 | 0.011 | 0.034 | 0.046 | 0.055 | 0.065 | 0.065 | 0.065 | 0.058 | cat, \(F_{\mathrm{cat}}=0.927\) |

N=8 reaches overlap 1 by L=2 (uniform) or L=4 (pair). That is easy, in a 256-dimensional space whose ground state is a classical-code superposition. Half-cut entropy of that state is about 2 bits, product overlap equals \(F_0\) (0.07–0.18), so it is not a product state, and it is still easy for this circuit.

N=10 uniform's late jump lands on the cat (entropy 1 bit, \(F_{\mathrm{cat}}=0.975\)). Block and pair energy-opt curves stay below 0.10 through L=6 while the ground state remains that cat. Neither curve is a slow non-cat ramp with a low fidelity-max at L=1.

Above the crossover the screen is a paramagnet. N=10 pair, \(\Gamma=0.25\) (\(\Gamma_X=3.0\)): \(F_0=0.906\), \(F_{1E}=0.975\). N=8 affine off-origin is already there at \(\Gamma=0.02\) (\(\Gamma_X=0.14\)): \(F_0=0.790\), \(F_{1\max}=0.977\), product overlap 0.79, entropy 1.02 bits. The certified maximum-frustration signing is the easiest quantum problem in the batch, because 112 of 256 basis states are classical minima and \(\lvert+\rangle\) already covers them.

Shared-\(\gamma\) at L=6 matches the saturated N=8 energies and is not a separate success.

## Why N=14 and N=16 were not given a QAOA budget

Their small-frustration signings have the same classical type as N=10 (degeneracy 2, global flips) or N=8 (degeneracy \(\lvert\ker\rvert=32\) on AG(4,2)). The N=8 code case saturates by L≤4. The N=10 flip-pair case is a cat, and energy-opt either jumps onto that cat at L=5 or never leaves \(F<0.1\) through L=6. The N=8 covering-radius signing is an easy paramagnet, and the same formula on N=16 has 896 classical minima with agreement 0.50. Running Lanczos there would not test a new mechanism. This is a stop, not a finite-size scaling claim. No 0.25 fidelity gate was used as a theorem; the rejections are the cat overlap, the L=1 fidelity max, and the depth at which energy-opt \(F\) hits 1.

## Rejected

- Uniform control, N=8: easy (F=1 at L=2). Not a two-point cat; it is a 16-word code, and the circuit still finds it immediately.
- Uniform control, N=10: polarized cat. Energy-opt L=1 \(F=0.038\) against \(F_{1\max}=0.454\). Do not quote the energy-opt number alone.
- Single-block and pair-star, N=8: easy saturation by L=4, \(F_{1E}\approx F_{1\max}\).
- Single-block and pair-star, N=10: cat. Energy-opt \(F\) stays ≤0.095 through L=6.
- Affine off-origin, N=8: certified covering-radius coset, and an easy paramagnet at every \(\Gamma\) from 0.02 to 1.5.

Nothing in the batch is a low-\(F_{1\max}\) climb through L=3..6 on a non-cat ground state.

## Next

A different interaction than a constant-weight SQS signing is required before another depth scan. Two facts keep repeating: unfrustrated or lightly frustrated signings have classical minima equal to a coset of \(\ker B\), hence a cat or a code state; pushing frustration to the N=8 covering radius fills so much of the cube that \(\lvert+\rangle\) is already the ground state. A geometric 4-body term that is not a 2-design would change the kernel and is a different family, and it should be checked in the full space before any irrep is chosen.
