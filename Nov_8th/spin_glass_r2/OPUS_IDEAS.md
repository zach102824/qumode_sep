# Round 2 ideas from the single Opus 5.5 medium consultation

One consultation was made (model `claude-opus-5-5-medium`, read-only reviewer, agent id
`bc-57ec9c9e-6338-51f6-a110-4ec930707a9c`). No further consultations are allowed in this round.
The review below is a faithful condensation of what it returned. The notes marked "checked" were
verified against `Nov_8th/spin_glass_r1/run_sg.py` and `starts.jsonl` before the protocol was frozen.

## 1. Critical review of round 1

Round 1 does not support a claim in either direction. KM versus QAOA in round 1 compares optimizers,
not ansatzes.

- Cap confound. KM hit the 80-evaluation cap in 36 to 47 of 48 starts at L >= 3; QAOA hit it in at
  most 7 of 48. KM ended worse in energy than K in many cells, although K is KM at theta_M = 0. A
  nested family that ends worse than its subfamily is an optimizer failure, not a lack of expressivity.
- Depth monotonicity failed: best energy rose from L-1 to L in 5 KM cells and 2 K cells, never for QAOA.
- Bounds and scaling (main cause). The scalar box is (-12, 12) for gamma, theta_K, theta_M. Good L1
  optima sit at |gamma| about 0.13 to 0.24. The gamma gradient is O(N^2), so the first projected
  L-BFGS-B step from gamma = +-0.3 lands near the bound, in a rugged large-gamma region with
  E about -0.4 to -4.6. Checked: the deterministic starts l1_0 and l1_1 end at |gamma| 11.2 to 12.0
  in every N=6 and N=12 K and KM cell listed; every KM L1 start at N12 k3 ended in E in [-0.79, -0.37].
- Starts 0 and 1 are mirror duplicates. H and |+> are real and every generator is real-diagonal in
  either the Z or the X basis, so E(x) = E(-x). Checked: l1_0 and l1_1 give identical E and
  evaluation counts in every K and KM cell. Effectively five L1 starts, two of them wasted.
- No nested seeding and no common random numbers. The random stream is keyed on the family, so KM's
  random starts share nothing with K's.
- Shared gamma cannot embed depth L-1 in depth L: an extra layer always applies exp(-i gamma G).
  Appending gamma = beta = 0 is an exact identity for QAOA. KM's "copy" doubles the total phase.
  Continuation must compensate (time rescaling).
- Ill-conditioned product parametrization: gamma*theta_K and gamma*theta_M enter as products.
- Continuation too narrow: perturbations only around the first continuation seed, sigma 0.05, one
  parent. The QAOA "fourier1" sine envelope is not Zhou FOURIER and never won. QAOA was also
  under-explored in starts (not in evaluations).
- Parity. Every gate commutes with prod_i X_i and |+> is even, so the circuit lives in the even
  sector. The ground state of +0.1 sum X has X-parity (-1)^N, so at N=9 the single-vector ground
  state is odd and unreachable. F_sub at 1e-6 works because the doublet splitting is below 1e-6
  everywhere except N6 k2 (4.4e-6, even ground state). The even-sector ground state is the
  natural secondary target.
- Round-1 N=12 gap diagnostics are not reliable (mixed degenerate vectors, NaN coupled gaps).
  Not used in round 2.
- Comparison is equal-layer, not equal-parameter (3+3L vs 2L). Parameter counts must be printed.

## 2. Ranked remedies

Optimizer fixes on the existing families (A to F); the only new ansatz is G.

| rank | remedy | rationale | cost | fairness risk |
|---|---|---|---|---|
| A | Rescale and reparametrize, identically for all families: u = gamma * s_J with s_J = sqrt(mean_i sum_j J_ij^2) (classical J only); KM in linear coefficients (a, b, c) = (gamma, gamma theta_K, gamma theta_M), so K is c = 0 and nodata is b = c = 0 exactly; box abs(u) <= 2 pi for every gamma including each QAOA gamma_l | removes the first-step jump to the bound | none | low if applied to QAOA too; it is a domain change and must be labelled |
| B | Nested energy-only seeding: KM gets K's best (c = 0), K gets nodata's best (b = 0), at every L | guarantees E_KM <= E_K <= E_nodata at the start | a few starts | none; within-family nesting only, never across to or from QAOA |
| C | Per-start cap 2000 evaluations, gtol 1e-6, same for all families | removes the cap confound | dominant cost | none |
| D | Broader L1 with common random numbers: sign-canonical u >= 0, a cheap energy-only grid (16 u values times beta values) shared by all families, top grid points as starts, shared random draws | removes duplicates, gives every family the same entry points | small | none |
| E | Continuation that respects shared gamma: beam of the top 3 distinct basins from L-1, interp plus time-rescaled gamma (L-1)/L plus perturbations sigma 0.1 and 0.3 | the single-parent rule stalled | about 12 starts per depth | none if QAOA gets the same beam |
| F | Equal QAOA rescue: Zhou INTERP, append-zero-layer (exact identity), linear ramp, same beam, same cap, at least as many starts | QAOA was also under-explored | same | this is the fairness guard |
| G | New ansatz "KM-gamma_l": shared (theta_K, theta_M) shape but gamma_l per layer, 3+4L parameters, contains QAOA | exact depth nesting | extra family | high to interpret; separate table only |
| H | Reporting: F_even (even-sector ground state), parity-sector gaps | correctness of the target | none | none |

## 3. Recommended bounded round-2 protocol (as returned)

- maxfun 2000, gtol 1e-6, ftol 1e-12, parametrization and box from A for every family.
- L1: 4 grid starts, 8 common-random-number starts, round-1 best re-optimized, nested seeds for K and KM;
  QAOA at least as many starts.
- L >= 2: beam 3 parents x 4 continuation starts, 2 fresh ramp starts, round-1 best at that cell, nested seed.
- Selection: lowest energy over all round-1 and round-2 starts in the cell. Never fidelity.
- Convergence flag per cell: best energy reached by at least 2 starts within 1e-6 and fewer than half the starts capped.
- Claim rule (frozen before the fit): "reliable advantage" only if KM F >= QAOA F + 0.05 in at least 3 of 4
  instances at each N, at both L = 5 and L = 6, both cells converged and KM energy not worse.
- Hard wall about 70 minutes; log truncations; no third round of tuning after seeing round-2 fidelities.
- G runs only if more than 15 minutes remain, in its own table.

## What this round adopts

A, B, C, D, E, F and H are adopted, with the details frozen in `PREREG.json`. K and nodata are run at
the same budget as KM and QAOA (not half), because nodata seeds K and K seeds KM, and K is a retained
shared-gamma family worth its own row. G is not run in this round (stated as untested): the budget is
spent on the retained families, and the stop rule forbids extending the round. Fidelity is the
user-mandated F_sub at tolerance 1e-6 (primary); F_even and single-vector F are secondary.
