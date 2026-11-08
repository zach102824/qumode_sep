# Spin glass round 2: diagnosis and outcome

Protocol frozen at commit `6fbab02e` (`PREREG.json` sha256 `41f48303...`) before any fit. Ideas came from one
Opus 5.5 medium consultation (`OPUS_IDEAS.md`). Tables: `RESULTS.md`. Every start, with its initial and final
angles: `logs/`. Per-cell selected states: `states/`. Post-hoc polish check: `polish_check.json`.

## Outcome

Under the frozen claim rule, neither KM nor K shows a reliable equal-layer advantage over QAOA. The rule needs, at
every N and at both L=5 and L=6, F_sub at least 0.05 above QAOA in 3 of 4 instances, with both cells converged
and energy not worse. KM passes at N=6 L=5 (4/4) and at N=9 L=5 (3/4), and fails the other four (N, L) checks.

Descriptively, after the fix KM has higher F_sub than QAOA in 11 of 12 instances at L=6 and higher mean F_sub at every
(N, L). Mean F_sub at L=6 is 0.945 vs 0.736 at N=6, 0.639 vs 0.496 at N=9, and 0.350 vs 0.286 at N=12. The rule
fails for three reasons:

- at N=6, QAOA is close to 1 on k0 and k3 (0.970, 0.950), so the 0.05 margin cannot be met;
- at N=9, KM's energy is worse than QAOA's on k2 (L=4 to 6) and k3 (L=6), even where its F_sub is higher;
- at N=12, k2 is a clear QAOA win in both energy and F_sub at L>=3 (L=6: 0.144 vs 0.073), and k1 is near zero
  for every family.

## Diagnosis of round 1

- **Underoptimization, not feature failure, explains the large round-1 KM losses.** The (-12, 12) product
  box let the deterministic gamma = +-0.3 starts jump to |gamma| about 12, which is a poor region. The two
  deterministic starts were also mirror copies, and KM had no nested seed. The same circuits under the linear
  coefficients and the 2pi/s_J box give these L=6 changes in KM F_sub:

  | cell | round 1 | round 2 |
  |---|---|---|
  | N6 k0 | 0.464 | 0.990 |
  | N9 k0 | 0.024 | 0.908 |
  | N12 k3 | 0.001 | 0.596 |

  KM energy is now never above K's in any cell, and best energy never rises with L.
- **The 80-evaluation cap mattered, but 2000 was never binding.** No round-2 start reached the cap. 4437 of 4440
  starts stopped on L-BFGS-B's own convergence tests, and 3 ended with an abnormal line-search return. Median
  evaluations per L=6 start were 41 to 48 for QAOA and 143 to 205 for KM. A tight re-polish of every selected L=6
  cell (ftol 0, gtol 1e-10) moved no energy by more than 1.2e-10.
- **QAOA was not under-optimized in round 1.** Its round-1 L=6 states are still its lowest energy. The full
  rescue re-found them from 5 to 9 independent starts per instance: interp, append-zero, ramp, grid and
  perturbations. QAOA had the same cap, beam, and grid entry points as the other families, and one extra start per depth.
- **Where QAOA still wins (N9 k2, N9 k3 at L=6, N12 k2), the loss is to QAOA's per-layer gamma.** The shared-gamma
  families reach a lower energy at small L but are overtaken as L grows. The KM best energy was reached
  independently by 12 to 13 of 16 starts in those cells. That suggests a limit of the shared-gamma family at fixed
  depth, not a missed basin, although a global optimum cannot be certified.
- **K and KM are nearly indistinguishable.** Their mean F_sub differs by at most 0.05 at any (N, L) (KM ahead at N=6
  and 9 for L>=5, K slightly ahead at N=12 L=5 and 6), and K beats KM in F_sub in some cells despite KM's lower energy. Four instances do not support claims about
  which feature helps.
- **Fidelity is not monotone in energy.** Lower energy does not always mean higher F_sub (for example N12 k3 L=6:
  KM has lower E than K but F_sub 0.596 vs 0.620). The selection stayed energy-only as preregistered.

## Parity and fidelity checks

The circuit conserves prod X and starts even. At N=9, the single-vector F is 0 because eigh column 0 is the odd
member of the doublet. F_sub and F_even agree there. At every N and k, the even-sector ground state lies inside
the 1e-6 window. N6 k2 has a nondegenerate ground state (n_sub = 1).

## Limits and untested items

- The KM-gamma_l ansatz (per-layer gamma with a shared theta shape) was not run. Nothing in this round uses a new
  ansatz.
- The equal-layer comparison is not parameter-matched: KM has 3+3L parameters, QAOA 2L.
- Sixteen starts per depth find a best-found local optimum, not a certified global one, for every family.
- The box change (|a| <= 2pi/s_J) is a domain restriction for all families. No selected round-1 state lay outside
  it as the winner: round-2 starts matched or beat every round-1 cell, so no preserved round-1 state was selected.
- Gaps were not recomputed. No claim links gap to fidelity on four instances.
- Wall time was 152 s of the 4500 s allowed. The bounded protocol, not compute, set the limit. No extra round
  was run after the fidelities were seen.
