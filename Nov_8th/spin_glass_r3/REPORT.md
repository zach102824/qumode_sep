# Spin glass round 3: robustness of the KM equal-depth fidelity advantage

Protocol and code frozen at `15da8c83` (`PREREG.json` sha256 `c46c578f...`) before any fit, after the second and last
Opus 5.5 medium consultation (`OPUS_IDEAS.md`). Full tables: `RESULTS.md`. Every start: `logs/`. Per-repeat selections:
`jobs/`. Main-simulator evaluation: `eval/`. Independent replay: `replay/`. Verdict: `verdict.json`.

## Answer

The advantage generalizes at N=6 and is not established at N=9 or N=12. The frozen verdict is **partial**.

- **N=6 (22 fresh instances):** KM's F_sub beats equal-depth QAOA in 18 of 22 instances at L=6, all of them joint
  wins on both fidelity and energy, with 1 loss and 3 ties. Median paired dF is +0.150 (95% bootstrap CI
  [+0.042, +0.304]), mean +0.199; the sign test gives p = 3.8e-5. Every criterion passes: it holds at L=5, in all
  three per-repeat selections, against the 3x-start QAOA-heavy control, and under independent replay.
- **N=9 (22 instances), L=6:** 9 wins (6 joint, 3 fidelity-only), 4 losses, 9 ties. Median dF is +0.018
  [-0.038, +0.075] and sign p = 0.13. At L=5 the counts are 7 wins and 6 losses, with median dF -0.001.
- **N=12 (22 instances), L=6:** 7 wins (all joint), 3 losses, 11 ties, and 1 instance where both families
  fail. Median dF is -0.015 [-0.026, +0.088], although the mean is +0.043; sign p = 0.17. KM wins large on a few
  instances (up to +0.33) and loses or ties small on most.
- Mean F_sub is higher for KM at every (N, L), but at N=9 and 12 the margin is small:

  | N | KM mean / median (L=6) | QAOA mean / median (L=6) |
  |---|---|---|
  | 9 | 0.399 / 0.350 | 0.378 / 0.319 |
  | 12 | 0.334 / 0.251 | 0.290 / 0.275 |

  At N=12 the QAOA median is higher.

So the round-2 picture on k=0..3 (KM ahead in 11 of 12 instances) was optimistic for N=9 and N=12. Four hand-picked
instances overstated the advantage there. At N=6 it holds up on fresh instances.

## Losses and failures kept visible (L=6, best over repeats)

| N | k | KM F | QAOA F | KM E − QAOA E | outcome |
|---|---|---|---|---|---|
| 6 | 12 | 0.031 | 0.146 | −0.375 | fidelity-only loss |
| 9 | 7 | 0.000 | 0.279 | −1.06 | fidelity-only loss |
| 9 | 17 | 0.129 | 0.235 | +0.26 | joint loss |
| 9 | 18 | 0.180 | 0.317 | +0.12 | joint loss |
| 9 | 20 | 0.143 | 0.321 | −0.24 | fidelity-only loss |
| 12 | 17 | 0.526 | 0.591 | −0.74 | fidelity-only loss |
| 12 | 19 | 0.250 | 0.380 | +0.43 | joint loss |
| 12 | 24 | 0.120 | 0.216 | +1.05 | joint loss |
| 12 | 23 | 0.001 | 0.000 | | tie, both failed |

In several of the losses (N6 k12, N9 k7, N9 k20, N12 k17), KM reaches a much lower energy than QAOA but a state with little ground-space weight. At
these depths, energy-only selection does not track fidelity for this ansatz. The secondary instances k=0..3 (including
N12 k2 and the low-F k1 from round 2) were not reached before the wall; their round-2 values are unchanged and not re-tested.

## Optimizer reliability (separate from the physics)

- **N=9 and N=12 repeats always agree.** In all 22 instances, at every L and for every family, the three repeats
  reached the same best energy within 1e-6, so the across-repeat F SD is 0. This is not a seeding bug: the winning
  starts differ between repeats (grid, random, nested, perturbation labels), and the energies agree to about
  1e-10 but not bitwise.
- **N=6 KM is the noisiest cell.** The repeats are within 1e-6 of the best in 53 to 56 of 66 cells, and 7 to 8 of
  22 instances have energy spread above 1e-4. The largest F_sub range is 0.835 at L=1 and 0.099 at L=6. Examples
  are N6 k8 at L=5 (0.715 / 0.860 / 0.615) and N6 k11 at L=6 (0.948 / 0.848 / 0.948). QAOA is steadier: 64 of 66
  cells at best at L=6.
- **More QAOA starts change almost nothing.** QAOA-heavy, with 27 L1 starts and doubled perturbations, found lower
  energy than standard QAOA in 5 of 132 N=6 cells (largest gain 0.088) and in none at N=9 or N=12. The
  standard-QAOA comparator is therefore not effort-limited here.
- No equal-energy pair of repeats differed in F by more than 0.01.
- Energy rose with depth within a single repeat in 5 KM cells, all at N=6. KM never ended above K.
- 375 of 1199 selected KM cells have a nested-seed ancestor.
- Projected gradient at the selected point:

  | family | median | max |
  |---|---|---|
  | KM | 1.5e-5 | 6.9e-3 |
  | QAOA | 9e-7 | 1.5e-4 |

- 14 of 96,213 starts hit the 2000 cap, all KM at N=6.

## Bounds do constrain coverage in a few N=6 cells

The R2 box (|diagonal coefficient| ≤ 2π/s_J) was kept, but it is not always slack:

- **Selected cells on a bound.** Five selected per-repeat cells, all at N=6, have a coordinate within 1e-6 of a bound.
- **N6 k8, repeat 1.** Its KM selections at L=4 and L=5 sit on the gamma bound (a = 2.85 = box) at a lower energy
  than the other repeats' small-gamma basin. The best-over-repeats state at L=5 comes from it (F 0.860).
- **Starts that ended on a bound.** 9151 of 96,213 starts (9.5%) ended with a coordinate on a bound.

A larger box might give lower-energy KM or QAOA states in such cells. This was not rerun.

## Independent verification

`replay_independent.py` imports no runner code. It does the following:

1. Regenerates J from the seed.
2. Rebuilds K-hat and M-hat from their definitions.
3. Builds every operator as a sparse Pauli string, including the XX and XXX cavity mixers.
4. Evolves |+> with `expm_multiply`.
5. Takes the ground space from `eigsh`.

It replayed 5991 selected states (every repeat, family and L) on 84 instances. The run-time tolerances are frozen in the
preregistration. The table gives the worst case per N:

| check | N=6 | N=9 | N=12 | tolerance |
|---|---|---|---|---|
| relative energy | 1.2e-12 | 3.5e-12 | 7.9e-13 | 1e-8 |
| F_sub | 1.1e-12 | 4.8e-13 | 4.7e-13 | 1e-6 |
| E0 | 2.8e-14 | 1.4e-13 | 1.4e-13 | 1e-9 |
| norm | 6.2e-13 | 1.7e-12 | 4.0e-13 | 1e-10 |

Window sizes match the main code on every instance, and eigensolver residuals are at most 1.5e-13. The seed formula
reproduced the R1 manifest hashes for k=0..3.

No fresh instance was structurally unreachable: the even-sector ground state was inside the 1e-6 window in every
case. One instance (N9 k11) is tolerance-sensitive. Its doublet splitting is 3.1e-8, so a 1e-8 window holds only
the odd state and gives F = 0 for every family. The preregistered 1e-6 window is consistent.

## Equal-parameter view (descriptive only)

KM L3 has 12 parameters, the same as QAOA L6. Mean F_sub for that pairing is 0.653 vs 0.589 at N=6, 0.242 vs
0.378 at N=9, and 0.160 vs 0.290 at N=12. On a per-parameter basis QAOA is ahead at N=9 and N=12.

## Scope, limits, and what was not run

- **Instances.** 22 of 24 fresh instances per N completed all three repeats. k=26 and 27 were cut by the
  4500 s optimization wall, and the secondary k=0..3 were never started. All are listed in `RESULTS.md`; none
  was replaced.
- **Wall time.** Total was 77 min: 75 min of optimization, then evaluation, replay and summary.
- **Optimality.** Results are best-found local optima from about 13 to 21 starts per depth in each of three
  repeats. No statement of global optimality is made.
- **What the comparison measures.** It is "KM package (shared gamma, K and M features, and the XX and XXX cavity
  shells) vs standard QAOA". It does not isolate the K and M features from the richer mixers.
- **Untested ideas** (listed in `OPUS_IDEAS.md`, not run): per-layer-gamma KM, QAOA with the cavity shells, KM
  without the w2/w3 shells, an equal-gate-count comparison, packing permutations, larger N, other fields and ±J
  disorder, and warm-start QAOA.
- **Gaps** were not recomputed, and no gap–fidelity correlation is claimed.
