# N3INIT v4 preregistration

`center4` is a development candidate only. It is not adopted into the frozen 1000-seed benchmark. This file is the rule for the next fit. Prior cells are not overwritten and are not relabeled.

## Derived QAOA center

Nov3 start 0 is `γ = 0.25`, `θ_K = θ_M = 0`, and every layer `(s1, s2, s3) = (−0.5, 0, 0)`.

The shell with `s2 = s3 = 0` is `exp(−i s1 n_Z)` on each cavity. Hadamard conjugation gives `exp(−i s1 n_X)` with `n_X = (I − X)/2`, which is `exp(−i β X)` up to a global phase when `β = −s1/2`. Thus `β = +0.25`. The cost angle is the same shared `γ`, because `θ = 0` leaves the cost diagonal equal to the `J` diagonal used by QAOA. Every QAOA layer is therefore `(γ, β) = (0.25, 0.25)`.

That vector is also the historical `equal4` QAOA start 0. The overlap check below is what makes it the derived control, rather than an unexamined default. The sign-flipped mixer `β = −0.25` is not the control. Fidelity in that check is a circuit identity. It is not an optimizer input.

Starts 1–3 stay in the historical boxes: Nov3 `γ, θ ~ U(−0.5, 0.5)`, shells `~ U(−0.3, 0.3)`; QAOA angles `~ U(−0.5, 0.5)`. Four equal SPSA blocks, same frozen gains, same energy-only selection (init, last, Polyak of the last `min(64, T)`, midpoint). QAOA gets that same four-start opportunity. It is not left on a different split.

## Arms

New directory `results/v4/`. Method ids stay 1 (Nov3) and 0 (QAOA).

- `nov3_equal4`: historical shells-at-0 start 0.
- `nov3_center4`: coherent start 0. Starts 1–3 must match `nov3_equal4` to `1e-8`.
- `qaoa_center4`: derived start 0 above. One QAOA arm, not a second hidden baseline. Its procedural queries are new at these depths.

`q = 128` is not run. `L = 1..3` already measured that budget, and the depth question is at the primary caps. `Q = q L` with `q ∈ {512, 1024}`.

- Development, not a fresh holdout: seeds `10018..10025` at `N ∈ {9, 12}`, `L ∈ {4, 6}`. Those seeds were already used at these `N`.
- `N = 15`, `L = 1`, same seeds, development because the width is new. Run only if the conformance probe records one ground-state solve under 2s and one L1 Nov3 energy under 0.05s. Otherwise skip and keep the probe.
- No `N = 18`. No dense `2^15` or `2^18` Hamiltonian matrix. `B*` stays the sparse interior entropy maximum. `N = 9` and `N = 12` reuse the existing `B*` files.

## Read rule

Energy only. No fidelity win is required. `n = 8` seeds, development.

For each `(N, L, q)`, and pooled over `N` at fixed `(L, q)` for the `N = 9/12` depth grid:

- `d_self = E(nov3_center4) − E(nov3_equal4)`
- `d_pair = E(nov3_center4) − E(qaoa_center4)`

`center_helps` if `mean(d_self) < 0`. `beats_fair_qaoa` if `mean(d_pair) < 0`. A flag whose seed-bootstrap interval contains 0 is reported as a point estimate, not as a confirmed edge. Bootstrap: 4000 resamples, `default_rng(3334)`, seeds resampled, `N` held when a cell is one `N`. Call order: `N` ascending, `L` ascending, `q` ascending, then `d_self`, `d_pair`.

`L = 4` and `L = 6` both run. Neither flag drops the other depth. `N = 15` does not retune the center. Nothing here authorizes reading validation, final, or the running benchmark.

## Queries

Each cell’s ledger equals its own `Q`. Report that procedural cap and the sum of new v4 queries. Queries already stored under `results/v1`, `v2`, `v3`, and `l2` are historical research spend, counted once in the cumulative total, not added again, and not treated as free training. This phase does not train on them.
