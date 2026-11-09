# N3INIT v7 preregistration

Exact v4 four-start recipes, frozen before any N=15 L=2 or L=3 SPSA. Not retuned from N=18 L=1–3 or from N=15 L=1. `center4` remains a development candidate. It is not a GR benchmark adoption. Prior cells are not overwritten. Historical research queries `1826816` are spent once.

This batch is N=15 depth after the already-measured N=15 L=1 shallow edge. It is not LE S1 schedule work and it is not local R10 ordinary-IID budget/depth on seeds `10026..31`.

## Recipe freeze (exact v4)

Original Nov3, all `Uc` controls shared across layers, full 3-shell parameters per layer, unchanged circuit. QAOA `p = L`. SPSA only, frozen gains: `c = 0.1`, `A = 25`, exponents `0.602/0.101`, 4 two-sided calibration pairs, `a0 = 0.5 / median(|g|/σ)` or 0 if that median `< 1e-15`, clip `2σ`, Rademacher `±1`. σ: Nov3 `γ 0.10`, `θ 0.25`, shells `0.15`; QAOA angles `0.10`. Four equal blocks, `T = Q/8 − 6`. Inclusive `Q = q L`. Energy selection among init, last iterate, Polyak of last `min(64, T)`, midpoint. Lowest energy; energy tie keeps init; across starts lower `start_id` wins. Fidelity is post hoc.

Nov3 start 0: `γ = 0.25`, `θ_K = θ_M = 0`, every layer `(s1, s2, s3) = (−0.5, 0, 0)`. QAOA start 0: every layer `(γ, β) = (0.25, 0.25)`. Starts 1–3 stay in the historical boxes. Method ids remain 1 (Nov3) and 0 (QAOA).

## Arms and grid

New directory `results/v7/`. Arms: `nov3_center4`, `qaoa_center4`. Seeds `10018..10025` only, n=8 development. `N = 15`, `L ∈ {2, 3}`, `q ∈ {512, 1024}`, `Q = q L`. Both depths and both budgets run unless a measured gate or per-cell blocker stops a cell. Energy-only selection. Report paired means, CIs, medians, wins, tails for `E`, `(E − E0)/N`, and F, plus start-0 rates.

Reuse the v4 N=15 `B*` files. Do not recompute. Require each reused file to have interior cut `0..6` (`floor(15/2)−1` is 6, sites `0..6`), `gap > 1e-8`, `gs_parity > 1 − 1e-6`, `gs_min > −1e-8`, residual `< 1e-8`, both sector gaps positive, matching `npz` of length `2^15`. Classical prep cost stays the v4 charge.

Do not rerun N=18 L=1 or the completed N=18 L=2,3 matrix.

## Resource gate (before SPSA)

Smoke on seed `10018` only, loading the existing v4 instance, no B* recompute:

1. One original-Nov3 L=2 energy and one L=3 energy at the frozen center.
2. RSS of that process.
3. Projected cell wall `3072 × L3_energy_seconds`. Four workers, BLAS 1.

Gate `n15_l23_allowed` if all of: L=2 energy `< 0.4 s`, L=3 energy `< 0.6 s`, process RSS `< 1 GiB`, projected four-worker RSS `< 4 GiB`, projected one L=3 q=1024 cell `< 1800 s`. If the gate fails, stop and write the probe. Do not switch seeds. Do not approximate the circuit.

If a started cell is infeasible (non-finite energy, OOM, wall > 1800 s), stop that cell, record the measured blocker, and leave remaining cells to finish or to hit the same stop.

## Read rule

Energy only. No fidelity win is required.

- `d_pair = E(nov3_center4) − E(qaoa_center4)`
- `nov3_lower` if `mean(d_pair) < 0` and the seed-bootstrap 95% interval excludes 0.
- If the interval contains 0, report the point estimate; do not call a confirmed edge.
- Call order: L ascending, then q ascending. Outcomes do not change Q, gains, or the center.
- Do not attribute an energy edge to the coherent center unless start 0 is actually selected.

Bootstrap: 4000 resamples, `default_rng(3334)`, seeds resampled.

## Queries

Each cell ledger equals its own `Q`. Report the procedural cap, the sum of new v7 quantum queries, and the cumulative total `1826816 + new`. Reused B* is not added again.

## Off limits

GR frozen 1000 panel. Validation `11000..11031`. Final `20000..20999`. LE `10032..47` and LE S1 schedules. Local R10 ordinary-IID on `10026..31`. Nov9, SYK, AF, N21, extra agents, extra parameters, conditioning, frame combo, N=18 reruns, L>3 at N=15.
