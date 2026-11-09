# N3INIT v6 preregistration

Exact v4 four-start recipes, frozen before any N=18 L=2 or L=3 SPSA. Not retuned from the N=18 L=1 outcomes. `center4` remains a development candidate. It is not a GR benchmark adoption. Prior cells are not overwritten. Historical research queries `1703936` are spent once.

## Recipe freeze (exact v4 / v5)

Original Nov3, all `Uc` controls shared across layers, full 3-shell parameters per layer, unchanged circuit. QAOA `p = L`. SPSA only, frozen gains: `c = 0.1`, `A = 25`, exponents `0.602/0.101`, 4 two-sided calibration pairs, `a0 = 0.5 / median(|g|/σ)` or 0 if that median `< 1e-15`, clip `2σ`, Rademacher `±1`. σ: Nov3 `γ 0.10`, `θ 0.25`, shells `0.15`; QAOA angles `0.10`. Four equal blocks, `T = Q/8 − 6`. Inclusive `Q = q L`. Energy selection among init, last iterate, Polyak of last `min(64, T)`, midpoint. Lowest energy; energy tie keeps init; across starts lower `start_id` wins. Fidelity is post hoc. No GS/fidelity init or start selection. No architecture change, no extra parameters, no conditioning, no frame combo.

Nov3 start 0: `γ = 0.25`, `θ_K = θ_M = 0`, every layer `(s1, s2, s3) = (−0.5, 0, 0)`. QAOA start 0: every layer `(γ, β) = (0.25, 0.25)`. Starts 1–3 stay in the historical boxes. Method ids remain 1 (Nov3) and 0 (QAOA).

## Arms and grid

New directory `results/v6/`. Arms: `nov3_center4`, `qaoa_center4`. Seeds `10018..10025` only, n=8 development, not a population. `N = 18`, `L ∈ {2, 3}`, `q ∈ {512, 1024}`, `Q = q L` so L=2 uses 1024 and 2048, L=3 uses 1536 and 3072. Both depths and both budgets run unless a measured resource gate or a per-cell blocker stops a cell. Energy-only selection. Report paired means, CIs, medians, wins, tails for `E`, `(E − E0)/N`, and F.

Reuse the v5 N=18 `B*` files. Do not recompute. Require each reused file to have interior cut `0..8`, `gap > 1e-8`, `gs_parity > 1 − 1e-6`, `gs_min > −1e-8`, residual `< 1e-8`, both sector gaps positive, matching `npz` of length `2^18`. Classical prep cost stays the v5 charge; v6 records reuse, not a second B* wall.

## Resource gate (before SPSA)

Smoke on seed `10018` only, loading the existing v5 instance, no B* recompute, no other seeds:

1. One original-Nov3 L=2 energy and one L=3 energy at the frozen center.
2. RSS of that process.
3. Projected cell wall: `Q_max × energy_seconds` with `Q_max = 3072` (L=3, q=1024). Four workers, BLAS 1. Peak RSS of four concurrent cells estimated as `4 × process RSS`.

Gate `n18_l23_allowed` if all of: L=2 energy `< 0.8 s`, L=3 energy `< 1.2 s`, process RSS `< 2 GiB`, projected four-worker RSS `< 8 GiB`, projected one L=3 q=1024 cell `< 3600 s`. If the gate fails, stop, write the probe, and do not open SPSA. Do not switch seeds. Do not drop L. Do not approximate the circuit.

If a started cell is infeasible (non-finite energy, OOM, wall > 3600 s), stop that cell, record the measured blocker, and leave remaining cells to finish or to hit the same stop. Do not invent a cheaper physics.

## Read rule

Energy only. No fidelity win is required.

- `d_pair = E(nov3_center4) − E(qaoa_center4)`
- `nov3_lower` if `mean(d_pair) < 0` and the seed-bootstrap 95% interval excludes 0.
- If the interval contains 0, report the point estimate; do not call a confirmed edge.
- Call order: L ascending, then q ascending. Outcomes do not change Q, gains, or the center.
- Do not attribute an energy edge to the coherent center unless start 0 is actually selected.

Bootstrap: 4000 resamples, `default_rng(3334)`, seeds resampled.

## Queries

Each cell ledger equals its own `Q`. Report the procedural cap, the sum of new v6 quantum queries, and the cumulative total `1703936 + new`. Reused B* is not added again. No uncharged training.

## Off limits

GR frozen 1000 panel. Validation `11000..11031`. Final `20000..20999`. LE `10032..47`. Local R8 `10026..31`. Nov9, SYK, AF, N21, extra agents, extra parameters, conditioning, frame combo, L>3 at N=18.
