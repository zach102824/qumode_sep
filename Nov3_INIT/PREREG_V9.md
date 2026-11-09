# N3INIT v9 preregistration

N=15 L=2 Q=2048 effect-size panel. Estimation only, not adoption. Exact v4 four-start center-4 / equal-SPSA-block recipe, not retuned from train `10018..10025` or holdout `10144..10151`. No GR patch. No candidacy file. Historical research queries `2048000` are spent once.

## Recipe freeze (exact v4)

Original Nov3, all `Uc` controls shared across layers, full 3-shell parameters per layer, unchanged circuit. QAOA `p = L = 2`. SPSA only, frozen gains: `c = 0.1`, `A = 25`, exponents `0.602/0.101`, 4 two-sided calibration pairs, `a0 = 0.5 / median(|g|/σ)` or 0 if that median `< 1e-15`, clip `2σ`, Rademacher `±1`. σ: Nov3 `γ 0.10`, `θ 0.25`, shells `0.15`; QAOA angles `0.10`. Four equal blocks, `T = Q/8 − 6`. Inclusive `Q = q L = 2048`. Energy selection among init, last iterate, Polyak of last `min(64, T)`, midpoint. Lowest energy; energy tie keeps init; across starts lower `start_id` wins. Fidelity is post hoc.

Nov3 start 0: `γ = 0.25`, `θ_K = θ_M = 0`, every layer `(s1, s2, s3) = (−0.5, 0, 0)`. QAOA start 0: every layer `(γ, β) = (0.25, 0.25)`. Starts 1–3 stay in the historical boxes. Method ids remain 1 (Nov3) and 0 (QAOA). No schedule change.

## Seeds

Exclusive unread band `10200..10215`, n=16. Provenance, verified before any instance or SPSA:

- Not train `10018..10025`.
- Not holdout `10144..10151`.
- Not LE `≤ 10199`.
- Not local R19 `10184..10191`.
- Not validation `11000..11031`.
- Not GR final `20000..20999`.
- No existing `Nov3_INIT/results/**` cell or instance file for these seeds.

The runner refuses every other seed. `B*` is new and classical, not a quantum query.

## Grid

New directory `results/v9/`. Arms: `nov3_center4`, `qaoa_center4`. `N = 15`, `L = 2` only, `q = 1024`, `Q = 2048`. No L=3. No N=18. Energy-only selection.

If a cell is infeasible, stop that cell with a measured blocker. Do not approximate physics.

## Gates (prespecified, energy only)

`d_pair = E(nov3_center4) − E(qaoa_center4)`. Bootstrap 4000, `default_rng(3334)`, seeds resampled.

Label **N15-L2 ESTIMATE_NEGATIVE_CI** only if mean `d_pair` CI is entirely `< 0` on this n=16 band. Otherwise **UNSTABLE / CI_CROSSES_0**. Either label: no GR patch, no candidacy file.

Report mean ΔE, CI, wins/tails, mean F both arms, start-0 rates, query ledger.

## Exploratory pool (not a gate)

After the n=16 label is written, a secondary table pools already-ledgered N=15 L=2 Q=2048 cells from train `10018..10025` (v7), holdout `10144..10151` (v8), and this band `10200..10215` (n=32). Marked exploratory. It does not retune, confirm, or adopt.

## Queries

Each cell ledger equals its own `Q = 2048`. Report new v9 quantum queries and cumulative total `2048000 + new`. Reused train/holdout queries stay historical, counted once.

## Off limits

GR PR #77. Validation. Final. LE. Local R19. Nov9, SYK, AF, N21, L=3, N=18, extra agents, candidacy notes.
