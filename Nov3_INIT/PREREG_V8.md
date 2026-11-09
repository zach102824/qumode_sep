# N3INIT v8 preregistration

Fresh-holdout L=2 confirmation. Exact v4 four-start center-4 / equal-SPSA-block recipe, not retuned from N=15/18 development on `10018..10025`. `center4` is not a GR patch in this file. Historical research queries `1949696` are spent once. Prior cells are not overwritten.

## Recipe freeze (exact v4)

Original Nov3, all `Uc` controls shared across layers, full 3-shell parameters per layer, unchanged circuit. QAOA `p = L = 2`. SPSA only, frozen gains: `c = 0.1`, `A = 25`, exponents `0.602/0.101`, 4 two-sided calibration pairs, `a0 = 0.5 / median(|g|/σ)` or 0 if that median `< 1e-15`, clip `2σ`, Rademacher `±1`. σ: Nov3 `γ 0.10`, `θ 0.25`, shells `0.15`; QAOA angles `0.10`. Four equal blocks, `T = Q/8 − 6`. Inclusive `Q = q L`. Energy selection among init, last iterate, Polyak of last `min(64, T)`, midpoint. Lowest energy; energy tie keeps init; across starts lower `start_id` wins. Fidelity is post hoc.

Nov3 start 0: `γ = 0.25`, `θ_K = θ_M = 0`, every layer `(s1, s2, s3) = (−0.5, 0, 0)`. QAOA start 0: every layer `(γ, β) = (0.25, 0.25)`. Starts 1–3 stay in the historical boxes. Method ids remain 1 (Nov3) and 0 (QAOA).

## Holdout seeds

Exclusive unread band `10144..10151`, n=8. Provenance, verified before any instance or SPSA:

- Not prior FT development `10018..10025`.
- Not local R15/R16 `≤ 10143`.
- Not LE warm-start `10128..10135`.
- Not validation `11000..11031`.
- Not GR final `20000..20999`.
- No existing `Nov3_INIT/results/**` cell or instance file for these seeds.

The runner refuses every other seed. These eight instances are new; `B*` is computed here and charged as classical, not as a quantum query.

## Grid

New directory `results/v8/`. Arms: `nov3_center4`, `qaoa_center4`. `N ∈ {15, 18}`, `L = 2` only, `q ∈ {512, 1024}` so `Q = 1024` and `Q = 2048`. Both budgets run; v6/v7 L=2 cell times make both cheap (N=15 ~8–16s, N=18 ~130–260s). No L=3, L=4, N=21.

If a cell is infeasible, stop that cell with a measured blocker. Do not approximate physics. Do not drop to a cheaper seed.

## Gates (prespecified, energy only)

`d_pair = E(nov3_center4) − E(qaoa_center4)`. Bootstrap 4000, `default_rng(3334)`, seeds resampled. Call order: N=15 then N=18, both at `q = 1024` (`Q = 2048`).

- **H1:** N=15, Q=2048, mean `d_pair` CI entirely `< 0`.
- **H2:** N=18, Q=2048, mean `d_pair` CI entirely `< 0`.

Label **L2-holdout CONFIRMED** only if both H1 and H2 pass. Otherwise **FAIL** and stop. Do not escalate to GR. The q=512 rows are reported but are not the confirmation gates.

If confirmed, write `results/v8/CANDIDACY_NOTE.md` with recipe hash, seed bands, and tables. Do not patch PR #77. Do not apply the recipe to the frozen 1000-seed benchmark.

## Queries

Each cell ledger equals its own `Q`. Report procedural cap, new v8 quantum queries, and cumulative total `1949696 + new`. Classical `B*` wall/CPU/RSS is separate.

## Off limits

GR PR #77 and its finals. Validation. Local R15/R16. LE warm-start. Nov9, SYK, AF, N21, extra agents, L>2, ALT/S1, conditioning, frame combo.
