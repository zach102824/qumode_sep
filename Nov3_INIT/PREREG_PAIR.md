# N3INIT pair-shell probe (arxiv 2204.02923)

Bounded development. Original Nov3 family only. SPSA only. Energy selection. Fidelity post hoc. No GR patch. No candidacy file. Holdout FAIL on `10144..10151` still stands. Do not retune the frozen 1000-seed benchmark.

Paper `2204.02923` pair-phase is \(\mathcal{V}(M)=\exp(-i/4\sum_{n<m}M_{nm}Z_n Z_m)\), an all-to-all independent ZZ graph-state layer. That is a different architecture (O(N²) pair angles plus local Bloch rotations), not Nov3 shared-Uc + SNAP shells. It is **not implemented**. The in-family probe unshares the existing SNAP polynomial so the paper-like pair term \(s_2\sum_{r<r'}n_r n_{r'}\) (and the rest of the 3-shell) can differ by cavity.

## Arms and param counts

Same Hamiltonian, same packing `[[a, a+N/3, a+2N/3]]`, same shared cost unitary \(U_c^Z=\exp[-i\gamma(J+\theta_K\hat K+\theta_M\hat M)]\).

| arm | method | controls | N=15 L=2 | N=15 L=3 |
|---|---|---|---:|---:|
| A `nov3_shared` | `Nov3rd` | shared \(\gamma,\theta_K,\theta_M\) + shared \((s_1,s_2,s_3)\) per layer | 9 | 12 |
| B `nov3_pair` | `Nov3pair` | same shared \(U_c\) + independent \((s_1,s_2,s_3)\) per cavity per layer | 33 | 48 |
| C `qaoa_center4` | `QAOA` | \(p=L\), \((\gamma,\beta)\) per layer | 4 | 6 |

Layout for B: `x[0:3] = (γ, θK, θM)`, `x[3:]` reshape `(L, C, 3)` with `C = N/3 = 5`. If every cavity of a layer uses the same `(s1,s2,s3)`, B is identical to A. Start 0 tiles the shared coherent center onto every cavity: `γ=0.25`, `θ=0`, every cavity every layer `(s1,s2,s3)=(−0.5,0,0)`. That still reduces algebraically to QAOA `(γ,β)=(0.25,0.25)`.

Method ids: QAOA `0`, shared Nov3 `1`, pair-cavity `2`. Distinct SPSA streams.

Not this round: paper \(\mathcal{V}(M)\); independent intra-cavity pair bonds (3 pairs/cavity instead of one \(s_2\)); Nov9/SYK; extra agents.

## Recipe freeze

SPSA only, frozen gains: `c=0.1`, `A=25`, exponents `0.602/0.101`, 4 two-sided calibration pairs, `a0=0.5/median(|g|/σ)` or 0 if that median `<1e-15`, clip `2σ`, Rademacher `±1`. σ: Nov3 `γ 0.10`, `θ 0.25`, shells `0.15` (pair-cavity uses the same shell σ on every cavity parameter); QAOA angles `0.10`. Four equal blocks, `T=Q/8−6`. Inclusive `Q=qL` with `q=1024`, so `Q=2048` at L=2 and `Q=3072` at L=3, **the same query budget for all three arms** even though param counts differ. Energy selection among init, last iterate, Polyak of last `min(64,T)`, midpoint. Lowest energy; energy tie keeps init; across starts lower `start_id` wins. Fidelity is post hoc.

Shared-shell start 0 and QAOA start 0 remain the v4 centers. Pair start 0 is the tiled shared center. Starts 1–3 stay in the historical boxes (`γ,θ ~ U(−0.5,0.5)`, shells `U(−0.3,0.3)`), drawn independently per cavity for B.

## Seeds

Exclusive unread band `10328..10335`, n=8. Provenance, verified before any instance or SPSA:

- Not LE `10320..10327`.
- Not local R23 `10304..10319`.
- Not FT / prior Nov3_INIT `≤10247` (covers train `10018..10025`, holdout `10144..10151`, v9 `10200..10215`, v10 `10232..10247`).
- Not validation `11000..11031`.
- Not GR final `20000..20999`.
- No existing `Nov3_INIT/results/**` cell or instance file for these seeds.

The runner refuses every other seed. `B*` is new, classical, interior entropy max on sites `0..floor(N/2)−1`, not a quantum query.

## Grid

New directory `results/pair/`. `N=15` only. `L∈{2,3}`. `q=1024`. Three arms. Energy-only selection.

If a cell is infeasible, stop that cell with a measured blocker. Do not approximate physics. Fail any cell → reject the probe; no GR.

## Gates (prespecified, energy only, before any SPSA)

Bootstrap 4000, `default_rng(3334)`, seeds resampled. `d_pairQ = E(nov3_pair) − E(qaoa_center4)`. Means are the n=8 seed means.

**Primary L=3 (dev only).** Pass only if both:

1. pair vs QAOA: mean `d_pairQ` CI entirely `< 0`
2. pair is not worse than shared: mean `E(nov3_pair) ≤` mean `E(nov3_shared)`

**L=2 sanity / regression only.** Pass only if both:

1. pair is not worse than shared: mean `E(nov3_pair) ≤` mean `E(nov3_shared)`
2. pair does not lose to QAOA: mean `d_pairQ` CI is **not** entirely `> 0`

Fail either layer, or any infeasible cell → label **REJECT_PAIR_SHELL**. Pass both → **PAIR_SHELL_L3_EDGE**. Either label: no GR patch, no candidacy file.

L=2 pair-vs-QAOA magnitude is reported, not a beat-QAOA requirement.

## Queries

Each cell ledger equals its `Q`. Report new pair-shell quantum queries and cumulative total `2179072 + new`. Historical v1–v10 queries stay spent once. Report wall and CPU. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, paper \(\mathcal{V}(M)\), extra agents, candidacy notes, frozen-benchmark retune.
