# FTREP: LE replicate of FT PR79 v4 center-4 at L=2 Q=2048

FT alloc.py sha256 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` from PR79 `347ac23369db`.
Exclusive unread seeds 10152–10159. N∈{12,15} L=2 Q=2048. Four equal SPSA blocks, frozen gains, center-4 starts.
Energy selects (init/last/Polyak/mid). F post hoc. GR not patched. FT holdout 10144–10151 unused.

Runs 32 ok 32 failures 0. Charged 65536. Outside cap 0. Rejected B* 0. Missing 0.

Gate R1 N15 mean ΔE -0.07079 med -0.04108 [-0.13629,-0.01273] tails [-0.23480,+0.02295] 5/8: **pass**.
Gate R2 N12 mean ΔE -0.10655 med -0.11279 [-0.18754,-0.02987] tails [-0.31360,+0.04535] 6/8: **pass**.
Decision: **FT-L2 REPLICATED on LE**.

## Nov3 − QAOA (center4)

| N | mean ΔE med [CI] tails wins(Nov3) | Δ(E−E0)/N | ΔF wins | start0 Nov3 |
|---|---|---|---|---|
| 12 | -0.10655 med -0.11279 [-0.18754,-0.02987] tails [-0.31360,+0.04535] 6/8 | -0.00888 med -0.00940 [-0.01547,-0.00247] tails [-0.02613,+0.00378] | +0.05985 med +0.06762 [+0.02875,+0.08923] tails [-0.00347,+0.11463] 6/8 | 2/8 |
| 15 | -0.07079 med -0.04108 [-0.13629,-0.01273] tails [-0.23480,+0.02295] 5/8 | -0.00472 med -0.00274 [-0.00916,-0.00094] tails [-0.01565,+0.00153] | +0.04241 med +0.03910 [+0.01721,+0.06933] tails [-0.00270,+0.09525] 6/8 | 5/8 |
