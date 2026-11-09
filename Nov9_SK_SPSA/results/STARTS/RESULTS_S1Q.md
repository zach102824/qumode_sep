# STARTS S1Q: frozen S1 Nov3 vs QAOA at N15 L=3 vs budget

Exclusive unread seeds 10072–10079. Frozen PR83/84/85 S1 (TQA-only C0). Energy selects. F post hoc.
Q ∈ {3072, 6144, 12288} inclusive. S1 K=1531 / 3067 / 6139. No S4/S8. GR not patched.

Runs 48 ok 48 failures 0. Charged 344064. Outside cap 0. Rejected B* 0. Missing 0.
Q=12288: ran. Blocker: None.

Soft note: **gap_persists_vs_budget**. Development energy-win Qs: none. Not a GR adoption.

## Primary: Nov3 − QAOA under S1

| Q | Δ(E−E0)/N mean/med [CI] tails Nov3 lower | ΔE | ΔF | energy-win candidate? |
|---|---|---|---|---|
| 3072 | +0.00206 med +0.00364 [-0.00165,+0.00491] tails [-0.00951,+0.00776] 1/8 | +0.03095 med +0.05456 [-0.02471,+0.07359] tails [-0.14263,+0.11635] | +0.03318 med +0.01220 [+0.00718,+0.06384] tails [-0.00709,+0.12004] 6/8 | no |
| 6144 | +0.00158 med +0.00312 [-0.00195,+0.00415] tails [-0.00964,+0.00577] 2/8 | +0.02371 med +0.04679 [-0.02929,+0.06223] tails [-0.14456,+0.08657] | +0.03651 med +0.01516 [+0.01062,+0.06704] tails [-0.00494,+0.12219] 7/8 | no |
| 12288 | +0.00116 med +0.00268 [-0.00209,+0.00343] tails [-0.00930,+0.00449] 1/8 | +0.01745 med +0.04024 [-0.03132,+0.05150] tails [-0.13944,+0.06732] | +0.04128 med +0.02291 [+0.01586,+0.07082] tails [-0.00415,+0.12502] 7/8 | no |

## Secondary: within-method vs own Q=3072 (negative energy = improved)

| method | Q | Δ(E−E0)/N vs Q=3072 mean/med [CI] tails | ΔF |
|---|---|---|---|
| Nov3 | 6144 | -0.00077 med -0.00067 [-0.00122,-0.00046] tails [-0.00217,-0.00021] 8/8 | +0.00677 med +0.00637 [+0.00469,+0.00877] tails [+0.00117,+0.01146] 8/8 |
| QAOA | 6144 | -0.00029 med -0.00022 [-0.00041,-0.00019] tails [-0.00058,-0.00012] 8/8 | +0.00343 med +0.00275 [+0.00244,+0.00473] tails [+0.00198,+0.00716] 8/8 |
| Nov3 | 12288 | -0.00148 med -0.00115 [-0.00235,-0.00091] tails [-0.00427,-0.00043] 8/8 | +0.01470 med +0.01481 [+0.00886,+0.02070] tails [-0.00038,+0.03049] 7/8 |
| QAOA | 12288 | -0.00058 med -0.00042 [-0.00086,-0.00036] tails [-0.00138,-0.00027] 8/8 | +0.00659 med +0.00503 [+0.00463,+0.00904] tails [+0.00406,+0.01306] 8/8 |
