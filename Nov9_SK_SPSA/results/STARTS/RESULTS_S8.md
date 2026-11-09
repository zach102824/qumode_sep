# STARTS S8: equal-budget 8 starts vs C0 4 starts at N15 L3

Exclusive unread development seeds 10048–10055. Ordinary IID C0 SPSA. Energy-only selection. F post hoc.
Q = 3072 inclusive. S4 K=379×4. S8 K=187×8. No continuation, no free prefix.

Runs 32 ok 32 failures 0. Charged 98304. Outside cap 0. Rejected instances 0. Missing 0.

Gate N15 L3 Nov3 mean ΔE (S8−S4) = +0.039561 med +0.045124 CI [+0.022026, +0.057274] n=8: **reject**. GR not patched.

N3COND 772224, LEXACT SPSA 2560, DEPTH C1 327680 are not revised. This stream is STARTS_S8.

## S8 vs S4

| circuit | Δ(E−E0)/N mean/med [CI] tails wins(S8) | ΔE | ΔF wins(S8) |
|---|---|---|---|
| Nov3 | +0.00264 med +0.00301 [+0.00147,+0.00382] tails [+0.00014,+0.00539] 0/8 | +0.03956 med +0.04512 [+0.02203,+0.05727] tails [+0.00209,+0.08079] | -0.02281 med -0.01479 [-0.04684,-0.00506] tails [-0.09360,+0.00405] 2/8 |
| QAOA | +0.00119 med +0.00082 [+0.00061,+0.00186] tails [+0.00025,+0.00288] 0/8 | +0.01783 med +0.01231 [+0.00918,+0.02786] tails [+0.00372,+0.04320] | -0.00965 med -0.00562 [-0.01712,-0.00384] tails [-0.03250,-0.00160] 0/8 |

## Nov3 vs QAOA under each start count

| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF Nov3 higher |
|---|---|---|
| S4 | +0.00396 med +0.00533 [+0.00088,+0.00654] tails [-0.00326,+0.00779] 2/8 | +0.01901 med +0.01588 [-0.00559,+0.04851] tails [-0.02673,+0.10038] 5/8 |
| S8 | +0.00541 med +0.00691 [+0.00141,+0.00910] tails [-0.00325,+0.01266] 2/8 | +0.00584 med -0.00767 [-0.03517,+0.04985] tails [-0.07347,+0.11967] 4/8 |
