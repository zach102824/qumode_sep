# STARTS S1: concentrate Q on TQA vs historical four-start C0

Read-dev seeds 10048–10055. S4 reused from PR82 after recipe audit. Ordinary C0 SPSA. Energy-only. F post hoc.
Q=3072 inclusive. S1 K=1531×1. S2 K=763×2 (not in the gate). S4 K=379×4 reused, not recharged.

New ok 32 fail 0. Newly charged 98304. Outside cap 0. S4 reused True (16 files, 49152 not added again).

Gate N15 L3 Nov3 mean ΔE (S1−S4) = -0.038441 med -0.033106 CI [-0.053317, -0.027036] n=8: **adopt_candidate**. GR not patched.

N3COND 772224, LEXACT 2560, DEPTH C1 327680, STARTS S8 98304 are not revised.

## S1/S2 vs reused S4

| contrast | Δ(E−E0)/N mean/med [CI] tails wins | ΔE | ΔF wins |
|---|---|---|---|
| Nov3 S1−S4 | -0.00256 med -0.00221 [-0.00355,-0.00180] tails [-0.00545,-0.00106] 8/8 | -0.03844 med -0.03311 [-0.05332,-0.02704] tails [-0.08168,-0.01583] | +0.01913 med +0.01757 [+0.01045,+0.02793] tails [-0.00350,+0.04227] 7/8 |
| Nov3 S2−S4 | -0.00151 med -0.00114 [-0.00237,-0.00090] tails [-0.00408,-0.00051] 8/8 | -0.02268 med -0.01709 [-0.03556,-0.01348] tails [-0.06114,-0.00770] | +0.01233 med +0.01499 [+0.00409,+0.01989] tails [-0.00958,+0.02747] 7/8 |
| QAOA S1−S4 | -0.00108 med -0.00072 [-0.00197,-0.00046] tails [-0.00394,-0.00017] 8/8 | -0.01620 med -0.01082 [-0.02962,-0.00685] tails [-0.05911,-0.00259] | +0.00960 med +0.00712 [+0.00405,+0.01671] tails [-0.00055,+0.03164] 7/8 |
| QAOA S2−S4 | -0.00048 med -0.00019 [-0.00094,-0.00007] tails [-0.00167,+0.00046] 7/8 | -0.00718 med -0.00292 [-0.01405,-0.00101] tails [-0.02500,+0.00686] | +0.00424 med +0.00269 [+0.00064,+0.00792] tails [-0.00365,+0.01184] 6/8 |

## Nov3 vs QAOA under S1 and S4

| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF Nov3 higher |
|---|---|---|
| S1 | +0.00248 med +0.00327 [+0.00011,+0.00445] tails [-0.00353,+0.00594] 2/8 | +0.02854 med +0.01999 [+0.01030,+0.04962] tails [-0.00291,+0.07941] 6/8 |
| S4 | +0.00396 med +0.00533 [+0.00088,+0.00654] tails [-0.00326,+0.00779] 2/8 | +0.01901 med +0.01588 [-0.00559,+0.04851] tails [-0.02673,+0.10038] 5/8 |
