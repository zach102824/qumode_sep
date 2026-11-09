# STARTS S1L6: S1 vs S4 at N15 L=6 Q=6144

Exclusive unread seeds 10064–10071. Frozen PR83/84 C0 recipes. Energy selects. F post hoc.
Q=1024·6=6144 inclusive. S1 K=3067×1. S4 K=763×4. No S1 retune. GR not patched.

Runs 32 ok 32 failures 0. Charged 196608. Outside cap 0. Rejected B* 0. Missing 0.

Gate N15 L6 Nov3 mean ΔE (S1−S4) = -0.086844 med -0.120332 CI [-0.144066, -0.013775] n=8: **s1_helps_at_L6**.

## S1 vs S4

| circuit | Δ(E−E0)/N mean/med [CI] tails wins(S1) | ΔE | ΔF wins(S1) |
|---|---|---|---|
| Nov3 | -0.00579 med -0.00802 [-0.00960,-0.00092] tails [-0.01155,+0.00919] 7/8 | -0.08684 med -0.12033 [-0.14407,-0.01378] tails [-0.17324,+0.13786] | +0.06175 med +0.08280 [+0.01270,+0.10391] tails [-0.07591,+0.12777] 7/8 |
| QAOA | -0.00130 med -0.00109 [-0.00224,-0.00039] tails [-0.00385,+0.00102] 7/8 | -0.01944 med -0.01629 [-0.03355,-0.00589] tails [-0.05774,+0.01534] | +0.01154 med +0.01144 [+0.00310,+0.01969] tails [-0.01100,+0.03340] 7/8 |

## Fair Nov3 vs QAOA (not the S1-vs-S4 gate)

Nov3 energy superiority is claimed only if that CI lies entirely below 0.

| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF | energy-superior? |
|---|---|---|---|
| S1 | +0.00741 med +0.00791 [+0.00208,+0.01297] tails [-0.00735,+0.02268] 1/8 | -0.01167 med -0.01122 [-0.06413,+0.04064] tails [-0.15265,+0.13562] 2/8 | no |
| S4 | +0.01190 med +0.01370 [+0.00781,+0.01561] tails [+0.00054,+0.01861] 0/8 | -0.06188 med -0.07867 [-0.10773,-0.01420] tails [-0.15430,+0.05136] 2/8 | no |
