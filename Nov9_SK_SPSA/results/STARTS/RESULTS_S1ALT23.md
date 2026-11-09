# STARTS S1ALT23: stacked S1 + PR88 ALT at N15 L=2,3 Q=qL

Exclusive unread seeds 10096–10103. Frozen S1JOINT and exact PR88 S1ALT. Energy selects. F post hoc.
Q=1024·L. K=1019 / 1531. Odd=Uc/γ even=SNAP/β. No JOINT prefix. GR not patched.

Runs 64 ok 64 failures 0. Charged 163840. Outside cap 0. Rejected B* 0. Missing 0.

Overall: **s1alt_does_not_hold_at_both_L**.

## Gate: Nov3 S1ALT − S1JOINT

| L | Q | mean ΔE med [CI] tails wins(S1ALT) | Δ(E−E0)/N | pass |
|---|---|---|---|---|
| 2 | 2048 | -0.03567 med -0.01577 [-0.08082,-0.01089] tails [-0.19075,-0.00775] 8/8 | -0.00238 med -0.00105 [-0.00539,-0.00073] tails [-0.01272,-0.00052] | yes |
| 3 | 3072 | -0.00620 med -0.00433 [-0.02233,+0.00976] tails [-0.03992,+0.02547] 4/8 | -0.00041 med -0.00029 [-0.00149,+0.00065] tails [-0.00266,+0.00170] | no |

## S1ALT vs S1JOINT (both circuits)

| L | circuit | Δ(E−E0)/N mean/med [CI] tails wins(S1ALT) | ΔF wins(S1ALT) |
|---|---|---|---|
| 2 | Nov3 | -0.00238 med -0.00105 [-0.00539,-0.00073] tails [-0.01272,-0.00052] 8/8 | +0.00421 med -0.00589 [-0.00848,+0.02555] tails [-0.01239,+0.07593] 2/8 |
| 2 | QAOA | -0.00027 med -0.00026 [-0.00031,-0.00023] tails [-0.00038,-0.00018] 8/8 | +0.00782 med +0.00859 [+0.00685,+0.00868] tails [+0.00533,+0.00905] 8/8 |
| 3 | Nov3 | -0.00041 med -0.00029 [-0.00149,+0.00065] tails [-0.00266,+0.00170] 4/8 | -0.01072 med -0.01576 [-0.02738,+0.00612] tails [-0.04135,+0.02503] 3/8 |
| 3 | QAOA | -0.00044 med -0.00046 [-0.00054,-0.00030] tails [-0.00062,-0.00003] 8/8 | +0.01525 med +0.01479 [+0.01394,+0.01655] tails [+0.01182,+0.01858] 8/8 |

## Fair Nov3 vs QAOA under S1ALT (not the S1ALT-vs-S1JOINT gate)

Energy-win candidate only if that CI lies entirely below 0.

| L | arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF | energy-win candidate? |
|---|---|---|---|---|
| 2 | S1JOINT | +0.00295 med +0.00334 [+0.00089,+0.00488] tails [-0.00240,+0.00740] 2/8 | +0.00445 med -0.00098 [-0.00428,+0.01464] tails [-0.00959,+0.02817] 3/8 | no |
| 2 | S1ALT | +0.00084 med +0.00236 [-0.00140,+0.00284] tails [-0.00509,+0.00486] 3/8 | +0.00084 med -0.01410 [-0.01717,+0.02204] tails [-0.02784,+0.06203] 3/8 | no |
| 3 | S1JOINT | +0.00263 med +0.00488 [-0.00326,+0.00638] tails [-0.01680,+0.00887] 1/8 | +0.02339 med +0.00921 [-0.00865,+0.07181] tails [-0.02518,+0.18193] 5/8 | no |
| 3 | S1ALT | +0.00265 med +0.00557 [-0.00372,+0.00626] tails [-0.01901,+0.00713] 1/8 | -0.00259 med -0.03299 [-0.03879,+0.05355] tails [-0.04742,+0.18303] 2/8 | no |
