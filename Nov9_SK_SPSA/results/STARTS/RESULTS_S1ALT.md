# STARTS S1ALT: block-ALT vs JOINT S1 at N15 L=2 Q=2048

Exclusive unread seeds 10088–10095. Frozen PR83/87 S1 TQA-only C0. Energy selects. F post hoc.
Q=2048 inclusive. K=1019. ALT odd=Uc/γ, even=SNAP/β. No JOINT prefix. GR not patched.

Runs 32 ok 32 failures 0. Charged 65536. Outside cap 0. Rejected B* 0. Missing 0.

Gate N15 L2 Nov3 mean ΔE (ALT−JOINT) = -0.009342 med -0.008744 CI [-0.011320, -0.007583] n=8: **alt_helps_at_L2**.

## ALT vs JOINT

| circuit | Δ(E−E0)/N mean/med [CI] tails wins(ALT) | ΔE | ΔF wins(ALT) |
|---|---|---|---|
| Nov3 | -0.00062 med -0.00058 [-0.00075,-0.00051] tails [-0.00101,-0.00035] 8/8 | -0.00934 med -0.00874 [-0.01132,-0.00758] tails [-0.01508,-0.00522] | -0.01263 med -0.01164 [-0.01682,-0.00867] tails [-0.02282,-0.00301] 0/8 |
| QAOA | -0.00026 med -0.00026 [-0.00032,-0.00021] tails [-0.00043,-0.00016] 8/8 | -0.00388 med -0.00391 [-0.00475,-0.00311] tails [-0.00638,-0.00238] | +0.00890 med +0.00839 [+0.00770,+0.01023] tails [+0.00669,+0.01210] 8/8 |

## Fair Nov3 vs QAOA (not the ALT-vs-JOINT gate)

Nov3 energy-win candidate only if that CI lies entirely below 0.

| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF | energy-win candidate? |
|---|---|---|---|
| JOINT | +0.00292 med +0.00251 [+0.00198,+0.00398] tails [+0.00121,+0.00589] 0/8 | +0.00549 med +0.00418 [-0.00127,+0.01279] tails [-0.00791,+0.02197] 5/8 | no |
| ALT | +0.00255 med +0.00212 [+0.00157,+0.00371] tails [+0.00091,+0.00583] 0/8 | -0.01604 med -0.01269 [-0.02557,-0.00727] tails [-0.04010,+0.00235] 1/8 | no |
