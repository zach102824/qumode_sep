# STARTS S1ALT — block-ALT beats JOINT at L=2; still not a GR patch

S1SH found a Nov3 energy-win at L=1 and a loss already at L=2 under joint IID S1. This batch asked whether **block-alternating SPSA** at the L1→L2 onset, with frozen S1 start/gains/readout and the same inclusive Q=2048, lowers Nov3 energy versus joint IID. Seeds **10088–10095**. Odd steps update shared Uc (Nov3) or all γ (QAOA); even steps update SNAP shells or all β. Calibration is the C0 4-probe full vector, charged. No JOINT K prefix.

**Gate passes. Do not patch GR.** Nov3 mean(E_ALT − E_JOINT) bootstrap 95% CI is entirely below 0:

| cell | mean ΔE | median | CI | ALT energy wins |
|---|---|---|---|---|
| N15 L2 Nov3 | −0.0093 | −0.0087 | [−0.0113, −0.0076] | 8/8 |

Δ(E−E0)/N = −0.00062, CI [−0.00075, −0.00051], 8/8. QAOA ALT also beat JOINT (mean ΔE −0.0039, CI [−0.0048, −0.0031], 8/8). Nov3 F is *lower* under ALT than JOINT (mean ΔF −0.013, 0/8). Energy selects; F does not override.

**Fair Nov3 vs QAOA under ALT is a different claim, and energy does not support Nov3 superiority.**

| arm | Δ(E−E0)/N Nov3−QAOA mean [CI] | Nov3 lower | energy-win candidate? |
|---|---|---|---|
| JOINT | +0.00292 [+0.00198, +0.00398] | 0/8 | no |
| ALT | +0.00255 [+0.00157, +0.00371] | 0/8 | no |

ALT shrinks the Nov3–QAOA energy gap slightly versus JOINT but does not reverse it. Both CIs lie entirely above 0.

32/32 fits ok. B* rejects 0. Charged 65536. Outside-cap 0. Frozen hashes matched. ALT 510/509 A/B steps, joint_prefix_K=0. Prior streams unchanged. No S4/S8, no continuation, no extra budget, no N18, no GR patch. Stop after this matrix.
