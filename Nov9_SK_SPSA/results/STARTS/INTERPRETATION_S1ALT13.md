# STARTS S1ALT13 — 1:3 does not beat PR88 1:1 at L=2; GR unpatched

PR88/89 showed 1:1 block-ALT beats joint IID S1 at L=2. This batch asked whether a **shell-heavy 1:3** duty cycle (one shared Uc/γ step, then three SNAP/β steps) beats that frozen 1:1 schedule at the same N15 L=2 Q=2048. Seeds **10104–10111**. No S1/ALT11 retune. No extra budget. Energy selects. F diagnostic only.

**Preregistered 1:3 phase.** Start at A. `k%4==0` updates A; otherwise B. Leftover K=1019 steps continue that phase: k=1016..1018 = A,B,B (255 A / 764 B). Not dumped to shells. No free prefix.

**Gate fails. Do not patch GR.**

Nov3 mean ΔE (S1ALT13−S1ALT11) = **+0.00146** med +0.00092, CI **[−0.00022, +0.00358]**, tails [−0.00218, +0.00785], S1ALT13 lower **2/8**. The CI is not entirely below 0. The point estimate favors 1:1.

Versus S1JOINT (reported, not the gate): both ALT schedules still lower Nov3 energy (S1ALT11 8/8, S1ALT13 7/8; both CIs entirely below 0). 1:3 does not add on top of 1:1.

**Fair Nov3 vs QAOA under S1ALT13 is not an energy-win candidate.**

| arm | Δ(E−E0)/N Nov3−QAOA mean [CI] | Nov3 lower | energy-win candidate? |
|---|---|---|---|
| S1JOINT | +0.00315 [+0.0023, +0.0040] | 0/8 | no |
| S1ALT11 | +0.00252 [+0.0016, +0.0035] | 0/8 | no |
| S1ALT13 | +0.00262 [+0.0017, +0.0037] | 0/8 | no |

All three CIs lie entirely above 0. Shell-heavy 1:3 does not reverse the L=2 energy loss to QAOA.

48/48 fits ok. B* rejects 0. Charged 98304. Outside-cap 0. Frozen hashes matched including unmodified alt_spsa.py. Prior streams unchanged. No S4/S8, no continuation, no extra L3 ALT stacking, no N18, no GR patch. Stop after this matrix.
