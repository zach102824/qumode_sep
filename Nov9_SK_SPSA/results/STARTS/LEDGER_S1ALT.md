# STARTS S1ALT query ledger

Prior streams are preserved. Frozen JOINT C0 was not edited. ALT is a new in-cap schedule. S4/S8/continuation were not run.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 | 98304 | that stream |
| STARTS S1 (PR83) | 98304 | that stream |
| STARTS S1C (PR84) | 196608 | that stream |
| STARTS S1L6 (PR85) | 196608 | that stream |
| STARTS S1Q (PR86) | 344064 | that stream |
| STARTS S1SH (PR87) | 49152 | that stream |
| STARTS S1ALT new fits | 65536 | yes: 32 × 2048 |
| STARTS S1ALT outside cap | 0 | |
| B* / GS, seeds 10088–10095, N15 | 0 | classical field prep (~37 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1ALT categories, all inside the 65536:

| category | JOINT (16) | ALT (16) | total |
|---|---|---|---|
| calibration | 128 | 128 | 256 |
| perturbations | 32608 | 32608 | 65216 |
| candidates | 32 | 32 | 64 |
| monitor | 0 | 0 | 0 |
| total | 32768 | 32768 | 65536 |

S1: 8 cal + 2 cand per fit, K=1019. ALT: 510 A steps + 509 B steps, joint_prefix_K=0. Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10087 unused.

STARTS S1ALT stream total: **65536**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 = 2151040, ten streams, not a revised N3COND number.

Process cost: B* prep ~37 s classical; SPSA wall ~52 s on 4 workers. 32 energy-selected fits, 0 refits.
