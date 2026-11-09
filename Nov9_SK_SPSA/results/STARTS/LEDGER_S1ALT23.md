# STARTS S1ALT23 query ledger

Prior streams are preserved. S1JOINT and PR88 ALT recipes were not retuned.

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
| STARTS S1ALT (PR88) | 65536 | that stream |
| STARTS S1ALT23 new fits | 163840 | yes: 32×2048 + 32×3072 |
| STARTS S1ALT23 outside cap | 0 | |
| B* / GS, seeds 10096–10103, N15 | 0 | classical field prep (~36 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1ALT23 categories, all inside the 163840:

| category | L=2 (32) | L=3 (32) | total |
|---|---|---|---|
| calibration | 256 | 256 | 512 |
| perturbations | 65216 | 97984 | 163200 |
| candidates | 64 | 64 | 128 |
| monitor | 0 | 0 | 0 |
| total | 65536 | 98304 | 163840 |

S1: 8 cal + 2 cand per fit. L=2 K=1019 (510/509 A/B). L=3 K=1531 (766/765 A/B). Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10095 unused.

STARTS S1ALT23 stream total: **163840**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 + 163840 = 2314880, eleven streams, not a revised N3COND number.

Process cost: B* prep ~36 s classical; SPSA wall ~158 s on 4 workers. 64 energy-selected fits, 0 refits.
