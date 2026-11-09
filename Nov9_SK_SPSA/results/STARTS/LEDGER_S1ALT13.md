# STARTS S1ALT13 query ledger

Prior streams are preserved. S1JOINT and PR88 S1ALT11 recipes were not retuned.

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
| STARTS S1ALT23 (PR89) | 163840 | that stream |
| STARTS S1ALT13 new fits | 98304 | yes: 48×2048 |
| STARTS S1ALT13 outside cap | 0 | |
| B* / GS, seeds 10104–10111, N15 | 0 | classical field prep (~34 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1ALT13 categories, all inside the 98304:

| category | S1JOINT (16) | S1ALT11 (16) | S1ALT13 (16) | total |
|---|---|---|---|---|
| calibration | 128 | 128 | 128 | 384 |
| perturbations | 32608 | 32608 | 32608 | 97824 |
| candidates | 32 | 32 | 32 | 96 |
| monitor | 0 | 0 | 0 | 0 |
| total | 32768 | 32768 | 32768 | 98304 |

S1: 8 cal + 2 cand per fit. K=1019. S1ALT11 510/509 A/B. S1ALT13 255/764 A/B leftover ABB. Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10103 unused.

STARTS S1ALT13 stream total: **98304**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 + 163840 + 98304 = 2413184, twelve streams, not a revised N3COND number.

Process cost: B* prep ~34 s classical; SPSA wall ~77 s on 4 workers. 48 energy-selected fits, 0 refits.
