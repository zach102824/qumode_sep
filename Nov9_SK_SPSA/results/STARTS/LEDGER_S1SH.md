# STARTS S1SH query ledger

Prior streams are preserved. S1 recipes were not retuned. S4/S8/continuation were not run. Local R10 seeds 10026–31 and FT 10018–25 unused.

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
| STARTS S1SH new fits | 49152 | yes: 16×1024 + 16×2048 |
| STARTS S1SH outside cap | 0 | |
| B* / GS, seeds 10080–10087, N15 | 0 | classical field prep (~36 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1SH categories, all inside the 49152:

| category | L=1 Q=1024 (16) | L=2 Q=2048 (16) | total |
|---|---|---|---|
| calibration | 128 | 128 | 256 |
| perturbations | 16224 | 32608 | 48832 |
| candidates | 32 | 32 | 64 |
| monitor | 0 | 0 | 0 |
| total | 16384 | 32768 | 49152 |

S1: 8 cal + 2 cand per fit. K=507 (L=1) / 1019 (L=2). Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10079 unused.

STARTS S1SH stream total: **49152**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 = 2085504, nine streams, not a revised N3COND number.

Process cost: B* prep ~36 s classical; SPSA wall ~35 s on 4 workers. 32 energy-selected fits, 0 refits.
