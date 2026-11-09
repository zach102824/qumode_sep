# STARTS S1Q query ledger

Prior streams are preserved. S1 recipes were not retuned. S4/S8 were not run.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 | 98304 | that stream |
| STARTS S1 (PR83) | 98304 | that stream |
| STARTS S1C (PR84) | 196608 | that stream |
| STARTS S1L6 (PR85) | 196608 | that stream |
| STARTS S1Q new fits | 344064 | yes: 16×3072 + 16×6144 + 16×12288 |
| STARTS S1Q outside cap | 0 | |
| B* / GS, seeds 10072–10079, N15 | 0 | classical field prep (~35 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1Q categories, all inside the 344064:

| category | Q=3072 (16) | Q=6144 (16) | Q=12288 (16) | total |
|---|---|---|---|---|
| calibration | 128 | 128 | 128 | 384 |
| perturbations | 48992 | 98144 | 196448 | 343584 |
| candidates | 32 | 32 | 32 | 96 |
| monitor | 0 | 0 | 0 | 0 |
| total | 49152 | 98304 | 196608 | 344064 |

S1: 8 cal + 2 cand per fit. K=1531 / 3067 / 6139. Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10071 unused.

STARTS S1Q stream total: **344064**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 = 2036352, eight streams, not a revised N3COND number.

Process cost: B* prep ~35 s classical; SPSA wall ~56 + 114 + 227 = 397 s on 4 workers, checkpointed after 6144. 48 energy-selected fits, 0 refits. Q=12288 was not a resource blocker.
