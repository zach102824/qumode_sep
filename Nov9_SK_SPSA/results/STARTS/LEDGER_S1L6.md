# STARTS S1L6 query ledger

Prior streams are preserved. S1 recipes were not retuned. S4 was fitted and charged on 10064–10071; no reuse from PR82/83/84.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 | 98304 | that stream |
| STARTS S1 (PR83) | 98304 | that stream |
| STARTS S1C (PR84) | 196608 | that stream |
| STARTS S1L6 new fits | 196608 | yes: 32 × 6144 |
| STARTS S1L6 outside cap | 0 | |
| B* / GS, seeds 10064–10071, N15 | 0 | classical field prep (~36 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1L6 categories, all inside the 196608:

| category | S1 (16 fits) | S4 (16 fits) | total |
|---|---|---|---|
| calibration | 128 | 512 | 640 |
| perturbations | 98144 | 97664 | 195808 |
| candidates | 32 | 128 | 160 |
| monitor | 0 | 0 | 0 |
| total | 98304 | 98304 | 196608 |

S1: 8 cal + 2 cand per fit, K=3067, Q=6144. S4: 32 cal + 8 cand per fit, K=763, Q=6144. Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10063 unused.

STARTS S1L6 stream total: **196608**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 = 1692288, seven streams, not a revised N3COND number.

Process cost: B* prep ~36 s classical; SPSA wall ~396 s on 4 workers. 32 energy-selected fits, 0 refits.
