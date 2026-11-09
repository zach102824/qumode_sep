# STARTS S8 query ledger

N3COND **772224**, LEXACT SPSA **2560**, and DEPTH C1 **327680** are preserved. Nothing here is moved into those numbers.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 fits | 98304 | yes: 32 × 3072 |
| STARTS S8 outside cap | 0 | |
| B* / GS, seeds 10048–10055 | 0 | classical field prep |
| post-hoc F rebuild | not Q | one state per ok fit, not used to select |

STARTS S8 categories, all inside the 98304:

| category | S4 (16 fits) | S8 (16 fits) | total |
|---|---|---|---|
| calibration | 512 | 1024 | 1536 |
| perturbations | 48512 | 47872 | 96384 |
| candidates | 128 | 256 | 384 |
| monitor | 0 | 0 | 0 |
| total | 49152 | 49152 | 98304 |

S4: 32 cal + 8 cand per fit, K=379. S8: 64 cal + 16 cand per fit, K=187. Failures: 0. Forbidden bands 10010–10047 / 11000–11031 / 20000–20999: unused.

STARTS S8 stream total: **98304**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 = 1200968, reported as four streams, not as a revised N3COND number.
