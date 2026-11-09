# STARTS S1 query ledger

N3COND **772224**, LEXACT SPSA **2560**, DEPTH C1 **327680**, and STARTS S8 **98304** are preserved. The 16 S4 files reused from PR82 stay inside the S8 98304 and are not added again.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 (incl. the reused S4) | 98304 | that stream |
| STARTS S1+S2 new fits | 98304 | yes: 32 × 3072 |
| S4 reuse | 0 new | 16 files audited, 49152 not recharged |
| STARTS S1 outside cap | 0 | |
| B* / GS | 0 | classical, already built on these seeds |
| post-hoc F rebuild | not Q | one state per new ok fit |

New STARTS S1 categories, all inside the 98304:

| category | S1 (16 fits) | S2 (16 fits) | total |
|---|---|---|---|
| calibration | 128 | 256 | 384 |
| perturbations | 48992 | 48832 | 97824 |
| candidates | 32 | 64 | 96 |
| monitor | 0 | 0 | 0 |
| total | 49152 | 49152 | 98304 |

S1: 8 cal + 2 cand per fit, K=1531. S2: 16 cal + 4 cand per fit, K=763. Failures: 0. Validation 11000–11031 and final 20000–20999 unused.

STARTS S1 stream total: **98304**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 = 1299272, five streams, not a revised N3COND number.
