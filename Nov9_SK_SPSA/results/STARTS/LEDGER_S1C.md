# STARTS S1C query ledger

Prior streams are preserved. This confirmation does not reuse 10048–55 S4 files.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 | 98304 | that stream |
| STARTS S1 (PR83) | 98304 | that stream |
| STARTS S1C new fits | 196608 | yes: 64 × 3072 |
| STARTS S1C outside cap | 0 | |
| B* / GS, seeds 10056–10063, N12 and N15 | 0 | classical field prep (~100 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1C categories, all inside the 196608:

| category | S1 (32 fits) | S4 (32 fits) | total |
|---|---|---|---|
| calibration | 256 | 1024 | 1280 |
| perturbations | 97984 | 97024 | 195008 |
| candidates | 64 | 256 | 320 |
| monitor | 0 | 0 | 0 |
| total | 98304 | 98304 | 196608 |

S1: 8 cal + 2 cand per fit, K=1531. S4: 32 cal + 8 cand per fit, K=379. Failures: 0. Validation 11000–11031 and final 20000–20999 unused.

STARTS S1C stream total: **196608**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 = 1495680, six streams, not a revised N3COND number.

Process cost: B* prep ~100 s classical; SPSA wall ~114 s on 4 workers. 64 energy-selected fits, 0 refits.
