# FTREP H query ledger

Prior streams are preserved. This stream is an intentional replay of FT holdout 10144–10151, not a collision and not a read of validation/final. GR was not patched. Local R19 10184–10191 was not run.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 | 98304 | that stream |
| STARTS S1 | 98304 | that stream |
| STARTS S1C | 196608 | that stream |
| STARTS S1L6 | 196608 | that stream |
| STARTS S1Q | 344064 | that stream |
| STARTS S1SH | 49152 | that stream |
| STARTS S1ALT | 65536 | that stream |
| STARTS S1ALT23 | 163840 | that stream |
| STARTS S1ALT13 | 98304 | that stream |
| DEPTH S1W | 245760 | that stream |
| FTREP N12/N15 L2 | 65536 | that stream |
| FTREP N18 L2 | 32768 | that stream |
| FTREP L3 | 98304 | that stream |
| FTREP H new fits | 65536 | yes: 32×2048 (N15 required + N18 optional) |
| FTREP H outside cap | 0 | |
| B* / GS, seeds 10144–10151, N15 and N18 | 0 | classical field prep (N15 wall 10.4 s; N18 wall 103.6 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

FTREP H categories, all inside the 65536:

| category | per fit | ×32 | total |
|---|---|---|---|
| init | 4 | 32 | 128 |
| calibration | 32 | 32 | 1024 |
| iteration (perturb) | 2000 | 32 | 64000 |
| candidates | 12 | 32 | 384 |
| screen / monitor | 0 | 32 | 0 |
| total | 2048 | 32 | 65536 |

T=250 per start, 4 starts. Failures: 0. N15: 16 fits, 32768 queries (Gate R). N18: 16 fits, 32768 queries (optional report). Validation 11000–11031 and final 20000–20999 unused.

FTREP H stream total: **65536**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 + 163840 + 98304 + 245760 + 65536 + 32768 + 98304 + 65536 = 2921088, seventeen streams, not a revised N3COND or FT number.

Process cost: N15 B* 10.4 s + fit 28.5 s; N18 B* 103.6 s + fit 259.9 s; total wall 402.4 s on 4 workers; sum of cell wall 1026.7 s; sum of cell CPU 1025.7 s; peak RSS 265460 KiB. 32 energy-selected fits, 0 refits.
