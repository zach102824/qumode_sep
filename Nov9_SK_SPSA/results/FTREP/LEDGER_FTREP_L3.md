# FTREP L3 query ledger

Prior streams are preserved. FT holdout 10144–10151 was not run. Local R19 10184–10191 was not run. GR was not patched. FTREP L2 N12/N15 (65536) and N18 (32768) are separate streams.

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
| FTREP L3 new fits | 98304 | yes: 32×3072 |
| FTREP L3 outside cap | 0 | |
| B* / GS, seeds 10192–10199, N12 and N15 | 0 | classical field prep (wall 28.2 s, 4 workers) |
| post-hoc F rebuild | not Q | one state per ok fit |

FTREP L3 categories, all inside the 98304 (16 fits per N, 8 Nov3 + 8 QAOA):

| category | per fit | ×32 | total |
|---|---|---|---|
| init | 4 | 32 | 128 |
| calibration | 32 | 32 | 1024 |
| iteration (perturb) | 3024 | 32 | 96768 |
| candidates | 12 | 32 | 384 |
| screen / monitor | 0 | 32 | 0 |
| total | 3072 | 32 | 98304 |

T=378 per start, 4 starts. Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds ≤10191 unused here. FT holdout 10144–10151 unused. Local R19 10184–10191 unused.

FTREP L3 stream total: **98304**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 + 163840 + 98304 + 245760 + 65536 + 32768 + 98304 = 2855552, sixteen streams, not a revised N3COND or FT number.

Process cost: B* prep wall 28.2 s; SPSA fit wall 59.5 s on 4 workers; sum of cell wall 219.5 s; sum of cell CPU 219.3 s; peak RSS 188636 KiB. 32 energy-selected fits, 0 refits.
