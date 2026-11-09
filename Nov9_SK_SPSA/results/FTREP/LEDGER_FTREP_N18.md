# FTREP N18 query ledger

Prior streams are preserved. FT holdout 10144–10151 was not run. GR was not patched. FTREP N12/N15 (65536) is a separate stream.

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
| FTREP N12/N15 | 65536 | that stream |
| FTREP N18 new fits | 32768 | yes: 16×2048 |
| FTREP N18 outside cap | 0 | |
| B* / GS, seeds 10168–10175, N=18 | 0 | classical field prep (wall 98.8 s, 4 workers) |
| post-hoc F rebuild | not Q | one state per ok fit |

FTREP N18 categories, all inside the 32768 (8 Nov3 + 8 QAOA):

| category | per fit | ×16 | total |
|---|---|---|---|
| init | 4 | 16 | 64 |
| calibration | 32 | 16 | 512 |
| iteration (perturb) | 2000 | 16 | 32000 |
| candidates | 12 | 16 | 192 |
| screen / monitor | 0 | 16 | 0 |
| total | 2048 | 16 | 32768 |

T=250 per start, 4 starts. Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds ≤10167 unused here. FT holdout 10144–10151 unused.

FTREP N18 stream total: **32768**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 + 163840 + 98304 + 245760 + 65536 + 32768 = 2757248, fifteen streams, not a revised N3COND or FT number.

Process cost: B* prep wall 98.8 s (sum of per-seed field_prep ≈ 381 s); SPSA fit wall 258.1 s on 4 workers; sum of cell wall 919.3 s; sum of cell CPU 918.5 s; peak RSS 265652 KiB. 16 energy-selected fits, 0 refits.
