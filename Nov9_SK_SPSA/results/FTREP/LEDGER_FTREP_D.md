# FTREP D query ledger

Prior streams are preserved. Characterization, not adoption. GR was not patched. Holdout 10144–10151 was not re-run here. Local R19 10184–10191 unused. FT N18×16 10232–47 unused.

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
| FTREP H holdout replay | 65536 | that stream |
| FTREP D new fits | 147456 | yes: 16×2048 + 16×3072 + 16×4096 |
| FTREP D outside cap | 0 | |
| B* / GS, seeds 10264–10271, N=15 | 0 | classical field prep (wall 9.9 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

FTREP D by depth, all inside the 147456 (8 Nov3 + 8 QAOA per L):

| L | Q | per fit | ×16 | total |
|---|---|---|---|---|
| 2 | 2048 | 2048 | 16 | 32768 |
| 3 | 3072 | 3072 | 16 | 49152 |
| 4 | 4096 | 4096 | 16 | 65536 |
| all | | | 48 | 147456 |

| category | L2 ×16 | L3 ×16 | L4 ×16 | total |
|---|---|---|---|---|
| init | 64 | 64 | 64 | 192 |
| calibration | 512 | 512 | 512 | 1536 |
| iteration (perturb) | 32000 | 48384 | 64768 | 145152 |
| candidates | 192 | 192 | 192 | 576 |
| screen / monitor | 0 | 0 | 0 | 0 |
| total | 32768 | 49152 | 65536 | 147456 |

T = Q/8−6 (250 / 378 / 506). Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds ≤10263 unused here.

FTREP D stream total: **147456**. Combined research total if summed: 2921088 + 147456 = 3068544, eighteen streams, not a revised N3COND or FT number.

Process cost: B* wall 9.9 s; fit wall 166.7 s on 4 workers; sum of cell wall 626.4 s; sum of cell CPU 625.9 s; peak RSS 193044 KiB. 48 energy-selected fits, 0 refits. L=4 finished; not downscoped.
