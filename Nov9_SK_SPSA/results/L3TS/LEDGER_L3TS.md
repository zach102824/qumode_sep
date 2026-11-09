# L3TS query ledger

Prior streams are preserved. Gate FAIL. GR was not patched. Local R25 10344+ unused. FT pair-phase 10328–35 unused. Holdout 10144–51 unused as this band. Center-4 depth-curve sampling stopped.

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
| FTREP D N15 curve | 147456 | that stream |
| FTREP D18 N18 curve | 147456 | that stream |
| L3TS new fits | 73728 | yes: 24×3072 |
| L3TS outside cap | 0 | |
| B* / GS, seeds 10360–10367, N=15 | 0 | classical field prep (wall 9.6 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

L3TS by arm, all inside the 73728 (8 seeds × Q=3072):

| arm | Q | per fit | ×8 | total | T / starts |
|---|---|---|---|---|---|
| JOINT | 3072 | 3072 | 8 | 24576 | T=378, 4 equal blocks, all params |
| TWO_STAGE | 3072 | 3072 | 8 | 24576 | T_joint=186 (4 starts) + T_shell=378 (2 blocks) |
| QAOA p=3 | 3072 | 3072 | 8 | 24576 | T=378, 4 equal blocks |
| all | | | 24 | 73728 | |

TWO_STAGE category (×8), inclusive:

| category | joint ×8 | shell ×8 | TWO_STAGE total | JOINT ×8 | QAOA ×8 |
|---|---|---|---|---|---|
| init | 32 | 16 | 48 | 32 | 32 |
| calibration | 256 | 128 | 384 | 256 | 256 |
| iteration (perturb) | 11904 | 12096 | 24000 | 24192 | 24192 |
| candidates | 96 | 48 | 144 | 96 | 96 |
| screen / monitor | 0 | 0 | 0 | 0 | 0 |
| total | 12288 | 12288 | 24576 | 24576 | 24576 |

JOINT/QAOA per fit: init 4, cal 32, perturb 3024, cand 12 = 3072.
TWO_STAGE per fit: init 6, cal 48, perturb 3000, cand 18 = 3072.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds ≤10359 unused here.

L3TS stream total: **73728**. Combined research total if summed: 3216000 + 73728 = 3289728, twenty streams, not a revised N3COND or FT number.

Process cost: B* wall 9.6 s; fit wall 86.5 s; sum of cell wall 326.7 s; sum of cell CPU 326.4 s; peak RSS 190692 KiB. 24 energy-selected fits, 0 refits.
