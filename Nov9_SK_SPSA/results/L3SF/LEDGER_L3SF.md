# L3SF query ledger

Prior streams are preserved. Gate FAIL. GR was not patched. D18 10320–27 not rerun. L3TS 10360–67 unused here. FT pair-phase 10328–35 unused. Local R25 10344–59 unused. 10376+ reserved unused. Holdout 10144–51 unused as this band. Center-4 depth-curve sampling stopped. No L2 center-4 copies.

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
| L3TS joint-then-shell | 73728 | that stream |
| L3SF new fits | 73728 | yes: 24×3072 |
| L3SF outside cap | 0 | |
| B* / GS, seeds 10368–10375, N=15 | 0 | classical field prep (wall 10.3 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

L3SF by arm, all inside the 73728 (8 seeds × Q=3072):

| arm | Q | per fit | ×8 | total | T / starts |
|---|---|---|---|---|---|
| JOINT | 3072 | 3072 | 8 | 24576 | T=378, 4 equal blocks, all params |
| SHELL_FIRST | 3072 | 3072 | 8 | 24576 | T_shell=186 (4 starts, Uc frozen) + T_uc=378 (2 blocks, shells frozen) |
| QAOA p=3 | 3072 | 3072 | 8 | 24576 | T=378, 4 equal blocks |
| all | | | 24 | 73728 | |

SHELL_FIRST category (×8), inclusive:

| category | shell ×8 | Uc ×8 | SHELL_FIRST total | JOINT ×8 | QAOA ×8 |
|---|---|---|---|---|---|
| init | 32 | 16 | 48 | 32 | 32 |
| calibration | 256 | 128 | 384 | 256 | 256 |
| iteration (perturb) | 11904 | 12096 | 24000 | 24192 | 24192 |
| candidates | 96 | 48 | 144 | 96 | 96 |
| screen / monitor | 0 | 0 | 0 | 0 | 0 |
| total | 12288 | 12288 | 24576 | 24576 | 24576 |

JOINT/QAOA per fit: init 4, cal 32, perturb 3024, cand 12 = 3072.
SHELL_FIRST per fit: init 6, cal 48, perturb 3000, cand 18 = 3072.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds ≤10367 unused here. 10376+ unused.

L3SF stream total: **73728**. Combined research total if summed: 3289728 + 73728 = 3363456, twenty-one streams, not a revised N3COND or FT number.

Process cost: B* wall 10.3 s; fit wall 84.8 s; sum of cell wall 320.8 s; sum of cell CPU 320.5 s; peak RSS 190716 KiB. 24 energy-selected fits, 0 refits.
