# L3S8 query ledger

Prior streams are preserved. Gate FAIL. GR was not patched. Seeds ≤10391 unused (FT gain-ratio and prior LE/FT/local including 10368–75). Local R25 10344–59 unused. Holdout 10144–51 unused as this band. No two-stage freeze-order this stream.

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
| L3SF shell-first | 73728 | that stream |
| L3S8 new fits | 73728 | yes: 24×3072 |
| L3S8 outside cap | 0 | |
| B* / GS, seeds 10400–10407, N=15 | 0 | classical field prep (wall 10.7 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

L3S8 by arm, all inside the 73728 (8 seeds × Q=3072):

| arm | Q | per fit | ×8 | total | T / starts |
|---|---|---|---|---|---|
| C4 Nov3 | 3072 | 3072 | 8 | 24576 | T=378, 4 equal blocks |
| S8 Nov3 | 3072 | 3072 | 8 | 24576 | T=186, 8 equal blocks |
| QAOA p=3 | 3072 | 3072 | 8 | 24576 | T=186, 8 equal blocks (matched) |
| all | | | 24 | 73728 | |

Category (×8), inclusive:

| category | C4 ×8 | S8 ×8 | QAOA ×8 | total |
|---|---|---|---|---|
| init | 32 | 64 | 64 | 160 |
| calibration | 256 | 512 | 512 | 1280 |
| iteration (perturb) | 24192 | 23808 | 23808 | 71808 |
| candidates | 96 | 192 | 192 | 480 |
| screen / monitor | 0 | 0 | 0 | 0 |
| total | 24576 | 24576 | 24576 | 73728 |

C4 per fit: init 4, cal 32, perturb 3024, cand 12 = 3072.
S8/QAOA per fit: init 8, cal 64, perturb 2976, cand 24 = 3072.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds ≤10391 unused here.

L3S8 stream total: **73728**. Combined research total if summed: 3363456 + 73728 = 3437184, twenty-two streams, not a revised N3COND or FT number.

Process cost: B* wall 10.7 s; fit wall 86.7 s; sum of cell wall 326.4 s; sum of cell CPU 326.1 s; peak RSS 193828 KiB. 24 energy-selected fits, 0 refits.
