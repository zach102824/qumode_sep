# L3TIE query ledger

Prior streams are preserved. Gate FAIL. GR was not patched. Seeds 10552–10559 exclusive unread (not R31 10536–51; not R30 10504–19; not FT 10520–35 / 10568+; not LE 10432–39; not L3S8 10400–07; not holdout 10144–51). 10560–67 held unused. Not start-count / freeze-order / shell-unshare / two-stage this stream.

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
| L3S8 eight starts vs C4 | 73728 | that stream |
| N21L1 N=21 L=1 | 16384 | that stream |
| L3TIE new fits | 73728 | yes: 24×3072 |
| L3TIE outside cap | 0 | |
| B* / GS, seeds 10552–10559, N=15 | 0 | classical field prep (wall 9.4 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

L3TIE by arm, all inside the 73728 (8 seeds × Q=3072):

| arm | Q | per fit | ×8 | total | T / dim |
|---|---|---|---|---|---|
| JOINT Nov3 | 3072 | 3072 | 8 | 24576 | T=378, dim 12, 4 equal blocks |
| TIE Nov3 | 3072 | 3072 | 8 | 24576 | T=378, SPSA dim 6, circuit dim 12 |
| QAOA p=3 | 3072 | 3072 | 8 | 24576 | T=378, dim 6, 4 equal blocks |
| all | | | 24 | 73728 | |

Category (×8), inclusive:

| category | JOINT ×8 | TIE ×8 | QAOA ×8 | total |
|---|---|---|---|---|
| init | 32 | 32 | 32 | 96 |
| calibration | 256 | 256 | 256 | 768 |
| iteration (perturb) | 24192 | 24192 | 24192 | 72576 |
| candidates | 96 | 96 | 96 | 288 |
| screen / monitor | 0 | 0 | 0 | 0 |
| total | 24576 | 24576 | 24576 | 73728 |

Per fit: init 4, cal 32, perturb 3024, cand 12 = 3072. T = Q/8−6 = 378.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused. 10560–67 unused here.

L3TIE stream total: **73728**. Combined research total if summed: 3453568 + 73728 = 3527296, twenty-four streams, not a revised N3COND or FT number.

Process cost: B* wall 9.4 s; fit wall 83.7 s; sum of cell wall 314.2 s; sum of cell CPU 314.0 s; peak RSS 190280 KiB. 24 energy-selected fits, 0 refits.
