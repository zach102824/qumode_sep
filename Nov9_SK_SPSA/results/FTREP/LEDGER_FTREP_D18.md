# FTREP D18 query ledger

Prior streams are preserved. Characterization, not adoption. GR was not patched. L=2 and L=3 finished before L=4 started. Local R23 10304–19 unused. Holdout 10144–51 unused as this band.

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
| FTREP D18 new fits | 147456 | yes: 16×2048 + 16×3072 + 16×4096 |
| FTREP D18 outside cap | 0 | |
| B* / GS, seeds 10320–10327, N=18 | 0 | classical field prep (wall 106.6 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

FTREP D18 by depth, all inside the 147456 (8 Nov3 + 8 QAOA per L):

| L | Q | per fit | ×16 | total | role |
|---|---|---|---|---|---|
| 2 | 2048 | 2048 | 16 | 32768 | required |
| 3 | 3072 | 3072 | 16 | 49152 | required |
| 4 | 4096 | 4096 | 16 | 65536 | optional, finished after L2/L3 |
| all | | | 48 | 147456 | |

| category | L2 ×16 | L3 ×16 | L4 ×16 | total |
|---|---|---|---|---|
| init | 64 | 64 | 64 | 192 |
| calibration | 512 | 512 | 512 | 1536 |
| iteration (perturb) | 32000 | 48384 | 64768 | 145152 |
| candidates | 192 | 192 | 192 | 576 |
| screen / monitor | 0 | 0 | 0 | 0 |
| total | 32768 | 49152 | 65536 | 147456 |

T = Q/8−6 (250 / 378 / 506). Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds ≤10319 unused here.

FTREP D18 stream total: **147456**. Combined research total if summed: 3068544 + 147456 = 3216000, nineteen streams, not a revised N3COND or FT number.

Process cost: B* wall 106.6 s; L=2,3 fit wall 781.3 s; L=4 fit wall 908.3 s; total wall 1689.6 s on 4 workers; sum of cell wall 6182.2 s; sum of cell CPU 6176.4 s; peak RSS 269568 KiB. 48 energy-selected fits, 0 refits. L=4 finished without truncating L=2/L=3.
