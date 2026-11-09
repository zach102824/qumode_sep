# N21L1 query ledger

Prior streams are preserved. Characterization estimate, not adoption. GR was not patched. Seeds 10432–10439 exclusive unread (not local R26 10416–31; not LE ≤10407; not FT ≤10415 / 10368–75; not holdout 10144–51; not validation 11000–11031; not GR/final 20000–20999). No L2+. No L=3 start-count / freeze-order / shell-unshare this stream.

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
| N21L1 new fits | 16384 | yes: 16×1024 |
| N21L1 outside cap | 0 | |
| B* / GS, seeds 10432–10439, N=21 | 0 | classical field prep (wall 3015.0 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

N21L1 by arm, all inside the 16384 (8 seeds × Q=1024):

| arm | Q | per fit | ×8 | total | T / starts |
|---|---|---|---|---|---|
| Nov3 L=1 | 1024 | 1024 | 8 | 8192 | T=122, 4 equal blocks |
| QAOA p=1 | 1024 | 1024 | 8 | 8192 | T=122, 4 equal blocks |
| all | | | 16 | 16384 | |

Category (×8), inclusive:

| category | Nov3 ×8 | QAOA ×8 | total |
|---|---|---|---|
| init | 32 | 32 | 64 |
| calibration | 256 | 256 | 512 |
| iteration (perturb) | 7808 | 7808 | 15616 |
| candidates | 96 | 96 | 192 |
| screen / monitor | 0 | 0 | 0 |
| total | 8192 | 8192 | 16384 |

Per fit: init 4, cal 32, perturb 976, cand 12 = 1024. T = Q/8−6 = 122.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused.

N21L1 stream total: **16384**. Combined research total if summed: 3437184 + 16384 = 3453568, twenty-three streams, not a revised N3COND or FT number.

Process cost: B* wall 3015.0 s on 2 workers (no fallback); fit wall 1244.1 s on 2 workers (no fallback); total wall 4260 s; sum of cell wall 2479.2 s; sum of cell CPU 2479.0 s; peak RSS 1245060 KiB (B*), 951028 KiB (fits). 16 energy-selected fits, 0 refits. Serial 1-worker fallback was coded and not needed.
