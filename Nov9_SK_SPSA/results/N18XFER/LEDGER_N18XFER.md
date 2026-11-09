# N18XFER query ledger

Prior streams are preserved. Exploratory transfer label, not adoption. GR was not patched. Seeds 10648–10663 exclusive unread (not R31 10536–51; not R32 10584–99; not R30 10504–19; not local-next 10632–47; not FT 10520–35 / 10552–59 (PR112) / 10568–83 (PR113) / 10616–23 (PR117) / unused-hold 10624–31 / next 10664–79; not LE 10432–39; not L3TIE 10552–59 (PR115); not PR116 10600–15; not unused hole 10560–67; not holdout 10144–51). Train 10648–55 fitted before freeze; test 10656–63 and transfer evals only after freeze. No L2/L3 this stream. N=18 not skipped. 4 workers, BLAS 1, peak RSS 265532 KiB.

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
| L3TIE tied SNAP | 73728 | that stream |
| L1XFER independent fits | 32768 | that stream |
| L1XFER transfer evals | 32 | that stream |
| N18XFER independent fits | 32768 | yes: 32×1024 |
| N18XFER transfer evals | 32 | yes: 32×1 |
| N18XFER outside cap | 0 | |
| B* / GS, seeds 10648–10663, N=18 | 0 | classical field prep (wall 199.7 s) |
| post-hoc F rebuild | not Q | from the same state as the charged energy |

N18XFER independent fits, all inside 32768 (16 seeds × 2 methods × Q=1024):

| split | arm | Q | ×8 | total |
|---|---|---|---|---|
| train | Nov3 | 1024 | 8 | 8192 |
| train | QAOA | 1024 | 8 | 8192 |
| test | Nov3 | 1024 | 8 | 8192 |
| test | QAOA | 1024 | 8 | 8192 |
| all independent | | | 32 | 32768 |

Transfer evals, all charged (8 test × 2 methods × 2 aggregators × 1):

| vector | ×8 | total |
|---|---|---|
| Nov3 mean | 8 | 8 |
| Nov3 median | 8 | 8 |
| QAOA mean | 8 | 8 |
| QAOA median | 8 | 8 |
| all transfer | 32 | 32 |

Category, inclusive:

| category | independent ×32 | transfer ×32 | total |
|---|---|---|---|
| init | 128 | 0 | 128 |
| calibration | 1024 | 0 | 1024 |
| iteration (perturb) | 31232 | 0 | 31232 |
| candidates | 384 | 0 | 384 |
| transfer | 0 | 32 | 32 |
| screen / monitor | 0 | 0 | 0 |
| total | 32768 | 32 | 32800 |

Independent per fit: init 4, cal 32, perturb 976, cand 12 = 1024. T = Q/8−6 = 122.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused. 10552–59 not reused. 10560–67 unused.

N18XFER stream total: **32800**. Combined research total if summed: 3560096 + 32800 = 3592896, twenty-six streams, not a revised N3COND or FT number.

Process cost: B* wall 199.7 s; train fit wall 76.7 s; test+transfer wall 76.0 s; sum of cell wall 539.5 s; sum of cell CPU 538.9 s; peak RSS 265532 KiB. 32 energy-selected independent fits, 32 charged transfer evals, 0 refits. N=18 not skipped; 4 workers, no OOM.
