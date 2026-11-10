# N18GEOM query ledger

Prior streams are preserved. Exploratory SK-geometry freeze transfer label, not adoption. GR was not patched. Seeds 10872–10887 exclusive unread (not other-stream 10888–10903; not FT this-hour 10856–71; not PR129 10840–55; not local R35 10824–39; not FT PR127 10808–23; not PR126 10792–10807; not FT PR124 10776–91; not PR125 10760–75; not FT PR122 10744–59; not PR123 10728–43; not local R34 10712–27; not unused-hold 10624–31; not unused hole 10560–67; not PR121 10680–95; not PR119 10648–63; not PR116 10600–15; not holdout 10144–51). Train 10872–79 fitted before freeze; test 10880–87 independent and geom SPSA only after freeze. Geometry freeze is an energy-map projection inside already-charged SPSA (no extra query). No L2/L3 this stream. N=18 not skipped. 4 workers, BLAS 1, peak RSS 265708 KiB.

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
| L1XFER independent + transfer | 32800 | that stream |
| N18XFER independent + transfer | 32800 | that stream |
| N18ALIGN independent + transfer | 32800 | that stream |
| N18TWO independent + transfer | 32824 | that stream |
| N15TWO independent + transfer | 32824 | that stream |
| N12TWO independent + transfer | 32824 | that stream |
| N9TWO independent + transfer | 32824 | that stream |
| N18GEOM independent + geom | 32768 | yes: 32×1024 |
| N18GEOM outside cap | 0 | |
| B* / GS, seeds 10872–10887, N=18 | 0 | classical field prep (wall 184.0 s) |
| post-hoc F rebuild | not Q | from the same state as the charged energy |
| geometry projection | 0 extra | overwrite inside charged energy calls |

N18GEOM fits, all inside 32768 (8 train Nov3 + 8 test Nov3 + 8 test QAOA + 8 test geom):

| split | arm | Q | ×8 | total |
|---|---|---|---|---|
| train | Nov3 independent | 1024 | 8 | 8192 |
| test | Nov3 independent | 1024 | 8 | 8192 |
| test | QAOA independent | 1024 | 8 | 8192 |
| test | Nov3 geom θ_K,θ_M freeze | 1024 | 8 | 8192 |
| all | | | 32 | 32768 |

Category, inclusive:

| category | ×32 | total |
|---|---|---|
| init | 128 | 128 |
| calibration | 1024 | 1024 |
| iteration (perturb) | 31232 | 31232 |
| candidates | 384 | 384 |
| transfer | 0 | 0 |
| screen / monitor | 0 | 0 |
| total | 32768 | 32768 |

Independent and geom per fit: init 4, cal 32, perturb 976, cand 12 = 1024. T = Q/8−6 = 122.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused. 10552–59 not reused. 10560–67 unused. 10624–31 unused. 10840–55 not reused. 10856–71 unused. 10888–10903 unused.

N18GEOM stream total: **32768**. Combined research total if summed: 3756992 + 32768 = 3789760, thirty-two streams, not a revised N3COND or FT number.

Process cost: B* wall 184.0 s; train fit wall 41.8 s; test wall 116.0 s; sum of cell wall 583.3 s; sum of cell CPU 582.9 s; peak RSS 265708 KiB. 32 energy-selected fits, 0 transfer evals, 0 refits. N=18 not skipped; 4 workers, no OOM. Two-template remains closed. Not adoption.
