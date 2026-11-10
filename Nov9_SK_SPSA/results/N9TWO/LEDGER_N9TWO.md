# N9TWO query ledger

Prior streams are preserved. Exploratory two-template confirmatory transfer label, not adoption. GR was not patched. Seeds 10840–10855 exclusive unread (not PR126 10792–10807; not PR125 10760–75; not PR123 10728–43; not FT PR122 10744–59; not FT PR124 10776–91; not FT PR127 10808–23; not local R35 10824–39; not FT this hour 10856–71; not local R34 10712–27; not unused-hold 10624–31; not unused hole 10560–67; not PR121 10680–95; not PR119 10648–63; not PR116 10600–15; not FT 10664–79 / 10696–10711; not R33 10632–47; not PR117 10616–23; not R31/R32/R30; not FT 10520–35 / 10552–59 / 10568–83; not LE 10432–39; not holdout 10144–51). Train 10840–47 fitted before freeze; test 10848–55 and transfer evals only after freeze. Two-template min uses already-charged cluster evals (no extra query). No L2/L3 this stream. N=9 not skipped. Independent-stream confirmation of FT PR122; does not copy FT PR122. 4 workers, BLAS 1, peak RSS 180764 KiB.

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
| N9TWO independent fits | 32768 | yes: 32×1024 |
| N9TWO transfer evals | 56 | yes: 56×1 |
| N9TWO outside cap | 0 | |
| B* / GS, seeds 10840–10855, N=9 | 0 | classical field prep (wall 8.5 s) |
| post-hoc F rebuild | not Q | from the same state as the charged energy |
| two-template min | 0 extra | min of charged cluster-template energies |

N9TWO independent fits, all inside 32768 (16 seeds × 2 methods × Q=1024):

| split | arm | Q | ×8 | total |
|---|---|---|---|---|
| train | Nov3 | 1024 | 8 | 8192 |
| train | QAOA | 1024 | 8 | 8192 |
| test | Nov3 | 1024 | 8 | 8192 |
| test | QAOA | 1024 | 8 | 8192 |
| all independent | | | 32 | 32768 |

Transfer evals, all charged (8 test × 7 vectors × 1; all four clusters nonempty):

| vector | ×8 | total |
|---|---|---|
| Nov3 cluster+ | 8 | 8 |
| Nov3 cluster− | 8 | 8 |
| QAOA cluster+ | 8 | 8 |
| QAOA cluster− | 8 | 8 |
| Nov3 unsigned mean | 8 | 8 |
| Nov3 sign-aligned mean | 8 | 8 |
| QAOA unsigned mean | 8 | 8 |
| all transfer | 56 | 56 |

Category, inclusive:

| category | independent ×32 | transfer ×56 | total |
|---|---|---|---|
| init | 128 | 0 | 128 |
| calibration | 1024 | 0 | 1024 |
| iteration (perturb) | 31232 | 0 | 31232 |
| candidates | 384 | 0 | 384 |
| transfer | 0 | 56 | 56 |
| screen / monitor | 0 | 0 | 0 |
| total | 32768 | 56 | 32824 |

Independent per fit: init 4, cal 32, perturb 976, cand 12 = 1024. T = Q/8−6 = 122.

Failures: 0. Validation 11000–11031 and final 20000–20999 unused. 10552–59 not reused. 10560–67 unused. 10624–31 unused. 10744–59 not reused. 10792–10807 not reused. 10824–39 unused. 10856–71 unused.

N9TWO stream total: **32824**. Combined research total if summed: 3724168 + 32824 = 3756992, thirty-one streams, not a revised N3COND or FT number.

Process cost: B* wall 8.5 s; train fit wall 0.7 s; test+transfer wall 0.7 s; sum of cell wall 1.7 s; sum of cell CPU 1.7 s; peak RSS 180764 KiB. 32 energy-selected independent fits, 56 charged transfer evals, 0 refits. N=9 not skipped; 4 workers, no OOM. After this confirmatory, no further two-template size in this stream without a new parent steer.
