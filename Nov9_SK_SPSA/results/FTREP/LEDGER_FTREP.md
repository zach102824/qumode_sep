# FTREP query ledger

Prior streams are preserved. FT holdout 10144–10151 was not run. GR was not patched.

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
| FTREP new fits | 65536 | yes: 32×2048 |
| FTREP outside cap | 0 | |
| B* / GS, seeds 10152–10159, N12 and N15 | 0 | classical field prep (~100 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

FTREP categories, all inside the 65536 (16 fits per N, 8 Nov3 + 8 QAOA):

| category | per fit | ×32 | total |
|---|---|---|---|
| init | 4 | 32 | 128 |
| calibration | 32 | 32 | 1024 |
| iteration (perturb) | 2000 | 32 | 64000 |
| candidates | 12 | 32 | 384 |
| screen / monitor | 0 | 32 | 0 |
| total | 2048 | 32 | 65536 |

T=250 per start, 4 starts. Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10151 unused.

FTREP stream total: **65536**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 + 163840 + 98304 + 245760 + 65536 = 2724480, fourteen streams, not a revised N3COND or FT number.

Process cost: B* prep ~100 s classical; SPSA wall ~29 s on 4 workers. 32 energy-selected fits, 0 refits.
