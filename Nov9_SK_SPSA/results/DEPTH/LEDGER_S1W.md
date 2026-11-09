# DEPTH S1W query ledger

Prior streams are preserved. S1JOINT was not retuned. DEPTH C1 recipes were not edited.

| block | queries | inside Q |
|---|---|---|
| N3COND (closed) | 772224 | that stream |
| LEXACT M1–M3 SPSA | 2560 | that stream |
| DEPTH C1 | 327680 | that stream |
| STARTS S8 | 98304 | that stream |
| STARTS S1 (PR83) | 98304 | that stream |
| STARTS S1C (PR84) | 196608 | that stream |
| STARTS S1L6 (PR85) | 196608 | that stream |
| STARTS S1Q (PR86) | 344064 | that stream |
| STARTS S1SH (PR87) | 49152 | that stream |
| STARTS S1ALT (PR88) | 65536 | that stream |
| STARTS S1ALT23 (PR89) | 163840 | that stream |
| STARTS S1ALT13 (PR90) | 98304 | that stream |
| DEPTH S1W new fits | 245760 | yes: 48×2048 + 48×3072 |
| DEPTH S1W outside cap | 0 | |
| B* / GS, seeds 10128–10135, N12 and N15 | 0 | classical field prep (~101 s) |
| post-hoc F rebuild | not Q | one state per ok fit |

S1W categories, all inside the 245760 (32 fits per arm: 16 at L=2 Q=2048, 16 at L=3 Q=3072):

| arm | L=2 queries | L=3 queries | cal | perturb | cand | total |
|---|---|---|---|---|---|---|
| COLD | 16×2048 | 16×3072 | 256 | 81600 | 64 | 81920 |
| QAOA | 16×2048 | 16×3072 | 256 | 81600 | 64 | 81920 |
| WARM | 16×2048 | 16×3072 | 512 | 80896 | 128 | 81920 |
| total | 98304 | 147456 | 1024 | 244096 | 256 | 245760 |

WARM L1 is 1024/fit (8 cal + 1014 perturb + 2 cand, K=507). Continue is Q−1024 (L=2: K=507; L=3: K=1019). Failures: 0. Validation 11000–11031 and final 20000–20999 unused. Seeds 10010–10127 unused.

DEPTH S1W stream total: **245760**. Combined research total if summed: 772224 + 2560 + 327680 + 98304 + 98304 + 196608 + 196608 + 344064 + 49152 + 65536 + 163840 + 98304 + 245760 = 2658944, thirteen streams, not a revised N3COND number.

Process cost: B* prep ~101 s classical; SPSA wall ~128 s on 4 workers. 96 energy-selected fits, 0 refits.
