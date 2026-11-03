# beat_qaoa_12h STATUS

**Updated:** 2026-11-03 20:49 Asia/Shanghai  
**Phase:** Round 0 **PASS**; Round 1 L=2 trio training; Idea1 L=16 queued

## Round 0 — PASS
| L | n6 R | n9 R | n12 R | n15 R |
|---|------|------|-------|-------|
| 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 16 | 1.000000 | 1.000004 | ~1.000 | 1.000005 |

Hash disjoint: PASS. Stale L2/L16 QAOA ladders force-refreshed; `ensure_qaoa.py` patched.

Fresh fair QAOA Med (paper-n6):  
L2: 5.326 / 12.642 / 29.606 / 76.088  
L16: 1.226 / 1.579 / 2.071 / 2.888

## Round 1 (running)
- Idea1 CORE-RATIO L=2 seed0 — Adam in progress
- Idea2 RULE-SCALED L=2 seed0 — Adam ~500/800
- Idea3 TRUST-RESIDUAL L=2 seed0 ρ=0.1 — Adam in progress
- Idea1 L=16 seed0 — starts after R0 L16 (now PASS)
