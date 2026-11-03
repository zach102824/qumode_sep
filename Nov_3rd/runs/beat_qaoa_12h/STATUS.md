# beat_qaoa_12h STATUS

**Updated:** 2026-11-03 20:51 Asia/Shanghai  
**Phase:** R0 PASS; R1 in progress

## Round 0 — PASS (L∈{1,2,16} dual R=1.000)

## Round 1
### Idea2 RULE-SCALED L=2 seed0 — DONE
κ≈1.107 | train {9,12}

| n | R | verdict |
|---|---|---------|
| 6 | 0.987 | TIE |
| 9 | 0.980 | TIE |
| 12 | **0.979** | **BEAT** |
| 15 | 0.999 | TIE |

→ BEAT n12 but n15 not yet ≤0.98 (holdout soft).

### Still running
- Idea1 CORE-RATIO L=2
- Idea3 TRUST-RESIDUAL L=2 ρ=0.1
- Idea1 CORE-RATIO L=16 (restarted batch_per_n=8)
