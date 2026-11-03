# Round 1 RESULTS

_Updated 2026-11-03 20:54 Asia/Shanghai_

Eval: scale1000 B=1000 vs fair paper-n6 QAOA. BEAT = R≤0.98.

## L=2 seed0 summary

| Idea | n6 R | n9 R | n12 R | n15 R | n12+n15 clear? |
|------|------|------|-------|-------|----------------|
| 1 CORE-RATIO | **0.959 BEAT** | **0.978 BEAT** | **0.976 BEAT** | 0.993 TIE | n12 yes, n15 no |
| 2 RULE-SCALED | 0.987 TIE | 0.980 TIE | **0.979 BEAT** | 0.999 TIE | n12 yes, n15 no |
| 3 TRUST-RESIDUAL ρ=0.1 | **0.973 BEAT** | 0.992 TIE | 0.990 TIE | 0.995 TIE | no |

**Best shallow so far:** Idea1 CORE-RATIO (beats n6/9/12; n15 short by ~1.3%).

### Idea1 CORE-RATIO L=2 detail
| n | QAOA Med | Med | R | verdict |
|---|----------|-----|---|---------|
| 6 | 5.326 | 5.105 | 0.9586 | BEAT |
| 9 | 12.642 | 12.365 | 0.9781 | BEAT |
| 12 | 29.606 | 28.896 | 0.9760 | BEAT |
| 15 | 76.088 | 75.546 | 0.9929 | TIE |

### Idea2 RULE-SCALED L=2 (κ≈1.107)
| n | R | verdict |
|---|---|---------|
| 6 | 0.987 | TIE |
| 9 | 0.980 | TIE |
| 12 | 0.979 | BEAT |
| 15 | 0.999 | TIE |

### Idea3 TRUST-RESIDUAL L=2 (ρ=0.1)
| n | R | verdict |
|---|---|---------|
| 6 | 0.973 | BEAT |
| 9 | 0.992 | TIE |
| 12 | 0.990 | TIE |
| 15 | 0.995 | TIE |

## L=16
Idea1 CORE-RATIO seed0 training (batch_per_n=8) — in progress.

## Notes
- Gate sequence unchanged; training-only.
- R0 dual identity PASS at L∈{1,2,16} after refreshing stale QAOA ladders.
