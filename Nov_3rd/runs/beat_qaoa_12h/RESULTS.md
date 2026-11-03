# beat_qaoa_12h RESULTS

_Updated 2026-11-03 21:02 Asia/Shanghai_

**Locked:** Hamming-shell SNAP + Nov1st X-sandwich; training-only.  
**BEAT** = R≤0.98 vs fair paper-n6 QAOA, scale1000 B=1000.

## Champion so far

### Deep L=16 — CLEAR BEAT (Idea1 CORE-RATIO seed0)
| n | QAOA Med | Med | R | verdict |
|---|----------|-----|---|---------|
| 6 | 1.226 | 1.156 | **0.942** | **BEAT** |
| 9 | 1.579 | 1.417 | **0.898** | **BEAT** |
| 12 | 2.071 | 1.833 | **0.885** | **BEAT** |
| 15 | 2.888 | 2.547 | **0.882** | **BEAT** |

Params: `round_1/idea1_core_ratio/params_L16_seed0.pt`

### Shallow L=2 — BEAT n6/9/12; n15 still ~0.98–0.99 TIE
Best Idea1 seed0: n12 **0.976 BEAT**, n15 0.993 TIE.  
Minimax polish best seed1: n12 **0.979 BEAT**, n15 0.984 TIE.

### Shallow L=1 — BEAT n6/12; n9/n15 TIE
n12 **0.979 BEAT**, n15 0.984 TIE.

## Success gate
| Requirement | Status |
|-------------|--------|
| BEAT n12+n15 at L=16 | **PASS** |
| BEAT n12+n15 at L=1 or 2 | **OPEN** (n15 ~0.984–0.993) |

## Round 0
Dual identity PASS R=1.000 at L∈{1,2,16} (after QAOA ladder refresh).
