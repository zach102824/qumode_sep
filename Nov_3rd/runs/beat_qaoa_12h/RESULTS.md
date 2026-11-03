# beat_qaoa_12h RESULTS

_Updated 2026-11-03 21:36 Asia/Shanghai_

**Locked:** Hamming-shell SNAP + Nov1st X-sandwich (training-only).  
**BEAT** = R≤0.98 vs fair paper-n6 QAOA · scale1000 B=1000.

---

## Deep L=16 — CLEAR BEAT (confirmed 3 seeds)

Idea1 CORE-RATIO train{6,9,12,15}, Adam→LBFGS, poly (c1,c2,c3).

| seed | n6 R | n9 R | n12 R | n15 R |
|------|------|------|-------|-------|
| 0 | **0.942** | **0.898** | **0.885** | **0.882** |
| 1 | **0.944** | **0.908** | **0.888** | **0.886** |
| 2 | **0.946** | **0.900** | **0.901** | **0.900** |

All CORE BEAT on every seed. Champion: `champion/params_L16.pt` (seed0).

---

## Shallow — n12 BEAT; n15 capacity-limited ~0.983–0.985

### L=2 best tables
| Method | n6 | n9 | n12 | n15 |
|--------|----|----|-----|-----|
| CORE-RATIO s0 | **0.959** | **0.978** | **0.976** | 0.993 |
| Minimax polish s1 | **0.960** | **0.978** | **0.979** | 0.984 |
| free-φ weighted | **0.960** | **0.970** | 0.986 | 0.983 |
| **n15-only probe** | — | — | **0.976** | train **0.985** / eval 0.993 |

**Capacity note:** optimizing free φ+γ on n15 train alone cannot push train R below ~0.985 at L=2 → Hamming-shell appears capacity-limited for n15 BEAT at shallow depth.

### L=1 CORE-RATIO s0
| n | R | verdict |
|---|---|---------|
| 6 | **0.979** | BEAT |
| 9 | 0.988 | TIE |
| 12 | **0.979** | BEAT |
| 15 | 0.984 | TIE |

---



### Global search L=2 shell on n15 train (Opus capacity check)
- Method: 801 scrambled-uniform starts over (γ,c1,c2,c3) ± box around QAOA dual; L-BFGS refine top 20.
- **best_train_R=0.9807** (capacity_limited=False vs kill 0.982; NOT ≤0.978 reopen bar).
- scale1000 eval of that point:

| n | R | verdict |
|---|---|---------|
| 6 | **0.971** | BEAT |
| 9 | 0.983 | TIE |
| 12 | **0.977** | BEAT |
| 15 | 0.997 | TIE |

Train scrape does **not** transfer to eval n15. Best *eval* n15 remains ~0.983 (free-φ weighted). Artifacts: `round_2/global_search_n15_L2/`.

## Success gate
| Goal | Status |
|------|--------|
| BEAT n12+n15 @ L=16 | **PASS** (3/3 seeds) |
| BEAT n12+n15 @ L=1 or 2 | **OPEN** — n12 yes; n15 stuck ~0.983–0.985 (likely capacity) |

## Round 0
Dual identity PASS R=1.000 at L∈{1,2,16} after refreshing stale QAOA ladders.


---

## Closing check DECLARE — shallow n15 capacity (2026-11-03 22:21 Asia/Shanghai)

Dense disjoint n15 pack N=1000; global search → **best_dense_R=0.9803** (`capacity_proven=true`, reopen=false).
Per Opus R3: stop shallow-n15 training hunt. Best *eval* n15 remains ~0.983 (free-φ). L=16 clear beat stands.
