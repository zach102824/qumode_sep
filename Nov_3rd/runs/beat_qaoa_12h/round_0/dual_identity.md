# Round 0 dual identity — PASS

_Finished 2026-11-03 20:49:51 +0800 Asia/Shanghai_

**Overall:** **PASS**
**Hash disjoint:** **PASS**

### L=1 — **PASS**

| n | QAOA Med | warm Med | R | verdict |
|---|----------|----------|---|---------|
| 6 | 9.8097 | 9.8097 | 1.000000 | PASS |
| 9 | 29.3085 | 29.3085 | 1.000000 | PASS |
| 12 | 91.0422 | 91.0422 | 1.000000 | PASS |
| 15 | 293.2323 | 293.2324 | 1.000000 | PASS |

### L=2 — **PASS**

| n | QAOA Med | warm Med | R | verdict |
|---|----------|----------|---|---------|
| 6 | 5.3257 | 5.3257 | 1.000000 | PASS |
| 9 | 12.6416 | 12.6416 | 1.000000 | PASS |
| 12 | 29.6063 | 29.6063 | 1.000000 | PASS |
| 15 | 76.0878 | 76.0879 | 1.000001 | PASS |

### L=16 — **PASS**

| n | QAOA Med | warm Med | R | verdict |
|---|----------|----------|---|---------|
| 6 | 1.2261 | 1.2261 | 1.000002 | PASS |
| 9 | 1.5787 | 1.5787 | 1.000004 | PASS |
| 12 | 2.0707 | 2.0707 | 1.000004 | PASS |
| 15 | 2.8881 | 2.8881 | 1.000005 | PASS |

## Notes

- R0 PASS dual identity L∈{1,2,16} after refreshing stale L2/L16 QAOA ladders
- Fresh L16 QAOA Med: n6=1.226 n9=1.579 n12=2.071 n15=2.888

