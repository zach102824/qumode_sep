# Round 0 dual identity (aggregate)

_Updated 2026-11-03 20:47:30 +0800 Asia/Shanghai_

**Overall (so far):** PENDING L[16]
**Hash disjoint:** PASS

## Per L

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

## Notes

- L16 QAOA results.json was STALE (legacy ladder kept after paper-n6 retrain). Force-refreshed in R0.
- L2 QAOA was non-paper; archived to L2_legacy_nonpaper, retrained paper-n6, ladder refreshed.
- ensure_qaoa.py patched: retrain now forces full CORE re-eval.
