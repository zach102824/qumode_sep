# Nov1st-final angle + SNAP explore — RESULTS

_Updated 2026-11-01 ~23:20 Asia/Shanghai — wave-1 complete; polish wave launched._

Fixed `max_train=200`, train n∈{6,9}. Motif **Nov1st-final**. Ping only on **dual-beat** sealed Med **and** B90 at n9/12/15/18.

## Sealed bar (L=32)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

## Wave-1 closest: progressive shell L32 (scale1000 B=1000)

| n | Med | B90 | Med× | B90× |
|---|-----|-----|------|------|
| 9 | 1.103 | 1.772 | 1.010× | **0.998×** |
| 12 | 1.267 | 2.997 | 1.058× | 1.054× |
| 15 | 1.559 | 5.923 | 1.099× | 1.076× |
| 18 | 2.049 | 15.262 | 1.169× | 1.344× |

n9 B90 already ≤ sealed; Med still 1.010×. **No dual-beat.**

## SNAP bakeoff (full B=1000 on top heads)

Closest bakeoff head by n9 Med: `snap_free_perL` 1.130/1.897 (1.035×/1.068×).  
`snap_bits_perL` n18 B90 10.557 (0.930×) but Med worse. Details: `snap_bakeoff/RESULTS.md`.

## Wave-2 (running)

Polish best shell/free/bits warm-starts: more Adam steps, heavier LBFGS, CVaR / lr sweeps. Outputs under `polish/`. Flag file if beat: `BEAT_SEALED.json`.
