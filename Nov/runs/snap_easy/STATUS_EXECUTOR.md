# SNAP-easy executor status

_Updated 2026-11-02 22:24 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — training `kp_fewm2_cvar096_n12n15_hiLBFGS_s22`
- angle polish **624526** (untouched)
- watch **1073382** (explore paused; EXPLORE_PAUSED present)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×** (s55 tied 1.235×).

## cvar097_n12n15_long — DONE (all 6 seeds)
| seed | bottleneck | n18 B90× |
|------|------------|----------|
| s22 | **1.235×** | 1.235 |
| s55 | **1.235×** | 1.235 |
| s44 | 1.237× | 1.237 |
| s33 | 1.245× | 1.245 |
| s10 | 1.265× | 1.265 |
| s0 | 1.289× | 1.289 |

## Now / next
1. `cvar096_n12n15_hiLBFGS` (in progress s22)
2. `cvar099/098/096_n12n15_long`, `adam1800_n12n15`
3. Anti-overfit: **`slow`**, **`cvar975`**, **`hiLeak`**
4. n15-from_polish / leftover n18-in-train

## Guard
Explore paused. Do not revive heavy explore while polish runs.
