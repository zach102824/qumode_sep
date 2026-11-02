# SNAP-easy executor status

_Updated 2026-11-03 02:24 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — `kp_fewm2_cvar098_n12n15_long_s44` training (~step 160+/1000)
- angle polish **624526** (untouched)
- watch **1073382** (explore paused)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×**

## cvar098_n12n15_long so far
| seed | n18 Med/B90 | bottleneck |
|------|-------------|------------|
| 22 | 2.004/14.799 | **1.304×** |
| 10 | 2.038/14.037 | **1.236×** (near-tie) |
| 0 | 2.022/14.632 | **1.289×** |
| 33 | 2.028/14.167 | **1.248×** |
| 44 | training | — |
| 55 | pending | — |

Floor still ~1.235; no beat. Running polish **missing NEW IDEA arms** — after slow/cvar975/hiLeak will kill+relaunch so watch injects gproj/shared/softplus/shell (skip-done preserves finished).

## Now / next
finish cvar098 (s44→s55) → cvar096_long remain (s0/33/44/55) → adam1800 → slow/cvar975/hiLeak → **NEW IDEA relaunch** → n15-from_polish
