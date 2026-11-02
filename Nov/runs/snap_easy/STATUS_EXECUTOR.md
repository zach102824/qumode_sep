# SNAP-easy executor status

_Updated 2026-11-02 21:53 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — training `kp_fewm2_cvar097_n12n15_long_s44` (n12n15-first relaunch)
- angle polish **624526** (untouched)
- watch **1073382** (restarted with new arm order; explore paused)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×**.

## n15n18 cvar097 wave — DONE, no beat (tiny-val overfit)
| seed | bottleneck | n18 B90× |
|------|------------|----------|
| s10 | **1.331×** | 1.331 |
| s0 | **1.369×** | 1.369 |
| s33 | **1.380×** | 1.380 |
| s22 | **1.386×** | 1.386 |
| s44 | **1.396×** | 1.396 |

Killed mid-`cvar098_n15n18` after wave; relaunched **n12n15-first**.

## Now / next
1. `cvar097_n12n15_long` s44 (+s55) — was missing
2. `cvar096_n12n15_hiLBFGS`, `cvar099_n12n15_long`, `cvar098_n12n15_long`, …
3. New anti-overfit: `cvar097_n12n15_slow`, `cvar975_n12n15_long`, `cvar097_n12n15_hiLeak`
4. n15-from_polish arms, then leftover n18-in-train (max_val=24 now)

## Guard
Explore paused. Do not revive heavy explore while polish runs.
