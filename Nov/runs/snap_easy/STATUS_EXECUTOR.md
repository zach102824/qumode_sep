# SNAP-easy executor status

_Updated 2026-11-03 07:46 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — `kp_fewm2_cvar975_n12n15_long_s22` ~80/1400
- angle polish **624526** (untouched)
- watch **1073382** (explore paused; NEW IDEA in watch arm-list, NOT in current polish argv)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
`kp_fewm2_adam1800_n12n15_s55_full` bottleneck **1.233×** (n18 2.037/13.998) — no dual-beat

## Slow arm DONE (all ≥1.235×)
| tag | n18 Med/B90 | bottleneck |
|-----|-------------|------------|
| adam1800 s55 | 2.037/13.998 | **1.233×** BEST |
| slow s22 | 2.025/14.577 | 1.284× |
| slow s10 | 2.037/14.024 | 1.235× |
| slow s0 | 2.025/14.395 | 1.268× |
| slow s33 | 2.033/14.066 | 1.239× |
| slow s44 | 2.034/14.240 | 1.254× |
| slow s55 | 2.036/14.017 | 1.235× |

## Now / next
cvar975 (6 seeds) → hiLeak (6 seeds) → **kill+relaunch NEW IDEA** → n15-from-polish → n15n18 last

Floor still ~1.23–1.24×. Current polish argv lacks NEW IDEA — relaunch after hiLeak.
