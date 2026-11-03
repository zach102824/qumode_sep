# SNAP-easy executor status

_Updated 2026-11-03 09:32 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — `kp_fewm2_cvar097_n12n15_hiLeak_s22` just started (~40/1200)
- angle polish **624526** (untouched)
- watch **1073382** (explore paused; NEW IDEA in watch launch arm-list, NOT in live polish argv)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
`kp_fewm2_adam1800_n12n15_s55_full` bottleneck **1.233×** (n18 2.037/13.998) — no dual-beat

## cvar975 DONE (all 6 seeds) — no beat; floor intact
| seed | n18 Med/B90 | bottleneck |
|------|-------------|------------|
| 22 | 2.017/14.671 | 1.292× |
| 10 | 2.038/14.028 | 1.236× |
| 0 | 2.025/14.531 | 1.280× |
| 33 | 2.032/14.099 | 1.242× |
| 44 | 2.039/14.034 | 1.236× |
| 55 | 2.037/14.017 | 1.235× |

## Now / next
hiLeak (6 seeds, 1200 steps) → **kill+relaunch NEW IDEA** (watch arm-list: gproj/shared/softplus/shell; skip-done) → n15-from-polish → n15n18 last

Floor still **1.233×**. Live polish argv still lacks NEW IDEA.
ETA hiLeak ≈ 6×~15 min ≈ 1.5h.
