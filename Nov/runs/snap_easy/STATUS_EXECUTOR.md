# SNAP-easy executor status

_Updated 2026-11-03 00:38 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — `kp_fewm2_cvar099_n12n15_long_s33` (~700/1200)
- angle polish **624526** (untouched)
- watch **1073382** (explore paused)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×**
(near-ties: long_s55 / hiLBFGS_s55 **1.235×**, cvar099_s10 **1.236×**, … — 9 runs in 1.225–1.245).

## Recent
- hiLBFGS DONE — floor ~1.235×
- cvar099: s22 **1.297×**, s10 **1.236×**, s0 **1.285×**; s33 training; s44/s55 remain
- Prepared **NEW IDEA** arms (gauss_project / shared / softplus / shell) — see `NEW_IDEA_2PARAM.md`
- Watch `--arms` updated to run new ideas **after** slow/cvar975/hiLeak (on next relaunch; live PID still on old in-memory list through hiLeak)

## Now / next
finish cvar099 → cvar098/096/adam1800 → slow/cvar975/hiLeak → **NEW IDEA wave** → n15-from_polish
(n15n18 last; explore stays paused)

## Guard
Explore paused. Do not revive heavy explore while polish runs.
