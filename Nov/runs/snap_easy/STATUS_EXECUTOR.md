# SNAP-easy executor status

_Updated 2026-11-03 01:21 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — training `kp_fewm2_cvar098_n12n15_long_s22` (~260/1000)
- angle polish **624526** (untouched)
- watch **1073382** (explore paused)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×**

## Recent
- **cvar099 wave DONE** (all 6 seeds): best near-tie s55/s10 **~1.235–1.236×**; no beat
- Now on cvar098_n12n15_long s22 (train-val briefly improved S→0.655)
- NEW IDEA arms queued after hiLeak (gauss_project / shared / softplus / shell) — `NEW_IDEA_2PARAM.md`

## Now / next
cvar098 → cvar096_long remain → adam1800 → slow/cvar975/hiLeak → **NEW IDEA** → n15-from_polish
(n15n18 last; explore stays paused)

## Guard
Explore paused. Do not revive heavy explore while polish runs.
