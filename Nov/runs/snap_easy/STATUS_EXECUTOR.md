# SNAP-easy executor status

_Updated 2026-11-03 00:34 Asia/Shanghai_

## Alive
- kerr_polish **1073389** — training `kp_fewm2_cvar099_n12n15_long_s33` (~60/1200)
- angle polish **624526** (untouched)
- watch **1073382** (explore paused)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×**
(near-ties: long_s55 / hiLBFGS_s55 **1.235×**, cvar099_s10 **1.236×**, hiLBFGS_s10 **1.236×**).

## Recent
- hiLBFGS wave DONE — no beat (best ~1.235×)
- cvar099 so far: s22 **1.297×**, s10 **1.236×**, s0 **1.285×**; s33 training; s44/s55 remain
- Floor stuck ~1.235 across arms/seeds

## Now / next
finish cvar099 → cvar098/096/adam1800 → slow/cvar975/hiLeak → n15-from_polish
(n15n18 last; explore stays paused)

## Guard
Explore paused. Do not revive heavy explore while polish runs.
