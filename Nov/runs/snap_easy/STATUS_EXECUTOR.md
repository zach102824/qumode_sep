# SNAP-easy executor status

_Updated 2026-11-02 18:51 Asia/Shanghai_

## Alive
- ladder 714841 · watch **974174** · kerr_polish **974182** (cvar097_n12n15_long s33 training; warm=1.235x)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best <=2-param
`kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235x** (n18 B90 14.015 / sealed 11.353).
Trend …→1.301→1.296→1.264→**1.235**. Need ≤1.000 on all Med+B90.
- s0 of same arm: **1.289x** (worse; n18 B90 14.629)

## In flight / next
- Finishing cvar097 seeds 33,44 then remaining filter arms on live PID.
- Patched ARMS + watch: **n18-in-train** arms (`cvar097/098/099_n15n18`, `cvar096_n12n15n18`, `adam1200_n15n18`, `cvar097_n18_only`) will run on next polish relaunch (skip-done preserves finished fulls).
- Plan: after s33/s44 full evals, soft-restart polish so n18 arms start immediately.

## Ping rule
Ping parent ONLY on <=2-param strict dual-beat.
