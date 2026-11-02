# SNAP-easy executor status

_Updated 2026-11-02 18:20 Asia/Shanghai_

## Alive
- ladder 714841 · watch **957583** · kerr_polish **957591** (cvar097_n12n15_long s10 training)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best <=2-param
`kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235x** (n18 B90 14.015 / sealed 11.353).
Prior best was 1.264x. Trend …→1.301→1.296→1.264→**1.235**.
Near: cvar096_n12_from_polish_s22 1.264x · cvar096_n12n15_long_s10 1.265x.

## In flight
cvar097 seeds 10,0,33,44 then hiLBFGS / cvar099 / n9n15 / adam1800 / n15.
Will rechain warm from 1.235x ckpt after s10 (prefs patched; need polish restart to pick up).

## Ping rule
Ping parent ONLY on <=2-param strict dual-beat.
