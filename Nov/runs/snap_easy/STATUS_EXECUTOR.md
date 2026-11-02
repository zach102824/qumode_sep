# SNAP-easy executor status

_Updated 2026-11-02 18:05 Asia/Shanghai_

## Alive
- ladder 714841 · watch **957583** · kerr_polish **957591** (cvar097_n12n15_long s22 just started)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best <=2-param
`kp_fewm2_cvar096_n12_from_polish_s22_full` bottleneck **1.264x** (n18 B90 14.352 / sealed 11.353).
Near-tie: `cvar096_n12n15_long_s10` **1.265x** · `cvar096_n12n15_long_s22` **1.270x**.
Trend 1.494->1.389->1.301->1.296->**1.264**.

## In flight (n18 B90 push wave)
cvar097_n12n15_long → hiLBFGS → cvar099_n12n15 → n9n15 → cvar097_n15 → adam1800_n12n15 → prior n15/cvar098
seeds 22,10,0,33,44. Warm from 1.264x polish. Gate uses largest train-val n.

## Ping rule
Ping parent ONLY on <=2-param strict dual-beat.
