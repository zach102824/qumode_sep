# SNAP-easy executor status

_Updated 2026-11-02 16:56 Asia/Shanghai_

## Alive
- ladder 714841 · watch 853191 · kerr_polish **918955** cvar096_n12_from_polish s22 warm from cvar098-s22 1.296×
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best ≤2-param
`kp_fewm2_cvar098_n12_from_polish_s22_full` bottleneck **1.296×**
n18 B90 14.718 vs sealed 11.353 (1.296×). Improving 1.494→1.389→1.301→**1.296**.

## cvar098 wave (done enough)
s22 **1.296** · s0 1.310 · s10 1.334 · s33 1.342 · s44 (see RESULTS). Skipped remaining s55/66/2; restarted with s22 warm onto next arms.

## In flight
cvar096 → adam1500 → cvar09 → adam1200 → n15 CVaR/Adam (warm=s22)

## Ping rule
Ping parent ONLY on ≤2-param strict dual-beat.
