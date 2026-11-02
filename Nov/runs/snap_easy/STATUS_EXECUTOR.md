# SNAP-easy executor status

_Updated 2026-11-02 16:25 Asia/Shanghai_

## Alive
- ladder 714841 · watch 853191 · kerr_polish **893614** (cvar098_n12_from_polish s0 finishing LBFGS → full)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best ≤2-param
`kp_fewm2_cvar095_n12_from_polish_s10_full` bottleneck **1.301×**
n18 gap: Med 1.159× / B90 **1.301×** (14.771 vs sealed 11.353).

## Latest arm
`kp_fewm2_cvar098_n12_from_polish_s10_full` bottleneck **1.334×** (n18 B90 15.140) — **worse** than best.
cvar098 s0 in flight; then s22/33/44/55/66/2 → cvar096 → adam1500 → …

## Ping rule
Ping parent ONLY on ≤2-param strict dual-beat.
