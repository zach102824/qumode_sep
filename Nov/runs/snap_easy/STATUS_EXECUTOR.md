# SNAP-easy executor status

_Updated 2026-11-02 14:41 Asia/Shanghai_

## Alive
- ladder 714841 (wave 0, gauss_plus1 full B=1000; nice +10)
- watch 838921 (FEWM n12/n15 relaunch enabled)
- kerr_polish 775868 (legacy CVaR/adam n6+n9 arms; mid adam800 s2)
- few_m n12 side polish **821866** (adam900 s44 full-eval; then cvar095/09/08)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best ≤2-param
`kp_fewm2_adam900_n12_from_best_s0_full` bottleneck **1.494×** (n18 B90).
s33 full: **1.510×** (no improve). s44 in flight.

## Next arms (queued in watch relaunch)
Polish-chain warm from adam900 best:
- `fewm2_cvar095_n12_from_polish`, `fewm2_cvar09_n12_from_polish`, `fewm2_adam1200_n12_from_polish`
- `fewm2_cvar095_n15_from_polish`, `fewm2_adam900_n15_from_polish`, `fewm2_cvar09_n15_from_polish`
seeds 0,33,44,55,66

## Ping rule
Ping parent ONLY on ≤2-param strict dual-beat.
