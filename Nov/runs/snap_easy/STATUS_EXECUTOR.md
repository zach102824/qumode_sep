# SNAP-easy executor status

_Updated 2026-11-02 15:12 Asia/Shanghai_

## Alive
- ladder 714841 · watch 853191 · kerr_polish **852435** (polish-chain)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best ≤2-param
`kp_fewm2_cvar095_n12_from_polish_s0_full` bottleneck **1.389×** (was 1.494× adam900).
n18 still the gap: Med 1.191× / B90 **1.389×** (B90=15.774 vs sealed 11.353).

## In flight
- `fewm2_cvar095_n12_from_polish` s2 full-eval; then seeds 10,22,33,44,55
- then cvar09/adam1200 n12_from_polish + n15 CVaR/Adam arms

## Ping rule
Ping parent ONLY on ≤2-param strict dual-beat.
