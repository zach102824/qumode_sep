# SNAP-easy executor status

_Updated 2026-11-02 16:15 Asia/Shanghai_

## Alive
- ladder 714841 · watch 853191 · kerr_polish **893614** (cvar098_n12_from_polish s10 train, warm=s10 1.301×)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best ≤2-param
`kp_fewm2_cvar095_n12_from_polish_s10_full` bottleneck **1.301×**
n18 gap: Med 1.159× / B90 **1.301×** (14.771 vs sealed 11.353).
cvar095 wave done (s0/2/10/22/33/44/55); s55 bottleneck 1.361× (worse).

## Restart note
Stale pid 852435 (pre-ARMS-patch) skipped cvar098/096/adam1500 → killed after s55 full.
Relaunched 893614 onto patched priority arms with live s10 warm.

## In flight
- **fewm2_cvar098_n12_from_polish** s10 (then 0,22,33,44,55,66,2)
- then cvar096 → adam1500 → cvar09 → adam1200 → n15 CVaR/Adam

## Ping rule
Ping parent ONLY on ≤2-param strict dual-beat.
