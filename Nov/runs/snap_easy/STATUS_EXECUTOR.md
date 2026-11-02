# SNAP-easy executor status

_Updated 2026-11-02 17:13 Asia/Shanghai_

## Alive
- ladder 714841 · watch 853191 · kerr_polish **928634** (retargeted: adam1500/cvar099/n15 warm from cvar096-s22)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best <=2-param
`kp_fewm2_cvar096_n12_from_polish_s22_full` bottleneck **1.264x**
n18 B90 14.352 vs sealed 11.353. Trend 1.494->1.389->1.301->1.296->**1.264**.
cvar096 s10: 1.287x (kept; not better).

## In flight
adam1500 -> cvar099 -> adam2000 -> cvar098_n12n15_long -> cvar09/adam1200 -> n15 arms (seeds 22,10,0,33)

## Ping rule
Ping parent ONLY on <=2-param strict dual-beat.
