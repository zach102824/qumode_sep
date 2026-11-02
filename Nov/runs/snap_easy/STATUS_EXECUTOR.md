# SNAP-easy executor status

_Updated 2026-11-02 16:40 Asia/Shanghai_

## Alive
- ladder 714841 · watch 853191 · kerr_polish **893614** (cvar098 s33 train)
- angle polish 624526 (untouched)
- STOP_SNAP_EASY: absent
- BEAT_SEALED_2PARAM: absent

## Best ≤2-param
`kp_fewm2_cvar098_n12_from_polish_s22_full` bottleneck **1.296×** (was 1.301×)
n18 gap: Med ~1.16× / B90 **1.296×** (14.718 vs sealed 11.353).
cvar098: s22 **1.296** · s0 1.310 · s10 1.334.

## In flight
cvar098 remaining s33/44/55/66/2 → cvar096 → adam1500 → cvar09 → adam1200 → n15

## Note
Script FEWM_POLISH_CKPTS now prefers s22; live process still uses in-memory list until next restart (first_existing will pick s22 file once listed — **needs restart after cvar098 wave or mid-arm for warm to flip**). Will restart onto cvar096 with s22 warm after cvar098 seed wave or sooner if plateaus.

## Ping rule
Ping parent ONLY on ≤2-param strict dual-beat.
