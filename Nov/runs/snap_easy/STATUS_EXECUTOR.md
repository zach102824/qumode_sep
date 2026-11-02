# SNAP-easy executor status

_Updated 2026-11-02 21:08 Asia/Shanghai_

## Alive
- kerr_polish **1009137** — training `kp_fewm2_cvar097_n15n18_s33` (~step 80/900); LBFGS off for 18∈train
- angle polish **624526** (untouched)
- watch **1002328** (explore paused; OOM guard on)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×**.

## n15n18 cvar097 wave (full B=1000) — NO beat; tiny-val overfits
| seed | bottleneck | n18 B90× | note |
|------|------------|----------|------|
| s10 | **1.331×** | 1.331 | best of arm so far |
| s0 | **1.369×** | 1.369 | |
| s22 | **1.386×** | 1.386 | |
| s33 | training | — | |
| s44 | queued | — | then cvar098_n15n18, n12n15n18, n18_only, …

Tiny-val n18 B90 ~6.0–6.4 ≠ scale1000 B90 ~15–15.7.

## Guard
Explore paused. RAM currently ~6.8 GiB avail (recovered mid-eval) — still do not revive heavy explore while n18-in-train polish runs.
