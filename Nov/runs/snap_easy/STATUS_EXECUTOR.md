# SNAP-easy executor status

_Updated 2026-11-02 20:42 Asia/Shanghai_

## Alive
- kerr_polish **1009137** — training `kp_fewm2_cvar097_n15n18_s0` (~step 80/900); LBFGS off for 18∈train
- angle polish **624526** (untouched)
- watch **1002328** (explore paused; OOM guard on)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param (completed full)
Still `kp_fewm2_cvar097_n12n15_long_s22_full` bottleneck **1.235×**.

## n15n18 wave (full B=1000)
| seed | bottleneck | n18 B90× | note |
|------|------------|----------|------|
| s22 | **1.386×** | 1.386 | tiny-val B90~6 → full 15.74 |
| s10 | **1.331×** | 1.331 | better than s22; still worse than 1.235 |
| s0 | training | — | in progress |
| s33,s44 | queued | — | then cvar098 / n12n15n18 / n18_only |

## Fix / guard
LBFGS disabled when 18∈train_ns. Explore paused (EXPLORE_PAUSED). RAM ~3.4 GiB avail — do not revive explore.
