# SNAP-easy executor status

_Updated 2026-11-03 10:57 Asia/Shanghai_

## Live
- kerr_polish **1504756** — NEW IDEA arms first (`fewm2_gproj_cvar097_n12n15_s22` training)
- angle polish **624526** (untouched)
- watch **1073382** (explore paused)
- STOP / BEAT_SEALED_2PARAM: absent

## Best ≤2-param
`kp_fewm2_adam1800_n12n15_s55_full` **1.233×** — no dual-beat

## hiLeak
All 6 seeds done; best ~1.235× (floor intact). Killed same-basin CVaR grind → relaunched NEW IDEA.

## L=1 vs QAOA (done)
3p gauss fresh L=1 (`gauss_L1_cvar08_gauss_trunc_s10`) vs QAOA L=1:
| n | QAOA | 3p gauss | gauss÷QAOA |
|---|------|----------|------------|
| 9 | 29.3 | 55.9 | 1.91× |
| 12 | 90.0 | 249.2 | 2.77× |
| 15 | 289.9 | 1093.0 | 3.77× |
| 18 | 920.5 | 4342.4 | 4.72× |
Writeup: `Nov/runs/snap_easy/gauss_L1_vs_qaoa/RESULTS.md`
