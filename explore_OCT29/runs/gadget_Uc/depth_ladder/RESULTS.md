# Depth ladder — projector-Uc BinH lineage (L16→32)

_Generated 2026-10-31 01:00:43 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. Sealed scale1000 Med/B90; b from log-linear fit on n∈{9,12,15,18}.

## Ladder

| L | tag | n9 | n12 | n15 | n18 | b_Med | b_B90 |
|---|-----|----|-----|-----|-----|-------|-------|
| 16 | L16_train6912 | 1.406/3.288 | 1.864/6.106 | 2.754/10.244 | 4.194/24.302 | 1.1301 | 1.2427 |
| 20 | L20_retrain | 1.261/2.657 | 1.590/4.760 | 2.168/8.296 | 3.170/19.916 | 1.1080 | 1.2460 |
| 24 | L24_retrain | 1.185/2.245 | 1.392/3.865 | 1.820/7.283 | 2.492/17.005 | 1.0868 | 1.2506 |
| 28 | L28_retrain | 1.144/1.962 | 1.317/3.364 | 1.655/6.385 | 2.183/15.419 | 1.0749 | 1.2555 |
| 32 | L32_plain | 1.109/1.828 | 1.258/3.013 | 1.537/5.767 | 1.995/14.117 | 1.0676 | 1.2536 |
| 32 | L32_softn15_t91215 | 1.100/1.759 | 1.231/2.919 | 1.483/5.847 | 1.884/13.704 | 1.0619 | 1.2567 |
| 32 | L32_tail_plain | 1.086/1.799 | 1.192/2.829 | 1.411/5.520 | 1.752/11.813 | 1.0549 | 1.2343 |
| 32 | **L32_lbfgs_seed0** | 1.092/1.776 | 1.197/2.843 | 1.418/5.505 | 1.752/11.353 | 1.0543 | 1.2306 |

## Notes

- Primary depth path: L16→20→24→28→32_plain→soft-n15→tail_plain→**lbfgs_seed0** (champ).
- Extra L32 rows show intermediate checkpoints.
- Sources: `depth/`, `train6912/`, `soft_n15/`, `tail_retrain/`, `lbfgs_polish/` RESULTS.
- No blind-cost / stencil arms.

Files: `ladder.csv`, `ladder.json`.

