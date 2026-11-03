# Nov_3rd — RESULTS L=4 (core)

_Asia/Shanghai 2026-11-03 17:18_

Phase: **core**  ns=[6, 9, 12, 15]
Overall vs QAOA (accepted): **UNDER**
Warm-only vs QAOA: **TIE**
Best seed = 0
Warm recovery: no (polish train Med ≤ warm)

QAOA ref: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000.

| n | QAOA Med | QAOA B90 | warm Med | warm B90 | R_warm | warm | acc Med | acc B90 | R_acc | acc |
|---|----------|----------|----------|----------|--------|------|---------|---------|-------|-----|
| 6 | 2.824 | 6.100 | 2.824 | 6.100 | 1.000× | TIE | 2.520 | 5.163 | 0.892× | BEAT |
| 9 | 5.324 | 13.687 | 5.377 | 13.831 | 1.010× | TIE | 5.062 | 12.745 | 0.951× | BEAT |
| 12 | 9.958 | 31.325 | 9.992 | 32.010 | 1.003× | TIE | 10.224 | 31.991 | 1.027× | UNDER |
| 15 | 20.314 | 65.365 | 20.191 | 66.293 | 0.994× | TIE | 21.863 | 74.379 | 1.076× | UNDER |

## STOP
At least one n UNDER QAOA (R>1.02) on accepted ckpt — do not climb L.

