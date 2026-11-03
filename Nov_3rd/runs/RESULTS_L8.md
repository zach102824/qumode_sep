# Nov_3rd — RESULTS L=8 (core)

_Asia/Shanghai 2026-11-03 18:33_

Phase: **core**  ns=[6, 9, 12, 15]
Overall vs QAOA (accepted): **UNDER**
Warm-only vs QAOA: **UNDER**
Best seed = 0
Warm recovery: **yes** (transfer_under_fallback; exact QAOA dual)

Motif: Nov1st-final X-sandwich + **Hamming-shell SNAP** (QAOA-warm).
QAOA ref: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000.

| n | QAOA Med | QAOA B90 | warm Med | warm B90 | R_warm | warm | acc Med | acc B90 | R_acc | acc |
|---|----------|----------|----------|----------|--------|------|---------|---------|-------|-----|
| 6 | 1.678 | 3.780 | 1.678 | 3.780 | 1.000× | TIE | 1.678 | 3.780 | 1.000× | TIE |
| 9 | 2.396 | 7.169 | 2.543 | 7.599 | 1.061× | UNDER | 2.543 | 7.599 | 1.061× | UNDER |
| 12 | 3.649 | 14.375 | 3.953 | 14.980 | 1.083× | UNDER | 3.953 | 14.980 | 1.083× | UNDER |
| 15 | 5.966 | 27.257 | 6.559 | 27.368 | 1.099× | UNDER | 6.559 | 27.368 | 1.099× | UNDER |

## Transfer-UNDER → ship warm (keep climbing)
Polished ckpt had R>1.02 on at least one CORE n; discarded polish,
shipped warm Hamming-shell SNAP dual (expect TIE). Ladder continues.

