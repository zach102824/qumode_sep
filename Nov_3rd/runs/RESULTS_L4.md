# Nov_3rd — RESULTS L=4 (core)

_Asia/Shanghai 2026-11-03 17:51_

Phase: **core**  ns=[6, 9, 12, 15]
Overall vs QAOA (accepted): **TIE**
Warm-only vs QAOA: **TIE**
Best seed = 0
Warm recovery: **yes** (transfer_under_fallback; exact QAOA dual)

Motif: Nov1st-final X-sandwich + **Hamming-shell SNAP** (QAOA-warm).
QAOA ref: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000.

| n | QAOA Med | QAOA B90 | warm Med | warm B90 | R_warm | warm | acc Med | acc B90 | R_acc | acc |
|---|----------|----------|----------|----------|--------|------|---------|---------|-------|-----|
| 6 | 2.824 | 6.100 | 2.824 | 6.100 | 1.000× | TIE | 2.824 | 6.100 | 1.000× | TIE |
| 9 | 5.324 | 13.687 | 5.377 | 13.831 | 1.010× | TIE | 5.377 | 13.831 | 1.010× | TIE |
| 12 | 9.958 | 31.325 | 9.992 | 32.010 | 1.003× | TIE | 9.992 | 32.010 | 1.003× | TIE |
| 15 | 20.314 | 65.365 | 20.191 | 66.293 | 0.994× | TIE | 20.191 | 66.293 | 0.994× | TIE |

## Transfer-UNDER → ship warm (keep climbing)
Polished ckpt had R>1.02 on at least one CORE n; discarded polish,
shipped warm Hamming-shell SNAP dual (expect TIE). Ladder continues.

