# Nov_3rd — RESULTS L=12 (core)

_Asia/Shanghai 2026-11-03 20:04_

Phase: **core**  ns=[6, 9, 12, 15]
Overall vs QAOA (accepted): **TIE**
Warm-only vs QAOA: **TIE**
Best seed = 0
Warm recovery: **yes** (transfer_under_fallback; exact QAOA dual)

Motif: Nov1st-final X-sandwich + **Hamming-shell SNAP** (QAOA-warm).
QAOA ref: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000.

| n | QAOA Med | QAOA B90 | warm Med | warm B90 | R_warm | warm | acc Med | acc B90 | R_acc | acc |
|---|----------|----------|----------|----------|--------|------|---------|---------|-------|-----|
| 6 | 1.355 | 2.905 | 1.355 | 2.905 | 1.000× | TIE | 1.355 | 2.905 | 1.000× | TIE |
| 9 | 1.865 | 5.028 | 1.865 | 5.028 | 1.000× | TIE | 1.865 | 5.028 | 1.000× | TIE |
| 12 | 2.620 | 9.653 | 2.620 | 9.653 | 1.000× | TIE | 2.620 | 9.653 | 1.000× | TIE |
| 15 | 3.924 | 16.983 | 3.924 | 16.983 | 1.000× | TIE | 3.924 | 16.983 | 1.000× | TIE |

## Transfer-UNDER → ship warm (keep climbing)
Polished ckpt had R>1.02 on at least one CORE n; discarded polish,
shipped warm Hamming-shell SNAP dual (expect TIE). Ladder continues.

