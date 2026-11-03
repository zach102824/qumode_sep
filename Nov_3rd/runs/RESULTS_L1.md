# Nov_3rd — RESULTS L=1 (core)

_Asia/Shanghai 2026-11-03 16:37_

Phase: **core**  ns=[6, 9, 12, 15]
Overall vs QAOA (accepted): **TIE**
Warm-only vs QAOA: **TIE**
Best seed = 2
Warm recovery: **yes** (polish rejected; exact QAOA dual shipped)

QAOA ref: paper init β=0.01 γ=-0.01 nstart=1, train n6×1000.

| n | QAOA Med | QAOA B90 | warm Med | warm B90 | R_warm | warm | acc Med | acc B90 | R_acc | acc |
|---|----------|----------|----------|----------|--------|------|---------|---------|-------|-----|
| 6 | 9.810 | 18.704 | 9.810 | 18.704 | 1.000× | TIE | 9.810 | 18.704 | 1.000× | TIE |
| 9 | 29.309 | 62.636 | 29.309 | 62.636 | 1.000× | TIE | 29.309 | 62.636 | 1.000× | TIE |
| 12 | 91.042 | 214.158 | 91.042 | 214.158 | 1.000× | TIE | 91.042 | 214.158 | 1.000× | TIE |
| 15 | 293.232 | 747.637 | 293.232 | 747.637 | 1.000× | TIE | 293.232 | 747.637 | 1.000× | TIE |

