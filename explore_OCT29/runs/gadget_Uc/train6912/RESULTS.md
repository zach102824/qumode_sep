# Projector-Uc train {6,9,12} probe

_Generated 2026-10-30 19:37:58 +0800 (Asia/Shanghai)_

**keep=True** — n15 Med 2.754 < 2.820*0.98; n18 Med 4.194 < 4.329*0.98

## scale1000 vs baseline projector / Arm C

| n | train6912 | baseline proj | Arm C | vs base |
|---|-----------|---------------|-------|---------|
| 9 | 1.406/3.288 | 1.445/3.222 | 1.441/3.314 | ×1.028 |
| 12 | 1.864/6.106 | 1.926/6.148 | 1.991/6.252 | ×1.033 |
| 15 | 2.754/10.244 | 2.820/10.063 | 2.921/10.959 | ×1.024 |
| 18 | 4.194/24.302 | 4.329/24.342 | 4.573/27.810 | ×1.032 |

## Verdict

PROMOTE — replace unsat_params.pt with unsat_6912_params.pt.

