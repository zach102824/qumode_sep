# L=36 pad-only — **PROMOTED**

_Generated 2026-10-31 02:50:09 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. Interp soft-n18 L32→36, **no retrain**.

Gate: n15 **and** n18 Med ≤98% of soft-n18 → thr 1.367 / 1.675. Both pass (1.312 / 1.571).

| n | soft-n18 L32 | **L36 pad** | ΔMed |
|---|--------------|-------------|------|
| 9 | 1.098/1.719 | **1.072/1.613** | -2.4% |
| 12 | 1.197/2.779 | **1.156/2.532** | -3.4% |
| 15 | 1.395/5.460 | **1.312/4.918** | -5.9% |
| 18 | 1.709/11.107 | **1.571/10.312** | -8.1% |

b_Med≈**1.0434**, b_B90≈**1.2308**.

Params: `unsat_params.pt`. Backup soft-n18: `unsat_params_pre_L36_backup.pt`.
