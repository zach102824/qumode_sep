# Paper table — scale1000 B=1000 (refreshed)

_Generated 2026-10-31 02:50:09 +0800 (Asia/Shanghai)_

**Metric:** quantum = Med/B90 of **1/p**; WalkSAT = Med/B90 **flips** (and evals). Different units.

Honesty: champ = **L36 pad** from soft-n18 L32 projector-Uc ≡ `exp(-iγ Hc)`;
n18 val was in soft-n18 pool; L36 is depth pad only (no extra train). scale1000 holdout. No blind-cost.

| n | **champ L36** Med/B90 | ArmC L16 | QAOA L16 | WalkSAT flips | WalkSAT evals |
|---|----------------------|----------|----------|---------------|---------------|
| 9 | **1.072/1.613** | 1.441/3.314 | 2.242/5.534 | 12.5/40.1 | 51.0/161.4 |
| 12 | **1.156/2.532** | 1.991/6.252 | 3.263/10.868 | 26.5/65.9 | 107.0/264.6 |
| 15 | **1.312/4.918** | 2.921/10.959 | 5.099/18.647 | 43.8/128.1 | 176.0/513.6 |
| 18 | **1.571/10.312** | 4.573/27.810 | 7.894/34.070 | 71.5/203.6 | 287.0/815.4 |

## b fits (9→18)

- champ b_Med≈**1.0434**, b_B90≈**1.2308**
- ArmC b_Med≈**1.1368**
- QAOA b_Med≈**1.1511**
- WalkSAT flips b_Med≈**1.2106**

## vs ArmC / QAOA (champ Med ratio)

| n | champ/ArmC | champ/QAOA |
|---|------------|------------|
| 9 | ×0.744 | ×0.478 |
| 12 | ×0.581 | ×0.354 |
| 15 | ×0.449 | ×0.257 |
| 18 | ×0.344 | ×0.199 |

## Lineage

LBFGS seed0 → soft-n18 w1 L32 → **L36 pad (promoted)**.
Multi-seed LBFGS on L32 soft-n18: no promote (flat).

