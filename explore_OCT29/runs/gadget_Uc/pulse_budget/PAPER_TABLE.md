# Paper table — scale1000 B=1000 (refreshed)

_Generated 2026-10-31 02:53:57 +0800 (Asia/Shanghai)_

**Metric:** quantum = Med/B90 of **1/p**; WalkSAT = Med/B90 **flips** (and evals). Different units.

Honesty: champ = **L40 pad** (L32 soft-n18 → L36 pad → L40 pad) projector-Uc ≡ `exp(-iγ Hc)`.
n18 val in soft-n18 pool only; pads = interp depth, no retrain. No blind-cost.

| n | **champ L40** Med/B90 | ArmC L16 | QAOA L16 | WalkSAT flips | WalkSAT evals |
|---|----------------------|----------|----------|---------------|---------------|
| 9 | **1.058/1.532** | 1.441/3.314 | 2.242/5.534 | 12.5/40.1 | 51.0/161.4 |
| 12 | **1.129/2.367** | 1.991/6.252 | 3.263/10.868 | 26.5/65.9 | 107.0/264.6 |
| 15 | **1.263/4.563** | 2.921/10.959 | 5.099/18.647 | 43.8/128.1 | 176.0/513.6 |
| 18 | **1.467/10.012** | 4.573/27.810 | 7.894/34.070 | 71.5/203.6 | 287.0/815.4 |

## b fits (9→18)

- champ b_Med≈**1.0371**, b_B90≈**1.2332**
- ArmC b_Med≈**1.1368**
- QAOA b_Med≈**1.1511**
- WalkSAT flips b_Med≈**1.2106**

## vs ArmC / QAOA

| n | champ/ArmC | champ/QAOA |
|---|------------|------------|
| 9 | ×0.734 | ×0.472 |
| 12 | ×0.567 | ×0.346 |
| 15 | ×0.432 | ×0.248 |
| 18 | ×0.321 | ×0.186 |

## Depth pad ladder (soft-n18 lineage)

| L | tag | n15 Med/B90 | n18 Med/B90 | b_Med |
|---|-----|-------------|-------------|-------|
| 32 | soft-n18 w1 | 1.395/5.460 | 1.709/11.107 | 1.0506 |
| 36 | pad | 1.312/4.918 | 1.571/10.312 | 1.0434 |
| **40** | **pad PROMOTED** | **1.263/4.563** | **1.467/10.012** | **1.0371** |

