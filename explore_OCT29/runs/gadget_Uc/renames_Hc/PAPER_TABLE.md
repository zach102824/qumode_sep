# Paper mega-table — scale1000 B=1000 (L44 finalize)

_Generated 2026-10-31 03:59:56 +0800 (Asia/Shanghai)_

**Metric:** quantum = Med/B90 of **1/p**; WalkSAT = flips/evals (different unit).

Honesty: champ = **L44 pad** projector-Uc ≡ `exp(-iγ Hc)`. L48 STOP. No blind-cost. ArmC L44 = interp of L16 shell (fair same-L densified baseline).

| n | **champ L44** | ArmC L16 | ArmC L44 pad | QAOA L16 | WalkSAT flips | WalkSAT evals |
|---|---------------|----------|----------------|----------|---------------|---------------|
| 9 | **1.051/1.473** | 1.441/3.314 | 1.162/1.754 | 2.242/5.534 | 12.5/40.1 | 51.0/161.4 |
| 12 | **1.113/2.263** | 1.991/6.252 | 1.353/3.291 | 3.263/10.868 | 26.5/65.9 | 107.0/264.6 |
| 15 | **1.233/4.243** | 2.921/10.959 | 1.728/6.399 | 5.099/18.647 | 43.8/128.1 | 176.0/513.6 |
| 18 | **1.409/9.645** | 4.573/27.810 | 2.160/20.294 | 7.894/34.070 | 71.5/203.6 | 287.0/815.4 |

## b fits (9→18)

- champ L44 b_Med≈**1.0333**, b_B90≈**1.2323**
- ArmC L16 b_Med≈**1.1368**
- ArmC L44 pad b_Med≈**1.0726**
- QAOA L16 b_Med≈**1.1511**
- WalkSAT flips b_Med≈**1.2106**

## Kerr one-liner

L44 mean |γ|=4.20 → ~6.06 µs/layer raw; fold→[0,π] mean=1.51 → ~2.18 µs/layer; sum_L fold≈96 µs (cross-Kerr only, excl. SNAP/ECD).

**Note:** ArmC L44 n9/12/15 = B=1000; **n18 B=100** (full B=1000 ETA~5h deferred; L16 n18 was 106 min).
