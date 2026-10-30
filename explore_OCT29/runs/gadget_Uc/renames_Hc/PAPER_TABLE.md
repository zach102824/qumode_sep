# Paper mega-table — scale1000 B=1000 (CVaR@L44 champ)

_Generated 2026-10-31 04:35:15 +0800 (Asia/Shanghai)_

**Metric:** quantum = Med/B90 of **1/p**; WalkSAT = flips/evals (different unit).

Honesty: champ = **L44 CVaR_q80** (+ optional LBFGS polish if promoted) projector-Uc ≡ `exp(-iγ Hc)`. L48 STOP. No blind-cost.

| n | **champ** | ArmC L16 | ArmC L44 pad | QAOA L16 | WalkSAT flips | WalkSAT evals |
|---|-----------|----------|--------------|----------|---------------|---------------|
| 9 | **1.045/1.420** | 1.441/3.314 | 1.162/1.754 | 2.242/5.534 | 12.5/40.1 | 51.0/161.4 |
| 12 | **1.109/2.163** | 1.991/6.252 | 1.353/3.291 | 3.263/10.868 | 26.5/65.9 | 107.0/264.6 |
| 15 | **1.228/4.067** | 2.921/10.959 | 1.728/6.399 | 5.099/18.647 | 43.8/128.1 | 176.0/513.6 |
| 18 | **1.412/9.145** | 4.573/27.810 | 2.160/20.294 | 7.894/34.070 | 71.5/203.6 | 287.0/815.4 |

## b fits (9→18)

- champ b_Med≈**1.0341**, b_B90≈**1.2304**
- ArmC L16 b_Med≈**1.1368**
- ArmC L44 pad b_Med≈**1.0727** (n18 B=100)
- QAOA L16 b_Med≈**1.1511**
- WalkSAT flips b_Med≈**1.2106**

## Notes

- Mixture Spearman>0.95 kill; freedom span_w miss 2% Med gate.
- CVaR_q80 promoted over L44 pad (n18 B90 −5.2%).
- ArmC L44 n18 B=100; else B=1000.
