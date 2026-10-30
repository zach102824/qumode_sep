# Paper mega-table — scale1000 B=1000 (L44 CVaR champ)

_Generated 2026-10-31 04:33:54 +0800 (Asia/Shanghai)_

**Metric:** quantum = Med/B90 of **1/p**; WalkSAT = flips/evals (different unit).

Honesty: champ = **L44 CVaR_q80** projector-Uc ≡ `exp(-iγ Hc)`. L48 STOP. No blind-cost.
Equal-shot mixture Spearman>0.95 (no promote). Freedom span_w @L44 no promote (<2% Med).

| n | **champ L44 CVaR** | ArmC L16 | ArmC L44 pad | QAOA L16 | WalkSAT flips | WalkSAT evals |
|---|--------------------|----------|--------------|----------|---------------|---------------|
| 9 | **1.045/1.420** | 1.441/3.314 | 1.162/1.754 | 2.242/5.534 | 12.5/40.1 | 51.0/161.4 |
| 12 | **1.109/2.163** | 1.991/6.252 | 1.353/3.291 | 3.263/10.868 | 26.5/65.9 | 107.0/264.6 |
| 15 | **1.228/4.067** | 2.921/10.959 | 1.728/6.399 | 5.099/18.647 | 43.8/128.1 | 176.0/513.6 |
| 18 | **1.412/9.145** | 4.573/27.810 | 2.160/20.294 | 7.894/34.070 | 71.5/203.6 | 287.0/815.4 |

## b fits (9→18)

- champ L44 CVaR b_Med≈**1.0341**, b_B90≈**1.2304**
- ArmC L16 b_Med≈**1.1368**
- ArmC L44 pad b_Med≈**1.0727** (n18 B=100)
- QAOA L16 b_Med≈**1.1511**
- WalkSAT flips b_Med≈**1.2106**

## NON-DEPTH #3+#4 notes

- Equal-shot mixture: max Spearman(log inv_p)=0.997 → **kill** (tail shared); best mix L44+L40 n18 B90=9.501 (miss ≥3% gate 9.356).
- CVaR_q80 @L44: **PROMOTE** n18 B90 9.145 (−5.2%).
- Freedom span_w @L44: n15 Med 1.218 vs thr 1.208; n18 1.394 vs 1.381 → **NO PROMOTE**.
