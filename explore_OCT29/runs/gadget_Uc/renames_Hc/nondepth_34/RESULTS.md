# NON-DEPTH #3+#4 — mixture / CVaR / freedom@L44

_Generated 2026-10-31 04:28:26 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. No L48+. Blind banned. Champ L44 sealed.

Champ Med/B90: n9 1.051/1.473, n12 1.113/2.263, n15 1.233/4.243, n18 1.409/9.645

## #3a Equal-shot schedule mixture

**Spearman kill:** max corr=0.9974 > 0.95 → tail largely shared across members (paper sentence).

### Solo members

| member | n9 | n12 | n15 | n18 |
|--------|----|-----|-----|-----|
| L44 | 1.051/1.473 | 1.113/2.263 | 1.233/4.243 | 1.409/9.645 |
| L40 | 1.058/1.532 | 1.129/2.367 | 1.263/4.563 | 1.467/10.012 |
| L36 | 1.072/1.613 | 1.156/2.532 | 1.312/4.918 | 1.571/10.312 |
| soft18 | 1.098/1.719 | 1.197/2.779 | 1.395/5.460 | 1.709/11.107 |

### Mixtures (equal-shot)

Promote gate: n18 B90 ≤ **9.356** (≥3% vs 9.645) AND Med ≤ 1.05× champ.

| mix | n9 | n12 | n15 | n18 | promote |
|-----|----|-----|-----|-----|---------|
| all4 | 1.070/1.576 | 1.147/2.455 | 1.296/4.710 | 1.529/10.348 | **False** |
| L44_L40_L36 | 1.061/1.533 | 1.131/2.377 | 1.269/4.575 | 1.477/9.956 | **False** |
| L44_L40 | 1.055/1.504 | 1.123/2.307 | 1.247/4.394 | 1.439/9.501 | **False** |
| L44_soft18 | 1.075/1.580 | 1.152/2.462 | 1.306/4.795 | 1.537/10.316 | **False** |
| L44_L40_soft18 | 1.069/1.563 | 1.145/2.432 | 1.291/4.642 | 1.514/10.299 | **False** |

## #3b CVaR_q80 retrain on L44

best step 33 score=0.3179; promote=**True**

| n | CVaR Med/B90 | champ |
|---|--------------|-------|
| 9 | 1.045/1.420 | 1.051/1.473 |
| 12 | 1.109/2.163 | 1.113/2.263 |
| 15 | 1.228/4.067 | 1.233/4.243 |
| 18 | 1.412/9.145 | 1.409/9.645 |

## #4 Freedom span_w @ L44

w_span softplus=[0.9569617509841919, 0.9860987067222595, 1.040842056274414]; promote=**False** gates={'thr15': 1.20834, 'thr18': 1.38082, 'ok15': False, 'ok18': False}

| n | freedom Med/B90 | champ | thr (0.98×) |
|---|-----------------|-------|-------------|
| 9 | 1.040/1.460 | 1.051/1.473 | 1.030 |
| 12 | 1.097/2.232 | 1.113/2.263 | 1.091 |
| 15 | 1.218/4.040 | 1.233/4.243 | 1.208 |
| 18 | 1.394/9.601 | 1.409/9.645 | 1.381 |

## Verdict

- mixture any promote: **False**
- cvar promote: **True**
- freedom promote: **False**

**CVaR PROMOTED** → `unsat_params.pt` now L44 CVaR_q80 (backup `unsat_params_pre_cvar_L44_backup.pt`). Freedom/mixture discarded.
