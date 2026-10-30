# Paper table — scale1000 B=1000 (refreshed)

_Generated 2026-10-31 03:14:44 +0800 (Asia/Shanghai)_

**Metric:** quantum = Med/B90 of **1/p**; WalkSAT flips/evals.

Honesty: champ = **L44 pad** ≡ `exp(-iγ Hc)`. L48 pad failed ≥2% gate → **STOP depth ladder**. No blind-cost.

| n | **champ** Med/B90 | ArmC L16 | QAOA L16 | WalkSAT flips | WalkSAT evals |
|---|-------------------|----------|----------|---------------|---------------|
| 9 | **1.051/1.473** | 1.441/3.314 | 2.242/5.534 | 12.5/40.1 | 51.0/161.4 |
| 12 | **1.113/2.263** | 1.991/6.252 | 3.263/10.868 | 26.5/65.9 | 107.0/264.6 |
| 15 | **1.233/4.243** | 2.921/10.959 | 5.099/18.647 | 43.8/128.1 | 176.0/513.6 |
| 18 | **1.409/9.645** | 4.573/27.810 | 7.894/34.070 | 71.5/203.6 | 287.0/815.4 |

## b fits

- champ b_Med≈**1.0333**, b_B90≈**1.2323**

## Depth pad ladder (diminishing returns)

| L | tag | n15 Med/B90 | n18 Med/B90 | Δn15 Med | Δn18 Med |
|---|-----|-------------|-------------|----------|----------|
| 32 | soft-n18 w1 | 1.395/5.460 | 1.709/11.107 | — | — |
| 36 | pad | 1.312/4.918 | 1.571/10.312 | -5.9% | -8.1% |
| 40 | pad | 1.263/4.563 | 1.467/10.012 | -3.7% | -6.6% |
| **44** | **pad **CHAMP**** | **1.233/4.243** | **1.409/9.645** | -2.4% | -4.0% |
| 48 | pad NO | 1.212/3.872 | 1.379/9.706 | -1.7% | -2.1% |

L48: n15 1.212 vs thr 1.208 (miss); n18 1.379 vs thr 1.381 (ok). Gate AND → **no promote**. L52 not tried.

### Depth diminishing returns

- L32→36: n15 −6.0%, n18 −8.1%
- L36→40: n15 −3.7%, n18 −6.6%
- L40→44: n15 −2.4%, n18 −4.0% (last promote, ≥2% gate)
- L44→48: n15 −1.7%, n18 −2.1% (**below** 2% n15 gate) → stop
- Marginal gain per +4 layers halves ~each step; pad-only interp saturates near L44.
## WalkSAT R=50 confirm

max_inst=200 scale1000; see `renames_Hc/walksat_R50/RESULTS.md`. Med flips ≈ R=20 (within sampling noise).

