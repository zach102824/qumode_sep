# Gadget-Uc — **L40 pad PROMOTED**

_Generated 2026-10-31 02:53:57 +0800 (Asia/Shanghai)_

BinH, Arm C shell. Still `Uc=exp(-iγ Hc)`. No blind-cost.

## Champ — L=40 pad

| n | **L40** | L36 | soft-n18 L32 |
|---|---------|-----|--------------|
| 9 | **1.058/1.532** | 1.072/1.613 | 1.098/1.719 |
| 12 | **1.129/2.367** | 1.156/2.532 | 1.197/2.779 |
| 15 | **1.263/4.563** | 1.312/4.918 | 1.395/5.460 |
| 18 | **1.467/10.012** | 1.571/10.312 | 1.709/11.107 |

b_Med≈**1.0371**, b_B90≈**1.2332**.

Params: `unsat_params.pt`. L36 backup: `unsat_params_pre_L40_backup.pt`.
Paper: `pulse_budget/PAPER_TABLE.md`.

## WalkSAT (scale1000 R=20 B=200)

| n | flips Med/B90 | evals Med/B90 |
|---|---------------|---------------|
| 9 | 12.5/40.1 | 51.0/161.4 |
| 12 | 26.5/65.9 | 107.0/264.6 |
| 15 | 43.8/128.1 | 176.0/513.6 |
| 18 | 71.5/203.6 | 287.0/815.4 |

## Probes

- L40 pad: **PROMOTE** (n15 -3.7%, n18 -6.6% Med).
- L36 retrain: skipped (L40 succeeded).
- Multi-seed LBFGS on L32: no promote.
