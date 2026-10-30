# Gadget-Uc — **L36 pad PROMOTED**

_Generated 2026-10-31 02:50:09 +0800 (Asia/Shanghai)_

BinH W=H^{⊗3}, Arm C shell mixer. Still `Uc=exp(-iγ Hc)`.

## Champ — L=36 pad from soft-n18

| n | **L36 pad** | soft-n18 L32 | LBFGS seed0 |
|---|-------------|--------------|-------------|
| 9 | **1.072/1.613** | 1.098/1.719 | 1.092/1.776 |
| 12 | **1.156/2.532** | 1.197/2.779 | 1.197/2.843 |
| 15 | **1.312/4.918** | 1.395/5.460 | 1.418/5.505 |
| 18 | **1.571/10.312** | 1.709/11.107 | 1.752/11.353 |

b_Med≈**1.0434**, b_B90≈**1.2308**.

Params: `unsat_params.pt`. Soft-n18 backup: `unsat_params_pre_L36_backup.pt`.
Paper table: `pulse_budget/PAPER_TABLE.md`.

## WalkSAT (scale1000 R=20 B=200)

| n | flips Med/B90 | evals Med/B90 |
|---|---------------|---------------|
| 9 | 12.5/40.1 | 51.0/161.4 |
| 12 | 26.5/65.9 | 107.0/264.6 |
| 15 | 43.8/128.1 | 176.0/513.6 |
| 18 | 71.5/203.6 | 287.0/815.4 |

## Recent probes

- Multi-seed LBFGS on soft-n18: **no promote** (flat).
- L36 pad: **PROMOTE** (n15 −6.0%, n18 −8.1% Med).
