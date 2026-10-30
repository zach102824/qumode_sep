# Gadget-Uc — projector-gadget family (**soft-n18 PROMOTED** from locked LBFGS)

_Generated 2026-10-31 02:33:30 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer, **L=32**.

## Champ — soft-n18 `t9121518_w1` (warm from LBFGS seed0)

Params: `unsat_params.pt`. Backup locked: `unsat_params_pre_soft18_backup.pt`.

| n | **soft-n18** | locked LBFGS seed0 |
|---|--------------|--------------------|
| 9 | **1.098/1.719** | 1.092/1.776 |
| 12 | **1.197/2.779** | 1.197/2.843 |
| 15 | **1.395/5.460** | 1.418/5.505 |
| 18 | **1.709/11.107** | 1.752/11.353 |

b_Med≈**1.0506**, b_B90≈**1.2325**.

**Honesty:** soft-n18 uses n18 val in train pool; scale1000 holdout. Still `Uc=exp(-iγ Hc)`.

**CRITICAL:** prior renames_Hc soft-n18 (~9.8/38.6) was on bad retrain — **INVALID**. See `renames_Hc/RESULTS.md`.

## WalkSAT bars (scale1000, R=20, max_inst=200)

| n | flips Med/B90 | evals Med/B90 |
|---|---------------|---------------|
| 9 | 12.5/40.1 | 51.0/161.4 |
| 12 | 26.5/65.9 | 107.0/264.6 |
| 15 | 43.8/128.1 | 176.0/513.6 |
| 18 | 71.5/203.6 | 287.0/815.4 |

## GRAPE library — FREEZE

See `grape_library/FREEZE.md`. Still renames densify as clause→mask.

## Retired / invalid

- Prior renames_Hc soft-n18 on short retrain seed (INVALID)
- parity, stencil, Fourier, freedom, span-γ, CVaR/lme
