# Gadget-Uc — projector-gadget family (TAIL-PROMOTED plain)

_Generated 2026-10-31 00:34:00 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer, **L=32**.

## Champ — tail-retrain **plain** (warm from soft-n15 t91215)

Params: `unsat_params.pt`. Prior soft-n15: `unsat_params_softn15_t91215_backup.pt`.

| n | **tail plain** | soft-n15 t91215 | plain L32 | Arm C |
|---|----------------|-----------------|-----------|-------|
| 9 | **1.086/1.799** | 1.100/1.759 | 1.109/1.828 | 1.441/3.314 |
| 12 | **1.192/2.829** | 1.231/2.919 | 1.258/3.013 | 1.991/6.252 |
| 15 | **1.411/5.520** | 1.483/5.847 | 1.537/5.767 | 2.921/10.959 |
| 18 | **1.752/11.813** | 1.884/13.704 | 1.995/14.117 | 4.573/27.810 |

b_Med≈**1.055**, b_B90≈**1.234** (9→18).

## Tail-retrain arms

| arm | n15 | n18 | keep |
|-----|-----|-----|------|
| plain | 1.411/5.520 | 1.752/11.813 | 1 |
| lme_tau1 | 1.489/5.771 | 1.884/13.589 | 0 |
| lme_tau2 | 1.476/5.861 | 1.873/13.656 | 0 |
| cvar_q80 | 1.485/5.785 | 1.884/13.745 | 0 |

Detail: `tail_retrain/RESULTS.md`. Pulse budget: `pulse_budget/RESULTS.md` + `PAPER_TABLE.md`.

## GRAPE library — FREEZE

See `grape_library/FREEZE.md` + `LIBRARY.md`. Classical angle/depth search paused; hardware compile next.

Two-cavity sim compile: `grape_library/COMPILE_TWOCAV.md` — joint SNAP exact; local SNAP⊗SNAP / naive Ua⊗Ub fail AND-phase (needs entangling).

## Retired

parity, stencil, Fourier, freedom w/εΔ, span-γ, further depth >32, CVaR/lme tail losses (no beat plain).

## WalkSAT bars (existing scale1000 baseline)

| n | Med | B90 |
|---|-----|-----|
| 9 | None | None |
| 12 | None | None |
| 15 | None | None |
| 18 | None | None |
