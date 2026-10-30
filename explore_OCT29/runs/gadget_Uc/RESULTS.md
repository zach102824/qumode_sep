# Gadget-Uc — projector-gadget family (SOFT-PROMOTED)

_Generated 2026-10-30 20:32:36 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer, **L=32**.

## Champ — soft-n15 **t91215** (train includes n15 val)

Params: `unsat_params.pt`. Prior plain L32: `unsat_params_L32_plain_backup.pt`.

| n | **soft-n15** | plain L32 | Arm C |
|---|--------------|-----------|-------|
| 9 | **1.100/1.759** | 1.109/1.828 | 1.441/3.314 |
| 12 | **1.231/2.919** | 1.258/3.013 | 1.991/6.252 |
| 15 | **1.483/5.847** | 1.537/5.767 | 2.921/10.959 |
| 18 | **1.884/13.704** | 1.995/14.117 | 4.573/27.810 |

b_Med≈**1.0618**, b_B90≈**1.2566** (9→18).

## Soft-n15 arms

| arm | n15 | n18 | keep |
|-----|-----|-----|------|
| t691215 | 1.495/5.837 | 1.900/13.741 | 1 |
| t91215 | 1.483/5.847 | 1.884/13.704 | 1 |

Detail: `soft_n15/RESULTS.md`. n15 train empty → soft pool=val; scale1000=holdout.

## GRAPE library — FREEZE

See `grape_library/FREEZE.md` + `LIBRARY.md`. Classical angle/depth search paused; hardware compile next.

Two-cavity sim compile: `grape_library/COMPILE_TWOCAV.md` — joint SNAP exact; local SNAP⊗SNAP / naive Ua⊗Ub fail AND-phase (needs entangling).

## Retired

parity, stencil, Fourier, freedom w/εΔ, span-γ, further depth >32.


## WalkSAT bars (existing scale1000 baseline)

| n | Med | B90 |
|---|-----|-----|
| 9 | None | None |
| 12 | None | None |
| 15 | None | None |
| 18 | None | None |
