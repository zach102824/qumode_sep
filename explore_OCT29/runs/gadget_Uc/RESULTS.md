# Gadget-Uc — projector-gadget family (LBFGS-PROMOTED seed0)

_Generated 2026-10-31 01:00:00 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer, **L=32**.

## Champ — LBFGS polish **seed0** (warm from tail plain)

Params: `unsat_params.pt`. Prior tail plain: `unsat_params_tail_plain_backup.pt`.

| n | **LBFGS seed0** | tail plain | soft-n15 | Arm C |
|---|-----------------|------------|----------|-------|
| 9 | **1.092/1.776** | 1.086/1.799 | 1.100/1.759 | 1.441/3.314 |
| 12 | **1.197/2.843** | 1.192/2.829 | 1.231/2.919 | 1.991/6.252 |
| 15 | **1.418/5.505** | 1.411/5.520 | 1.483/5.847 | 2.921/10.959 |
| 18 | **1.752/11.353** | 1.752/11.813 | 1.884/13.704 | 4.573/27.810 |

b_Med≈**1.054**, b_B90≈**1.231** (9→18).

## Recent arms

| arm | n18 Med/B90 | keep |
|-----|-------------|------|
| LBFGS seed0 | 1.752/11.353 | 1 → **CHAMP** |
| LBFGS seed4 | 1.744/11.446 | 1 |
| LBFGS seed2 | 1.745/11.593 | 0 |
| tail plain | 1.752/11.813 | prior |
| soft-n15 t91215 | 1.884/13.704 | prior |

Detail: `lbfgs_polish/RESULTS.md`, `tail_retrain/RESULTS.md`. Depth ladder: `depth_ladder/`.
Pulse budget: `pulse_budget/`.

## GRAPE library — FREEZE

See `grape_library/FREEZE.md` + `LIBRARY.md`. Classical angle/depth search paused; hardware compile next.

## Retired

parity, stencil, Fourier, freedom w/εΔ, span-γ, further depth >32, CVaR/lme (no beat plain).

## WalkSAT bars (scale1000, R=20, max_inst=200)

| n | flips Med/B90 | evals Med/B90 |
|---|---------------|---------------|
| 9 | 12.5/40.1 | 51.0/161.4 |
| 12 | 26.5/65.9 | 107.0/264.6 |
| 15 | 43.8/128.1 | 176.0/513.6 |
| 18 | 71.5/203.6 | 287.0/815.4 |

Locked champ 1/p still 1.092/1.776 … 1.752/11.353 (units ≠ flips).

## renames_Hc wave (2026-10-31 ~02:19 CST)

Opus-style top-2 under renames_Hc: SNAP φ noise + soft-n18. Detail: `renames_Hc/RESULTS.md`.

- Soft-n18 improved a **short retrain seed** (params missing from git) but **does not** beat locked LBFGS.
- SNAP φ: σ≤2% mild; σ=5% ~+23% n9 Med (same ballpark as γ-jitter).
- L=36/40 deferred.
