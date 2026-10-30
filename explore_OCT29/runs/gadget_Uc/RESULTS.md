# Gadget-Uc — projector-gadget family (SOFT-PROMOTED)

_Generated 2026-10-30 19:39:52 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, mixer Arm C shell, L=16.

## Soft promote

- **KEEP:** per-clause projector gadgets `U=∏_c[I+(e^{-iγ}-1)C_c]` (GRAPE library: finite C / C^X; not densified global H_cost).
- **DROP:** XOR parity; stencil_x; Fourier-γ deepen.
- **Champ params:** `unsat_params.pt` ← train **{6,9,12}** (`train6912/`, promoted).
- Prior train{9,12} backup: `unsat_params_train912_backup.pt`.

## scale1000 Med/B90 (B=1000)

| n | projector {6,9,12} | prior {9,12} | Arm C | ×ArmC/new |
|---|--------------------|--------------|-------|-----------|
| 9 | **1.406/3.288** | 1.445/3.222 | 1.441/3.314 | ×1.025 |
| 12 | **1.864/6.106** | 1.926/6.148 | 1.991/6.252 | ×1.068 |
| 15 | **2.754/10.244** | 2.820/10.063 | 2.921/10.959 | ×1.061 |
| 18 | **4.194/24.302** | 4.329/24.342 | 4.573/27.810 | ×1.090 |

## GRAPE library (finite ops)

See `grape_library/LIBRARY.md`. **26** local projectors (k=1,2,3). Exact: SNAP↔UZ and W†SNAP W↔UX Fid=1. Mean SNAP·D fit to UX ≈0.90; D·SNAP·D ≈0.93.

## Retired / discarded

| item | note |
|------|------|
| parity / stencil | wrong / blind cost |
| Fourier-γ deepen | L16 scale n9 1.843 > 1.445 |

Artifacts: `unsat_params.pt`, `train6912/`, `grape_library/`, `fourier_deepen/`.

