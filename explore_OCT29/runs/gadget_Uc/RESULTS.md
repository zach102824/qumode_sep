# Gadget-Uc — projector-gadget family (SOFT-PROMOTED)

_Generated 2026-10-30 19:47:06 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer, L=16.

## Champ — train{6,9,12} projector-Uc

Params: `unsat_params.pt`

| n | Med/B90 | Arm C | ×ArmC/champ |
|---|---------|-------|-------------|
| 9 | **1.406/3.288** | 1.441/3.314 | ×1.025 |
| 12 | **1.864/6.106** | 1.991/6.252 | ×1.068 |
| 15 | **2.754/10.244** | 2.921/10.959 | ×1.061 |
| 18 | **4.194/24.302** | 4.573/27.810 | ×1.090 |

## GRAPE library

`grape_library/LIBRARY.md`: **26** local projectors; SNAP↔UZ and W†SNAP W↔UX **Fid=1**.
Two-cavity: \(C=C_a\otimes C_b\) for span-2 with \((k_a,k_b)\in\{(1,3),(2,2)\}\).

### Two-cavity frequency (packs)

| n | two-cav frac | (1,3) | (2,2) |
|---|--------------|-------|-------|
| 9 | 0.359 | 0.146 | 0.213 |
| 12 | 0.184 | 0.074 | 0.110 |
| 15 | 0.110 | 0.044 | 0.065 |
| 18 | 0.073 | 0.030 | 0.043 |

Tensor check kron projector: pass. Detail: `freedom/two_cavity_census.json`.

## Freedom ladder — DISCARD

Promote gate: n15/n18 Med < 0.98×champ (2.699 / 4.110).

| arm | n15 | n18 | keep |
|-----|-----|-----|------|
| span_w (clause weights by span) | 2.695/10.064 | 4.120/24.166 | False (n18 miss by ~0.01) |
| eps_delta (local Fock residual) | 2.697/10.261 | 4.123/23.955 | False (n18 miss; ε→0) |

Mild absolute Med gains but **do not replace champ**. See `freedom/RESULTS.md`.

## Retired

parity XOR, stencil_x, Fourier-γ deepen, freedom span_w / eps_delta.

