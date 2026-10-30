# Gadget-Uc — projector-gadget family (SOFT-PROMOTED)

_Generated 2026-10-30 20:03:19 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer.

## Champ — projector-Uc **L=24** retrain{6,9,12}

Params: `unsat_params.pt` ← `depth/retrain_L24_params.pt`.
Backups: `unsat_params_L16_train6912_backup.pt`, `unsat_params_L20_backup.pt`.

| n | **L24 champ** | L20 | L16 | Arm C |
|---|---------------|-----|-----|-------|
| 9 | **1.185/2.245** | 1.261/2.657 | 1.406/3.288 | 1.441/3.314 |
| 12 | **1.392/3.865** | 1.590/4.760 | 1.864/6.106 | 1.991/6.252 |
| 15 | **1.820/7.283** | 2.168/8.296 | 2.754/10.244 | 2.921/10.959 |
| 18 | **2.492/17.005** | 3.170/19.916 | 4.194/24.302 | 4.573/27.810 |

## Scaling b (Med ≈ C·bⁿ), window 9→18

| model | b_Med | b_B90 |
|-------|-------|-------|
| **projector L24** | **1.0868** | **1.2506** |
| projector L20 | 1.1080 | 1.2460 |
| projector L16 | 1.1301 | 1.2427 |
| Arm C B1000 | 1.1369 | 1.2604 |
| QAOA L16 | 1.1511 | 1.2211 |

## Depth probe summary

| variant | n15 | n18 | keep vs L16 gate |
|---------|-----|-----|------------------|
| pad L20 | 2.206/8.682 | 3.178/21.284 | True |
| pad L24 | 1.959/8.071 | 2.744/19.543 | True |
| retrain L20 | 2.168/8.296 | 3.170/19.916 | True |
| **retrain L24** | **1.820/7.283** | **2.492/17.005** | **True (champ)** |

Detail: `depth/RESULTS.md`.

## Next probe

L=28/32 retrain same recipe, or span-split γ_ℓ^(s) on L24 backbone.

## GRAPE library (unchanged)

26 local projectors; SNAP/W†SNAP W Fid=1; two-cav n15~11%.

