# Gadget-Uc — projector-gadget family (SOFT-PROMOTED)

_Generated 2026-10-30 20:10:58 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer.

## Champ — projector-Uc **L=28** retrain{6,9,12}

Params: `unsat_params.pt` ← `depth/retrain_L28_params.pt`.

| n | **L28** | L24 | L20 | L16 | Arm C |
|---|--------|-----|-----|-----|-------|
| 9 | **1.144/1.962** | 1.185/2.245 | 1.261/2.657 | 1.406/3.288 | 1.441/3.314 |
| 12 | **1.317/3.364** | 1.392/3.865 | 1.590/4.760 | 1.864/6.106 | 1.991/6.252 |
| 15 | **1.655/6.385** | 1.820/7.283 | 2.168/8.296 | 2.754/10.244 | 2.921/10.959 |
| 18 | **2.183/15.419** | 2.492/17.005 | 3.170/19.916 | 4.194/24.302 | 4.573/27.810 |

## Scaling b_Med / b_B90 (9→18)

| model | b_Med | b_B90 |
|-------|-------|-------|
| **projector L28** | **1.0749** | **1.2555** |
| projector L24 | 1.0868 | 1.2506 |
| projector L20 | 1.1080 | 1.2460 |
| projector L16 | 1.1301 | 1.2427 |
| Arm C B1000 | 1.1369 | 1.2604 |
| QAOA L16 | 1.1511 | 1.2211 |

## Depth ladder

Pad L20/24 and retrain L20/24/28 all promoted stepwise. Detail: `depth/RESULTS.md`.

## Next

Stop depth for now (diminishing / cost↑) **or** try L=32 once. Prefer **span-split γ_ℓ^(s)** on L28 backbone next.

## GRAPE library

26 local projectors; SNAP exact; two-cav documented.

