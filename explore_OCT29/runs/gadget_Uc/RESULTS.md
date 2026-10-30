# Gadget-Uc — projector-gadget family (SOFT-PROMOTED)

_Generated 2026-10-30 20:20:28 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** W=H^{⊗3}, Arm C shell mixer.

## Champ — projector-Uc **L=32** retrain{6,9,12}

Params: `unsat_params.pt` ← `span_gamma/retrain_L32_params.pt`.

| n | **L32** | L28 | L24 | Arm C |
|---|--------|-----|-----|-------|
| 9 | **1.109/1.828** | 1.144/1.962 | 1.185/2.245 | 1.441/3.314 |
| 12 | **1.258/3.013** | 1.317/3.364 | 1.392/3.865 | 1.991/6.252 |
| 15 | **1.537/5.767** | 1.655/6.385 | 1.820/7.283 | 2.921/10.959 |
| 18 | **1.995/14.117** | 2.183/15.419 | 2.492/17.005 | 4.573/27.810 |

## Scaling b (9→18)

| model | b_Med | b_B90 |
|-------|-------|-------|
| **projector L32** | **1.0676** | **1.2536** |
| projector L28 | 1.0749 | 1.2555 |
| projector L24 | 1.0868 | 1.2506 |
| Arm C B1000 | 1.1369 | 1.2604 |
| QAOA L16 | 1.1511 | 1.2211 |

## Span-split γ_ℓ^(s) on L28 — DISCARD

Gate n15/n18 < 1.622 / 2.139. Got 1.663 / 2.210 (slightly worse). γmean_s≈(4.20,4.19,4.20)≈tied.

| n | spanγ | L28 |
|---|-------|-----|
| 9 | 1.136/1.946 | 1.144/1.962 |
| 12 | 1.306/3.361 | 1.317/3.364 |
| 15 | 1.663/6.463 | 1.655/6.385 |
| 18 | 2.210/16.237 | 2.183/15.419 |

Detail: `span_gamma/RESULTS.md`.

## Next

**Stop depth ladder** (L32 enough). Next cheap: train-set soft include n15, or freeze and document GRAPE compile of library on hardware path.

