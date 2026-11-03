# QAOA L=1 — scale1000 B=1000

_Generated 2026-11-03 10:45:56 +0800 (Asia/Shanghai)_

**Ansatz:** ideal qubit QAOA `|+>^n → [U_C(γ) U_B(β)]^L`, L=1
**Cost:** `Cn = E·(n/m)`, `U_C = exp(-i γ Cn)` (unsat count / SAT Hamiltonian)
**Mixer:** `U_B = ⊗ exp(-i β X/2)` (var0 = MSB of E index)
**Angles:** trained shared (β,γ) on sealed `train` n=[9, 12] (n_per={9: 200, 12: 128}), zero-shot eval
**Pack:** `/workspace/qumode_sep/explore_OCT29/data/scale1000` split=`scale1000`
**Angles file:** `/workspace/qumode_sep/explore_OCT29/runs/qaoa_L1_scale1000/angles.json`

## Ladder (1/p)

| n | Med | B90 | p(gs)≈1/Med | L×Med | frac_inv≤2 | B | wall_min |
|---|-----|-----|-------------|-------|------------|---|----------|
| 9 | 29.279 | 63.429 | 0.034154 | 29.3 | 0.000 | 1000 | 0.0 |
| 12 | 90.033 | 216.627 | 0.011107 | 90.0 | 0.000 | 1000 | 0.0 |
| 15 | 289.892 | 750.514 | 0.003450 | 289.9 | 0.000 | 1000 | 0.0 |
| 18 | 920.513 | 2585.026 | 0.001086 | 920.5 | 0.000 | 1000 | 0.2 |

## vs Arm C L=1 + Nov progressive shell L=1 (same packs)

| n | QAOA Med | QAOA B90 | QAOA p(gs) | ArmC Med | ArmC B90 | ArmC p(gs) | NovShell Med | NovShell B90 | NovShell p(gs) |
|---|----------|----------|------------|----------|----------|------------|--------------|--------------|----------------|
| 9 | 29.279 | 63.429 | 0.034154 | 47.469 | 99.340 | 0.021066 | 43.410 | 91.832 | 0.023036 |
| 12 | 90.033 | 216.627 | 0.011107 | 194.177 | 425.207 | 0.005150 | 172.318 | 382.556 | 0.005803 |
| 15 | 289.892 | 750.514 | 0.003450 | 793.389 | 1870.020 | 0.001260 | 676.600 | 1643.524 | 0.001478 |
| 18 | 920.513 | 2585.026 | 0.001086 | 2993.655 | 8178.549 | 0.000334 | 2541.414 | 6948.326 | 0.000393 |

## Side-by-side Med (1/p) only

| n | QAOA L=1 | Arm C L=1 | Nov shell L=1 | QAOA / ArmC | QAOA / Nov |
|---|----------|-----------|---------------|-------------|------------|
| 9 | 29.279 | 47.469 | 43.410 | 0.617× | 0.674× |
| 12 | 90.033 | 194.177 | 172.318 | 0.464× | 0.522× |
| 15 | 289.892 | 793.389 | 676.600 | 0.365× | 0.428× |
| 18 | 920.513 | 2993.655 | 2541.414 | 0.307× | 0.362× |

## b fits

- **9_18** (ns=[9, 12, 15, 18]): b_Med≈**1.4678**, b_B90≈**1.5101**
- **9_15** (ns=[9, 12, 15]): b_Med≈**1.4654**, b_B90≈**1.5095**
- **12_18** (ns=[12, 15, 18]): b_Med≈**1.4732**, b_B90≈**1.5117**
- **9_12** (ns=[9, 12]): b_Med≈**1.4542**, b_B90≈**1.5059**
- **12_15** (ns=[12, 15]): b_Med≈**1.4767**, b_B90≈**1.5131**
- **15_18** (ns=[15, 18]): b_Med≈**1.4698**, b_B90≈**1.5102**

## Notes

- Comparators: Arm C L=1 (`runs/armC_L1_scale1000`) and Nov progressive shell L=1 (`Nov/runs/angle_explore/L_LADDER_scale1000.md`).
- QAOA angles trained from multi-start (paper p=10 grown→L, linear ramps, random).
- p(gs)≈1/Med is the median ground-state success probability.
