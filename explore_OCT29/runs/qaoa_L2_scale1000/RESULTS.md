# QAOA L=2 — scale1000 B=1000

_Generated 2026-11-03 11:14:05 +0800 (Asia/Shanghai)_

**Ansatz:** ideal qubit QAOA `|+>^n → [U_C(γ) U_B(β)]^L`, L=2
**Cost:** `Cn = E·(n/m)`, `U_C = exp(-i γ Cn)` (unsat count / SAT Hamiltonian)
**Mixer:** `U_B = ⊗ exp(-i β X/2)` (var0 = MSB of E index)
**Angles:** trained shared (β,γ) on sealed `train` n=[9, 12] (n_per={9: 200, 12: 128}), zero-shot eval
**Pack:** `/workspace/qumode_sep/explore_OCT29/data/scale1000` split=`scale1000`
**Angles file:** `/workspace/qumode_sep/explore_OCT29/runs/qaoa_L2_scale1000/angles.json`

## Ladder (1/p)

| n | Med | B90 | p(gs)≈1/Med | L×Med | frac_inv≤2 | B | wall_min |
|---|-----|-----|-------------|-------|------------|---|----------|
| 9 | 12.736 | 30.047 | 0.078516 | 25.5 | 0.000 | 1000 | 0.0 |
| 12 | 30.079 | 78.011 | 0.033245 | 60.2 | 0.000 | 1000 | 0.0 |
| 15 | 76.424 | 223.897 | 0.013085 | 152.8 | 0.000 | 1000 | 0.1 |
| 18 | 189.727 | 616.070 | 0.005271 | 379.5 | 0.000 | 1000 | 0.4 |

## vs Arm C L=1 + Nov progressive shell L=1 (reference only; L=1 nums)

| n | QAOA Med | QAOA B90 | QAOA p(gs) | ArmC Med | ArmC B90 | ArmC p(gs) | NovShell Med | NovShell B90 | NovShell p(gs) |
|---|----------|----------|------------|----------|----------|------------|--------------|--------------|----------------|
| 9 | 12.736 | 30.047 | 0.078516 | 47.469 | 99.340 | 0.021066 | 43.410 | 91.832 | 0.023036 |
| 12 | 30.079 | 78.011 | 0.033245 | 194.177 | 425.207 | 0.005150 | 172.318 | 382.556 | 0.005803 |
| 15 | 76.424 | 223.897 | 0.013085 | 793.389 | 1870.020 | 0.001260 | 676.600 | 1643.524 | 0.001478 |
| 18 | 189.727 | 616.070 | 0.005271 | 2993.655 | 8178.549 | 0.000334 | 2541.414 | 6948.326 | 0.000393 |

## Side-by-side Med (1/p) only

| n | QAOA L=2 | Arm C L=1 | Nov shell L=1 | QAOA / ArmC | QAOA / Nov |
|---|----------|-----------|---------------|-------------|------------|
| 9 | 12.736 | 47.469 | 43.410 | 0.268× | 0.293× |
| 12 | 30.079 | 194.177 | 172.318 | 0.155× | 0.175× |
| 15 | 76.424 | 793.389 | 676.600 | 0.096× | 0.113× |
| 18 | 189.727 | 2993.655 | 2541.414 | 0.063× | 0.075× |

## b fits

- **9_18** (ns=[9, 12, 15, 18]): b_Med≈**1.3515**, b_B90≈**1.4010**
- **9_15** (ns=[9, 12, 15]): b_Med≈**1.3480**, b_B90≈**1.3976**
- **12_18** (ns=[12, 15, 18]): b_Med≈**1.3593**, b_B90≈**1.4112**
- **9_12** (ns=[9, 12]): b_Med≈**1.3317**, b_B90≈**1.3744**
- **12_15** (ns=[12, 15]): b_Med≈**1.3645**, b_B90≈**1.4211**
- **15_18** (ns=[15, 18]): b_Med≈**1.3540**, b_B90≈**1.4013**

## Notes

- Comparators: Arm C L=1 (`runs/armC_L1_scale1000`) and Nov progressive shell L=1 (`Nov/runs/angle_explore/L_LADDER_scale1000.md`).
- QAOA angles trained from multi-start (paper p=10 grown→L, linear ramps, random).
- p(gs)≈1/Med is the median ground-state success probability.
