# Arm C L=2 — B=1000 scaling eval

_Generated 2026-11-03 11:35:09 +0800 (Asia/Shanghai)_

**Motif:** probe6 Arm C free shells — prep=`W_H`, layer `COST → W_H·SNAP(φ_pc)·W_H†`, L=2
**Train:** n∈{9,12} sealed train, steps=120, max_train=32, max_val=12, seed=0, lr=0.002, gamma0/beta0=None (fresh L=2)
**Params:** `/workspace/qumode_sep/explore_OCT29/runs/armC_L2_scale1000/arm_C_shell_params.pt`
**Pack:** `/workspace/qumode_sep/explore_OCT29/data/scale1000` split=`scale1000`
**B:** 1000 per n (requested)

## Train (val @ best step)

- best_step=119  Scomb=4.363012551590146
- train_wall_s=4.0

| n | Med | B90 | B |
|---|-----|-----|---|
| 9 | 24.848 | 65.971 | 12 |
| 12 | 90.580 | 164.432 | 12 |

## Ladder (scale1000)

| n | Med | B90 | leak | L×Med | frac_inv≤2 | B | wall_min |
|---|-----|-----|------|-------|------------|---|----------|
| 9 | 28.337 | 59.507 | 0.0000 | 56.7 | 0.000 | 1000 | 0.0 |
| 12 | 94.629 | 210.648 | 0.0000 | 189.3 | 0.000 | 1000 | 0.0 |
| 15 | 312.710 | 769.608 | 0.0000 | 625.4 | 0.000 | 1000 | 1.0 |
| 18 | 1035.629 | 2810.769 | 0.0000 | 2071.3 | 0.000 | 1000 | 20.5 |

## b fits

- **9_18** (ns=[9, 12, 15, 18]): b_Med≈**1.4914**, b_B90≈**1.5353**
- **9_15** (ns=[9, 12, 15]): b_Med≈**1.4921**, b_B90≈**1.5321**
- **12_18** (ns=[12, 15, 18]): b_Med≈**1.4900**, b_B90≈**1.5401**
- **9_12** (ns=[9, 12]): b_Med≈**1.4947**, b_B90≈**1.5240**
- **12_15** (ns=[12, 15]): b_Med≈**1.4895**, b_B90≈**1.5402**
- **15_18** (ns=[15, 18]): b_Med≈**1.4906**, b_B90≈**1.5400**

## Notes

- Fresh L=2 train; does not reuse L=16 champ reshape.
- Out dir separate from `runs/armC_B1000` (L=16).

## vs QAOA L=2 (scale1000 B=1000)

_Updated 2026-11-03 11:35:40 +0800 (Asia/Shanghai) — n18 finished_

| n | QAOA Med | QAOA B90 | ArmC Med | ArmC B90 | ArmC/QAOA |
|---|----------|----------|----------|----------|-----------|
| 9 | 12.736 | 30.047 | 28.337 | 59.507 | 2.225× |
| 12 | 30.079 | 78.011 | 94.629 | 210.648 | 3.146× |
| 15 | 76.424 | 223.897 | 312.710 | 769.608 | 4.092× |
| 18 | 189.727 | 616.070 | 1035.629 | 2810.769 | 5.459× |

- **b_Med (9–18)** Arm C ≈ **1.491** vs QAOA L=2 ≈ **1.35** (from prior table)
- This is the *old cold* Arm C L=2 (γ0/φ0 fresh, lr=2e-3×120) — NOT dual-QAOA warm. E3 replaces this protocol.
