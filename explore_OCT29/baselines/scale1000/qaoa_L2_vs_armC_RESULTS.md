# L=2 QAOA vs Arm C — scale1000 B=1000

_Generated 2026-11-03 11:36:09 +0800 (Asia/Shanghai)_

Fresh train+eval both methods at L=2 on the same scale1000 pack (B=1000, n∈{9,12,15,18}).

**L=2 scale1000** — QAOA (b_Med≈1.35) vs Arm C (b_Med≈1.49)

| n | QAOA Med/B90 | Arm C Med/B90 |
|---|--------------|---------------|
| 9 | 12.7 / 30.0 | 28.3 / 59.5 |
| 12 | 30.1 / 78.0 | 94.6 / 210.6 |
| 15 | 76.4 / 223.9 | 312.7 / 769.6 |
| 18 | 189.7 / 616.1 | 1035.6 / 2810.8 |


## Sources
- QAOA: `explore_OCT29/runs/qaoa_L2_scale1000/` (also `baselines/scale1000/qaoa_L2_RESULTS.md`)
- Arm C: `explore_OCT29/runs/armC_L2_scale1000/RESULTS.md`
- Protocol mirrors L=1 scripts (`eval_qaoa_L2_scale1000.py`, `eval_armC_L2_scale1000.py`)

## Fits (n9→18)
- QAOA b_Med≈**1.3515**, b_B90≈**1.4010**
- Arm C b_Med≈**1.4914**, b_B90≈**1.5353**

## Notes
- At L=2, QAOA still wins on level and slope vs Arm C (same pattern as L=1).
- Arm C L=2 improves vs Arm C L=1 on level (n18 Med 1036 vs 2994) but remains behind QAOA L=2.
