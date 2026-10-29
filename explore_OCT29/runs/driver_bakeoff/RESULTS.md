# Driver bake-off (Opus probe 1) — two arms
_Generated 2026-10-29 17:41:22 +0800 (Asia/Shanghai box clock)_
Ideal Gray-X ceiling **dropped** (Zach steer). Arms: **projector** vs **snap_d**.

## W compile
- F=0.9990028738975525 leak=0.0002765059471130371 nf=16 pairs=3

## Results table (val Med / B90)

| arm | phase | n9 | n12 | n15 | b_Med | b_B90 |
|-----|-------|----|-----|-----|-------|-------|
| projector | ramp | 274/579 | 2.55e+03/5.27e+03 | 1.56e+04/3.49e+04 | 1.961 | 1.980 |
| projector | trained | 9.89/25 | 36.8/65.3 | 76.5/152 | 1.406 | 1.352 |
| snap_d | ramp | 744/5.01e+03 | 1.21e+04/4.43e+04 | 5.67e+04/2.53e+05 | 2.059 | 1.923 |
| snap_d | trained | 289/6.24e+03 | 8.89e+03/1.57e+05 | 1.24e+05/1.11e+06 | 2.746 | 2.371 |

## Winner

Winner: **projector_trained** with b_Med=1.406, b_B90=1.352, n12 Med=36.8. Uniform b=2.0; want b≪2 for native projector to be interesting. Projector keeps b well below 2 → recommend probe 2 (train n9+n12) next.

## Notes

- Protocol: L=4; Phase A Δ ramp at n9; Phase B train at n9; eval n9/12/15.
- Uniform guessing: b=2.0. OCT20 ideal-X reference (dropped arm): b≈1.25, n12 Med≈13.
- No GNN.

## vs prior seq_best (COST→SNAP→D L5 fixed-angle, for context)

| | n9 Med/B90 | n12 Med/B90 |
|--|------------|-------------|
| seq_best L5 (prior) | 92.1/183 | 587/1574 |
| **projector L4 trained (this run)** | **9.89/25** | **36.8/65.3** |
| snap_d L4 trained (this run) | 289/6.24e3 | 8.89e3/1.57e5 |

Projector zero-shot from n9 angles already beats seq_best by ~16× at n12 Med, with b_Med≈1.41 ≪ 2.

## Recommended next

**Probe 2**: retrain projector on {n9,n12}, zero-shot n15/n18; measure Med15/Med12.
Optional parallel: probe 5 — add W / SNAP0 / FLAT_PREP atoms to seq-evolve vocab.
