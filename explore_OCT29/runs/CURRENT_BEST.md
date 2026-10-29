# CURRENT BEST — explore_OCT29 (Asia/Shanghai)

_Updated 2026-10-29 ~18:05 CST_

## Fixed-angle champion (transfer)

**Fourier q=3, L_train=6** on projector `W·SNAP₀(β)·W†`  
Path: `runs/probe3_fourier_deep/` (modes: `modes_q3_Ltrain6.pt`, commit `d1399b4d`)

| n | L4 | L12 | L20 |
|---|----|-----|-----|
| 15 Med/B90 | 46.2/90 | **7.70/10.6** | **5.99/16.6** |
| 18 Med/B90 | 176/621 | 32.9/461 | 16.0/102 |

- b_Med @ L20 ≈ **1.24** (≪2)
- Monotonic Med vs L on n15

Legacy: Probe2 free-angle L4 train{9,12} n15 Med≈30.9; Probe3 q=2 L12 n15≈9.67 — both superseded for ceiling.

## Probe5 (seq vocab with W atoms)

- Loop: `scripts/probe5_seq_vocab.py` — **raise gate to Fourier q3_Ltrain6** (n15≲7.7, b≲1.24)
- Seeds: `fourier_proj_L{4,6,8,12,16}` warm-started from q3 modes
- Promote only if competitive with Fourier; mediocre mutants rejected
- Status/champ: `runs/probe5_seq_vocab/`
- Stop: `touch runs/STOP_PROBE5`

Until a motif beats/matches Fourier on n15+b, **fixed-angle champ remains q3_Ltrain6**.

## Not in play

- GNN: dead (do not revive)
- Raw SNAP+D seq_best: obsolete vs projector/Fourier
