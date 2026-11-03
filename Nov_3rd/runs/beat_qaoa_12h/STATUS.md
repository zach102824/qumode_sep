# STATUS — 2026-11-03 22:21 Asia/Shanghai

## DONE
- R0 dual PASS L∈{1,2,16}
- **L=16 CORE-RATIO CLEAR BEAT** all CORE, 3 seeds (n15 R≈0.88–0.90)
- Shallow L=1/2: BEAT n12; **n15 capacity-limited** under Hamming-shell

## Closing check (Opus R3 #1) — DECLARE
Dense n15 pack N=1000 disjoint from train+scale1000; 803-start global search + LBFGS refine.
- **best_dense_R = 0.9803** (>0.980 declare bar; reopen needs ≤0.978)
- capacity_proven=True, reopen=False
- Artifacts: `round_2/dense_n15/`

## OPEN
- None for shallow n15 under locked gate sequence (training-only exhausted per Opus)
- Optional later: unlock gate/head (Arm C-style) — out of this plan's scope
