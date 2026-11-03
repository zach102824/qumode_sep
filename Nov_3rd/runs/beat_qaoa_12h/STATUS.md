# STATUS — 2026-11-03 21:36 Asia/Shanghai

## DONE
- R0 dual PASS L∈{1,2,16}
- **L=16 CORE-RATIO clear BEAT all CORE, 3 seeds** (R≈0.88–0.95)
- L=1/2 BEAT n12; best shallow n15 eval ≈0.983–0.984 TIE
- Global search L=2 shell on n15 train: best_train_R=**0.9807** (under Opus kill 0.982) but scale1000 eval n15 **R=0.9965** (no transfer). n12 still BEAT 0.977.

## OPEN
- Shallow n15 ≤0.98 — train can scrape ~0.981 but eval stays ~0.983–0.997. Asking Opus R3 with this evidence.

## Running
- Ask Opus R3 (capacity + basin evidence); then next 3 training-only shallow ideas if any remain.
