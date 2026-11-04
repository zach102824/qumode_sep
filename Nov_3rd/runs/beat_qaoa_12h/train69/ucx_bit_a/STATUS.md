# ucx_bit_a STATUS — 2026-11-04 ~11:45 Asia/Shanghai (CST, UTC+8)

## Idea1 Walsh-degree-graded γ — **PASS**

- Decisive L=2 train{6,9}→CORE: **ALL_CORE_BEAT** seeds 0–3 at B=1000 sealed Q.
- Best: seed1 knee **R=0.9567** (prior capacity knee 0.9858).
- Label: **graded H_c** (not non-Hc).
- Params: `walsh_degree/params_L2_seed1_B100.pt` (smoke max_train=100).

## Next

- [ ] Optional seal: retrain Idea1 with `--max_train 1000 --eval_B 1000` seeds 0–2.
- [ ] Idea2 cavity-locality γ_in/γ_x scaffold + L=2 B=100 smoke (only if combining with degree later).
- [ ] L=16 sanity: graded γ still beats on all CORE (Opus sanity check).

## L=16 baseline (unchanged)

power_p champ still sealed ALL_CORE under train{6,9}.
