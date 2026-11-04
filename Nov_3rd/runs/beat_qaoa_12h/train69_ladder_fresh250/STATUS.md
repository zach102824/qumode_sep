# STATUS — train69_ladder_fresh250 Phase 1

_Asia/Singapore (UTC+8) — started 2026-11-04 ~12:14_

## Locked recipe
- Motif: Hamming-shell SNAP |0⟩→[Uc^X→shell SNAP]^L→W→Fock
- Idea: **power_p** `a_n=(8/(n-1))^p`
- Fresh packs: 250 n6 + 250 n9 (new subsample RNG), disjoint from scale1000
- Train batch: 32+32; SNAP w6=1 w9=2
- QAOA: fresh train on same 250/250 mix, paper init β=0.01 γ=-0.01 nstart=1
- Phase 1 eval: n∈{6,9,12,15} B=1000; Med + B90; BEAT if raw Med < QAOA Med (same for B90)
- L_LIST: 1,4,8,12,16,20,24,28,32 small→large
- Phase 2 (n18, n21) **blocked** until Phase 1 complete

## Progress
| Step | Status | Notes |
|------|--------|-------|
| PLAN.md | done | locked text |
| Fresh packs | in progress | |
| QAOA train69 path | pending | |
| L=1 | pending | |
| L=4 | pending | |
| L=8 | pending | |
| L=12 | pending | |
| L=16 | pending | |
| L=20 | pending | |
| L=24 | pending | |
| L=28 | pending | |
| L=32 | pending | |
| Phase1 RESULTS.md | pending | |
| Phase2 | blocked | wait for parent |

## Paths
- Run: `Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/`
- Packs: `train_packs/`
- QAOA: `qaoa_angles/L{L}/`
- SNAP: `power_p/`

- [2026-11-04 12:16:34 +0800 Asia/Singapore] Fresh packs OK: n6+n9 N=250 each, disjoint scale1000, differ from 2026-11-03 packs.

- [2026-11-04 12:16:38 +0800 Asia/Singapore] L=1: train QAOA fresh…

- [2026-11-04 12:16:43 +0800 Asia/Singapore] L=1: train power_p seed0…

- [2026-11-04 12:17:21 +0800 Asia/Singapore] L=1: DONE — RESULTS_L1.md written

- [2026-11-04 12:17:21 +0800 Asia/Singapore] L=4: train QAOA fresh…

- [2026-11-04 12:17:31 +0800 Asia/Singapore] L=4: train power_p seed0…

- [2026-11-04 12:19:20 +0800 Asia/Singapore] L=4: DONE — RESULTS_L4.md written

- [2026-11-04 12:19:20 +0800 Asia/Singapore] L=8: train QAOA fresh…

- [2026-11-04 12:19:41 +0800 Asia/Singapore] L=8: train power_p seed0…

- [2026-11-04 12:23:10 +0800 Asia/Singapore] L=8: DONE — RESULTS_L8.md written

- [2026-11-04 12:23:10 +0800 Asia/Singapore] L=12: train QAOA fresh…

- [2026-11-04 12:23:39 +0800 Asia/Singapore] L=12: train power_p seed0…

- [2026-11-04 12:29:09 +0800 Asia/Singapore] L=12: DONE — RESULTS_L12.md written

- [2026-11-04 12:29:09 +0800 Asia/Singapore] L=16: train QAOA fresh…

- [2026-11-04 12:29:48 +0800 Asia/Singapore] L=16: train power_p seed0…

- [2026-11-04 12:37:28 +0800 Asia/Singapore] L=16: DONE — RESULTS_L16.md written

- [2026-11-04 12:37:28 +0800 Asia/Singapore] L=20: train QAOA fresh…

- [2026-11-04 12:38:23 +0800 Asia/Singapore] L=20: train power_p seed0…

- [2026-11-04 12:47:03 +0800 Asia/Singapore] L=20: DONE — RESULTS_L20.md written

- [2026-11-04 12:47:03 +0800 Asia/Singapore] L=24: train QAOA fresh…

- [2026-11-04 12:48:04 +0800 Asia/Singapore] L=24: train power_p seed0…
