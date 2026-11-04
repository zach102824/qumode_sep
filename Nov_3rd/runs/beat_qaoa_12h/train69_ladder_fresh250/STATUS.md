# STATUS — train69_ladder_fresh250 Phase 1 done / Phase 2

_Asia/Singapore (UTC+8) — started 2026-11-04 ~12:14_

## Locked recipe
- Motif: Hamming-shell SNAP |0⟩→[Uc^X→shell SNAP]^L→W→Fock
- Idea: **power_p** `a_n=(8/(n-1))^p`
- Fresh packs: 250 n6 + 250 n9 (new subsample RNG), disjoint from scale1000
- Train batch: 32+32; SNAP w6=1 w9=2
- QAOA: fresh train on same 250/250 mix, paper init β=0.01 γ=-0.01 nstart=1
- Phase 1 eval: n∈{6,9,12,15} B=1000; Med + B90; BEAT if raw Med < QAOA Med (same for B90)
- L_LIST: 1,4,8,12,16,20,24,28,32 small→large
- Phase 2: n18 **DONE**; n21 **IN PROGRESS**

## Progress
| Step | Status | Notes |
|------|--------|-------|
| PLAN.md | done | locked text |
| Fresh packs | done | 250 n6 + 250 n9 |
| QAOA train69 path | done | fresh per L |
| L=1 | done | all CORE Med BEAT |
| L=4 | done | all CORE Med BEAT |
| L=8 | done | all CORE Med BEAT |
| L=12 | done | all CORE Med BEAT |
| L=16 | done | all CORE Med BEAT |
| L=20 | done | all CORE Med BEAT |
| L=24 | done | all CORE Med BEAT; n15 B90 UNDER |
| L=28 | done | all CORE Med+B90 BEAT |
| L=32 | done | all CORE Med+B90 BEAT |
| Phase1 RESULTS.md | done | full L ladder Med+B90 |
| Phase2 n18 | done | all L Med BEAT; B90 BEAT at L=4,8,16; UNDER elsewhere |
| Phase2 n21 | in progress | eval existing angles on scale1000 n21 |

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

- [2026-11-04 12:58:07 +0800 Asia/Singapore] L=24: DONE — RESULTS_L24.md written

- [2026-11-04 12:58:07 +0800 Asia/Singapore] L=28: train QAOA fresh…

- [2026-11-04 12:59:18 +0800 Asia/Singapore] L=28: train power_p seed0…

- [2026-11-04 13:12:02 +0800 Asia/Singapore] L=28: DONE — RESULTS_L28.md written

- [2026-11-04 13:12:02 +0800 Asia/Singapore] L=32: train QAOA fresh…

- [2026-11-04 13:14:37 +0800 Asia/Singapore] L=32: train power_p seed0…

- [2026-11-04 13:31:24 +0800 Asia/Singapore] L=32: DONE — RESULTS_L32.md written

- [2026-11-04 13:31:24 +0800 Asia/Singapore] PHASE1 COMPLETE — all L in L_LIST have CORE Med/B90. Do NOT start Phase2.

- [2026-11-04 13:33:52 +0800 Asia/Singapore] STATUS progress table marked Phase1 complete; Phase2 still blocked (do not start n18/n21).

- [2026-11-04 13:36:51 +0800 Asia/Singapore] Phase2 START — n18 eval for all L (no retrain).

- [2026-11-04 13:36:51 +0800 Asia/Singapore] Phase2 IN PROGRESS — n18 then n21; no retrain.

- [2026-11-04 13:36:51 +0800 Asia/Singapore] scale1000 pack n18 present: /workspace/qumode_sep/Nov_3rd/data/scale1000/data_n18.npz

- [2026-11-04 13:36:51 +0800 Asia/Singapore] Phase2 n18: start eval ns=[18] for all L

- [2026-11-04 13:36:51 +0800 Asia/Singapore] Phase2 n18: L=1 QAOA eval…

- [2026-11-04 13:37:24 +0800 Asia/Singapore] Phase2 n18: L=1 SNAP power_p eval…

- [2026-11-04 13:37:50 +0800 Asia/Singapore] Phase2 n18: L=1 DONE

- [2026-11-04 13:37:50 +0800 Asia/Singapore] Phase2 n18: L=4 QAOA eval…

- [2026-11-04 13:39:17 +0800 Asia/Singapore] Phase2 n18: L=4 SNAP power_p eval…

- [2026-11-04 13:40:51 +0800 Asia/Singapore] Phase2 n18: L=4 DONE

- [2026-11-04 13:40:51 +0800 Asia/Singapore] Phase2 n18: L=8 QAOA eval…

- [2026-11-04 13:43:30 +0800 Asia/Singapore] Phase2 n18: L=8 SNAP power_p eval…

- [2026-11-04 13:45:13 +0800 Asia/Singapore] Phase2 n18: L=8 DONE

- [2026-11-04 13:45:13 +0800 Asia/Singapore] Phase2 n18: L=12 QAOA eval…

- [2026-11-04 13:49:12 +0800 Asia/Singapore] Phase2 n18: L=12 SNAP power_p eval…

- [2026-11-04 13:52:03 +0800 Asia/Singapore] Phase2 n18: L=12 DONE

- [2026-11-04 13:52:03 +0800 Asia/Singapore] Phase2 n18: L=16 QAOA eval…

- [2026-11-04 13:57:41 +0800 Asia/Singapore] Phase2 n18: L=16 SNAP power_p eval…

- [2026-11-04 14:01:14 +0800 Asia/Singapore] Phase2 n18: L=16 DONE

- [2026-11-04 14:01:14 +0800 Asia/Singapore] Phase2 n18: L=20 QAOA eval…

- [2026-11-04 14:08:06 +0800 Asia/Singapore] Phase2 n18: L=20 SNAP power_p eval…

- [2026-11-04 14:12:28 +0800 Asia/Singapore] Phase2 n18: L=20 DONE

- [2026-11-04 14:12:28 +0800 Asia/Singapore] Phase2 n18: L=24 QAOA eval…

- [2026-11-04 14:20:22 +0800 Asia/Singapore] Phase2 n18: L=24 SNAP power_p eval…

- [2026-11-04 14:25:20 +0800 Asia/Singapore] Phase2 n18: L=24 DONE

- [2026-11-04 14:25:20 +0800 Asia/Singapore] Phase2 n18: L=28 QAOA eval…

- [2026-11-04 14:35:16 +0800 Asia/Singapore] Phase2 n18: L=28 SNAP power_p eval…

- [2026-11-04 14:41:10 +0800 Asia/Singapore] Phase2 n18: L=28 DONE

- [2026-11-04 14:41:10 +0800 Asia/Singapore] Phase2 n18: L=32 QAOA eval…

- [2026-11-04 14:49:55 +0800 Asia/Singapore] Phase2 n18: L=32 SNAP power_p eval…

- [2026-11-04 14:57:26 +0800 Asia/Singapore] Phase2 n18: L=32 DONE

- [2026-11-04 14:57:26 +0800 Asia/Singapore] Phase2 n18: COMPLETE for all L

- [2026-11-04 14:57:26 +0800 Asia/Singapore] PHASE2 n18 COMPLETE — n21 not requested this run.

- [2026-11-04 15:01:01 +0800 Asia/Singapore] Phase2 n18 COMPLETE — all L Med BEAT. Watcher died on git status|head pipefail; executor resumed commit+n21.
