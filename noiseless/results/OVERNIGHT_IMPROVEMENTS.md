# Overnight improvements (one shared rule per variant, all n=8..20)

Baseline = FINAL one-recipe (r150, noise 0.3; MIN_LOOKUP.md): success 1.00 all n; mean p(GS) 0.987,0.986,0.990,0.961,0.972,0.963,0.957;
lookups 152,190,228,266,400,558,740; evals 505,585,705,727,1153,1579,2005.
Everything else identical to the final recipe (layouts, E, r150, sanoise 0.3, K=2n, polt 1, fixed cost). 10-trial screens on the 70 instances,
fresh outdirs `noiseless/results/relayout_runs/ovn_<name>`, queue logs `logs/ovn/`. Times CST (box + 6h).

Variants
- satlike10 / satlike5: classical stage = SATLike-style search (noiseless/cls_search.py: unit weights, swt 20, sp 0.01, BMS t=3; lookup = every
  candidate flip scored + starts, i.e. the strict convention of STRONG_CLASSICAL_BASELINES; no 64 probes) with budget 10n / 5n. Explore runs are seeded
  from its lowest-energy distinct strings (same marginals/layout_for_run code).
- tunedsa10: classical stage = random-restart SA, absolute T 2.0 -> 1.0, chains 16n, no probes, budget 10n.

| variant | n | env | trials | success | mean p(GS) | evals | SA / readout / fix-up lookups | lookups per trial | finished CST |
|---|---|---|---|---|---|---|---|---|---|
| satlike10 | 8 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.987 | 505 | 80 / 48 / 24 | 152 | 17:58 |
| satlike10 | 10 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.986 | 585 | 100 / 60 / 30 | 190 | 17:58 |
| satlike10 | 12 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.992 | 705 | 120 / 72 / 36 | 228 | 17:58 |
| satlike10 | 14 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.961 | 727 | 140 / 84 / 42 | 266 | 17:59 |
| satlike10 | 16 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.972 | 1153 | 160 / 160 / 80 | 400 | 18:13 |
| satlike10 | 18 | QM_CLS_ALGO=satlike | 10 | 0.90 | 0.869 | 1579 | 180 / 252 / 126 | 558 | 18:43 |
| satlike5 | 8 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.987 | 505 | 40 / 48 / 24 | 112 | 18:43 |
| satlike5 | 10 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.986 | 585 | 50 / 60 / 30 | 140 | 18:43 |
| satlike5 | 12 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.992 | 705 | 60 / 72 / 36 | 168 | 18:43 |
| satlike5 | 14 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.961 | 727 | 70 / 84 / 42 | 196 | 18:45 |
| satlike5 | 16 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.972 | 1153 | 80 / 160 / 80 | 320 | 18:54 |
| satlike5 | 18 | QM_CLS_ALGO=satlike | 10 | 1.00 | 0.963 | 1579 | 90 / 252 / 126 | 468 | 19:10 |
| tunedsa10 | 8 | QM_CLS_ALGO=tunedsa | 10 | 1.00 | 0.987 | 505 | 80 / 48 / 24 | 152 | 19:10 |
| tunedsa10 | 10 | QM_CLS_ALGO=tunedsa | 10 | 1.00 | 0.986 | 585 | 100 / 60 / 30 | 190 | 19:11 |
| tunedsa10 | 12 | QM_CLS_ALGO=tunedsa | 10 | 1.00 | 0.991 | 705 | 120 / 72 / 36 | 228 | 19:11 |
| tunedsa10 | 14 | QM_CLS_ALGO=tunedsa | 10 | 1.00 | 0.961 | 727 | 140 / 84 / 42 | 266 | 19:12 |
| tunedsa10 | 16 | QM_CLS_ALGO=tunedsa | 10 | 1.00 | 0.972 | 1153 | 160 / 160 / 80 | 400 | 19:15 |
| tunedsa10 | 18 | QM_CLS_ALGO=tunedsa | 10 | 1.00 | 0.963 | 1579 | 180 / 252 / 126 | 558 | 19:37 |
| satlike5 | 20 | QM_CLS_ALGO=satlike  r=150  | 10 | 0.90 | 0.861 | 2005 | 100 / 360 / 180 | 640 | 21:04 |
| sl5_Elow | 16 | QM_CLS_ALGO=satlike EMAP=16:2,18:3,20:4 r=150     | 10 | 0.90 | 0.876 | 727 | 80 / 96 / 48 | 224 | 21:06 |

## Status notes (auto)
- satlike5 (budget 5n, same quantum recipe): **passes n=8..18** with fewer lookups (112..468 vs 152..558), same evals; **fails n=20** (success 0.90, mean p(GS) 0.861; instance 9 collapsed, p=0.0007; lookups 640 vs 740).
- satlike10: fails n=18 (0.90 / 0.869).
- tunedsa10: passes n=8..18 at **same** lookups as final (not a cost win); n=20 not run.
- Next: sl5_Elow (fewer explore rounds), sl3, sl5_r100 (queue2); K=n / CW / r75 (queue3); sl5div + sl10rs (queue4 after q2+q3).
| sl5_Elow | 18 | killed after n16 failed (0.90) — variant rejected | | | | | | | 21:14 |
| sl3 | 18 | QM_CLS_ALGO=satlike  r=150     | 10 | 1.00 | 0.963 | 1579 | 54 / 252 / 126 | 432 | 21:25 |
| sl3 | 16 | QM_CLS_ALGO=satlike  r=150     | 10 | 1.00 | 0.972 | 1153 | 48 / 160 / 80 | 288 | 22:56 |
| sl3 | 14 | QM_CLS_ALGO=satlike  r=150     | 10 | 1.00 | 0.961 | 727 | 42 / 84 / 42 | 168 | 22:57 |
| sl3 | 12 | QM_CLS_ALGO=satlike  r=150     | 10 | 1.00 | 0.991 | 705 | 36 / 72 / 36 | 144 | 22:57 |
| sl3 | 10 | QM_CLS_ALGO=satlike  r=150     | 10 | 1.00 | 0.988 | 585 | 30 / 60 / 30 | 120 | 22:57 |
| sl3 | 8 | QM_CLS_ALGO=satlike  r=150     | 10 | 1.00 | 0.987 | 505 | 24 / 48 / 24 | 96 | 22:58 |
| sl5_r100 | 18 | QM_CLS_ALGO=satlike  r=100      | 10 | 1.00 | 0.909 | 1479 | 90 / 252 / 126 | 468 | 23:08 |
| sl5_r100 | 16 | QM_CLS_ALGO=satlike  r=100      | 10 | 1.00 | 0.923 | 1053 | 80 / 160 / 80 | 320 | 23:19 |
| sl5_r100 | 14 | QM_CLS_ALGO=satlike  r=100      | 10 | 1.00 | 0.912 | 627 | 70 / 84 / 42 | 196 | 23:20 |
| sl5_r100 | 12 | QM_CLS_ALGO=satlike  r=100      | 10 | 1.00 | 0.985 | 605 | 60 / 72 / 36 | 168 | 23:21 |
| sl5_r100 | 10 | QM_CLS_ALGO=satlike  r=100      | 10 | 1.00 | 0.956 | 485 | 50 / 60 / 30 | 140 | 23:21 |
| sl5_r100 | 8 | QM_CLS_ALGO=satlike  r=100      | 10 | 1.00 | 0.964 | 405 | 40 / 48 / 24 | 112 | 23:21 |
| sl5_r100 | 20 | QM_CLS_ALGO=satlike  r=100      | 10 | 0.90 | 0.796 | 1905 | 100 / 360 / 180 | 640 | 06:58 |
| sl3 | 20 | QM_CLS_ALGO=satlike  r=150      | 10 | 1.00 | 0.959 | 2005 | 60 / 360 / 180 | 600 | 10:13 |
| sl5_Kn | 8 | QM_CLS_ALGO=satlike  r=150  Kmul=1    | 10 | 1.00 | 0.987 | 505 | 40 / 24 / 24 | 88 | 10:13 |
| sl5_Kn | 10 | QM_CLS_ALGO=satlike  r=150  Kmul=1    | 10 | 1.00 | 0.986 | 585 | 50 / 30 / 30 | 110 | 10:14 |
