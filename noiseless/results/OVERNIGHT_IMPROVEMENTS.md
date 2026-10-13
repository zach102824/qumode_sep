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
