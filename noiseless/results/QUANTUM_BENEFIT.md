# Does the quantum part add value beyond classical search? (2026-10-14, CST)

Exact-cost (noiseless, no shots) '3n' recipe from OCT14_FINAL_DATA.md, same 10 instances per n, 1 trial each, 8 workers.
Lookups = strict fixed count (every evaluated string, repeats included). Evals = quantum cost evals (same counting as the slide).
Runner: `logs/qbenefit/run.sh VARIANT n...`; summary: `python logs/qbenefit/summ.py`. Data: `noiseless/results/relayout_runs/qbenefit/<VARIANT>/`.

New opt-in env flags in `noiseless/relayout_kbit.py` (default behaviour unchanged; BASE n=8 reproduces 10/10, 0.987):
- `QM_NOCLS=1`: no classical search. Explore centers are uniform random strings, marginals flat (layouts differ only by layout noise); 1 lookup for the first random center.
- `QM_RANDREAD=1`: every round's quantum top-2n readout replaced by 2n uniform random strings (same lookup count); fix-up and relabel unchanged.

## Test 1: quantum found the GS when the classical stage missed it (existing ovn_sl3 runs)
| n | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|
| classical best = GS | 1/10 | 4/10 | 5/10 | 3/10 | 2/10 |
| success anyway (classical missed) | 9/9 | 6/6 | 5/5 | 7/7 | 8/8 |

## Test 2: no classical search (QM_NOCLS)
| n | success | mean p(GS) | lookups | evals |
|---|---|---|---|---|
| 8 | 9/10 | 0.889 | 73 | 505 |
| 10 | 10/10 | 0.986 | 91 | 585 |
| 12 | 10/10 | 0.989 | 109 | 705 |
| 14 | 10/10 | 0.961 | 127 | 727 |
| 16 | 10/10 | 0.972 | 241 | 1153 |

## Test 3: quantum readout replaced by uniform random strings
With the classical stage (QM_RANDREAD):
| n | success | mean p(GS) | lookups | evals |
|---|---|---|---|---|
| 8 | 7/10 | 0.440 | 96 | 505 |
| 10 | 6/10 | 0.514 | 120 | 585 |
| 12 | 7/10 | 0.514 | 144 | 705 |
| 14 | 8/10 | 0.421 | 168 | 727 |
| 16 | 5/10 | 0.205 | 288 | 1153 |

No classical stage and random readout (QM_NOCLS + QM_RANDREAD), the control for Test 2:
| n | success | mean p(GS) | lookups | evals |
|---|---|---|---|---|
| 8 | 6/10 | 0.434 | 73 | 505 |
| 10 | 5/10 | 0.339 | 91 | 585 |
| 12 | 3/10 | 0.030 | 109 | 705 |
| 14 | 3/10 | 0.142 | 127 | 727 |
| 16 | 3/10 | 0.039 | 241 | 1153 |

## Strengthening: no classical search, E=3 explore runs (QM_NOCLS, --explore 3)
| n | success | mean p(GS) | lookups | evals |
|---|---|---|---|---|
| 8 | 10/10 | 0.974 | 97 | 607 |
| 10 | 10/10 | 0.992 | 121 | 727 |
| 12 | 10/10 | 0.986 | 145 | 907 |

## Conclusion
In the exact-cost setting the method does not rely on the classical search: with zero classical search it keeps 10/10 at n=10-16 (9/10 at n=8, fixed by E=3) and needs 24-47 fewer lookups (73/91/109/127/241 vs 96/120/144/168/288). Replacing the quantum readout with random strings drops success to 3/10-6/10 without classical search, so the quantum explore runs carry the result. For reference, standalone SATLike needs 357/280/1,047/422/752 lookups at matched success (STANDALONE_SATLIKE numbers in this chat's earlier work).

Caveats: noiseless exact cost (the explore cost uses exact probabilities, so the earlier finite-shot result, where shots miss the rare GS tail, still applies); quantum evals are not counted as lookups; 10-trial screens; when the final relabel guess is the GS, the relabel round is identical to the baseline, so p(GS) matches the baseline per instance.
