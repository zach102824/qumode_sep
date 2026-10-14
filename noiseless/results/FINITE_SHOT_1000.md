# Finite-shot rerun of the sl3 recipe: S = 1000 shots per circuit eval

Run: `logs/shots1000/run.sh N...` (sl3 recipe = `logs/ovn/run_variant.sh sl3 3` with `QM_CLS_ALGO=satlike`, same args, 10 instances x 1 trial, same seeds; `--workers 6`; 3cav adds `--ecd-fast`) plus the new flag `--shots 1000`.
Raw output: `noiseless/results/relayout_runs/shots1000/*.jsonl` (gitignored), logs `logs/shots1000/`. Tables: `python noiseless/analyze_finite_shot.py`.

## Model (`noiseless/shots.py`, default path unchanged when `--shots` is 0)
- **Every circuit eval** draws 1000 shots (multinomial) from the exact Born distribution: every SPSA objective call, every eta-controller refresh, every `evaluate` (per-stage/per-round/final). RNG: `default_rng` seeded from a copy of the trial rng (does not shift the trial's own draws).
- **Training cost** = Gibbs objective on the empirical shot histogram only: energies of sampled strings weighted by their counts. No full-table emin: the shift is taken as the min over that eval's sampled support, and it cancels exactly in `-log Σ p e^{-η(E-emin)} + η·emin` (numerical only). The eta controller's quantiles also use only the shot histogram.
- **Leaked (out-of-code) shots** stay in the cost at the simulator's fixed leak-penalty energy (as in the exact cost). That's a constant, not a lookup, so they're never counted as strings.
- **Readout top-2n** = the 2n most frequent strings in that run's final-state 1000 shots. If fewer than 2n distinct valid strings were sampled, the remaining slots fall back to the zero-count strings in index order (no quantum information). We kept that to preserve the fixed-K lookup accounting. Fix-up and the SATLike classical search are unchanged. Round selection (`fun_common_eta`) also uses the shot histograms.
- **Reported success / exact p(GS)** = exact final-state values, same definition as before. "Shot p(GS)" is the GS frequency in the selected run's final 1000 shots.
- **Unique counts per trial:** unique sampled = union of valid sampled strings over all shot evals, mapped to the original problem labels (undoing XOR mask and bit permutation). Unique lookups = `n_lookups_distinct` (SA/SATLike + readout + fix-up, each once). Unique total = union of both.
- The training path differs from exact sl3 (the noisy cost changes SPSA), so success/p(GS) are NOT expected to match sl3.
- Caveat: the `nf_check` diagnostic after each 2cav trial now compares shot histograms, so `nf_check_max_abs_dp` in the summary JSON is meaningless for these runs. It doesn't affect any reported number.

## Results (exact numbers; means over trials)
| n | trials | success | mean exact p(GS) | mean shot p(GS) | sl3 exact-cost p(GS) | SPSA evals (nfev) | shot evals (all circuit evals) | raw lookups | unique lookups | unique sampled | unique total | 2^n | total/2^n mean | total/2^n max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 10/10 | 6/10 | 0.609 | 0.608 | 0.987 | 505 | 567 | 96 | 63.9 | 236.2 | 238.1 | 256 | 93.01% | 100.00% |
|  | per-trial: i0:Y p=0.989 samp=253 look=70 tot=253 ; i1:Y p=0.984 samp=254 look=68 tot=254 ; i2:Y p=0.986 samp=250 look=56 tot=250 ; i3:Y p=0.989 samp=255 look=62 tot=255 ; i4:n p=0.000 samp=133 look=57 tot=147 ; i5:Y p=0.993 samp=193 look=60 tot=198 ; i6:n p=0.145 samp=256 look=72 tot=256 ; i7:Y p=0.993 samp=256 look=60 tot=256 ; i8:n p=0.010 samp=256 look=71 tot=256 ; i9:n p=0.005 samp=256 look=63 tot=256 | | | | | | | | | | | | | |
| 10 | 10/10 | 5/10 | 0.445 | 0.446 | 0.988 | 585 | 655 | 120 | 85.2 | 657.7 | 669.2 | 1024 | 65.35% | 98.14% |
|  | per-trial: i0:Y p=0.986 samp=597 look=96 tot=613 ; i1:n p=0.000 samp=356 look=74 tot=375 ; i2:n p=0.003 samp=1004 look=100 tot=1005 ; i3:n p=0.003 samp=986 look=96 tot=987 ; i4:Y p=0.988 samp=197 look=76 tot=230 ; i5:Y p=0.977 samp=376 look=70 tot=389 ; i6:Y p=0.520 samp=862 look=101 tot=869 ; i7:n p=0.000 samp=267 look=73 tot=288 ; i8:n p=0.006 samp=965 look=76 tot=965 ; i9:Y p=0.970 samp=967 look=90 tot=971 | | | | | | | | | | | | | |
| 12 | 10/10 | 5/10 | 0.494 | 0.495 | 0.991 | 705 | 787 | 144 | 98.0 | 708.7 | 746.4 | 4096 | 18.22% | 52.86% |
|  | per-trial: i0:n p=0.000 samp=456 look=97 tot=496 ; i1:n p=0.000 samp=263 look=97 tot=319 ; i2:Y p=0.988 samp=1073 look=107 tot=1119 ; i3:Y p=0.991 samp=677 look=88 tot=706 ; i4:n p=0.000 samp=372 look=90 tot=411 ; i5:n p=0.000 samp=2152 look=106 tot=2165 ; i6:Y p=0.993 samp=225 look=94 tot=274 ; i7:Y p=0.987 samp=266 look=104 tot=316 ; i8:Y p=0.982 samp=1021 look=118 tot=1060 ; i9:n p=0.000 samp=582 look=79 tot=598 | | | | | | | | | | | | | |
| 14 | 10/10 | 3/10 | 0.270 | 0.271 | 0.961 | 727 | 817 | 168 | 129.0 | 8039.7 | 8055.9 | 16384 | 49.17% | 75.24% |
|  | per-trial: i0:n p=0.000 samp=9854 look=139 tot=9876 ; i1:n p=0.000 samp=7115 look=125 tot=7124 ; i2:Y p=0.961 samp=9229 look=114 tot=9235 ; i3:Y p=0.788 samp=8890 look=122 tot=8917 ; i4:n p=0.000 samp=12322 look=116 tot=12328 ; i5:n p=0.000 samp=4068 look=140 tot=4095 ; i6:n p=0.000 samp=6062 look=140 tot=6075 ; i7:Y p=0.949 samp=7530 look=139 tot=7550 ; i8:n p=0.000 samp=10861 look=131 tot=10867 ; i9:n p=0.000 samp=4466 look=124 tot=4492 | | | | | | | | | | | | | |

n=16-20 rows fill in as the detached run finishes; re-run the analyzer to refresh.
