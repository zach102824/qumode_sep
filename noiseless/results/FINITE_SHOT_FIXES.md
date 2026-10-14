# Finite-shot fixes: 10-trial screens at n = 8, 10 (sl3 recipe, same 10 instances/seeds as FINITE_SHOT_1000.md)

Run: `logs/shotfix/run.sh V S n...` (driver `logs/shotfix/all.sh`). Same args as `logs/shots1000/run.sh` 2cav rows (sl3, `QM_CLS_ALGO=satlike`, `--workers 6`), plus `--shots S` and the fix flags below. Raw: `noiseless/results/relayout_runs/shotfix/<V>_S<S>/*.jsonl` (gitignored). Table: `python noiseless/analyze_finite_shot_fixes.py`.

## Variants (new flags, defaults unchanged; `--shots S` with no fix flags reproduces FINITE_SHOT_1000 bit-identically: checked per trial, p(GS) and unique totals, n=8 and 10)
- **V0** plain shot code.
- **V1** `--shot-readout lowE`: top-2n readout handed to fix-up = the 2n lowest-energy distinct strings among all strings sampled during that run (every SPSA eval, eta refresh, final evaluate; energies already known from the cost). Padded with the frequency order only if fewer than 2n were sampled. Round selection (`sel`) and best-so-far use the run's best sampled energy (ties: shot Gibbs at common eta, then index).
- **V2** V1 + `--shot-crn`: the +delta and -delta evals of each SPSA step draw their shots from one shared seed (taken from the trial's shot RNG). Rademacher stream unchanged.
- **V3** V2 + `--shot-cost cvar --cvar-alpha 0.1`: training cost = mean energy of the lowest 10% of shots (fractional weight at the boundary). The eta controller still runs, but the cost ignores eta.
- Success / exact p(GS) = exact final state of the selected round (same definition as before). Unique total = union of all sampled strings and all classical lookups, each counted once.

## Results (means over 10 trials, 1 trial per instance)
| variant | S | n | success | mean exact p(GS) | mean shot p(GS) | SPSA evals (nfev) | shot evals | raw lookups | unique lookups | unique total | 2^n | total/2^n mean | total/2^n max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 plain | 100 | 8 | 2/10 | 0.211 | 0.221 | 505 | 567 | 96 | 63.4 | 232.1 | 256 | 90.66% | 100.00% |
| V0 plain | 100 | 10 | 6/10 | 0.523 | 0.519 | 585 | 655 | 120 | 84.4 | 520.5 | 1024 | 50.83% | 84.18% |
| V1 lowE readout | 100 | 8 | 6/10 | 0.594 | 0.608 | 505 | 567 | 96 | 47.0 | 229.8 | 256 | 89.77% | 100.00% |
| V1 lowE readout | 100 | 10 | 6/10 | 0.523 | 0.519 | 585 | 655 | 120 | 69.9 | 514.5 | 1024 | 50.24% | 84.18% |
| V2 lowE + CRN | 100 | 8 | 9/10 | 0.866 | 0.870 | 505 | 567 | 96 | 49.7 | 218.3 | 256 | 85.27% | 99.22% |
| V2 lowE + CRN | 100 | 10 | 6/10 | 0.531 | 0.533 | 585 | 655 | 120 | 71.4 | 480.4 | 1024 | 46.91% | 78.32% |
| V3 lowE + CRN + CVaR0.1 | 100 | 8 | 7/10 | 0.345 | 0.348 | 505 | 567 | 96 | 50.9 | 207.2 | 256 | 80.94% | 92.58% |
| V3 lowE + CRN + CVaR0.1 | 100 | 10 | 4/10 | 0.250 | 0.243 | 585 | 655 | 120 | 70.1 | 404.9 | 1024 | 39.54% | 82.62% |
| V0 plain | 1000 | 8 | 6/10 | 0.609 | 0.608 | 505 | 567 | 96 | 63.9 | 238.1 | 256 | 93.01% | 100.00% |
| V0 plain | 1000 | 10 | 5/10 | 0.445 | 0.446 | 585 | 655 | 120 | 85.2 | 669.2 | 1024 | 65.35% | 98.14% |
| V1 lowE readout | 1000 | 8 | 9/10 | 0.878 | 0.878 | 505 | 567 | 96 | 46.3 | 237.3 | 256 | 92.70% | 100.00% |
| V1 lowE readout | 1000 | 10 | 8/10 | 0.741 | 0.741 | 585 | 655 | 120 | 70.8 | 667.2 | 1024 | 65.16% | 98.05% |
| V2 lowE + CRN | 1000 | 8 | 8/10 | 0.790 | 0.791 | 505 | 567 | 96 | 46.8 | 228.2 | 256 | 89.14% | 100.00% |
| V2 lowE + CRN | 1000 | 10 | 8/10 | 0.725 | 0.727 | 585 | 655 | 120 | 68.2 | 693.6 | 1024 | 67.73% | 98.63% |
| V3 lowE + CRN + CVaR0.1 | 1000 | 8 | 9/10 | 0.388 | 0.386 | 505 | 567 | 96 | 50.1 | 237.2 | 256 | 92.66% | 99.61% |
| V3 lowE + CRN + CVaR0.1 | 1000 | 10 | 5/10 | 0.246 | 0.248 | 585 | 655 | 120 | 70.0 | 538.7 | 1024 | 52.61% | 97.27% |

Per-trial:
- V0 S=100 n=8: i0:n p=0.000 tot=242 ; i1:n p=0.000 tot=243 ; i2:n p=0.007 tot=238 ; i3:n p=0.016 tot=254 ; i4:n p=0.000 tot=152 ; i5:Y p=0.969 tot=256 ; i6:n p=0.173 tot=256 ; i7:Y p=0.950 tot=248 ; i8:n p=0.000 tot=223 ; i9:n p=0.000 tot=209
- V0 S=100 n=10: i0:Y p=0.982 tot=507 ; i1:n p=0.000 tot=265 ; i2:n p=0.000 tot=401 ; i3:Y p=0.952 tot=862 ; i4:Y p=0.950 tot=284 ; i5:Y p=0.977 tot=291 ; i6:Y p=0.390 tot=556 ; i7:n p=0.000 tot=778 ; i8:Y p=0.974 tot=627 ; i9:n p=0.000 tot=634
- V1 S=100 n=8: i0:n p=0.000 tot=242 ; i1:Y p=0.968 tot=248 ; i2:Y p=0.938 tot=223 ; i3:Y p=0.976 tot=254 ; i4:n p=0.000 tot=146 ; i5:Y p=0.969 tot=256 ; i6:n p=0.173 tot=256 ; i7:Y p=0.950 tot=247 ; i8:n p=0.000 tot=222 ; i9:Y p=0.968 tot=204
- V1 S=100 n=10: i0:Y p=0.982 tot=501 ; i1:n p=0.000 tot=248 ; i2:n p=0.000 tot=395 ; i3:Y p=0.952 tot=862 ; i4:Y p=0.950 tot=270 ; i5:Y p=0.977 tot=288 ; i6:Y p=0.390 tot=552 ; i7:n p=0.000 tot=777 ; i8:Y p=0.974 tot=626 ; i9:n p=0.000 tot=626
- V2 S=100 n=8: i0:Y p=0.975 tot=219 ; i1:Y p=0.977 tot=237 ; i2:Y p=0.979 tot=217 ; i3:Y p=0.977 tot=254 ; i4:n p=0.000 tot=125 ; i5:Y p=0.994 tot=219 ; i6:Y p=0.907 tot=238 ; i7:Y p=0.985 tot=249 ; i8:Y p=0.895 tot=212 ; i9:Y p=0.973 tot=213
- V2 S=100 n=10: i0:Y p=0.987 tot=495 ; i1:Y p=0.956 tot=621 ; i2:n p=0.000 tot=398 ; i3:Y p=0.989 tot=802 ; i4:Y p=0.992 tot=155 ; i5:Y p=0.973 tot=327 ; i6:Y p=0.411 tot=552 ; i7:n p=0.000 tot=311 ; i8:n p=0.000 tot=600 ; i9:n p=0.000 tot=543
- V3 S=100 n=8: i0:Y p=0.484 tot=172 ; i1:Y p=0.401 tot=237 ; i2:Y p=0.326 tot=204 ; i3:n p=0.263 tot=221 ; i4:Y p=0.528 tot=176 ; i5:Y p=0.355 tot=179 ; i6:Y p=0.466 tot=206 ; i7:n p=0.000 tot=217 ; i8:Y p=0.309 tot=235 ; i9:n p=0.320 tot=225
- V3 S=100 n=10: i0:n p=0.295 tot=391 ; i1:n p=0.000 tot=201 ; i2:n p=0.000 tot=216 ; i3:Y p=0.694 tot=699 ; i4:n p=0.340 tot=169 ; i5:Y p=0.380 tot=320 ; i6:Y p=0.402 tot=490 ; i7:n p=0.000 tot=242 ; i8:Y p=0.392 tot=846 ; i9:n p=0.000 tot=475
- V0 S=1000 n=8: i0:Y p=0.989 tot=253 ; i1:Y p=0.984 tot=254 ; i2:Y p=0.986 tot=250 ; i3:Y p=0.989 tot=255 ; i4:n p=0.000 tot=147 ; i5:Y p=0.993 tot=198 ; i6:n p=0.145 tot=256 ; i7:Y p=0.993 tot=256 ; i8:n p=0.010 tot=256 ; i9:n p=0.005 tot=256
- V0 S=1000 n=10: i0:Y p=0.986 tot=613 ; i1:n p=0.000 tot=375 ; i2:n p=0.003 tot=1005 ; i3:n p=0.003 tot=987 ; i4:Y p=0.988 tot=230 ; i5:Y p=0.977 tot=389 ; i6:Y p=0.520 tot=869 ; i7:n p=0.000 tot=288 ; i8:n p=0.006 tot=965 ; i9:Y p=0.970 tot=971
- V1 S=1000 n=8: i0:Y p=0.989 tot=253 ; i1:Y p=0.984 tot=254 ; i2:Y p=0.986 tot=250 ; i3:Y p=0.989 tot=255 ; i4:n p=0.000 tot=140 ; i5:Y p=0.993 tot=197 ; i6:Y p=0.984 tot=256 ; i7:Y p=0.993 tot=256 ; i8:Y p=0.891 tot=256 ; i9:Y p=0.976 tot=256
- V1 S=1000 n=10: i0:Y p=0.986 tot=610 ; i1:n p=0.000 tot=372 ; i2:Y p=0.994 tot=1004 ; i3:Y p=0.990 tot=990 ; i4:Y p=0.988 tot=220 ; i5:Y p=0.977 tot=386 ; i6:Y p=0.520 tot=869 ; i7:n p=0.000 tot=276 ; i8:Y p=0.984 tot=974 ; i9:Y p=0.970 tot=971
- V2 S=1000 n=8: i0:Y p=0.985 tot=239 ; i1:Y p=0.984 tot=245 ; i2:Y p=0.992 tot=242 ; i3:Y p=0.992 tot=225 ; i4:n p=0.000 tot=162 ; i5:Y p=0.996 tot=221 ; i6:Y p=0.981 tot=256 ; i7:Y p=0.990 tot=256 ; i8:Y p=0.983 tot=256 ; i9:n p=0.000 tot=180
- V2 S=1000 n=10: i0:Y p=0.983 tot=733 ; i1:n p=0.000 tot=326 ; i2:Y p=0.992 tot=1010 ; i3:Y p=0.988 tot=849 ; i4:Y p=0.994 tot=218 ; i5:Y p=0.983 tot=794 ; i6:Y p=0.968 tot=904 ; i7:n p=0.000 tot=287 ; i8:Y p=0.988 tot=875 ; i9:Y p=0.356 tot=940
- V3 S=1000 n=8: i0:Y p=0.339 tot=235 ; i1:Y p=0.351 tot=253 ; i2:Y p=0.414 tot=238 ; i3:Y p=0.353 tot=255 ; i4:Y p=0.528 tot=215 ; i5:Y p=0.291 tot=234 ; i6:Y p=0.543 tot=224 ; i7:Y p=0.337 tot=246 ; i8:Y p=0.457 tot=217 ; i9:n p=0.270 tot=255
- V3 S=1000 n=10: i0:Y p=0.359 tot=620 ; i1:n p=0.000 tot=308 ; i2:n p=0.000 tot=346 ; i3:Y p=0.504 tot=996 ; i4:n p=0.340 tot=212 ; i5:Y p=0.344 tot=376 ; i6:Y p=0.454 tot=843 ; i7:n p=0.000 tot=363 ; i8:Y p=0.458 tot=760 ; i9:n p=0.000 tot=563

Exact-cost sl3 reference: 10/10, p(GS) 0.987 (n=8), 0.988 (n=10).

## Reading
- **Lowest-energy readout (V1) is the main fix at S=1000**: 6->9/10 (n=8), 5->8/10 (n=10), and mean p(GS) 0.609->0.878, 0.445->0.741, at zero extra quantum cost, and unique total strings stay essentially the same. At S=100 it helps at n=8 (2->6) but not at n=10 (6->6).
- **CRN (V2)** helps most at S=100: 9/10 at n=8, 6/10 at n=10. At S=1000 it's neutral (8/10, 8/10).
- **CVaR alpha=0.1 (V3)** often finds the GS as most-likely string, but p(GS) stalls at ~0.25-0.39: the cost is already minimal once ~10% of the mass is on the GS, so it doesn't concentrate. Not useful as-is (it would need alpha annealed toward 1 or a final Gibbs/mean-energy stage).
- Remaining failures (e.g. n=10 inst 1, 7) fail in all variants: the GS never reached the readout/fix-up.
- Unique strings checked stay large (~40-68% of 2^n at n=10, ~81-93% at n=8). Shots dominate the count.
- Caveat: 10 trials, so differences of 1-2 successes are within noise.

## n = 14, 16 screens (3cav, --ecd-fast, sl3 args as logs/shots1000/run.sh: n=14 ks 4,4,3 E=2; n=16 ks 5,4,4 E=4; 8 workers)

Run: `logs/shotfix/all1416.sh` (V2 S=100, V1 S=1000, V0 S=100). Table: `python noiseless/analyze_finite_shot_fixes.py 14 16`.

| variant | S | n | success | mean exact p(GS) | mean shot p(GS) | SPSA evals (nfev) | shot evals | raw lookups | unique lookups | unique total | 2^n | total/2^n mean | total/2^n max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 plain | 100 | 14 | 4/10 | 0.378 | 0.383 | 727 | 817 | 168 | 137.3 | 4358.5 | 16384 | 26.60% | 46.77% |
| V0 plain | 100 | 16 | 2/10 | 0.159 | 0.160 | 1153 | 1301 | 288 | 226.6 | 14208.2 | 65536 | 21.68% | 29.13% |
| V2 lowE + CRN | 100 | 14 | 6/10 | 0.500 | 0.511 | 727 | 817 | 168 | 104.3 | 3025.2 | 16384 | 18.46% | 36.00% |
| V2 lowE + CRN | 100 | 16 | 3/10 | 0.254 | 0.251 | 1153 | 1301 | 288 | 184.8 | 10477.1 | 65536 | 15.99% | 26.80% |
| V1 lowE readout | 1000 | 14 | 8/10 | 0.752 | 0.755 | 727 | 817 | 168 | 110.9 | 8002.9 | 16384 | 48.85% | 74.92% |
| V1 lowE readout | 1000 | 16 | 7/10 | 0.671 | 0.672 | 1153 | 1301 | 288 | 190.6 | 30469.1 | 65536 | 46.49% | 62.80% |

Per-trial:
- V0 S=100 n=14: i0:n p=0.000 tot=4494 ; i1:n p=0.000 tot=3896 ; i2:Y p=0.942 tot=7662 ; i3:n p=0.069 tot=3538 ; i4:n p=0.000 tot=2766 ; i5:Y p=0.898 tot=4888 ; i6:Y p=0.960 tot=4290 ; i7:Y p=0.908 tot=2436 ; i8:n p=0.000 tot=4016 ; i9:n p=0.000 tot=5599
- V0 S=100 n=16: i0:n p=0.000 tot=13405 ; i1:Y p=0.662 tot=19090 ; i2:n p=0.000 tot=14108 ; i3:Y p=0.929 tot=15746 ; i4:n p=0.000 tot=13461 ; i5:n p=0.000 tot=11185 ; i6:n p=0.000 tot=10652 ; i7:n p=0.000 tot=18076 ; i8:n p=0.000 tot=10152 ; i9:n p=0.000 tot=16207
- V2 S=100 n=14: i0:Y p=0.936 tot=2827 ; i1:n p=0.000 tot=1932 ; i2:Y p=0.976 tot=1483 ; i3:Y p=0.269 tot=3053 ; i4:n p=0.000 tot=1687 ; i5:Y p=0.883 tot=4593 ; i6:Y p=0.977 tot=5899 ; i7:Y p=0.959 tot=1904 ; i8:n p=0.000 tot=1506 ; i9:n p=0.000 tot=5368
- V2 S=100 n=16: i0:Y p=0.951 tot=16557 ; i1:Y p=0.634 tot=7993 ; i2:n p=0.000 tot=17565 ; i3:Y p=0.958 tot=11565 ; i4:n p=0.000 tot=5857 ; i5:n p=0.000 tot=15017 ; i6:n p=0.000 tot=3902 ; i7:n p=0.000 tot=10441 ; i8:n p=0.000 tot=6876 ; i9:n p=0.000 tot=8998
- V1 S=1000 n=14: i0:Y p=0.975 tot=9894 ; i1:n p=0.000 tot=7170 ; i2:Y p=0.961 tot=9234 ; i3:Y p=0.788 tot=8917 ; i4:Y p=0.958 tot=12275 ; i5:Y p=0.942 tot=4159 ; i6:Y p=0.983 tot=5354 ; i7:Y p=0.949 tot=7546 ; i8:Y p=0.961 tot=10850 ; i9:n p=0.000 tot=4630
- V1 S=1000 n=16: i0:Y p=0.951 tot=21649 ; i1:Y p=0.931 tot=25974 ; i2:Y p=0.982 tot=40667 ; i3:Y p=0.973 tot=37573 ; i4:Y p=0.976 tot=30502 ; i5:Y p=0.935 tot=39711 ; i6:Y p=0.966 tot=41159 ; i7:n p=0.000 tot=22914 ; i8:n p=0.000 tot=25397 ; i9:n p=0.000 tot=19145

Exact-cost sl3 reference: 10/10, p(GS) 0.961 (n=14), 0.972 (n=16).

## Reading (n=14, 16)
- Success degrades with n under shots: V2 S=100 6/10 -> 3/10, V1 S=1000 8/10 -> 7/10, V0 S=100 4/10 -> 2/10 (exact cost: 10/10 both).
- Unique share of 2^n does drop with n, but slowly: V2 S=100 46.91% (n=10) -> 18.46% (n=14) -> 15.99% (n=16); V1 S=1000 65.16% -> 48.85% -> 46.49%. n=12 was not run in this screen.
- S=1000 is the only setting that stays reasonably accurate, at ~3x the unique strings of S=100.
