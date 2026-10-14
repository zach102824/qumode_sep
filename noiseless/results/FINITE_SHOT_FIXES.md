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

## R1-R3 screens (Oct 14, n = 10, 12, 14, 16; lowE readout + CRN always on; same instances/seeds/sl3 args, --ecd-fast for 3cav; 8 workers)

New flags (defaults unchanged): `--shot-s0 S0` (R2: shots grow geometrically S0 -> `--shots` over every SPSA stage, restarting each stage), `--shot-eta-anneal F` (R3: the shot Gibbs cost uses eta*F^(frac-1), i.e. eta/F at the start of every stage growing geometrically to the controller's eta at its end). Both track stage progress in the CRN SPSA loop, so they require `--shot-crn`. R1 (warm start) uses the existing `--xbeta 1`: the first explore growth stage starts with |beta| ~ U(0,1) instead of U(0,3), so the explore state starts near the vacuum, which the SA explore mask maps to the classical guess; early training is less broad. Records now carry `n_shots_total`.
Settings: B100 = S=100 lowE+CRN (= V2 S100; n=12 newly run); R1 = B100 + xbeta 1; R2 = S 50->1000; R3 = S=100 + eta anneal F=16; C23 = R2+R3; C123 = R1+R2+R3; C3S1000 = S=1000 + R3; C23b = S 100->3000 + R3.
Run: `logs/shotfix2/all.sh`, `logs/shotfix2/combo.sh`. Table: `python noiseless/analyze_finite_shot_r123.py B100 R1 R2 R3 C23 C123 C3S1000 C23b`. "unique strings checked" = union of all sampled strings and all lookups, each counted once.

| setting | n | success | mean exact p(GS) | unique strings checked (mean) | share of 2^n mean | share max | total shots (mean) | SPSA evals (nfev) | shot evals |
|---|---|---|---|---|---|---|---|---|---|
| B100 | 10 | 6/10 | 0.531 | 480.4 | 46.91% | 78.32% | 65,500 | 585 | 655 |
| B100 | 12 | 6/10 | 0.587 | 929.5 | 22.69% | 57.40% | 78,700 | 705 | 787 |
| B100 | 14 | 6/10 | 0.500 | 3025.2 | 18.46% | 36.00% | 81,700 | 727 | 817 |
| B100 | 16 | 3/10 | 0.254 | 10477.1 | 15.99% | 26.80% | 130,100 | 1153 | 1301 |
| R1 | 10 | 6/10 | 0.518 | 552.5 | 53.96% | 89.06% | 65,500 | 585 | 655 |
| R1 | 12 | 5/10 | 0.493 | 276.5 | 6.75% | 9.77% | 78,700 | 705 | 787 |
| R1 | 14 | 5/10 | 0.424 | 3218.1 | 19.64% | 31.13% | 81,700 | 727 | 817 |
| R1 | 16 | 2/10 | 0.149 | 6447.8 | 9.84% | 21.30% | 130,100 | 1153 | 1301 |
| R2 | 10 | 7/10 | 0.629 | 559.0 | 54.59% | 89.26% | 219,768 | 585 | 655 |
| R2 | 12 | 5/10 | 0.492 | 447.3 | 10.92% | 17.65% | 261,648 | 705 | 787 |
| R2 | 14 | 7/10 | 0.552 | 5184.3 | 31.64% | 43.03% | 276,788 | 727 | 817 |
| R2 | 16 | 6/10 | 0.542 | 13028.3 | 19.88% | 31.10% | 445,848 | 1153 | 1301 |
| R3 | 10 | 7/10 | 0.689 | 473.7 | 46.26% | 84.67% | 65,500 | 585 | 655 |
| R3 | 12 | 5/10 | 0.494 | 793.8 | 19.38% | 46.92% | 78,700 | 705 | 787 |
| R3 | 14 | 6/10 | 0.558 | 4405.0 | 26.89% | 40.97% | 81,700 | 727 | 817 |
| R3 | 16 | 3/10 | 0.280 | 9843.3 | 15.02% | 29.33% | 130,100 | 1153 | 1301 |
| C23 | 10 | 8/10 | 0.778 | 490.3 | 47.88% | 93.07% | 219,768 | 585 | 655 |
| C23 | 12 | 5/10 | 0.494 | 481.6 | 11.76% | 32.91% | 261,648 | 705 | 787 |
| C23 | 14 | 8/10 | 0.755 | 6257.6 | 38.19% | 64.66% | 276,788 | 727 | 817 |
| C23 | 16 | 6/10 | 0.541 | 13114.3 | 20.01% | 37.60% | 445,848 | 1153 | 1301 |
| C123 | 10 | 7/10 | 0.691 | 453.4 | 44.28% | 89.84% | 219,768 | 585 | 655 |
| C123 | 12 | 5/10 | 0.495 | 287.8 | 7.03% | 9.81% | 261,648 | 705 | 787 |
| C123 | 14 | 5/10 | 0.450 | 3968.9 | 24.22% | 40.08% | 276,788 | 727 | 817 |
| C123 | 16 | 4/10 | 0.376 | 10125.9 | 15.45% | 24.82% | 445,848 | 1153 | 1301 |
| C3S1000 | 10 | 8/10 | 0.788 | 667.2 | 65.16% | 99.12% | 655,000 | 585 | 655 |
| C3S1000 | 12 | 6/10 | 0.589 | 789.4 | 19.27% | 58.25% | 787,000 | 705 | 787 |
| C3S1000 | 14 | 7/10 | 0.653 | 5567.6 | 33.98% | 66.31% | 817,000 | 727 | 817 |
| C3S1000 | 16 | 3/10 | 0.285 | 14676.3 | 22.39% | 46.09% | 1,301,000 | 1153 | 1301 |
| C23b | 10 | 7/10 | 0.681 | 640.7 | 62.57% | 96.39% | 597,064 | 585 | 655 |
| C23b | 12 | 6/10 | 0.588 | 908.4 | 22.18% | 65.84% | 709,560 | 705 | 787 |
| C23b | 14 | 8/10 | 0.743 | 5230.2 | 31.92% | 55.13% | 753,220 | 727 | 817 |
| C23b | 16 | 5/10 | 0.479 | 17293.8 | 26.39% | 62.41% | 1,215,688 | 1153 | 1301 |

Per-trial:
- B100 n=10: i0:Y p=0.987 tot=495 ; i1:Y p=0.956 tot=621 ; i2:n p=0.000 tot=398 ; i3:Y p=0.989 tot=802 ; i4:Y p=0.992 tot=155 ; i5:Y p=0.973 tot=327 ; i6:Y p=0.411 tot=552 ; i7:n p=0.000 tot=311 ; i8:n p=0.000 tot=600 ; i9:n p=0.000 tot=543
- B100 n=12: i0:n p=0.000 tot=679 ; i1:n p=0.000 tot=179 ; i2:Y p=0.965 tot=1026 ; i3:Y p=0.977 tot=753 ; i4:n p=0.000 tot=361 ; i5:Y p=0.990 tot=2351 ; i6:Y p=0.990 tot=192 ; i7:Y p=0.968 tot=373 ; i8:Y p=0.979 tot=1645 ; i9:n p=0.000 tot=1736
- B100 n=14: i0:Y p=0.936 tot=2827 ; i1:n p=0.000 tot=1932 ; i2:Y p=0.976 tot=1483 ; i3:Y p=0.269 tot=3053 ; i4:n p=0.000 tot=1687 ; i5:Y p=0.883 tot=4593 ; i6:Y p=0.977 tot=5899 ; i7:Y p=0.959 tot=1904 ; i8:n p=0.000 tot=1506 ; i9:n p=0.000 tot=5368
- B100 n=16: i0:Y p=0.951 tot=16557 ; i1:Y p=0.634 tot=7993 ; i2:n p=0.000 tot=17565 ; i3:Y p=0.958 tot=11565 ; i4:n p=0.000 tot=5857 ; i5:n p=0.000 tot=15017 ; i6:n p=0.000 tot=3902 ; i7:n p=0.000 tot=10441 ; i8:n p=0.000 tot=6876 ; i9:n p=0.000 tot=8998
- R1 n=10: i0:Y p=0.986 tot=381 ; i1:n p=0.000 tot=659 ; i2:Y p=0.720 tot=661 ; i3:n p=0.000 tot=674 ; i4:Y p=0.992 tot=162 ; i5:Y p=0.973 tot=416 ; i6:Y p=0.522 tot=912 ; i7:Y p=0.987 tot=565 ; i8:n p=0.000 tot=377 ; i9:n p=0.000 tot=718
- R1 n=12: i0:n p=0.000 tot=400 ; i1:n p=0.000 tot=191 ; i2:Y p=0.992 tot=360 ; i3:Y p=0.994 tot=377 ; i4:n p=0.000 tot=160 ; i5:n p=0.000 tot=187 ; i6:Y p=0.995 tot=337 ; i7:Y p=0.959 tot=259 ; i8:Y p=0.987 tot=301 ; i9:n p=0.000 tot=193
- R1 n=14: i0:Y p=0.977 tot=2899 ; i1:n p=0.000 tot=1282 ; i2:Y p=0.978 tot=4868 ; i3:Y p=0.357 tot=5101 ; i4:n p=0.000 tot=2557 ; i5:n p=0.000 tot=3117 ; i6:Y p=0.971 tot=3297 ; i7:Y p=0.960 tot=3762 ; i8:n p=0.000 tot=2152 ; i9:n p=0.000 tot=3146
- R1 n=16: i0:n p=0.000 tot=7389 ; i1:Y p=0.569 tot=4368 ; i2:n p=0.000 tot=8841 ; i3:Y p=0.920 tot=5390 ; i4:n p=0.000 tot=5775 ; i5:n p=0.000 tot=4271 ; i6:n p=0.000 tot=6667 ; i7:n p=0.000 tot=13957 ; i8:n p=0.000 tot=1582 ; i9:n p=0.000 tot=6238
- R2 n=10: i0:Y p=0.990 tot=661 ; i1:n p=0.000 tot=170 ; i2:n p=0.000 tot=851 ; i3:Y p=0.989 tot=914 ; i4:Y p=0.990 tot=148 ; i5:Y p=0.974 tot=647 ; i6:Y p=0.480 tot=778 ; i7:n p=0.000 tot=194 ; i8:Y p=0.986 tot=533 ; i9:Y p=0.882 tot=694
- R2 n=12: i0:n p=0.000 tot=599 ; i1:n p=0.000 tot=196 ; i2:Y p=0.978 tot=723 ; i3:Y p=0.993 tot=608 ; i4:n p=0.000 tot=259 ; i5:n p=0.000 tot=561 ; i6:Y p=0.985 tot=420 ; i7:Y p=0.983 tot=300 ; i8:Y p=0.985 tot=310 ; i9:n p=0.000 tot=497
- R2 n=14: i0:Y p=0.966 tot=3277 ; i1:n p=0.000 tot=7050 ; i2:Y p=0.972 tot=4670 ; i3:Y p=0.252 tot=5499 ; i4:n p=0.000 tot=5635 ; i5:Y p=0.728 tot=6420 ; i6:Y p=0.888 tot=4476 ; i7:Y p=0.952 tot=4588 ; i8:Y p=0.758 tot=6848 ; i9:n p=0.000 tot=3380
- R2 n=16: i0:Y p=0.938 tot=18885 ; i1:Y p=0.766 tot=13184 ; i2:Y p=0.953 tot=15831 ; i3:Y p=0.961 tot=8119 ; i4:n p=0.000 tot=4728 ; i5:n p=0.000 tot=10689 ; i6:n p=0.000 tot=9845 ; i7:Y p=0.864 tot=14906 ; i8:n p=0.000 tot=13716 ; i9:Y p=0.935 tot=20380
- R3 n=10: i0:Y p=0.989 tot=574 ; i1:n p=0.000 tot=181 ; i2:n p=0.000 tot=425 ; i3:Y p=0.981 tot=741 ; i4:Y p=0.985 tot=156 ; i5:Y p=0.989 tot=211 ; i6:Y p=0.979 tot=867 ; i7:Y p=0.973 tot=505 ; i8:Y p=0.991 tot=318 ; i9:n p=0.000 tot=759
- R3 n=12: i0:n p=0.000 tot=737 ; i1:n p=0.000 tot=178 ; i2:Y p=0.988 tot=1409 ; i3:Y p=0.990 tot=847 ; i4:n p=0.000 tot=319 ; i5:n p=0.000 tot=289 ; i6:Y p=0.985 tot=172 ; i7:Y p=0.991 tot=376 ; i8:Y p=0.984 tot=1922 ; i9:n p=0.000 tot=1689
- R3 n=14: i0:Y p=0.963 tot=6712 ; i1:n p=0.000 tot=2659 ; i2:Y p=0.970 tot=3617 ; i3:Y p=0.892 tot=3851 ; i4:n p=0.000 tot=2291 ; i5:Y p=0.931 tot=5113 ; i6:Y p=0.915 tot=4939 ; i7:Y p=0.911 tot=5618 ; i8:n p=0.000 tot=3787 ; i9:n p=0.000 tot=5463
- R3 n=16: i0:n p=0.000 tot=19220 ; i1:Y p=0.908 tot=10360 ; i2:n p=0.000 tot=4667 ; i3:Y p=0.927 tot=6826 ; i4:Y p=0.968 tot=7272 ; i5:n p=0.000 tot=5104 ; i6:n p=0.000 tot=10386 ; i7:n p=0.000 tot=14488 ; i8:n p=0.000 tot=7834 ; i9:n p=0.000 tot=12276
- C23 n=10: i0:Y p=0.991 tot=485 ; i1:n p=0.000 tot=176 ; i2:Y p=0.994 tot=689 ; i3:Y p=0.989 tot=953 ; i4:Y p=0.985 tot=149 ; i5:Y p=0.976 tot=307 ; i6:Y p=0.951 tot=638 ; i7:n p=0.000 tot=174 ; i8:Y p=0.985 tot=628 ; i9:Y p=0.909 tot=704
- C23 n=12: i0:n p=0.000 tot=325 ; i1:n p=0.000 tot=192 ; i2:Y p=0.984 tot=1348 ; i3:Y p=0.994 tot=379 ; i4:n p=0.000 tot=266 ; i5:n p=0.000 tot=373 ; i6:Y p=0.991 tot=343 ; i7:Y p=0.989 tot=339 ; i8:Y p=0.987 tot=433 ; i9:n p=0.000 tot=818
- C23 n=14: i0:Y p=0.907 tot=4221 ; i1:Y p=0.969 tot=10594 ; i2:Y p=0.931 tot=5867 ; i3:Y p=0.960 tot=9356 ; i4:n p=0.000 tot=1521 ; i5:Y p=0.968 tot=6043 ; i6:Y p=0.926 tot=7877 ; i7:Y p=0.928 tot=4803 ; i8:Y p=0.960 tot=7054 ; i9:n p=0.000 tot=5240
- C23 n=16: i0:Y p=0.963 tot=13051 ; i1:Y p=0.703 tot=10005 ; i2:n p=0.000 tot=18324 ; i3:Y p=0.911 tot=16961 ; i4:Y p=0.963 tot=9876 ; i5:Y p=0.931 tot=12630 ; i6:n p=0.000 tot=7993 ; i7:Y p=0.935 tot=9874 ; i8:n p=0.000 tot=7785 ; i9:n p=0.000 tot=24644
- C123 n=10: i0:Y p=0.986 tot=920 ; i1:n p=0.000 tot=344 ; i2:Y p=0.996 tot=379 ; i3:Y p=0.983 tot=812 ; i4:Y p=0.988 tot=156 ; i5:Y p=0.986 tot=245 ; i6:Y p=0.983 tot=336 ; i7:n p=0.000 tot=223 ; i8:Y p=0.985 tot=873 ; i9:n p=0.000 tot=246
- C123 n=12: i0:n p=0.000 tot=402 ; i1:n p=0.000 tot=207 ; i2:Y p=0.983 tot=381 ; i3:Y p=0.990 tot=274 ; i4:n p=0.000 tot=192 ; i5:n p=0.000 tot=355 ; i6:Y p=0.997 tot=212 ; i7:Y p=0.991 tot=313 ; i8:Y p=0.989 tot=350 ; i9:n p=0.000 tot=192
- C123 n=14: i0:Y p=0.906 tot=4417 ; i1:n p=0.000 tot=5436 ; i2:Y p=0.957 tot=4499 ; i3:Y p=0.791 tot=6567 ; i4:n p=0.000 tot=4427 ; i5:n p=0.000 tot=4057 ; i6:Y p=0.916 tot=3135 ; i7:Y p=0.927 tot=1549 ; i8:n p=0.000 tot=2425 ; i9:n p=0.000 tot=3177
- C123 n=16: i0:Y p=0.963 tot=16266 ; i1:Y p=0.940 tot=8278 ; i2:n p=0.000 tot=14839 ; i3:Y p=0.885 tot=13992 ; i4:Y p=0.967 tot=13328 ; i5:n p=0.000 tot=4878 ; i6:n p=0.000 tot=12228 ; i7:n p=0.000 tot=8130 ; i8:n p=0.000 tot=3114 ; i9:n p=0.000 tot=6206
- C3S1000 n=10: i0:Y p=0.988 tot=654 ; i1:n p=0.000 tot=344 ; i2:Y p=0.990 tot=990 ; i3:Y p=0.976 tot=1015 ; i4:Y p=0.992 tot=218 ; i5:Y p=0.985 tot=626 ; i6:Y p=0.987 tot=688 ; i7:n p=0.000 tot=285 ; i8:Y p=0.987 tot=873 ; i9:Y p=0.973 tot=979
- C3S1000 n=12: i0:n p=0.000 tot=1056 ; i1:n p=0.000 tot=247 ; i2:Y p=0.995 tot=635 ; i3:Y p=0.969 tot=497 ; i4:n p=0.000 tot=376 ; i5:Y p=0.988 tot=1412 ; i6:Y p=0.992 tot=376 ; i7:Y p=0.973 tot=352 ; i8:Y p=0.974 tot=557 ; i9:n p=0.000 tot=2386
- C3S1000 n=14: i0:Y p=0.962 tot=10864 ; i1:n p=0.000 tot=5073 ; i2:Y p=0.968 tot=4204 ; i3:Y p=0.899 tot=7141 ; i4:Y p=0.890 tot=8643 ; i5:Y p=0.961 tot=1899 ; i6:Y p=0.921 tot=4324 ; i7:Y p=0.928 tot=9262 ; i8:n p=0.000 tot=2168 ; i9:n p=0.000 tot=2098
- C3S1000 n=16: i0:n p=0.000 tot=4879 ; i1:Y p=0.963 tot=12758 ; i2:Y p=0.935 tot=23565 ; i3:Y p=0.956 tot=30203 ; i4:n p=0.000 tot=11597 ; i5:n p=0.000 tot=16806 ; i6:n p=0.000 tot=5663 ; i7:n p=0.000 tot=11869 ; i8:n p=0.000 tot=7606 ; i9:n p=0.000 tot=21817
- C23b n=10: i0:Y p=0.984 tot=956 ; i1:n p=0.000 tot=232 ; i2:Y p=0.994 tot=890 ; i3:Y p=0.990 tot=987 ; i4:Y p=0.992 tot=187 ; i5:Y p=0.977 tot=596 ; i6:Y p=0.983 tot=950 ; i7:n p=0.000 tot=530 ; i8:n p=0.000 tot=280 ; i9:Y p=0.894 tot=799
- C23b n=12: i0:n p=0.000 tot=1438 ; i1:n p=0.000 tot=180 ; i2:Y p=0.958 tot=2697 ; i3:Y p=0.987 tot=447 ; i4:n p=0.000 tot=346 ; i5:Y p=0.986 tot=994 ; i6:Y p=0.998 tot=1025 ; i7:Y p=0.968 tot=404 ; i8:Y p=0.985 tot=1179 ; i9:n p=0.000 tot=374
- C23b n=14: i0:Y p=0.965 tot=9033 ; i1:n p=0.000 tot=4707 ; i2:Y p=0.976 tot=2973 ; i3:Y p=0.873 tot=2813 ; i4:Y p=0.911 tot=6255 ; i5:Y p=0.967 tot=7535 ; i6:Y p=0.953 tot=4511 ; i7:Y p=0.917 tot=3039 ; i8:Y p=0.872 tot=8507 ; i9:n p=0.000 tot=2929
- C23b n=16: i0:Y p=0.973 tot=19132 ; i1:Y p=0.974 tot=4770 ; i2:Y p=0.957 tot=40901 ; i3:Y p=0.949 tot=17285 ; i4:n p=0.000 tot=9902 ; i5:n p=0.000 tot=22979 ; i6:n p=0.000 tot=6677 ; i7:Y p=0.942 tot=16438 ; i8:n p=0.000 tot=23091 ; i9:n p=0.000 tot=11763

Reference: S=1000 lowE (V1): 8/10 n=10, 8/10 n=14, 7/10 n=16 at 65%, 49%, 46% of 2^n.

## Reading (R1-R3)
- No setting reaches 9/10 at n=14 or n=16. Best success: C23 (S 50->1000 + eta anneal) 8/10, 5/10, 8/10, 6/10 at n=10, 12, 14, 16, at 47.9%, 11.8%, 38.2%, 20.0% of 2^n, which is about half V1's share at n=14 and 16, with fewer shots (219,768 to 445,848 vs 655,000 to 1,301,000), but one fewer success at n=16.
- R1 (xbeta 1) cuts the unique share (n=12 6.75%, n=16 9.84%) but lowers success (5/10, 2/10 at n=14, 16). Narrower starts reach fewer strings, so the GS is reached less often.
- R2 (shot schedule) is the only single fix that helps n=16 (3 -> 6/10). R3 alone is roughly neutral (within 10-trial noise).
- More shots don't fix it: C23b (100->3000) and C3S1000 are no better than C23. n=12 is stuck at 5-6/10 in every setting, so the same instances fail across all settings.
