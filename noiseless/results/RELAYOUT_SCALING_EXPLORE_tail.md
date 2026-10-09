## Recommended K=1 recipe and per-n best (100 trials, 20 H × 5 seeds)

Recipe: L4, s=50, `--explore-mask perm --xeta 16 --reta 2 --lr 0.05`, exploit R=4 × r=200, top-1 candidate per
round, Hamming-1 fix-up. Only E grows with n. evals = 404·E + 1604.

| n | setting | success | mean p(GS) | leakage | evals | distinct lookups | lookups/2^n |
|---|---|---|---|---|---|---|---|
| 8 | E2, lr 0.07 | 1.000 | 0.995 | 0.0000 | 2412 | 10 | 0.040 |
| 10 | E3, lr 0.05 | 1.000 | 0.997 | 0.0000 | 2816 | 20 | 0.020 |
| 12 | E5, lr 0.05 | 1.000 | 0.997 | 0.0000 | 3624 | 46 | 0.011 |
| 14 | E8 top-2, lr 0.1 | 0.950 | 0.940 | 0.0021 | 4836 | 251 | 0.015 |
| 14 | E13 top-1, lr 0.1 | 0.940 | 0.929 | 0.0027 | 6856 | 181 | 0.011 |
| 16 | E8 top-1, lr 0.1 | 0.430 | 0.429 | 0.0000 | 4836 | 159 | 0.002 |

Second seed block (trials 5–9) for the lr 0.1 recipe reproduces n=8/10/12: 1.000/0.991, 1.000/0.988, 1.000/0.988.

Interim (83/100 trials, 01:55 CST): the first cold stage (n=14 E8, cool 50 steps, η×64, lr 0.05) does **not**
help top-1 discovery: guess hit 0.23 / 0.35 / 0.48 / 0.52 / 0.60 / 0.65 / 0.71 / 0.75 after runs 1–8 (no-cool
discovery scan: 0.26 / 0.43 / 0.52 / 0.59 / 0.71 / 0.80 / 0.83 / 0.89); success 0.77 so far.

## Status / next steps (for resuming)

- Running detached (resumable): `logs/queue_o.sh` — cold concentration stage after each explore growth
  (`--cool 50/100 --ceta 16/64/256 --clr 0.05/0.1`) at n=14 E8 top-1 and n=12 E3/E4, plus lr 0.05/0.07 at small E;
  then `logs/queue_p.sh` — exploit lr 0.03/0.05 at n=8/10/12, R3 / r150 with lr 0.05, and a second seed block for
  the lr 0.05 recipe.
- Rebuild this file with `noiseless/results/build_explore_md.sh`.
- Next: if the cold stage raises n=14 top-1 discovery, apply it at n=16 (E8–12); otherwise try explore depth growing
  with bits per cavity together with more steps (L6 s100), or an explore on a reduced cavity code (part of the
  bits on the transmon slots per perm).
