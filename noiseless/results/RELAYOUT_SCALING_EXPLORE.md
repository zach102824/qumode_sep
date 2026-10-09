# Relayout scaling exploration (n = 8, 10, 12, …)

Running log of the explore-then-exploit relayout study (started Oct 9 2026, 10 PM CST). Dense
`four_sat_scaling` sets, 20 H per n, 100 trials (20 H × 5 paired seeds) per setting, noiseless, Fock cutoff
code levels + 16 per cavity, leakage counted against p(GS). Fix-up radius fixed at Hamming 1 (n+1 lookups per new candidate).

## Protocol (`relayout_kbit.explore_exploit_trial_kbit`, runner `run_relayout_kbit.py --explore E --topk K ...`)

1. **Explore:** E layer-growth runs (L=1→4, s SPSA-Adam steps/stage). Run 0: identity layout. Runs 1..E-1:
   `--explore-mask perm` = fresh random assignment of the n logical bits to the hardware slots (transmons d,e,
   cavity Fock digits) plus a random cavity XOR; `random` = XOR only; `zero` = identity every time.
   `--xeta X` multiplies the sampled-tail η controller (inverse temperature) during explore growth.
2. After every round the K most probable code states are fixed up (radius 1) and pooled; guess = pool member with
   the lowest classical energy.
3. **Exploit:** R relabel rounds (r steps, lr 0.1, small-β init) with the guess at Fock (0,0).
4. Returned round: lowest Gibbs cost at the common η (no GS knowledge).

Setting names: `L4s50_r200_R4_E5K1perm_xeta16` = L=4, s=50, exploit r=200 × R=4, E=5 explore runs, top-1, perm mask,
explore η×16. evals = E·4·(2s+1) + R·(2r+1). lookups = distinct energy-table entries read (cache), shown as a
fraction of 2^n; rows marked "(lookups w/ repeats)" predate the cache counter and count (n+1) per pooled candidate.

## Key findings so far

1. **Exploit is not the bottleneck at any n.** Given the right guess, R=4 × r=200 gives p(GS) ≈ 0.987–0.99 at
   n = 8, 10, 12. Longer relabel rounds (r ≥ 400) drift and leak (see `RELAYOUT_N10_COST_DIAGNOSIS.md`).
2. **Discovery is the bottleneck**, and it is fixed by three explore knobs (discovery-only scans,
   `noiseless/explore_discovery.py`, 100 trials, top-1 guess-hit after E runs of L4s50):

| n | explore | hit after 1 / 2 / 3 / 4 / 5 runs |
|---|---|---|
| 12 | zero mask, η×1 | 0.31 / 0.40 / 0.45 / 0.47 / 0.48 (plateau 0.54 at 12) |
| 12 | random XOR, η×1 | 0.31 / 0.37 / 0.43 / 0.52 / 0.57 (0.84 at 12) |
| 12 | perm, η×1 | 0.31 / 0.36 / 0.50 / 0.56 / 0.60 (0.92 at 12) |
| 12 | perm, η×2 | 0.39 / 0.53 / 0.68 / 0.79 / 0.80 |
| 12 | perm, η×4 | 0.58 / 0.73 / 0.82 / 0.87 / 0.92 |
| 12 | perm, η×8 | 0.62 / 0.83 / 0.98 / 0.98 / 0.99 |
| 12 | perm, η×16 | 0.66 / 0.88 / 0.97 / 0.99 / 1.00 |
| 12 | random XOR, η×8 | 0.62 / 0.75 / 0.89 / 0.95 / 0.96 |
| 12 | perm, η×0.5 | 0.22 / 0.26 / 0.35 / 0.42 / 0.47 |
| 10 | perm, η×1 | 0.46 / 0.77 / 0.89 / 0.97 / 0.99 |
| 10 | perm, η×4 | 0.71 / 0.93 / 0.98 / 0.99 / 0.99 |
| 10 | perm, η×8 | 0.74 / 0.90 / 0.97 / 0.98 / 1.00 |
| 8 | perm, η×1 | 0.83 / 0.96 / 1.00 (random XOR) |
| 8 | perm, η×4 | 0.94 / 1.00 / 1.00 / 1.00 |

   Other explore knobs tried at n=12 (η×1) and not helpful per eval: L=3/5/6, s=25/30/100, start growth at L=2,
   explore lr ×0.5/×2, SPSA c=0.3, Gray cavity code (neutral at n=12, worse at n=10), +8 instead of +16 Fock margin
   (no change). The default η controller (ln 20 / (q25 − q05) of the sampled energy tail) is too hot for search at
   larger n; a colder explore (η×8–16) is the single biggest lever.

## All full runs (100 trials each, sorted by evals)

### n = 8 (2^n = 256)

| setting | success | mean p(GS) | leakage | evals | lookups | lookups/2^n | trials |
|---|---|---|---|---|---|---|---|
| L4s50_r200_R4_E2K1perm_xeta16 | 1.000 | 0.989 | 0.0000 | 2412 | 10 | 0.040 | 100 |
| L4s50_r200_R4_E2K1perm_xeta4 | 1.000 | 0.989 | 0.0000 | 2412 | 10 | 0.040 | 100 |
| L4s50_r200_R4_E3K1perm (lookups w/ repeats) | 1.000 | 0.988 | 0.0000 | 2816 | 63 | 0.246 | 100 |
| L4s50_r200_R4_E3K1perm_xeta16 | 1.000 | 0.988 | 0.0000 | 2816 | 12 | 0.045 | 100 |
| L4s50_r200_R4_E3K1perm_xeta4 | 1.000 | 0.988 | 0.0000 | 2816 | 11 | 0.045 | 100 |
| L4s50_r200_R4_E4K1perm (lookups w/ repeats) | 1.000 | 0.988 | 0.0000 | 3220 | 72 | 0.281 | 100 |
| L4s50_r200_R4_E5K1 (lookups w/ repeats) | 1.000 | 0.990 | 0.0000 | 3624 | 81 | 0.316 | 100 |
| L4s50_r200_R4_E5K4 (lookups w/ repeats) | 1.000 | 0.990 | 0.0000 | 3624 | 324 | 1.266 | 100 |
| L4s50_r400_R8 (lookups w/ repeats) | 1.000 | 0.994 | 0.0000 | 6812 | 81 | 0.316 | 100 |

### n = 10 (2^n = 1024)

| setting | success | mean p(GS) | leakage | evals | lookups | lookups/2^n | trials |
|---|---|---|---|---|---|---|---|
| L4s50_r200_R4_E3K1perm_xeta16 | 1.000 | 0.988 | 0.0000 | 2816 | 20 | 0.020 | 100 |
| L4s50_r200_R4_E3K1perm_xeta8 | 1.000 | 0.987 | 0.0000 | 2816 | 20 | 0.020 | 100 |
| L4s25_r200_R4_E6K4 (lookups w/ repeats) | 0.960 | 0.947 | 0.0024 | 2828 | 440 | 0.430 | 100 |
| L4s50_r200_R4_E4K1perm_xeta16 | 1.000 | 0.986 | 0.0000 | 3220 | 22 | 0.022 | 100 |
| L4s50_r200_R4_E4K1perm_xeta4 | 1.000 | 0.985 | 0.0000 | 3220 | 25 | 0.024 | 100 |
| L4s50_r200_R4_E4K1perm_xeta8 | 1.000 | 0.986 | 0.0000 | 3220 | 23 | 0.023 | 100 |
| L4s50_r200_R4_E4K4 (lookups w/ repeats) | 0.980 | 0.968 | 0.0020 | 3220 | 352 | 0.344 | 100 |
| L4s50_r200_R3_E5K4 (lookups w/ repeats) | 1.000 | 0.986 | 0.0000 | 3223 | 352 | 0.344 | 100 |
| L4s50_r200_R4_E5K1 (lookups w/ repeats) | 0.980 | 0.970 | 0.0047 | 3624 | 99 | 0.097 | 100 |
| L4s50_r200_R4_E5K1perm (lookups w/ repeats) | 0.990 | 0.977 | 0.0016 | 3624 | 99 | 0.097 | 100 |
| L4s50_r200_R4_E5K2 (lookups w/ repeats) | 0.990 | 0.979 | 0.0020 | 3624 | 198 | 0.193 | 100 |
| L4s50_r200_R4_E5K4 (lookups w/ repeats) | 1.000 | 0.989 | 0.0000 | 3624 | 396 | 0.387 | 100 |
| L4s50_r200_R4_E5K4zero (lookups w/ repeats) | 0.850 | 0.841 | 0.0288 | 3624 | 396 | 0.387 | 100 |
| L4s50_r200_R4_E6K1 (lookups w/ repeats) | 0.970 | 0.958 | 0.0044 | 4028 | 110 | 0.107 | 100 |
| L4s50_r200_R4_E6K1perm (lookups w/ repeats) | 1.000 | 0.988 | 0.0000 | 4028 | 110 | 0.107 | 100 |
| L4s50_r200_R4_E6K2 (lookups w/ repeats) | 1.000 | 0.986 | 0.0000 | 4028 | 220 | 0.215 | 100 |
| L4s50_r200_R4_E7K1 (lookups w/ repeats) | 0.990 | 0.977 | 0.0015 | 4432 | 121 | 0.118 | 100 |
| L4s50_r200_R6_E6K4 (lookups w/ repeats) | 1.000 | 0.990 | 0.0000 | 4830 | 528 | 0.516 | 100 |
| L4s50_r200_R8_E5K4 (lookups w/ repeats) | 1.000 | 0.993 | 0.0000 | 5228 | 572 | 0.559 | 100 |
| L4s50_r200_R6_E7K1 (lookups w/ repeats) | 0.990 | 0.981 | 0.0015 | 5234 | 143 | 0.140 | 100 |
| L4s100_r300_R4_E4K4 (lookups w/ repeats) | 0.990 | 0.981 | 0.0017 | 5620 | 352 | 0.344 | 100 |
| L4s50_r300_R6_E5K4 (lookups w/ repeats) | 1.000 | 0.992 | 0.0000 | 5626 | 484 | 0.473 | 100 |
| L4s50_r400_R8 (lookups w/ repeats) | 0.710 | 0.705 | 0.0444 | 6812 | 99 | 0.097 | 100 |
| L4s50_r300_R8_E5K4 (lookups w/ repeats) | 1.000 | 0.995 | 0.0000 | 6828 | 572 | 0.559 | 100 |
| L4s100_r800_R8 (lookups w/ repeats) | 0.880 | 0.858 | 0.0166 | 13612 | 99 | 0.097 | 100 |

### n = 12 (2^n = 4096)

| setting | success | mean p(GS) | leakage | evals | lookups | lookups/2^n | trials |
|---|---|---|---|---|---|---|---|
| L4s50_r200_R4_E4K1perm_xeta16 | 0.990 | 0.978 | 0.0000 | 3220 | 40 | 0.010 | 100 |
| L4s50_r200_R4_E4K1perm_xeta32 | 0.970 | 0.959 | 0.0048 | 3220 | 40 | 0.010 | 100 |
| L4s50_r200_R4_E4K1perm_xeta8 | 0.980 | 0.968 | 0.0026 | 3220 | 42 | 0.010 | 100 |
| L4s50_r200_R4_E5K1perm_xeta16 | 1.000 | 0.987 | 0.0000 | 3624 | 46 | 0.011 | 100 |
| L4s50_r200_R4_E5K1perm_xeta16_reta0.5 | 1.000 | 0.981 | 0.0000 | 3624 | 46 | 0.011 | 100 |
| L4s50_r200_R4_E5K1perm_xeta16_reta2 | 1.000 | 0.989 | 0.0000 | 3624 | 46 | 0.011 | 100 |
| L4s50_r200_R4_E5K1perm_xeta32 | 0.990 | 0.978 | 0.0042 | 3624 | 46 | 0.011 | 100 |
| L4s50_r200_R4_E5K1perm_xeta8 | 0.990 | 0.977 | 0.0016 | 3624 | 49 | 0.012 | 100 |
| L4s50_r200_R4_E5K4 (lookups w/ repeats) | 0.680 | 0.673 | 0.0043 | 3624 | 468 | 0.114 | 100 |
| L4s50_r300_R3_E5K1perm_xeta16 | 1.000 | 0.976 | 0.0000 | 3823 | 46 | 0.011 | 100 |
| L4s50_r200_R5_E5K1perm_xeta16 | 1.000 | 0.989 | 0.0000 | 4025 | 46 | 0.011 | 100 |
| L4s50_r200_R4_E12K1perm (lookups w/ repeats) | 0.920 | 0.910 | 0.0009 | 6452 | 208 | 0.051 | 100 |

## Best K=1 settings per n so far (~0.99 / ~0.99)

| n | setting | success | mean p(GS) | evals | distinct lookups | lookups/2^n |
|---|---|---|---|---|---|---|
| 8 | L4s50_r200_R4_E2K1perm_xeta16 | 1.000 | 0.989 | 2412 | 10 | 0.040 |
| 10 | L4s50_r200_R4_E3K1perm_xeta16 | 1.000 | 0.988 | 2816 | 20 | 0.020 |
| 12 | L4s50_r200_R4_E5K1perm_xeta16_reta2 | 1.000 | 0.989 | 3624 | 46 | 0.011 |

Trend so far: exploit cost fixed (R=4 × r=200 = 1604 evals); explore runs E ≈ 2, 3, 4–5 at n = 8, 10, 12 (≈ +1 run per +2 qubits,
404 evals each), total evals 2412 → 2816 → 3220–3624, i.e. ≈ 1.08–1.1^n.

## Status / next steps (for resuming)

- Driver scripts in `logs/queue_*.sh` (detached, resumable: finished trials are skipped).
- Next: exploit η scale (reta 2–8), cheaper exploit (r=150, R=3), n=14 (E5–7, xeta16), possibly n=16;
  then fit E(n), η(n) and refresh the per-n best table with distinct lookup fractions.
