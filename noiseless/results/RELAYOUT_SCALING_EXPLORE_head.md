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
