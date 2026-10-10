# Relayout scaling exploration (n = 8, 10, 12, …)

## FINAL SUMMARY (Oct 10 2026, 9:30 AM CST)

All rows 100 trials (20 H × 5 seeds), noiseless, Fock cutoff code + 16, leakage counted against p(GS), fix-up
always Hamming 1. Shared recipe: L4, s=50, explore `--explore-mask perm --xeta 16`, exploit `--reta 2`, R=4 × r=200,
lr 0.05 (0.07 at n=8). evals = 404·E + 1604. "lookups" = distinct classical energy-table entries read.

| n | best setting | success | mean p(GS) | leakage | evals | distinct lookups | share of 2^n |
|---|---|---|---|---|---|---|---|
| 8 | E2 top-1, lr 0.07 | 1.000 | 0.995 | 0.0000 | 2412 | 10 | 0.040 |
| 10 | E3 top-1 | 1.000 | 0.997 | 0.0000 | 2816 | 20 | 0.020 |
| 12 | E5 top-1 | 1.000 | 0.997 | 0.0000 | 3624 | 46 | 0.011 |
| 14 | E5 rawk64 t1 (seed block A) | 0.980 | 0.977 | 0.0021 | 3624 | 384 | 0.023 |
| 14 | E5 rawk64 t1 (seed block B, fresh) | 0.990 | 0.987 | 0.0006 | 3624 | 383 | 0.023 |
| 16 | E8 top-1, lr 0.1 (probe) | 0.430 | 0.429 | 0.0000 | 4836 | 159 | 0.002 |

Cheaper variant: R3 instead of R4 keeps quality at n=10 (E3, 2415 evals, 1.000 / 0.997) and n=12 (E5, 3223 evals,
1.000 / 0.997).

**Seed-block reproducibility.** Block B (trials 5–9, `--trial-offset 5`): n=8 1.000 / 0.997, n=10 0.990 / 0.988,
n=12 0.990 / 0.987 (lr 0.05 recipe); 10-seed averages ≈ 0.995 / 0.992 / 0.992 mean p(GS). n=14 rawk64: A 0.980 / 0.977,
B 0.990 / 0.987 → 200-trial average 0.985 / 0.982.

**How knobs scale with n.** Everything is fixed except the number of explore runs E: 2, 3, 5 at n = 8, 10, 12
(top-1), and E=5 with a raw top-64 pool at n=14. Top-1 alone would need E ≈ 12–14 at n=14 and ≈ 36 at n=16 (single-run
hit probability 0.66 → 0.26 → 0.12 at n = 12, 14, 16). evals grow ≈ 1.08–1.14× per bit through n=12.

**Raw top-K option (`rawk<K>t<T>`, `--rawk K --polt T`).** After *every* round (explore and exploit), the K most
probable code states of that round's output distribution are put into a running "raw pool", and their classical
energies are looked up (1 lookup each). The T lowest-energy members of the whole pool (T=1 here) then get the usual
Hamming-1 fix-up (itself + n one-bit flips, cached), and the guess is the lowest-energy fix-up result. `--topk` is
ignored. Lookups are counted as distinct entries (`n_lookups == n_lookups_distinct`, a set of every state whose energy
was read). The fix-up itself is still strictly Hamming 1; but the K raw-pool reads are extra classical lookups on
top of it, so this is a *larger classical budget*, not a pure top-1 + Hamming-1 rule. Its benefit is only where the GS
is sampled but not the most probable state: at n=10 E3 and n=12 E5 rawk64 gives the same 1.000 / 0.997 as top-1 but
needs 211 vs 20 lookups (21% vs 2% of 2^10) and 327 vs 46 (8.0% vs 1.1% of 2^12), so it is only worth using at n≥14.
At n=14 it reaches 0.985 / 0.982 with 2.3% of 2^14 lookups at 3624 evals (vs E13 top-1: 0.940 / 0.929, 6856 evals).

**What did not help.** Cold concentration stage after explore (n=14 0.900 → 0.48–0.78; also hurts n=12); deeper explore
(L=5/6, L=6 at n=14 0.76) or more explore steps (s=100, worse per eval); explore η×4 / η×64 at n=14; start growth at L=2;
explore lr ×0.5/×2; SPSA c=0.3; Gray cavity code; smaller Fock margin; exploit r≥400 or more relabel rounds (drift and
leak); exploit lr 0.05 with only E1 at n=8; fixed (identity) explore layout (n=10 drops to 0.850 / 0.841).

**Open:** n=16 needs either many more explore runs or a better explore circuit; the raw-pool lookup share
(2.3% at n=14) should be checked at n=16 before calling the scaling sub-exponential in classical cost.


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

3. **n = 14 discovery (beyond the n = 8–12 brief, to see the trend).** Same recipe, η×16, top-1 / top-8 hit after
   1 / 2 / 4 / 6 / 8 / 10 runs: 0.26 / 0.43 / 0.59 / 0.80 / 0.89 / 0.92 and 0.52 / 0.75 / 0.90 / 0.99 / 1.00 / 1.00.
   Run-0 p(GS) drops to 0.015 (0.08–0.10 at n = 12), and the GS is often 2nd–8th most probable, so top-1 needs
   ~E = 12–14 at n = 14. Variants at n = 14: η×4 worse (0.63 at E10), η×64 worse (0.83), s = 100 worse per eval
   (0.91 at E8 = 6432 explore evals), L = 6 worse (0.76 at E8). n=14 full run with E5: 0.730 / 0.723 at 3624 evals
   (guess hit 0.71 after explore; given the right guess, p(GS) = 0.987).
4. **Exploit η.** η×2 in the exploit rounds lifts mean p(GS) from 0.987 to 0.989–0.991 at no cost (n = 8, 10, 12);
   η×4/×8 the same as ×2, η×0.5 worse (0.981). R=3 or r=150 lose ~0.005–0.01 in mean p(GS).
5. **Exploit learning rate.** Lowering the relabel Adam lr from 0.1 to 0.07 / 0.05 lifts mean p(GS) further at no
   cost: n=10 E3 0.990 → 0.995 / 0.997, n=12 E5 0.989 → 0.995 / 0.997, n=8 E2 0.991 → 0.995 (lr 0.07). lr 0.15
   drops n=12 to 0.946. (lr 0.03 and second seed blocks queued.)
6. **Smallest E.** n=8: E1 = 0.989 (lr 0.1) at 2008 evals; n=10: E2 = 0.980/0.970 (E3 needed for ≥0.99); n=12: E4 =
   0.990/0.986 (lr 0.07), E5 = 1.000/0.997. Random XOR (no permutation) at n=12 with η×16 is as good as perm
   (1.000/0.989), so at n ≤ 12 the cold explore matters more than the mask type.
7. **n = 16 probe** (E8, K1, lr 0.1): 0.430 / 0.429 at 4836 evals; guess hit 0.12 after run 0, 0.43 after 8 runs.
   Discovery per explore run collapses past n = 12 with this L=4 explore.

### E vs n (fit)

Per-run top-1 discovery probability q(n) for the cold (η×8–16) perm explore, from the discovery curves
(first-run hit; later runs similar or a bit lower): q ≈ 0.94, 0.74, 0.66, 0.26, 0.12 at n = 8, 10, 12, 14, 16.
E needed for 99 % hit, E99 = ⌈ln 0.01 / ln(1 − q)⌉ ≈ 2, 4, 5, 16, 36; observed minimal E for ≥ 0.99 success
(100 trials): 1–2, 3, 4–5, > 13, > 8 (0.43 at 8).
Up to n = 12 E grows by ~+1 to +2 per +2 qubits (evals ≈ 404·E + 1604: 2412 → 2816 → 3624, ≈ 1.1^n); from
n = 12 to 14 q drops by 2.5×, so E would have to grow ~3× — the explore circuit (L = 4, 6–7 bits per cavity) is the
limit, not the exploit. The cold concentration stage (queued) is the first attempt to raise q at n = 14/16.
