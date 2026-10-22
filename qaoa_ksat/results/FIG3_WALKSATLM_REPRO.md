# Fig. 3 WalkSATlm reproduction (Boulebnane & Montanaro, PRX Quantum 5, 030348)

## What the paper says, and what it leaves out
- Ensemble: random k-SAT, Definition 1. m ~ Poisson(r n), k literals per clause drawn uniformly
  with replacement, k=8, r=176.54 (Table 1). Only satisfiable instances are kept.
- WalkSATlm with p=0.15, w1=6, w2=5, citing Cai et al. 2014. Classical running time is the
  "number of evaluations of the Boolean formula". The figure reports the median over instances,
  with a least-squares fit of log2(median) vs n.
- **The paper does not specify** what one evaluation is, the per-try cutoff, whether there are
  restarts, or which WalkSATlm source was used.

## What their data repository shows
GitHub `sami-b95/qaoa_ksat_paper_data` (`lib/analysis_helpers.py::load_walksatlm_benchmarks`) sets
`walksatlm_runtime = num_trials`. The Zenodo data (record 7764484) has one run per instance, with
fields `p=0.15, w1=6, w2=5, max_flips=n, max_tries=1e20`. There are 1415–6336 instances per n
(n=12..20). So **one "formula evaluation" = one try of n flips**. Recomputing medians and the fit
from their raw files gives **log2 = -0.3085 + 0.3246 n**, which exactly matches Table II
(-0.309 + 0.325 n). Their separate "probability" dataset (10000 tries of n flips, number of
successes counted) agrees: median 1/p_try ≈ 17.7 at n=12, which is about 12/ln 2.
- The WalkSATlm implementation itself is **not** in the repo. PySAT has no WalkSAT-family
  solver; its solvers are CDCL.
- Cai's official WalkSATlm2013 (lcs.ios.ac.cn/~caisw/Code/WalkSATlm2013.tar) hard-codes, for
  k≥7, wp≈0.115 with a=b=1 and **no restarts**. That is not the paper's p/w1/w2, so the paper
  used a parameterised implementation of its own or a modified one.

## Our rerun
`walksatlm_c/wslm.c` uses Cai's `pick_bbreak_gmake` rule with wp=p=0.15, a=w1=6, b=w2=5.
Each try starts from a fresh uniform random assignment and runs n flips; tries are counted until
SAT. It also counts flips. Instances are fresh from Definition 1 with seed 1000+n, brute-force
filtered to SAT (~58% satisfiable). We did one run per instance and took the median over
instances. The saved QAOA hamiltonians/ subsets were not rerun here.

| n | instances | tries (cutoff n) | flips | flips+starts | paper raw median | Table II fit | tries c=2n | tries c=4n |
|---|---|---|---|---|---|---|---|---|
| 8 | 2000 | 5 | 34 | 39 | — | 4.9 | 3 | 2 |
| 9 | 2000 | 6 | 54 | 60 | — | 6.1 | 4 | 3 |
| 10 | 2000 | 8 | 72 | 80 | — | 7.7 | 5 | 3 |
| 11 | 2000 | 10 | 102 | 112 | — | 9.6 | 6 | 3 |
| 12 | 2000 | 11 | 130 | 141 | 12.0 | 12.1 | 7 | 4 |
| 13 | 2000 | 15 | 185 | 200 | 14.0 | 15.1 | 8 | 5 |
| 14 | 2000 | 17 | 236.5 | 253.5 | 19.0 | 18.9 | 10 | 6 |
| 15 | 2000 | 22 | 328 | 350 | 24.0 | 23.7 | 12 | 7 |
| 16 | 2000 | 26 | 413 | 439 | 31.0 | 29.7 | 13 | 7 |
| 17 | 2000 | 32 | 537 | 569 | 39.0 | 37.2 | 16 | 9 |
| 18 | 2000 | 37 | 655.5 | 692.5 | 47.0 | 46.6 | 20 | 10 |
| 19 | 2000 | 49 | 927.5 | 976.5 | 58.0 | 58.3 | 23 | 11 |
| 20 | 1459 | 58 | 1150 | 1208 | 68.0 | 73.1 | 28 | 13 |
| 21 | 751 | 68 | 1419 | 1487 | — | 91.5 | 33 | 16 |
| 22 | 340 | 85 | 1859.5 | 1944.5 | — | 114.6 | 37 | 16 |

Fits over n=12..20 (slope, intercept):

| counter | slope | intercept |
|---|---|---|
| paper Table II | 0.325 | -0.309 |
| paper raw data (recomputed) | 0.3246 | -0.3085 |
| **ours, tries of n flips (paper unit)** | **0.292** | **0.032** |
| ours, flips | 0.387 | 2.47 |
| ours, tries of 2n / 4n flips | 0.250 / 0.201 | -0.21 / -0.31 |

![plot](fig3_walksatlm_repro.png)

## Verdict (first pass, p=0.15. Superseded by the follow-up below)
- **Unit and gap: resolved.** Fig. 3 counts n-flip tries (restarts), not flips. One paper unit is
  about n flips. Our old Python numbers were flips from a single unrestarted walk, which is
  where the ~17–21× gap came from (the gap is ≈n). It was never a weaker solver.
- **Exact numbers: close, not exact.** With the paper's protocol our medians are within 10% of
  theirs for n=12..16. Ours fall below at larger n (58 vs 68 at n=20), so our slope is 0.292 vs
  0.325. Likely causes, none confirmed: details of their unreleased implementation (tie-breaking,
  whether each try restarts from random, how the noise step is applied) and sampling (ours has
  1459 instances at n=20). Using their raw data, Table II reproduces exactly.
- Note that the restart cutoff max_flips=n is far from optimal. With 4n cutoffs the *flip* scaling
  improves to 0.313 vs 0.387, so the paper's classical exponent depends on this protocol choice.


## Follow-up (2026-10-22 16:30 UTC+8): exact reproduction attempt

### 1. Are their instances available? No.
The Zenodo tarball contains only `instance_benchmark_*.json` results plus `summary.json`. QAOA records
refer to `instance_description_*.json` and `instance_enumeration_*` files, and pysat records to
`instance_file`, but **none of those files is in the archive**. `instance_id`
(e.g. `21012_17_1653510490174244696`) has the form pid_index_timestamp, not a generator seed, and the GitHub
repo has no generator code. So a per-instance comparison is impossible. What we do have per instance
is the number of satisfying assignments (`num_valid_assignments`, from the QAOA records). It
covers 35098/35098 WalkSATlm instances.

### 2. Diagnosis on fresh Definition-1 instances
- **Ensemble.** It is essentially the same. Fraction with a unique solution: ours 0.50–0.54, theirs
  0.52–0.57. Mean #solutions: ours 1.8, theirs 1.65–1.76. That difference is slightly in the
  *easier* direction for us, so it cannot explain medians that come out too low.
- **Restricted to unique-solution instances, p=0.15 is still low** (n=20: 86 vs 114 tries), so the gap
  is in the algorithm, not the instances. The distribution shape is otherwise geometric-like in both
  (mean/median ≈ 1.8–1.9). P(tries=1) agrees within noise.
- **Variants tested.** n=12..20, 2000 instances, medians of tries:

| variant | n=12 | 14 | 16 | 18 | 20 | slope |
|---|---|---|---|---|---|---|
| theirs (raw) | 12 | 19 | 31 | 47 | 68 | 0.325 |
| p=0.15 lmake (as stated) | 11 | 16 | 27 | 40 | 61 | 0.31* |
| random tie-break (no w1/w2) | 12 | 19 | 28 | 40 | 66 | 0.300 |
| noise also when break=0 | 12 | 17 | 26 | 38.5 | 56 | 0.281 |
| max_flips n-1 | 12 | 17 | 28 | 43 | 56 | 0.289 |
| max_flips n+1 | 10.5 | 16 | 23 | 37 | 54 | 0.297 |
| w1=w2=1 (Cai default) | 11 | 17 | 26 | 40 | 50 | 0.280 |
| w1=5,w2=6 swapped | 11 | 17 | 27 | 40 | 54 | 0.291 |
| p=0 | 11.5 | 17 | 25 | 38 | 50 | 0.270 |
| **p=0.30, w1=6, w2=5** | 13 | 19 | 31 | 45 | 69 | 0.303* |

*These are single-seed runs; the medians move by about ±5% between seeds.

The noise sweep pools 3 seeds per instance (6000 runs per n). Medians for n=12,14,16,18,20:

| p | medians | slope |
|---|---|---|
| 0.15 | 12, 17, 26, 40, 55 | 0.281 |
| 0.20 | 12, 19, 27, 41, 58 | 0.283 |
| 0.25 | 12, 18, 28, 42, 64 | 0.303 |
| **0.30** | **12, 19, 30, 47, 71** | **0.322** |
| 0.35 | 13, 20, 31, 49, 75 | 0.317 |

Full n table with p=0.30 (3 seeds pooled) against their raw medians and Table II:

| n | ours p=0.15 | ours p=0.30 | theirs raw | Table II |
|---|---|---|---|---|
| 12 | 11 | 12 | 12 | 12.0 |
| 13 | 15 | 16 | 14 | 15.0 |
| 14 | 17 | 19 | 19 | 18.8 |
| 15 | 22 | 25 | 24 | 23.6 |
| 16 | 26 | 30 | 31 | 29.6 |
| 17 | 32 | 37 | 39 | 37.0 |
| 18 | 37 | 47 | 47 | 46.3 |
| 19 | 49 | 57 | 58 | 58.0 |
| 20 | 58 | 71 | 68 | 72.7 |

Fit n=12..20 for p=0.30: **log2 = -0.146 + 0.316 n**. Theirs: -0.3085 + 0.3246 n. Every per-n
median is within ±2 tries / ≤7% of theirs, which is the seed-to-seed noise of a median over about 2000
instances.

**Verdict.** We **cannot reproduce exactly**, because the instances and the solver code are not
published. With the stated p=0.15 our medians run 10–20% low at n≥15. The single change that
reproduces their medians at every n within sampling noise is **an effective noise of about 0.30**
(lmake tie-break with w1=6, w2=5 kept). No other variant we tried (off-by-one cutoffs, tie-break
rules, noise on zero-break, make weights) does. One guess, **unverified**: their implementation
applies the p=0.15 noise in a way that doubles its effect, e.g. two independent noise coins, or noise
counted per literal. We have no code to confirm this.

### 3. Why flips/n from one long walk disagreed with the per-try count
- **The protocols differ.** The paper restarts from a fresh random string every n flips and counts
  restarts. A long walk without restarts keeps a strongly correlated trajectory. Median flips to
  solution are 202 (C, no restart, n=12) and 255 (old Python) instead of about n×tries ≈ 130–145.
  WalkSATlm at r=176.54 tends to get trapped in local minima, so n-flip random restarts are
  *more* efficient per flip here. Dividing a single-walk flip count by n does not give tries.
- **The units differ.** One paper unit is n flips plus one fresh random string. The old ~20× gap is
  ≈ n (12–16) × the 1.3–1.8× restart advantage. The old Python run also used 40 QAOA-subset
  instances, not about 2000.
- **The median differs.** The paper takes the median over instances of a *single* geometric draw
  of tries (≈ ln2 × mean tries per instance). A per-instance mean over many runs is about 1.45× larger.
