# Dense n=8 relayout diagnosis

**Question.** Why does denser `four_sat_scaling/n08` (m ≈ 25–30, m/n ≈ 3.45) drop headline `L4s10-r200-R4` from 0.998/0.981 (19-clause legacy set) to 0.820/0.798, and how do we recover ~0.99/0.99?

**Setup.** Scaling path uses `nf = 24` Fock levels/cavity (leakage counted against p(GS)). Headline used `nf = 8` (no margin). Absolute success / mean p(GS) / evals below.

## 1. Truncation / leakage is the main cause (not clause count alone)

| set | nf | config | trials | success | mean p(GS) | evals | mean leak |
|---|---:|---|---:|---:|---:|---:|---:|
| legacy8 | 8 | L4s10-r200-R4 | 500 | 0.998 | 0.981 | 1688 | 0 |
| legacy8 | 24 | L4s10-r200-R4 | 100 | 0.950 | 0.923 | 1688 | 0.013 |
| **dense** | **8** | L4s10-r200-R4 | 500 | **0.986** | **0.965** | 1688 | 0 |
| dense | 24 | L4s10-r200-R4 | 100 | 0.820 | 0.798 | 1688 | 0.044 |

Paired seeds (100 trials): every nf=24 failure that nf=8 got right — nf=24 never uniquely succeeds. Failures have high leakage (mean ≈ 0.22). So the headline→dense drop at default scaling truncation is mostly **nf=24 leakage / truncated dynamics**, not “more clauses.”

Clause density still matters a little at nf=8 (0.986/0.965 vs 0.998/0.981), but is recoverable with a few more relabel rounds (below).

Energy spectra: dense has fewer E=1 states (≈119 vs ≈216) but more mass at E≥2. Both sets are locally rigid (min 1-flip = 1). Classical greedy basin is not worse for dense (mean 15.5 vs 11.3). Per-H mean p correlates strongly with round-0 polished hit (ρ ≈ 0.74), weakly with m.

## 2. Recovery on dense set

### A. Same truncation as headline (`nf = 8`) — cheapest ~0.99/0.99

| config | trials | success | mean p(GS) | evals |
|---|---:|---:|---:|---:|
| L4s10-r200-R4 | 500 | 0.986 | 0.965 | 1688 |
| L4s10-r200-R6 | 500 | 0.990 | 0.980 | 2490 |
| **L4s10-r200-R8** | **~500** | **~0.998** | **~0.988** | **3292** |
| L4s25-r200-R4 | 500 | 0.988 | 0.972 | 1808 |

**Recommended (nf=8):** `L4s10-r200-R8` — matches headline quality at ~2× evals.

### B. Scaling truncation (`nf = 24`) — need more relabel steps (r)

Zach’s suspicion that r=200 is too few is confirmed at nf=24:

| config | trials | success | mean p(GS) | evals | mean leak |
|---|---:|---:|---:|---:|---:|
| L4s25-r200-R4 | 500 | 0.928 | 0.911 | 1808 | 0.015 |
| L4s25-r200-R8 | 100 | 0.940 | 0.926 | 3412 | 0.012 |
| L4s50-r200-R8 | 100 | 0.970 | 0.964 | 3612 | 0.005 |
| L4s25-r200-R12 | 100 | 0.950 | 0.946 | 5016 | 0.010 |
| **L4s25-r400-R8** | 100 | **0.980** | **0.975** | 6612 | 0.002 |
| **L4s50-r400-R8** | (confirming) | ~1.00 | ~0.99 | 6812 | ~0 |

Raising **r** (relabel steps) cuts leakage and lifts mean p more than raising R alone. Screens at **r=300** (and 500/600) are in progress to find the cheapest ≥0.99/0.99 under nf=24.

lr 0.1 remains best among {0.05, 0.1, 0.2} on the screens tried; lr 0.2 collapses mean p.

## 3. Cross-check

`method=fast` vs `legacy` at nf=8 on legacy8: bit-for-bit identical 0.998/0.981 (500 trials). Dense at nf=8 uses the same k-bit path with `nf=8`.

## 4. Takeaway

1. **Why the drop:** default scaling `nf=24` lets population leak above Fock 7; denser Hams amplify that failure mode. Same dense Hams at `nf=8` are already near headline.
2. **Recover ~0.99/0.99 at modest cost:** use `nf=8` + `L4s10-r200-R8` (3292 evals), **or** stay on `nf=24` and raise relabel steps to **r≈400** (with s≥25, R=8) — r=300 grid pending for the cheapest nf=24 setting.
