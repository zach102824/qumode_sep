# Why relayout costs so much more at n=10, and the fix

Dense `four_sat_scaling` sets (20 H each), noiseless, code levels + 16 Fock levels per cavity (n=8: 24, n=10: 32, n=12: 48),
leakage counted against p(GS). 100 trials = 20 H x 5 paired seeds. evals = mean SPSA function evaluations per trial.
Classical lookups (energy table reads for the radius-1 fix-up and the candidate pool) are reported separately.
Analysis script: `noiseless/analyze_n10_cost.py` (per-round stats from the stored per-trial files).

## 1. Diagnosis (existing runs, no new simulation)

**The relabel ("exploit") round is not the bottleneck.** Once the guess is the GS, the old protocol already reaches
n=8 quality at n=10:

| run | trials whose round-0 guess was GS | best-so-far p(GS) after 1 / 2 / 4 / 8 relabel rounds |
|---|---|---|
| n08 L4s50-r400-R8 | 409/500 | 0.807 / 0.920 / 0.977 / 0.993 |
| n10 L4s50-r400-R8 | 46/100 | 0.770 / 0.914 / 0.971 / 0.994 |

Selected-round p(GS) given that some round had the right guess: 0.992 (n=8) vs 0.973–0.990 (n=10).
Shorter relabel rounds are better per round: at n=8 (325 paired trials with a correct round-0 guess) one r=200 round
gives 0.925, r=300 0.882, r=400 0.799; longer rounds drift off the vacuum (n=10 r=800: leakage 0.036–0.040 and
max|β| ≈ 2.1 in correct-guess rounds vs 0.001 and 1.1 at r=400).

**The cost is in *finding* the GS.**

| | n=8 L4s50-r400-R8 | n=10 L4s50-r400-R8 | n=10 L4s100-r800-R16 |
|---|---|---|---|
| round-0 guess = GS (after radius-1 fix-up) | 0.818 | 0.460 | 0.572 |
| a relabel round started from a *wrong* guess finds the GS | 0.635 | 0.099 | 0.143 |
| guess = GS after all rounds | 0.996 | 0.710 | 0.886 |

* Wrong guesses are almost all E=1 states 3–9 bits from the GS (n=10 L4s100-r800-R16, round 0: 1 bit: 6, 2: 1,
  3: 20, 4: 28, 5: 40, 6: 31, 7: 60, 8: 25 trials). A radius-2 fix-up would rescue < 2% of them, so it was not pursued.
* A relabel round from a wrong E=1 guess starts at an E=1 vacuum; at n=10 it escapes to the GS only 10–14% of the
  time (63% at n=8), so the old protocol adds about 3–10 points of guess hit rate per extra round. Reaching 0.99
  that way would need roughly 25 rounds — which is the extra cost.
* It is a few Hamiltonians, systematically: round-0 hit rate is 0.00 over 25 seeds on H3, H4, H19 and ≤0.16 on H6, H12
  (n10 L4s100-r800-R16), and those H have success 0.51–0.72. Growth from the same mask keeps landing on the same
  wrong E=1 state.
* Restarting the round-0 growth with the **same** (zero) mask plateaus: pooled guess hit rate 0.59 / 0.73 / 0.85 / 0.85
  after 1 / 2 / 3 / 6 growths. With a **fresh random cavity XOR mask** per restart the bias is broken:
  0.59 / 0.89 / 0.95 / 0.98 / 1.00 (s=50, top-4 candidates per run, 100 trials).

## 2. Fix: explore-then-exploit (`explore_exploit_trial_kbit`, runner flags `--explore E --topk K`)

1. **Explore:** E independent layer-growth runs (L=1→4, s steps/stage, the round-0 recipe). Run 0 uses mask (0,0),
   the others a uniformly random cavity XOR mask.
2. After every round, the K most probable code states are fixed up (radius 1) and added to a candidate pool;
   the guess is the pool member with the lowest classical energy (no GS knowledge).
3. **Exploit:** R relabel rounds of r steps from small β with the guess at Fock (0,0) (the old relabel round).
4. Returned round: lowest Gibbs cost at the common η, unchanged.

evals = E·L·(2s+1) + R·(2r+1); lookups = (E+R)·K·(n+1).

| setting | success | mean p(GS) | leakage | evals | lookups | trials |
|---|---|---|---|---|---|---|
| n=8 old L4s50-r400-R8 | 1.000 | 0.994 | 0.0000 | 6812 | 81 | 100 |
| n=8 E5K4 L4s50-r200-R4 | 1.000 | 0.990 | 0.0000 | 3624 | 324 | 100 |
| n=8 E5K1 L4s50-r200-R4 | 1.000 | 0.990 | 0.0000 | 3624 | 81 | 100 |
| n=10 old L4s50-r400-R8 | 0.710 | 0.705 | 0.0444 | 6812 | 99 | 100 |
| n=10 old L4s100-r800-R8 | 0.880 | 0.858 | 0.0166 | 13612 | 99 | 100 |
| n=10 old L4s100-r800-R16 | 0.886 | 0.876 | 0.0237 | 26420 | 187 | 500 |
| n=10 E5K4 zero-mask L4s50-r200-R4 | 0.850 | 0.841 | 0.0288 | 3624 | 396 | 100 |
| n=10 E5K1 L4s50-r200-R4 | 0.980 | 0.970 | 0.0047 | 3624 | 99 | 100 |
| n=10 E5K2 L4s50-r200-R4 | 0.990 | 0.979 | 0.0020 | 3624 | 198 | 100 |
| n=10 E7K1 L4s50-r200-R4 | 0.990 | 0.977 | 0.0015 | 4432 | 121 | 100 |
| n=10 E7K1 L4s50-r200-R6 | 0.990 | 0.981 | 0.0015 | 5234 | 143 | 100 |
| n=10 E6K2 L4s50-r200-R4 | 1.000 | 0.986 | 0.0000 | 4028 | 220 | 100 |
| n=10 E4K4 L4s50-r200-R4 | 0.980 | 0.968 | 0.0020 | 3220 | 352 | 100 |
| n=10 E6K4 L4s25-r200-R4 | 0.960 | 0.947 | 0.0024 | 2828 | 440 | 100 |
| n=10 E5K4 L4s50-r200-R3 | 1.000 | 0.986 | 0.0000 | 3223 | 352 | 100 |
| **n=10 E5K4 L4s50-r200-R4** | **1.000** | **0.989** | **0.0000** | **3624** | 396 | 100 |
| n=10 E6K4 L4s50-r200-R6 | 1.000 | 0.990 | 0.0000 | 4830 | 528 | 100 |
| n=10 E5K4 L4s50-r200-R8 | 1.000 | 0.993 | 0.0000 | 5228 | 572 | 100 |
| n=10 E5K4 L4s50-r300-R6 | 1.000 | 0.992 | 0.0000 | 5626 | 484 | 100 |
| n=10 E5K4 L4s50-r300-R8 | 1.000 | 0.995 | 0.0000 | 6828 | 572 | 100 |
| n=10 E4K4 L4s100-r300-R4 | 0.990 | 0.981 | 0.0017 | 5620 | 352 | 100 |

Lookups are counted with repeats (each pooled candidate costs n+1 table reads), so they are an upper bound;
2^10 = 1024. The old protocol's n=10 numbers in the first rows are the existing runs (100-trial subsets where marked 100).

**Result:** n=10 reaches 1.000 / 0.989 at 3624 evals (E5K4 L4s50-r200-R4), below the ~8200 the 1.1^n rule allows,
and n=8 drops from 6812 to 3624 evals at the same quality. The random explore mask is essential (zero mask: 0.850).
With K=1 (same 99-lookup budget as the old protocol) n=10 gives 0.980 / 0.970 at 3624 evals, or 0.990 / 0.981 at 5234.

n=12 with the same setting: 0.680 / 0.673 at 3624 evals (100 trials) — the knobs have to grow with n; that study
continues in `RELAYOUT_SCALING_EXPLORE.md`.
