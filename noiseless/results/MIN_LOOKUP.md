# Minimum classical lookups, one relabel round (R1), Gray code, fixed_low

10-trial screens (1 per instance), bar: success >= 0.9 AND mean p(GS) >= 0.9. Descend one knob at a time, keeping the cheapest passing value: sab (halve; floor 64, since sa_sample always spends 64 random probes, so sab below 64 changes nothing), then rawk 24, 12, then polt 1. Lookups = mean distinct strings per trial (total) and SA part. Screens are 10 trials, so a pass needs a 50-trial confirm before it is final.

### n=18

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n18_base_sab162_k48_p3 | 162 | 48 | 3 | 10 | 1.00 | 0.930 | 1053 | 503.9 | 162.0 | PASS |
| n18_sab81_k48_p3 | 81 | 48 | 3 | 10 | 0.90 | 0.836 | 1053 | 408.8 | 81.0 | fail |
| n18_sab162_k24_p3 | 162 | 24 | 3 | 10 | 1.00 | 0.930 | 1053 | 402.7 | 162.0 | PASS |
| n18_sab162_k12_p3 | 162 | 12 | 3 | 10 | 1.00 | 0.930 | 1053 | 343.6 | 162.0 | PASS |
| n18_sab162_k12_p1 | 162 | 12 | 1 | 10 | 1.00 | 0.930 | 1053 | 247.9 | 162.0 | PASS |

**n=18 cheapest passing: sab 162, rawk 12, polt 1**

### n=16

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n16_base_sab128_k48_p3 | 128 | 48 | 3 | 10 | 1.00 | 0.914 | 1053 | 408.3 | 128.0 | PASS |
| n16_sab64_k48_p3 | 64 | 48 | 3 | 10 | 1.00 | 0.914 | 1053 | 369.5 | 64.0 | PASS |
| n16_sab64_k24_p3 | 64 | 24 | 3 | 10 | 1.00 | 0.914 | 1053 | 260.4 | 64.0 | PASS |
| n16_sab64_k12_p3 | 64 | 12 | 3 | 10 | 1.00 | 0.914 | 1053 | 216.3 | 64.0 | PASS |
| n16_sab64_k12_p1 | 64 | 12 | 1 | 10 | 1.00 | 0.914 | 1053 | 145.1 | 64.0 | PASS |

**n=16 cheapest passing: sab 64, rawk 12, polt 1**

### n=14

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n14_base_sab64_k48_p3 | 64 | 48 | 3 | 10 | 1.00 | 0.891 | 627 | 239.3 | 64.0 | fail |
| n14_sab64_k24_p3 | 64 | 24 | 3 | 10 | 1.00 | 0.891 | 627 | 180.7 | 64.0 | fail |
| n14_sab64_k48_p1 | 64 | 48 | 1 | 10 | 1.00 | 0.891 | 627 | 200.9 | 64.0 | fail |

**n=14 cheapest passing: sab 64, rawk 48, polt 3**

### n=12

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n12_base_sab72_k48_p3 | 72 | 48 | 3 | 10 | 1.00 | 0.986 | 503 | 192.4 | 72.0 | PASS |
| n12_sab64_k48_p3 | 64 | 48 | 3 | 10 | 1.00 | 0.986 | 503 | 184.3 | 64.0 | PASS |
| n12_sab64_k24_p3 | 64 | 24 | 3 | 10 | 1.00 | 0.986 | 503 | 144.5 | 64.0 | PASS |
| n12_sab64_k12_p3 | 64 | 12 | 3 | 10 | 0.70 | 0.695 | 503 | 130.4 | 64.0 | fail |
| n12_sab64_k24_p1 | 64 | 24 | 1 | 10 | 1.00 | 0.986 | 503 | 114.9 | 64.0 | PASS |

**n=12 cheapest passing: sab 64, rawk 24, polt 1**

### n=10

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n10_base_sab50_k48_p3 | 50 | 48 | 3 | 10 | 1.00 | 0.991 | 443 | 161.1 | 62.6 | PASS |
| n10_sab50_k24_p3 | 50 | 24 | 3 | 10 | 0.90 | 0.891 | 443 | 125.8 | 62.6 | fail |
| n10_sab50_k48_p1 | 50 | 48 | 1 | 10 | 1.00 | 0.991 | 443 | 145.4 | 62.6 | PASS |

**n=10 cheapest passing: sab 50, rawk 48, polt 1**

### n=8

| screen | sab | rawk | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n8_base_sab32_k48_p3 | 32 | 48 | 3 | 10 | 1.00 | 0.990 | 403 | 127.5 | 56.7 | PASS |
| n8_sab32_k24_p3 | 32 | 24 | 3 | 10 | 1.00 | 0.990 | 403 | 102.9 | 56.7 | PASS |
| n8_sab32_k12_p3 | 32 | 12 | 3 | 10 | 1.00 | 0.990 | 403 | 88.6 | 56.7 | PASS |
| n8_sab32_k12_p1 | 32 | 12 | 1 | 10 | 1.00 | 0.990 | 403 | 76.8 | 56.7 | PASS |

**n=8 cheapest passing: sab 32, rawk 12, polt 1**

## Scaling fit of the minimum lookups (written 2026-10-12 23:50 CST)

Cheapest passing 10-trial screen per n (brute-force descent above). n=14 has no passing row: its R1 base (3-cav 4,4,3, E=2) already fails on mean p(GS) (0.891), so the miss is the quantum stage, not a lookup cut, and n=14 is left out of the fit.

| n | setting (sab, rawk, polt) | evals | success | mean p(GS) | total lookups | SA lookups | share of 2^n |
|---|---|---|---|---|---|---|---|
| 8 | 32, 12, 1 | 403 | 1.00 | 0.990 | 76.8 | 56.7 | 30.0% |
| 10 | 50, 48, 1 | 443 | 1.00 | 0.991 | 145.4 | 62.6 | 14.2% |
| 12 | 64, 24, 1 | 503 | 1.00 | 0.986 | 114.9 | 64.0 | 2.81% |
| 14 | none passes (base 64, 48, 3) | 627 | 1.00 | 0.891 | 239.3 | 64.0 | 1.46% |
| 16 | 64, 12, 1 | 1053 | 1.00 | 0.914 | 145.1 | 64.0 | 0.22% |
| 18 | 162, 12, 1 | 1053 | 1.00 | 0.930 | 247.9 | 162.0 | 0.095% |

Fits over n = 8, 10, 12, 16, 18 (R2 on log lookups):
- a*2^n: a = 1.0e-3, R2 = -93. Ruled out; lookups are nowhere near exponential.
- a*n^2: a = 0.75 (48, 75, 108, 191, 242), R2 = -0.03. Right at n = 16-18, too low at n = 8-10.
- power law: 8.9 * n^1.09, R2 = 0.72.
- exponential: 44 * 1.091^n, R2 = 0.73.
- linear: 12.9 n - 18.8.

Reading: over n = 8-18 the minimum lookup count grows roughly linearly (n^1.1), with jumps where E changes (1 run at n <= 12, 4 at n = 16-18) and where SA must stay at 162 (n=18; 81 fails, and the rule run with 64 probes, K 20 also fails at 0.854). The share of 2^n falls from 30% at n=8 to 0.095% at n=18. All rows are 10-trial screens and need 50-trial confirms. The rule validation (MIN_LOOKUP_RULE.md) found n=16 at 64, 10, 1 passing 1.00 / 0.914 at 135.9 lookups (10 trials), slightly below the 145.1 here; its 50-trial confirm is running.


## Rule vs brute-force (written 2026-10-13 00:30 CST)

`q_3cav_rule.sh` finished. Fitted K(n)/polt(n) from GS-rank diags, then 10-trial rule screens (SA probes 16 at n>=12, else 64) with Kx2+64-probe fallback, then 50-trial confirms of each pass.

| n | brute cheapest (sab, K, polt) | total lookups | rule setting (sab, probes, K, polt) | total lookups | 50-trial success / mean p(GS) |
|---|---|---|---|---|---|
| 8 | 32, 12, 1 | 76.8 | 32, 64, 9, 1 | 73.5 | 0.98 / 0.967 PASS |
| 10 | 50, 48, 1 | 145.4 | 50, 64, 37, 1 | 129.2 | 1.00 / 0.990 PASS |
| 12 | 64, 24, 1 | 114.9 | 16, 16, 23, 1 | 67.2 | 0.92 / 0.909 PASS |
| 14 | none (base 64, 48, 3 fails mean p(GS) 0.891) | 239.3 | 16/64 + K=2/4 all fail at 0.891 | — | quantum stage, not lookup cuts |
| 16 | 64, 12, 1 | 145.1 | 64, 64, 10, 1 | 129.4 | 1.00 / 0.909 PASS |
| 18 | 162, 12, 1 | 247.9 | sab16/K10 and sab64/K20 fail (0.838 / 0.854) | — | 50-trial confirm of brute 162/12/1 queued |

Rule beats brute on n=8, 10, 12, 16 (confirmed). At n=18 the systematic K/probe cut undershoots mean p(GS); keep sab 162, K 12, polt 1 pending confirm. Updated min-lookup scaling using rule-confirmed rows where they exist:

| n | setting | evals | success | mean p(GS) | total lookups | SA lookups | share of 2^n | source |
|---|---|---|---|---|---|---|---|---|
| 8 | sab32 K9 polt1 | 403 | 0.98 | 0.967 | 73.5 | 56.8 | 28.7% | rule 50-trial |
| 10 | sab50 K37 polt1 | 443 | 1.00 | 0.990 | 129.2 | 62.2 | 12.6% | rule 50-trial |
| 12 | sab16 K23 polt1 (probes 16) | 503 | 0.92 | 0.909 | 67.2 | 16.0 | 1.64% | rule 50-trial |
| 14 | — | 627 | 1.00 | 0.891 | 239.3 | 64.0 | 1.46% | R1 base fails bar |
| 16 | sab64 K8 polt1 | 1053 | 1.00 | 0.909 | 124.3 | 64.0 | 0.190% | cutnext4 confirm (5×10) |
| 18 | sab162 K12 polt1 | 1053 | 1.00 | 0.930 | 247.9 | 162.0 | 0.095% | 10-trial screen only |

Refit over confirmed (or best available) n = 8, 10, 12, 16, 18 total lookups [73.5, 129.2, 67.2, 129.4, 247.9]: still roughly flat-to-linear, not exponential. n=12 with probes 16 is an outlier low; without it the rest track closer to ~8–14*n. Full detail: `MIN_LOOKUP_RULE.md`.

## Cutnext probes (auto, started 2026-10-13 00:31 CST)

| run | sab | SA probes | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n18_confirm_sab162_k12_p1 | 162 | 64 | 12 | 1 | 50 | 0.96 | 0.858 | 1053 | 248.3 | 162.0 | fail |
| n18_sab162_k10_p1 | 162 | 64 | 10 | 1 | 10 | 1.00 | 0.930 | 1053 | 241.1 | 162.0 | PASS |
| n18_sab162_k8_p1 | 162 | 64 | 8 | 1 | 10 | 0.80 | 0.748 | 1053 | 232.1 | 162.0 | fail |
| n18_sab162_k12_p0 | 162 | 64 | 12 | 0 | 10 | 1.00 | 0.930 | 1053 | 247.9 | 162.0 | PASS |
| n16_sab64_k10_p0 | 64 | 64 | 10 | 0 | 10 | 1.00 | 0.914 | 1053 | 135.9 | 64.0 | PASS |
| n16_sab64_k8_p1 | 64 | 64 | 8 | 1 | 10 | 1.00 | 0.914 | 1053 | 127.1 | 64.0 | PASS |
| n16_sab64_k6_p1 | 64 | 64 | 6 | 1 | 10 | 1.00 | 0.914 | 1053 | 118.1 | 64.0 | PASS |

## n=18 re-confirms after K12 confirm fail (auto, started 2026-10-13 01:38 CST)

K12/polt1 50-trial: 0.96 / 0.858 (median p 0.924; 2 misses, one with GS at readout rank 19, so K>=24 would have caught it). The 10-trial screens reuse the trial-0 seeds, which are easier; K10/K8/K6 screens are therefore optimistic.

| run | sab | SA probes | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n18_confirm_sab162_k24_p1 | 162 | 64 | 24 | 1 | 50 | 0.96 | 0.880 | 1053 | 303.9 | 162.0 | fail |
| n18_confirm_sab162_k48_p1 | 162 | 64 | 48 | 1 | 50 | 0.98 | 0.897 | 1053 | 415.5 | 162.0 | fail |
| n18_sab162_pr32_k12_p1_sep | 162 | 32 | 12 | 1 | 10 | 0.80 | 0.748 | 1053 | 251.3 | 162.0 | fail |
| n18_sab162_pr16_k12_p1_sep | 162 | 16 | 12 | 1 | 10 | 1.00 | 0.930 | 1053 | 241.5 | 162.0 | PASS |

## Cutnext3: n=16 confirm + SA probes, n=18 E=6 fallback (auto, started 2026-10-13 03:48 CST)

| run | sab | SA probes | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n16_confirm_sab64_k6_p1 | 64 | 64 | 6 | 1 | 50 | 0.98 | 0.890 | 1053 | 116.7 | 64.0 | fail |
| n16_sab32_pr32_k10_p1 | 32 | 32 | 10 | 1 | 10 | 0.90 | 0.843 | 1053 | 94.2 | 32.0 | fail |
| n16_sab16_pr16_k10_p1 | 16 | 16 | 10 | 1 | 10 | 0.90 | 0.819 | 1053 | 83.4 | 16.0 | fail |
| n18_confirm_E6_sab162_k12_p1 | 162 | 64 | 12 | 1 | 50 | 0.98 | 0.895 | 1479 | 271.8 | 162.0 | fail |

## Cutnext4: n16 K8/polt0 confirms, n18 E6 follow-up, n12 probes8 (auto, started 2026-10-13 07:36 CST)

| run | sab | SA probes | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n16_confirm_sab64_k8_p1 | 64 | 64 | 8 | 1 | 50 | 1.00 | 0.909 | 1053 | 124.3 | 64.0 | PASS |

**n=16 new cheapest (2026-10-13 01:44 CST):** sab 64, rawk 8, polt 1 confirmed at 124.3 lookups (success 1.00, mean p(GS) 0.909), below the prior rule K10 confirm at 129.4. Scaling table above updated.

| n16_confirm_sab64_k10_p0 | 64 | 64 | 10 | 0 | 50 | 1.00 | 0.909 | 1053 | 129.4 | 64.0 | PASS |

n16 K10/polt0 confirm (2026-10-13 08:55 CST): PASS at 129.4 lookups, not a new cheapest (above K8/polt1 at 124.3).

## Scaling note after n16 K8 (2026-10-13 08:54 CST)

Confirmed (or best available) total lookups now [73.5, 129.2, 67.2, 124.3, 247.9] for n = 8, 10, 12, 16, 18. Still flat-to-linear, not exponential; n=16 edged down from 129.4 to 124.3. K10/polt0 confirm at 129.4 does not change the cheapest.
| n18_E6_sab162_k24_p1 | 162 | 64 | 24 | 1 | 10 | 0.90 | 0.820 | 1479 | 351.5 | 162.0 | fail |
| n12_sab16_pr8_k23_p1 | 16 | 8 | 23 | 1 | ERROR ||||||

## Cutnext5: layout from explore (no SA) (auto, started 2026-10-13 09:05 CST)

| run | mask | sab | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n16_rand_k10_p1 | random | 0 | 10 | 1 | 10 | 1.00 | 0.900 | 1053 | 61.8 | 0.0 | PASS |
| n16_zero_k10_p1 | zero | 0 | 10 | 1 | 10 | 1.00 | 0.914 | 1053 | 61.5 | 0.0 | PASS |
| n16_rand_lmut2_k10_p1 | random | 0 | 10 | 1 | 10 | 1.00 | 0.900 | 1053 | 61.8 | 0.0 | PASS |
| n12_rand_k23_p1 | random | 0 | 23 | 1 | ERROR ||||||
| n18_E6_rand_k12_p1 | random | 0 | 12 | 1 | 10 | 0.90 | 0.826 | 1479 | 105.2 | 0.0 | fail |

## Cutnext6: n16 K8/polt0 confirm, n12 probe follow-up, n18 E8 (auto, started 2026-10-13 09:24 CST)

| run | sab | SA probes | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|---|

| n16_confirm_sab64_k8_p0 | killed 09:25 CST: 50-trial confirms dropped (10-trial screens only); cutnext6 superseded by no-SA sweep |

## No-SA sweep, 10-trial screens only (auto, started 2026-10-13 09:26 CST) — SUPERSEDED 2026-10-13 09:3x CST (Zach: keep SA; replaced by consistent linear-SA recipe)

Pass = success >= 0.9 and mean p(GS) >= 0.89. sab 0 (no SA lookups); K descended to smallest pass, then polt 0.

| run | mask | sab | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nosa_n8_zero_k9_p1 | zero | 0 | 9 | 1 | 10 | 1.00 | 0.896 | 403 | 22.3 | 0.0 | PASS |
| nosa_n8_zero_k8_p1 | zero | 0 | 8 | 1 | 10 | 0.90 | 0.803 | 403 | 20.7 | 0.0 | fail |
| nosa_n8_zero_k9_p0 | zero | 0 | 9 | 0 | 10 | 1.00 | 0.896 | 403 | 22.3 | 0.0 | PASS |
| nosa_n8_random_k9_p1 | random | 0 | 9 | 1 | 10 | 1.00 | 0.896 | 403 | 22.3 | 0.0 | PASS |
| nosa_n8_random_k8_p1 | random | 0 | 8 | 1 | 10 | 0.90 | 0.803 | 403 | 20.7 | 0.0 | fail |
| nosa_n8_random_k9_p0 | random | 0 | 9 | 0 | 10 | 1.00 | 0.896 | 403 | 22.3 | 0.0 | PASS |
| nosa_n10_zero_k37_p1 | zero | 0 | 37 | 1 | 10 | 0.90 | 0.894 | 443 | 70.5 | 0.0 | PASS |
| nosa_n10_zero_k32_p1 | zero | 0 | 32 | 1 | 10 | 0.90 | 0.887 | 443 | 62.8 | 0.0 | fail |
| nosa_n10_zero_k37_p0 | zero | 0 | 37 | 0 | 10 | 0.90 | 0.894 | 443 | 70.5 | 0.0 | PASS |
| nosa_n10_random_k37_p1 | random | 0 | 37 | 1 | 10 | 0.90 | 0.894 | 443 | 70.5 | 0.0 | PASS |
| nosa_n10_random_k32_p1 | random | 0 | 32 | 1 | 10 | 0.90 | 0.887 | 443 | 62.8 | 0.0 | fail |
| nosa_n10_random_k37_p0 | random | 0 | 37 | 0 | 10 | 0.90 | 0.894 | 443 | 70.5 | 0.0 | PASS |
| nosa_n12_zero_k23_p1 | zero | 0 | 23 | 1 | 10 | 0.90 | 0.817 | 503 | 50.4 | 0.0 | fail |
| nosa_n12_zero_k24_p1 | zero | 0 | 24 | 1 | 10 | 0.90 | 0.817 | 503 | 52.1 | 0.0 | fail |
| nosa_n12_zero_k32_p1 | zero | 0 | 32 | 1 | 10 | 1.00 | 0.986 | 503 | 64.9 | 0.0 | PASS |
| nosa_n12_zero_k32_p0 | zero | 0 | 32 | 0 | 10 | 1.00 | 0.986 | 503 | 64.9 | 0.0 | PASS |
| nosa_n12_random_k23_p1 | random | 0 | 23 | 1 | 10 | 0.90 | 0.817 | 503 | 50.4 | 0.0 | fail |
| nosa_n12_random_k24_p1 | random | 0 | 24 | 1 | 10 | 0.90 | 0.817 | 503 | 52.1 | 0.0 | fail |
| nosa_n12_random_k32_p1 | random | 0 | 32 | 1 | 10 | 1.00 | 0.986 | 503 | 64.9 | 0.0 | PASS |
| nosa_n12_random_k32_p0 | random | 0 | 32 | 0 | 10 | 1.00 | 0.986 | 503 | 64.9 | 0.0 | PASS |
| nosa_n14_zero_k2_p1 | zero | 0 | 2 | 1 | 10 | 0.80 | 0.700 | 627 | 24.0 | 0.0 | fail |
| nosa_n14_zero_k3_p1 | zero | 0 | 3 | 1 | 10 | 0.80 | 0.713 | 627 | 24.4 | 0.0 | fail |

## Consistent linear-SA recipe (10-trial screens, started 2026-10-13 09:29 CST)

Same steps at every n: SA with sab = 9n distinct lookups (QM_SA_PROBES = 4n random probes, rest in anneal chains of 8n steps), E explore runs (1,1,1,2,4,6), top-K readouts with K = n, polt 1 single-bit fix-ups, one relabel round (r 150 n<=12 on 2 cav, r 100 n>=14 on 3 cav), Gray code, fixed_low. If K = n fails, K steps up 1.5n, 2n, 3n; if it passes, K = n/2 is tried once. Pass = success >= 0.9 and mean p(GS) >= 0.89.

| n | lookups total | SA lookups | K | E | success | mean p(GS) | evals | share of 2^n | result |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 20.7 | 0.0 | 8 | 1 | 0.90 | 0.803 | 403 | 8.09% | fail |
| 8 | 24.9 | 0.0 | 12 | 1 | 1.00 | 0.990 | 403 | 9.73% | PASS |
| 10 | 27.1 | 0.0 | 10 | 1 | 0.80 | 0.795 | 443 | 2.65% | fail |
| 10 | 34.2 | 0.0 | 15 | 1 | 0.90 | 0.887 | 443 | 3.34% | fail |
| 10 | 42.7 | 0.0 | 20 | 1 | 0.90 | 0.887 | 443 | 4.17% | fail |
| 10 | 59.4 | 0.0 | 30 | 1 | 0.90 | 0.887 | 443 | 5.8% | fail |
| 12 | 31.7 | 0.0 | 12 | 1 | 0.80 | 0.796 | 503 | 0.774% | fail |
| 12 | 41.4 | 0.0 | 18 | 1 | 0.90 | 0.817 | 503 | 1.01% | fail |
| 12 | 52.1 | 0.0 | 24 | 1 | 0.90 | 0.817 | 503 | 1.27% | fail |
| 12 | 72.1 | 0.0 | 36 | 1 | 1.00 | 0.986 | 503 | 1.76% | PASS |

## Strict recipe (SA=10n raw, K=2n, polt 1, R1), 10-trial — SUPERSEDED by K=3n

**Superseded 2026-10-13: replaced by the K=3n strict recipe below. Queue stopped during n=20 (killed, no row); rows n=8..18 kept for reference only.**

Started 2026-10-13 11:05 CST. SA budget = 10n cost evaluations counted raw (QM_SA_COUNT=raw: 64 random probes + anneal steps, repeats count), K = 2n top readouts per explore run, polt 1 (n single-bit neighbours per new fix-up center), one relabel round, Gray code, fixed_low, 1 trial per instance x 10 instances. Raw lookups = SA + readouts + fix-ups as counted by the code; distinct = unique strings. Eval formula E*S*(2s+1)+(2r+1) with S = xL, s = xs.

| n | chip | E | xL | xs | r | success | mean p(GS) | evals | formula | SA lookups | readout lookups | fix-up lookups | total raw lookups | distinct lookups | share of 2^n (raw) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2cav 2+3/3 | 1 | 2 | 25 | 150 | 1.00 | 0.990 | 403 | 1*2*51+301=403 ok | 80 | 32 | 8 | 120 | 91 | 46.9% |
| 10 | 2cav 2+4/4 | 1 | 2 | 35 | 150 | 1.00 | 0.991 | 443 | 1*2*71+301=443 ok | 100 | 40 | 10 | 150 | 125.5 | 14.6% |
| 12 | 2cav 2+5/5 | 1 | 2 | 50 | 150 | 0.60 | 0.592 | 503 | 1*2*101+301=503 ok | 120 | 48 | 15.6 | 183.6 | 160.6 | 4.48% |
| 14 | 3cav 3+4/4/3 | 2 | 3 | 35 | 100 | 1.00 | 0.891 | 627 | 2*3*71+201=627 ok | 140 | 84 | 14 | 238 | 213.4 | 1.45% |
| 16 | 3cav 3+5/4/4 | 4 | 3 | 35 | 100 | 1.00 | 0.917 | 1053 | 4*3*71+201=1053 ok | 160 | 160 | 17.6 | 337.6 | 298.9 | 0.515% |
| 18 | 3cav 3+5/5/5 | 6 | 3 | 35 | 100 | 1.00 | 0.908 | 1479 | 6*3*71+201=1479 ok | 180 | 252 | 34.2 | 466.2 | 432.6 | 0.178% |

## Strict recipe (SA=10n raw, K=3n, polt 1, R1), 10-trial — SUPERSEDED by FINAL recipe (n=18 row kept as reference; n=20 stopped unfinished)

Started 2026-10-13 11:30 CST. SA budget = 10n cost evaluations counted raw (QM_SA_COUNT=raw: 64 random probes + anneal steps, repeats count), K = 3n top readouts per explore run, plus the relabel round also reads its own top K, so readout lookups = (E+1)*K (kept deliberately), polt 1 (n single-bit neighbours per new fix-up center), one relabel round, Gray code, fixed_low, 1 trial per instance x 10 instances. Raw lookups = SA + readouts + fix-ups as counted by the code; distinct = unique strings. Eval formula E*S*(2s+1)+(2r+1) with S = xL, s = xs.

| n | chip | E | xL | xs | r | success | mean p(GS) | evals | formula | SA lookups | readout lookups | fix-up lookups | total raw lookups | distinct lookups | share of 2^n (raw) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2cav 2+3/3 | 1 | 2 | 25 | 150 | 1.00 | 0.990 | 403 | 1*2*51+301=403 ok | 80 | 48 | 8 | 136 | 100.7 | 53.1% |
| 10 | 2cav 2+4/4 | 1 | 2 | 35 | 150 | 1.00 | 0.991 | 443 | 1*2*71+301=443 ok | 100 | 60 | 10 | 170 | 140.8 | 16.6% |
| 12 | 2cav 2+5/5 | 1 | 2 | 50 | 150 | 0.80 | 0.789 | 503 | 1*2*101+301=503 ok | 120 | 72 | 14.4 | 206.4 | 181.3 | 5.04% |
| 14 | 3cav 3+4/4/3 | 2 | 3 | 35 | 100 | 1.00 | 0.891 | 627 | 2*3*71+201=627 ok | 140 | 126 | 14 | 280 | 250.7 | 1.71% |
| 16 | 3cav 3+5/4/4 | 4 | 3 | 35 | 100 | 1.00 | 0.917 | 1053 | 4*3*71+201=1053 ok | 160 | 240 | 17.6 | 417.6 | 368.7 | 0.637% |
| 18 | 3cav 3+5/5/5 | 6 | 3 | 35 | 100 | 1.00 | 0.908 | 1479 | 6*3*71+201=1479 ok | 180 | 378 | 28.8 | 586.8 | 547.3 | 0.224% |

## FINAL strict recipe (SA=10n raw, K=2n, polt 1, R1, E=max(2,...)), 10-trial

**SUPERSEDED** (2026-10-13) by "FINAL fixed-cost strict recipe" below: fix-ups here skipped repeated centers, so lookups varied per trial; the n=20 run was stopped before finishing.

Started 2026-10-13 11:50 CST. SA = 10n raw cost evaluations (QM_SA_COUNT=raw), K = 2n top readouts per explore run and for the relabel round, polt 1, one relabel round, Gray, fixed_low, 1 trial x 10 instances. E = 2,2,2,2,4,6,8 for n=8..20. 2cav (n<=12): xL2, xs 25/35/50, r150; 3cav (n>=14): xL3, xs35, r100. L4 s50 lr0.05 xeta128 reta2.

Formulas: lookups = 10n + (E+1)*2n + E*n (upper bound; fix-ups are not repeated for a center already done). evals = E*S*(2s+1)+(2r+1), S = xL, s = xs. n=14/16/18 reused from kbit_strict (identical recipe).

| n | chip | E | xL | xs | r | success | mean p(GS) | evals | formula | SA lookups | readout lookups | fix-up lookups | total raw lookups | distinct lookups | share of 2^n (raw) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2cav 2+3/3 | 2 | 2 | 25 | 150 | 1.00 | 0.987 | 505 | 2*2*51+301=505 ok | 80 | 48 | 8 | 136 | 98.3 | 53.1% | kbit_final |
| 10 | 2cav 2+4/4 | 2 | 2 | 35 | 150 | 1.00 | 0.986 | 585 | 2*2*71+301=585 ok | 100 | 60 | 10 | 170 | 137.6 | 16.6% | kbit_final |
| 12 | 2cav 2+5/5 | 2 | 2 | 50 | 150 | 1.00 | 0.990 | 705 | 2*2*101+301=705 ok | 120 | 72 | 16.8 | 208.8 | 179.3 | 5.1% | kbit_final |
| 14 | 3cav 3+4/4/3 | 2 | 3 | 35 | 100 | 1.00 | 0.891 | 627 | 2*3*71+201=627 ok | 140 | 84 | 14 | 238 | 213.4 | 1.45% | kbit_strict |
| 16 | 3cav 3+5/4/4 | 4 | 3 | 35 | 100 | 1.00 | 0.917 | 1053 | 4*3*71+201=1053 ok | 160 | 160 | 17.6 | 337.6 | 298.9 | 0.515% | kbit_strict |
| 18 | 3cav 3+5/5/5 | 6 | 3 | 35 | 100 | 1.00 | 0.908 | 1479 | 6*3*71+201=1479 ok | 180 | 252 | 34.2 | 466.2 | 432.6 | 0.178% | kbit_strict |

## FINAL fixed-cost strict recipe

Started 2026-10-13 11:57 CST. Fixed cost per trial (QM_FIXED_COST=1): SA = 10n raw evaluations; readout = K = 2n top strings in each of the E+1 rounds (E explore + 1 relabel); fix-up = all n one-bit flips of the lowest-cost readout string so far, performed and counted every round even if repeated (n per round). Same guess logic as before. polt 1, R1, Gray, fixed_low, 1 trial x 10 instances. E = 2,2,2,2,4,6,8 for n=8..20. 2cav (n<=12): xL2, xs 25/35/50, r150; 3cav (n>=14): xL3, xs35, r100. L4 s50 lr0.05 xeta128 reta2. Fresh outdir relayout_runs/kbit_fixed.

Formulas: lookups per trial = 10n + (E+1)*2n + (E+1)*n = 10n + 3n(E+1), identical in every trial (asserted). evals = E*S*(2s+1)+(2r+1), S = xL, s = xs.

| n | chip | E | xL | xs | r | success | mean p(GS) | evals | formula check | SA / readout / fix-up lookups | total lookups per trial | share of 2^n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2cav 2+3/3 | 2 | 2 | 25 | 150 | 1.00 | 0.987 | 505 | 2*2*51+301=505 ok | 80 / 48 / 24 | 152 | 59.4% |
| 10 | 2cav 2+4/4 | 2 | 2 | 35 | 150 | 1.00 | 0.986 | 585 | 2*2*71+301=585 ok | 100 / 60 / 30 | 190 | 18.6% |
| 12 | 2cav 2+5/5 | 2 | 2 | 50 | 150 | 1.00 | 0.990 | 705 | 2*2*101+301=705 ok | 120 / 72 / 36 | 228 | 5.57% |
| 14 | 3cav 3+4/4/3 | 2 | 3 | 35 | 100 | 1.00 | 0.891 | 627 | 2*3*71+201=627 ok | 140 / 84 / 42 | 266 | 1.62% |
| 16 | 3cav 3+5/4/4 | 4 | 3 | 35 | 100 | 1.00 | 0.917 | 1053 | 4*3*71+201=1053 ok | 160 / 160 / 80 | 400 | 0.61% |
| 18 | 3cav 3+5/5/5 | 6 | 3 | 35 | 100 | 1.00 | 0.908 | 1479 | 6*3*71+201=1479 ok | 180 / 252 / 126 | 558 | 0.213% |

## FINAL one-recipe (r150, noise 0.3)

Started 2026-10-13 17:16 CST. Fixed cost per trial (QM_FIXED_COST=1): SA = 10n raw evaluations; readout = K = 2n top strings in each of the E+1 rounds (E explore + 1 relabel); fix-up = all n one-bit flips of the lowest-cost readout string so far, performed and counted every round even if repeated (n per round). Same guess logic as before. polt 1, R1, Gray, fixed_low, 1 trial x 10 instances. E = 2,2,2,2,4,6,8 for n=8..20. 2cav (n<=12): xL2, xs 25/35/50, r150; 3cav (n>=14): xL3, xs35, r150; SA layout noise 0.3 for all n. L4 s50 lr0.05 xeta128 reta2. Fresh outdir relayout_runs/final_r150nz03.

Formulas: lookups per trial = 10n + (E+1)*2n + (E+1)*n = 10n + 3n(E+1), identical in every trial (asserted). evals = E*S*(2s+1)+(2r+1), S = xL, s = xs.

| n | chip | E | xL | xs | r | success | mean p(GS) | evals | formula check | SA / readout / fix-up lookups | total lookups per trial | share of 2^n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 2cav 2+3/3 | 2 | 2 | 25 | 150 | 1.00 | 0.987 | 505 | 2*2*51+301=505 ok | 80 / 48 / 24 | 152 | 59.4% |
| 10 | 2cav 2+4/4 | 2 | 2 | 35 | 150 | 1.00 | 0.986 | 585 | 2*2*71+301=585 ok | 100 / 60 / 30 | 190 | 18.6% |
| 12 | 2cav 2+5/5 | 2 | 2 | 50 | 150 | 1.00 | 0.990 | 705 | 2*2*101+301=705 ok | 120 / 72 / 36 | 228 | 5.57% |
| 14 | 3cav 3+4/4/3 | 2 | 3 | 35 | 150 | 1.00 | 0.961 | 727 | 2*3*71+301=727 ok | 140 / 84 / 42 | 266 | 1.62% |
