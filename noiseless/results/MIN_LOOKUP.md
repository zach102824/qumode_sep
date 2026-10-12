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
