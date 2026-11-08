# Historical B=0.1 classical popcount initialization

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The packing rounds, the energy-only union, and the quarter-pi restarts are not this result.

KM uses classical popcount shells on the interleaved triples. QAOA uses the round-4 start list. Equal search is not claimed.

Primary repeats incomplete: []. Never started: [].

| N | primary complete |
|---|---|
| 6 | 12/12 |
| 9 | 12/12 |
| 12 | 12/12 |

KM has 3+3L parameters (21 at L=6). QAOA has 2L (12 at L=6).

## Independent replay

| N | states | max rel dE | max abs dF | max abs dE0 | n_sub mismatches | max norm err | pass |
|---|---|---|---|---|---|---|---|
| 6 | 288 | 4.5e-13 | 4.6e-13 | 2.0e-14 | 0 | 1.9e-13 | True |
| 9 | 288 | 8.5e-13 | 1.1e-13 | 9.9e-14 | 0 | 4.4e-13 | True |
| 12 | 288 | 5.2e-13 | 1.2e-13 | 8.5e-14 | 0 | 2.6e-13 | True |

## Equal depth, primary QAOA repeats 0 and 1

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 11 (11 / 0) | 0 | 1 | +0.253 [+0.127, +0.418] | +0.280 | 0.00635 |
| KM | 6 | 3 | 12 | 11 (11 / 0) | 0 | 1 | +0.302 [+0.124, +0.503] | +0.331 | 0.000488 |
| KM | 9 | 6 | 12 | 7 (7 / 0) | 3 | 2 | +0.111 [-0.040, +0.245] | +0.101 | 0.774 |
| KM | 9 | 3 | 12 | 6 (6 / 0) | 1 | 5 | +0.068 [+0.011, +0.203] | +0.086 | 0.0386 |
| KM | 12 | 6 | 12 | 2 (1 / 1) | 2 | 8 | +0.002 [-0.025, +0.050] | +0.012 | 0.774 |
| KM | 12 | 3 | 12 | 1 (1 / 0) | 0 | 11 | +0.008 [-0.018, +0.028] | +0.012 | 0.774 |

## Secondary equal parameter count, 12 parameters, not the verdict

| N | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|
| 6 | 5 (4) | 4 | -0.004 [-0.033, +0.338] |
| 9 | 0 (0) | 8 | -0.144 [-0.299, -0.027] |
| 12 | 0 (0) | 4 | -0.062 [-0.107, -0.013] |

## Verdict

Overall: **partial**.
- N=6: pass on 12 complete instances.
- N=9: does not pass on 12 complete instances.
- N=12: does not pass on 12 complete instances.

## L=6 instances

| N | k | label | F KM | F QAOA | dF | E KM - E QAOA | outcome |
|---|---|---|---|---|---|---|---|
| 6 | 92 | p2_interp | 0.995 | 0.996 | -0.001 | +0.0164 | tie |
| 6 | 93 | p2_interp | 0.994 | 0.318 | +0.675 | -0.7258 | win_joint |
| 6 | 94 | p2_time_rescaled | 0.267 | 0.001 | +0.266 | -0.1300 | win_joint |
| 6 | 95 | p1_time_rescaled | 0.984 | 0.752 | +0.232 | -0.4287 | win_joint |
| 6 | 96 | p0_perturb_0.1 | 0.982 | 0.439 | +0.544 | -0.2620 | win_joint |
| 6 | 97 | cls_reseed_m | 0.982 | 0.844 | +0.138 | -0.3108 | win_joint |
| 6 | 98 | cls_reseed_p | 0.877 | 0.578 | +0.299 | -0.3418 | win_joint |
| 6 | 99 | p0_interp | 0.906 | 0.667 | +0.239 | -0.3090 | win_joint |
| 6 | 102 | p0_interp | 0.960 | 0.943 | +0.017 | -0.0777 | win_joint |
| 6 | 103 | p0_time_rescaled | 0.959 | 0.506 | +0.453 | -0.6217 | win_joint |
| 6 | 104 | p1_perturb_0.1 | 0.965 | 0.581 | +0.383 | -0.2801 | win_joint |
| 6 | 105 | p1_time_rescaled | 0.606 | 0.489 | +0.117 | -0.1536 | win_joint |
| 9 | 92 | p0_interp | 0.748 | 0.412 | +0.336 | -0.4773 | win_joint |
| 9 | 93 | p1_interp | 0.378 | 0.186 | +0.192 | -0.0741 | win_joint |
| 9 | 94 | cls_reseed_p | 0.638 | 0.737 | -0.098 | +0.2614 | loss_joint |
| 9 | 95 | p0_time_rescaled | 0.771 | 0.824 | -0.052 | +0.2431 | loss_joint |
| 9 | 96 | cls_reseed_m | 0.294 | 0.322 | -0.027 | +0.3623 | tie |
| 9 | 97 | p0_time_rescaled | 0.258 | 0.131 | +0.126 | -0.5951 | win_joint |
| 9 | 98 | p0_perturb_0.1 | 0.885 | 0.647 | +0.237 | -0.7427 | win_joint |
| 9 | 99 | p0_time_rescaled | 0.359 | 0.489 | -0.130 | +0.4951 | loss_joint |
| 9 | 102 | p0_interp | 0.549 | 0.296 | +0.253 | -0.5906 | win_joint |
| 9 | 103 | p1_interp | 0.547 | 0.561 | -0.014 | -0.4913 | tie |
| 9 | 104 | p1_perturb_0.1 | 0.929 | 0.833 | +0.096 | -0.6640 | win_joint |
| 9 | 105 | p1_time_rescaled | 0.512 | 0.221 | +0.291 | -0.5070 | win_joint |
| 12 | 92 | cls_reseed_m | 0.170 | 0.169 | +0.001 | +0.4772 | tie |
| 12 | 93 | p2_perturb_0.1 | 0.297 | 0.306 | -0.009 | -0.4180 | tie |
| 12 | 94 | p0_interp | 0.176 | 0.120 | +0.056 | +0.1176 | tie |
| 12 | 95 | p1_interp | 0.042 | 0.140 | -0.098 | +0.0807 | loss_joint |
| 12 | 96 | p2_perturb_0.1 | 0.000 | 0.249 | -0.249 | -0.3141 | loss_fidelity_only |
| 12 | 97 | p0_perturb_0.1 | 0.263 | 0.219 | +0.044 | +0.2510 | tie |
| 12 | 98 | p1_perturb_0.3 | 0.487 | 0.251 | +0.237 | -0.2866 | win_joint |
| 12 | 99 | cls_reseed_m | 0.011 | 0.018 | -0.007 | +0.0885 | tie_both_failed |
| 12 | 102 | p1_perturb_0.1 | 0.035 | 0.017 | +0.017 | +0.4453 | tie |
| 12 | 103 | p0_interp | 0.004 | 0.001 | +0.003 | -0.1488 | tie_both_failed |
| 12 | 104 | p1_time_rescaled | 0.020 | 0.060 | -0.040 | -0.4572 | tie |
| 12 | 105 | p0_interp | 0.233 | 0.046 | +0.187 | +0.3264 | win_fidelity_only |

## Cost

Optimizer wall 447.4 s of 3600. Starts 12240. Logged energy evaluations 925477, L=1 grid evaluations 4608, capped starts 25.
