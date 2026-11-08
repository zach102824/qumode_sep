# Historical B=0.1 KM scale-normalized restarts

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, the QAOA-depth diagnosis, the energy-legal selection diagnostic, and the uniform box initialization are not this result.

KM is the interleaved shared-gamma circuit with the round-4 starts and two repeats. KM_rad keeps that L=6 vector and adds twelve directions of radius 1 in box half-width units, then selects the lowest true energy. The continuation column is equal search. The radius column is not.

Incomplete: []. Never started: [].

| N | complete |
|---|---|
| 6 | 12/12 |
| 9 | 12/12 |
| 12 | 12/12 |

KM has 3+3L parameters (21 at L=6). QAOA has 2L (12 at L=6).

## Independent replay

| N | states | max rel dE | max abs dF | max abs dE0 | n_sub mismatches | max norm err | pass |
|---|---|---|---|---|---|---|---|
| 6 | 300 | 6.4e-13 | 6.1e-13 | 2.5e-14 | 0 | 3.2e-13 | True |
| 9 | 300 | 4.9e-13 | 2.7e-13 | 3.9e-14 | 0 | 2.9e-13 | True |
| 12 | 300 | 6.5e-13 | 2.9e-13 | 1.3e-13 | 0 | 3.1e-13 | True |

## Equal depth, equal search, continuation KM versus QAOA repeats 0 and 1

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 12 (12 / 0) | 0 | 0 | +0.050 [+0.013, +0.083] | +0.053 | 0.000488 |
| KM | 6 | 3 | 12 | 12 (12 / 0) | 0 | 0 | +0.234 [+0.158, +0.312] | +0.235 | 0.000488 |
| KM | 9 | 6 | 12 | 6 (6 / 0) | 2 | 4 | +0.110 [-0.051, +0.243] | +0.116 | 0.388 |
| KM | 9 | 3 | 12 | 6 (6 / 0) | 0 | 6 | +0.114 [-0.024, +0.273] | +0.139 | 0.388 |
| KM | 12 | 6 | 12 | 4 (4 / 0) | 0 | 8 | -0.008 [-0.047, +0.178] | +0.075 | 1 |
| KM | 12 | 3 | 12 | 5 (5 / 0) | 0 | 7 | +0.024 [+0.009, +0.135] | +0.076 | 0.00635 |

## Equal depth, radius-inclusive KM at L=6, unequal search

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM_rad | 6 | 6 | 12 | 12 (12 / 0) | 0 | 0 | +0.050 [+0.013, +0.083] | +0.053 | 0.000488 |
| KM_rad | 9 | 6 | 12 | 6 (6 / 0) | 2 | 4 | +0.110 [-0.051, +0.243] | +0.116 | 0.388 |
| KM_rad | 12 | 6 | 12 | 4 (4 / 0) | 0 | 8 | -0.008 [-0.047, +0.178] | +0.075 | 1 |

## Energy drop

An N is deeper when the median energy drop exceeds 0.01 or at least 4 instances drop by more than 0.01. The drop is continuation KM energy minus radius-inclusive energy.

| N | n | median drop | n drop > 0.01 | deeper | continuation pass | radius pass |
|---|---|---|---|---|---|---|
| 6 | 12 | 0.0000 | 0 | False | True | True |
| 9 | 12 | 0.0000 | 0 | False | False | False |
| 12 | 12 | 0.0000 | 0 | False | False | False |

N=6: Continuation KM clears the historical comparator on this sample. The radius column says whether a lower energy keeps that clearance.
N=9: The scale-normalized L=6 directions did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten on this sample.
N=12: The scale-normalized L=6 directions did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten on this sample.

## Secondary equal parameter count, 12 parameters, not the verdict

| N | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|
| 6 | 2 (1) | 8 | -0.013 [-0.043, +0.001] |
| 9 | 2 (0) | 7 | -0.095 [-0.108, +0.012] |
| 12 | 1 (0) | 7 | -0.104 [-0.132, -0.053] |

## Verdict

Continuation column: **partial**.
Radius-inclusive column: **partial**.
The verdict of this round is the radius-inclusive column. Equal depth is the comparison. Equal search is the continuation column only.
- N=6: continuation pass, radius pass, deeper False, on 12 complete instances.
- N=9: continuation does not pass, radius does not pass, deeper False, on 12 complete instances.
- N=12: continuation does not pass, radius does not pass, deeper False, on 12 complete instances.

## L=6 instances

| N | k | continuation label | radius label | F continuation | F radius | F QAOA | dF continuation | dF radius | E continuation-QAOA | energy drop | continuation | radius |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 154 | p1_perturb_0.1 | historical | 0.998 | 0.998 | 0.990 | +0.008 | +0.008 | -0.0511 | +0.0000 | win_joint | win_joint |
| 6 | 155 | nested_K | historical | 0.994 | 0.994 | 0.950 | +0.044 | +0.044 | -0.1840 | +0.0000 | win_joint | win_joint |
| 6 | 156 | p0_time_rescaled | historical | 0.997 | 0.997 | 0.942 | +0.055 | +0.055 | -0.2113 | +0.0000 | win_joint | win_joint |
| 6 | 157 | p1_interp | historical | 0.993 | 0.993 | 0.873 | +0.120 | +0.120 | -0.2372 | +0.0000 | win_joint | win_joint |
| 6 | 158 | p0_interp | historical | 0.995 | 0.995 | 0.984 | +0.011 | +0.011 | -0.0661 | +0.0000 | win_joint | win_joint |
| 6 | 159 | p1_time_rescaled | historical | 0.971 | 0.971 | 0.857 | +0.114 | +0.114 | -0.2531 | +0.0000 | win_joint | win_joint |
| 6 | 160 | p2_time_rescaled | historical | 0.996 | 0.996 | 0.924 | +0.072 | +0.072 | -0.3213 | +0.0000 | win_joint | win_joint |
| 6 | 161 | p2_interp | historical | 0.745 | 0.745 | 0.650 | +0.094 | +0.094 | -0.1973 | +0.0000 | win_joint | win_joint |
| 6 | 162 | p0_interp | historical | 0.989 | 0.989 | 0.929 | +0.060 | +0.060 | -0.1364 | +0.0000 | win_joint | win_joint |
| 6 | 163 | p0_time_rescaled | historical | 0.998 | 0.998 | 0.983 | +0.015 | +0.015 | -0.0569 | +0.0000 | win_joint | win_joint |
| 6 | 164 | p1_perturb_0.1 | historical | 0.994 | 0.994 | 0.956 | +0.039 | +0.039 | -0.1542 | +0.0000 | win_joint | win_joint |
| 6 | 165 | p2_time_rescaled | historical | 0.999 | 0.999 | 0.994 | +0.005 | +0.005 | -0.0406 | +0.0000 | win_joint | win_joint |
| 9 | 154 | p2_time_rescaled | historical | 0.890 | 0.890 | 0.701 | +0.190 | +0.190 | -0.7620 | +0.0000 | win_joint | win_joint |
| 9 | 155 | p2_time_rescaled | historical | 0.357 | 0.357 | 0.459 | -0.102 | -0.102 | +0.6419 | +0.0000 | loss_joint | loss_joint |
| 9 | 156 | fresh_ramp_jitter | historical | 0.215 | 0.215 | 0.143 | +0.071 | +0.071 | +0.0893 | +0.0000 | tie | tie |
| 9 | 157 | p1_interp | historical | 0.238 | 0.238 | 0.202 | +0.036 | +0.036 | -0.0386 | +0.0000 | tie | tie |
| 9 | 158 | p1_interp | historical | 0.060 | 0.060 | 0.132 | -0.072 | -0.072 | +0.1038 | +0.0000 | tie | tie |
| 9 | 159 | p2_interp | historical | 0.741 | 0.741 | 0.592 | +0.149 | +0.149 | -0.1001 | +0.0000 | win_joint | win_joint |
| 9 | 160 | p0_perturb_0.1 | historical | 0.861 | 0.861 | 0.601 | +0.261 | +0.261 | -0.8640 | +0.0000 | win_joint | win_joint |
| 9 | 161 | fresh_ramp_jitter | historical | 0.056 | 0.056 | 0.204 | -0.148 | -0.148 | -0.0557 | +0.0000 | loss_fidelity_only | loss_fidelity_only |
| 9 | 162 | p0_time_rescaled | historical | 0.768 | 0.768 | 0.543 | +0.225 | +0.225 | -0.3320 | +0.0000 | win_joint | win_joint |
| 9 | 163 | p2_interp | historical | 0.943 | 0.943 | 0.619 | +0.325 | +0.325 | -1.2238 | +0.0000 | win_joint | win_joint |
| 9 | 164 | p0_time_rescaled | historical | 0.141 | 0.141 | 0.171 | -0.030 | -0.030 | -0.4114 | +0.0000 | tie | tie |
| 9 | 165 | nested_K | historical | 0.688 | 0.688 | 0.203 | +0.484 | +0.484 | -0.5756 | +0.0000 | win_joint | win_joint |
| 12 | 154 | fresh_ramp_jitter | historical | 0.684 | 0.684 | 0.489 | +0.195 | +0.195 | -0.4527 | +0.0000 | win_joint | win_joint |
| 12 | 155 | p2_interp | historical | 0.386 | 0.386 | 0.225 | +0.161 | +0.161 | -0.9367 | +0.0000 | win_joint | win_joint |
| 12 | 156 | p2_interp | historical | 0.107 | 0.107 | 0.170 | -0.063 | -0.063 | +0.9416 | +0.0000 | tie | tie |
| 12 | 157 | fresh_ramp | historical | 0.189 | 0.189 | 0.139 | +0.049 | +0.049 | +0.3578 | +0.0000 | tie | tie |
| 12 | 158 | p1_time_rescaled | historical | 0.203 | 0.203 | 0.260 | -0.057 | -0.057 | -0.2546 | +0.0000 | tie | tie |
| 12 | 159 | p0_time_rescaled | historical | 0.199 | 0.199 | 0.224 | -0.025 | -0.025 | +0.3846 | +0.0000 | tie | tie |
| 12 | 160 | p1_time_rescaled | historical | 0.786 | 0.786 | 0.571 | +0.215 | +0.215 | -0.8637 | +0.0000 | win_joint | win_joint |
| 12 | 161 | p2_interp | historical | 0.119 | 0.119 | 0.156 | -0.038 | -0.038 | +0.2339 | +0.0000 | tie | tie |
| 12 | 162 | p2_interp | historical | 0.118 | 0.118 | 0.110 | +0.008 | +0.008 | +0.1287 | +0.0000 | tie | tie |
| 12 | 163 | p1_time_rescaled | historical | 0.098 | 0.098 | 0.171 | -0.072 | -0.072 | +0.3906 | +0.0000 | tie | tie |
| 12 | 164 | p0_time_rescaled | historical | 0.277 | 0.277 | 0.302 | -0.025 | -0.025 | -0.1748 | +0.0000 | tie | tie |
| 12 | 165 | p2_interp | historical | 0.757 | 0.757 | 0.203 | +0.553 | +0.553 | -2.4048 | +0.0000 | win_joint | win_joint |

## Cost

Optimizer wall 1131.8 s of 4200. Starts 25344, of which 432 are the extra L=6 KM directions. L=6 KM_rad states whose label is one of those directions: 0. Logged energy evaluations 2036775, L=1 grid evaluations 18432, capped starts 0.
