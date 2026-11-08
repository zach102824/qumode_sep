# Historical B=0.1 QAOA-depth diagnosis

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, and the repulsion escape are not this result.

KM is the interleaved shared-gamma circuit with the round-4 starts and two repeats. The historical comparator is QAOA with that same start list and the same two repeats. qaoa_deep keeps that L=6 vector and adds eight box starts and eight N(0, 0.75) kicks, then selects the lowest true energy. Equal search is not claimed for the deep column.

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
| 6 | 300 | 4.6e-13 | 4.4e-13 | 1.2e-14 | 0 | 2.2e-13 | True |
| 9 | 300 | 3.5e-13 | 8.0e-14 | 7.3e-14 | 0 | 1.8e-13 | True |
| 12 | 300 | 4.0e-13 | 6.4e-14 | 8.5e-14 | 0 | 2.0e-13 | True |

## Equal depth, historical QAOA repeats 0 and 1

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 11 (11 / 0) | 0 | 1 | +0.143 [+0.065, +0.262] | +0.196 | 0.000488 |
| KM | 6 | 3 | 12 | 12 (12 / 0) | 0 | 0 | +0.250 [+0.162, +0.484] | +0.334 | 0.000488 |
| KM | 9 | 6 | 12 | 7 (6 / 1) | 2 | 3 | +0.087 [-0.027, +0.131] | +0.064 | 0.388 |
| KM | 9 | 3 | 12 | 5 (5 / 0) | 0 | 7 | +0.050 [+0.014, +0.112] | +0.064 | 0.0386 |
| KM | 12 | 6 | 12 | 5 (5 / 0) | 0 | 7 | +0.045 [-0.026, +0.167] | +0.111 | 0.388 |
| KM | 12 | 3 | 12 | 3 (3 / 0) | 1 | 8 | +0.023 [-0.007, +0.096] | +0.040 | 0.388 |

## Equal depth against qaoa_deep at L=6, unequal search

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 11 (11 / 0) | 0 | 1 | +0.143 [+0.065, +0.262] | +0.196 | 0.000488 |
| KM | 9 | 6 | 12 | 7 (6 / 1) | 2 | 3 | +0.087 [-0.027, +0.131] | +0.064 | 0.388 |
| KM | 12 | 6 | 12 | 5 (5 / 0) | 0 | 7 | +0.045 [-0.026, +0.167] | +0.111 | 0.388 |

## Diagnosis

An N is under-optimized when the median energy drop exceeds 0.01 or at least 4 instances drop by more than 0.01. The drop is historical QAOA energy minus qaoa_deep energy.

| N | n | median drop | n drop > 0.01 | underoptimized | base pass | deep pass |
|---|---|---|---|---|---|---|
| 6 | 12 | 0.0000 | 0 | False | True | True |
| 9 | 12 | 0.0000 | 0 | False | False | False |
| 12 | 12 | 0.0000 | 0 | False | False | False |

N=6: Standard KM clears the historical comparator on this sample. The deep column says whether that clearance survives the heavier QAOA search.
N=9: The historical QAOA search was not the weak point on this sample. The miss is a KM result under energy-only selection.
N=12: The historical QAOA search was not the weak point on this sample. The miss is a KM result under energy-only selection.

## Secondary equal parameter count, 12 parameters, not the verdict

| N | comparator | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|---|
| 6 | historical QAOA | 6 (5) | 4 | +0.027 [-0.101, +0.269] |
| 6 | qaoa_deep | 6 (5) | 4 | +0.027 [-0.101, +0.269] |
| 9 | historical QAOA | 0 (0) | 9 | -0.113 [-0.125, -0.067] |
| 9 | qaoa_deep | 0 (0) | 9 | -0.113 [-0.125, -0.067] |
| 12 | historical QAOA | 1 (0) | 7 | -0.100 [-0.112, -0.019] |
| 12 | qaoa_deep | 1 (0) | 7 | -0.100 [-0.112, -0.019] |

## Verdict

Historical comparator: **partial**.
Heavier QAOA search: **partial**.
- N=6: historical pass, deep pass, underoptimized False, on 12 complete instances.
- N=9: historical does not pass, deep does not pass, underoptimized False, on 12 complete instances.
- N=12: historical does not pass, deep does not pass, underoptimized False, on 12 complete instances.

## L=6 instances

| N | k | KM label | deep label | F KM | F base | F deep | dF base | dF deep | E KM-base | energy drop | base | deep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 118 | nested_K | historical | 0.270 | 0.011 | 0.011 | +0.258 | +0.258 | -0.2461 | +0.0000 | win_joint | win_joint |
| 6 | 119 | p1_time_rescaled | historical | 0.994 | 0.906 | 0.906 | +0.089 | +0.089 | -0.2833 | +0.0000 | win_joint | win_joint |
| 6 | 120 | p2_time_rescaled | historical | 0.100 | 0.018 | 0.018 | +0.082 | +0.082 | -0.1058 | +0.0000 | tie | tie |
| 6 | 121 | p1_perturb_0.1 | historical | 0.931 | 0.565 | 0.565 | +0.366 | +0.366 | -0.3299 | +0.0000 | win_joint | win_joint |
| 6 | 122 | p2_time_rescaled | historical | 0.967 | 0.920 | 0.920 | +0.047 | +0.047 | -0.0847 | +0.0000 | win_joint | win_joint |
| 6 | 123 | nested_K | historical | 0.996 | 0.952 | 0.952 | +0.044 | +0.044 | -0.1778 | +0.0000 | win_joint | win_joint |
| 6 | 124 | p1_interp | historical | 0.325 | 0.059 | 0.059 | +0.266 | +0.266 | -0.0594 | +0.0000 | win_joint | win_joint |
| 6 | 125 | p0_interp | historical | 0.974 | 0.725 | 0.725 | +0.249 | +0.249 | -0.2717 | +0.0000 | win_joint | win_joint |
| 6 | 126 | p1_perturb_0.1 | historical | 0.964 | 0.352 | 0.352 | +0.612 | +0.612 | -0.3333 | +0.0000 | win_joint | win_joint |
| 6 | 127 | p2_perturb_0.1 | historical | 0.974 | 0.927 | 0.927 | +0.047 | +0.047 | -0.2500 | +0.0000 | win_joint | win_joint |
| 6 | 128 | fresh_ramp | historical | 0.933 | 0.837 | 0.837 | +0.096 | +0.096 | -0.1629 | +0.0000 | win_joint | win_joint |
| 6 | 129 | nested_K | historical | 0.824 | 0.634 | 0.634 | +0.190 | +0.190 | -0.3494 | +0.0000 | win_joint | win_joint |
| 9 | 118 | p1_time_rescaled | historical | 0.447 | 0.541 | 0.541 | -0.094 | -0.094 | +0.2900 | +0.0000 | loss_joint | loss_joint |
| 9 | 119 | p1_time_rescaled | historical | 0.083 | 0.205 | 0.205 | -0.122 | -0.122 | -0.2962 | +0.0000 | loss_fidelity_only | loss_fidelity_only |
| 9 | 120 | p0_time_rescaled | historical | 0.297 | 0.301 | 0.301 | -0.005 | -0.005 | +0.2247 | +0.0000 | tie | tie |
| 9 | 121 | p1_perturb_0.1 | historical | 0.154 | 0.045 | 0.045 | +0.109 | +0.109 | -0.0905 | +0.0000 | win_joint | win_joint |
| 9 | 122 | p2_interp | historical | 0.457 | 0.414 | 0.414 | +0.043 | +0.043 | +0.0623 | +0.0000 | tie | tie |
| 9 | 123 | p2_perturb_0.1 | historical | 0.199 | 0.249 | 0.249 | -0.050 | -0.050 | -0.0282 | +0.0000 | tie | tie |
| 9 | 124 | p1_time_rescaled | historical | 0.178 | 0.050 | 0.050 | +0.128 | +0.128 | +0.0293 | +0.0000 | win_fidelity_only | win_fidelity_only |
| 9 | 125 | p2_time_rescaled | historical | 0.512 | 0.375 | 0.375 | +0.138 | +0.138 | -0.3320 | +0.0000 | win_joint | win_joint |
| 9 | 126 | p1_time_rescaled | historical | 0.441 | 0.307 | 0.307 | +0.134 | +0.134 | -0.1918 | +0.0000 | win_joint | win_joint |
| 9 | 127 | p1_time_rescaled | historical | 0.910 | 0.781 | 0.781 | +0.128 | +0.128 | -0.6695 | +0.0000 | win_joint | win_joint |
| 9 | 128 | p1_perturb_0.1 | historical | 0.621 | 0.556 | 0.556 | +0.065 | +0.065 | -0.2382 | +0.0000 | win_joint | win_joint |
| 9 | 129 | p1_interp | historical | 0.620 | 0.328 | 0.328 | +0.292 | +0.292 | -0.3484 | +0.0000 | win_joint | win_joint |
| 12 | 118 | nested_K | historical | 0.211 | 0.114 | 0.114 | +0.097 | +0.097 | -0.4650 | +0.0000 | win_joint | win_joint |
| 12 | 119 | fresh_ramp_jitter | historical | 0.123 | 0.175 | 0.175 | -0.052 | -0.052 | +0.2865 | +0.0000 | tie | tie |
| 12 | 120 | p2_time_rescaled | historical | 0.332 | 0.283 | 0.283 | +0.050 | +0.050 | +0.1186 | +0.0000 | tie | tie |
| 12 | 121 | p2_interp | historical | 0.106 | 0.170 | 0.170 | -0.065 | -0.065 | +0.1526 | +0.0000 | tie | tie |
| 12 | 122 | nested_K | historical | 0.003 | 0.004 | 0.004 | -0.001 | -0.001 | -0.7290 | +0.0000 | tie_both_failed | tie_both_failed |
| 12 | 123 | p0_perturb_0.1 | historical | 0.091 | 0.154 | 0.154 | -0.063 | -0.063 | +0.4132 | +0.0000 | tie | tie |
| 12 | 124 | fresh_ramp | historical | 0.700 | 0.195 | 0.195 | +0.505 | +0.505 | -1.3855 | +0.0000 | win_joint | win_joint |
| 12 | 125 | fresh_ramp_jitter | historical | 0.763 | 0.277 | 0.277 | +0.486 | +0.486 | -0.6678 | +0.0000 | win_joint | win_joint |
| 12 | 126 | fresh_ramp_jitter | historical | 0.464 | 0.227 | 0.227 | +0.238 | +0.238 | -0.8095 | +0.0000 | win_joint | win_joint |
| 12 | 127 | nested_K | historical | 0.270 | 0.262 | 0.262 | +0.009 | +0.009 | +0.0952 | +0.0000 | tie | tie |
| 12 | 128 | nested_K | historical | 0.276 | 0.235 | 0.235 | +0.041 | +0.041 | -0.6523 | +0.0000 | tie | tie |
| 12 | 129 | p2_time_rescaled | historical | 0.453 | 0.370 | 0.370 | +0.083 | +0.083 | -1.1815 | +0.0000 | win_joint | win_joint |

## Cost

Optimizer wall 1159.0 s of 4200. Starts 25488, of which 576 are the extra L=6 QAOA starts. L=6 qaoa_deep states whose label is one of those starts: 0. Logged energy evaluations 1962925, L=1 grid evaluations 18432, capped starts 6.
