# Historical B=0.1 KM box initialization

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, the QAOA-depth diagnosis, and the energy-legal selection diagnostic are not this result.

KM is the interleaved shared-gamma circuit with the round-4 starts and two repeats. KM_box keeps that L=6 vector and adds twelve uniform draws in the KM box, then selects the lowest true energy. The continuation column is equal search. The box column is not.

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
| 6 | 300 | 1.8e-13 | 1.8e-13 | 2.0e-14 | 0 | 9.0e-14 | True |
| 9 | 300 | 2.9e-13 | 1.4e-13 | 1.4e-13 | 0 | 1.4e-13 | True |
| 12 | 300 | 1.7e-13 | 6.8e-14 | 7.1e-14 | 0 | 8.7e-14 | True |

## Equal depth, equal search, continuation KM versus QAOA repeats 0 and 1

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 8 (8 / 0) | 1 | 3 | +0.026 [-0.001, +0.234] | +0.131 | 0.146 |
| KM | 6 | 3 | 12 | 9 (9 / 0) | 1 | 2 | +0.122 [+0.060, +0.249] | +0.135 | 0.0386 |
| KM | 9 | 6 | 12 | 9 (8 / 1) | 1 | 2 | +0.161 [+0.052, +0.264] | +0.167 | 0.00635 |
| KM | 9 | 3 | 12 | 9 (9 / 0) | 0 | 3 | +0.151 [+0.065, +0.254] | +0.168 | 0.00635 |
| KM | 12 | 6 | 12 | 5 (5 / 0) | 0 | 7 | +0.024 [-0.007, +0.229] | +0.109 | 0.146 |
| KM | 12 | 3 | 12 | 5 (5 / 0) | 0 | 7 | +0.059 [+0.010, +0.128] | +0.062 | 0.146 |

## Equal depth, box-inclusive KM at L=6, unequal search

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM_box | 6 | 6 | 12 | 8 (8 / 0) | 1 | 3 | +0.026 [-0.001, +0.234] | +0.131 | 0.146 |
| KM_box | 9 | 6 | 12 | 9 (8 / 1) | 1 | 2 | +0.161 [+0.052, +0.264] | +0.167 | 0.00635 |
| KM_box | 12 | 6 | 12 | 5 (5 / 0) | 0 | 7 | +0.024 [-0.007, +0.229] | +0.109 | 0.146 |

## Energy drop

An N is deeper when the median energy drop exceeds 0.01 or at least 4 instances drop by more than 0.01. The drop is continuation KM energy minus box-inclusive energy.

| N | n | median drop | n drop > 0.01 | deeper | continuation pass | box pass |
|---|---|---|---|---|---|---|
| 6 | 12 | 0.0000 | 0 | False | False | False |
| 9 | 12 | 0.0000 | 0 | False | True | True |
| 12 | 12 | 0.0000 | 0 | False | False | False |

N=6: The uniform L=6 draws did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten on this sample.
N=9: Continuation KM clears the historical comparator on this sample. The box column says whether a lower energy keeps that clearance.
N=12: The uniform L=6 draws did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten on this sample.

## Secondary equal parameter count, 12 parameters, not the verdict

| N | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|
| 6 | 1 (0) | 9 | -0.047 [-0.098, -0.017] |
| 9 | 2 (0) | 6 | -0.061 [-0.145, +0.034] |
| 12 | 0 (0) | 7 | -0.097 [-0.148, -0.074] |

## Verdict

Continuation column: **not_confirmed**.
Box-inclusive column: **not_confirmed**.
The verdict of this round is the box-inclusive column. Equal depth is the comparison. Equal search is the continuation column only.
- N=6: continuation does not pass, box does not pass, deeper False, on 12 complete instances.
- N=9: continuation pass, box pass, deeper False, on 12 complete instances.
- N=12: continuation does not pass, box does not pass, deeper False, on 12 complete instances.

## L=6 instances

| N | k | continuation label | box label | F continuation | F box | F QAOA | dF continuation | dF box | E continuation-QAOA | energy drop | continuation | box |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 142 | p1_perturb_0.1 | historical | 0.784 | 0.784 | 0.783 | +0.001 | +0.001 | +0.1144 | +0.0000 | tie | tie |
| 6 | 143 | fresh_ramp_jitter | historical | 0.244 | 0.244 | 0.273 | -0.029 | -0.029 | -0.0446 | +0.0000 | tie | tie |
| 6 | 144 | p2_perturb_0.1 | historical | 0.996 | 0.996 | 0.960 | +0.036 | +0.036 | -0.1342 | +0.0000 | win_joint | win_joint |
| 6 | 145 | p0_time_rescaled | historical | 0.877 | 0.877 | 0.710 | +0.167 | +0.167 | -0.0981 | +0.0000 | win_joint | win_joint |
| 6 | 146 | p0_interp | historical | 0.786 | 0.786 | 0.628 | +0.158 | +0.158 | -0.0756 | +0.0000 | win_joint | win_joint |
| 6 | 147 | p0_time_rescaled | historical | 0.601 | 0.601 | 0.140 | +0.461 | +0.461 | -0.2904 | +0.0000 | win_joint | win_joint |
| 6 | 148 | p1_time_rescaled | historical | 0.947 | 0.947 | 0.393 | +0.554 | +0.554 | -0.2516 | +0.0000 | win_joint | win_joint |
| 6 | 149 | p1_interp | historical | 0.991 | 0.991 | 0.994 | -0.002 | -0.002 | +0.0249 | +0.0000 | tie | tie |
| 6 | 150 | p2_interp | historical | 0.992 | 0.992 | 0.976 | +0.016 | +0.016 | -0.1191 | +0.0000 | win_joint | win_joint |
| 6 | 151 | fresh_ramp | historical | 0.871 | 0.871 | 0.570 | +0.301 | +0.301 | -0.2496 | +0.0000 | win_joint | win_joint |
| 6 | 152 | p0_perturb_0.1 | historical | 0.676 | 0.676 | 0.787 | -0.111 | -0.111 | +0.1390 | +0.0000 | loss_joint | loss_joint |
| 6 | 153 | p1_interp | historical | 0.994 | 0.994 | 0.979 | +0.015 | +0.015 | -0.0534 | +0.0000 | win_joint | win_joint |
| 9 | 142 | p2_perturb_0.1 | historical | 0.855 | 0.855 | 0.425 | +0.430 | +0.430 | -0.7108 | +0.0000 | win_joint | win_joint |
| 9 | 143 | p0_interp | historical | 0.170 | 0.170 | 0.277 | -0.107 | -0.107 | -0.2621 | +0.0000 | loss_fidelity_only | loss_fidelity_only |
| 9 | 144 | fresh_ramp | historical | 0.245 | 0.245 | 0.157 | +0.088 | +0.088 | -0.2336 | +0.0000 | win_joint | win_joint |
| 9 | 145 | nested_K | historical | 0.782 | 0.782 | 0.403 | +0.378 | +0.378 | -0.7347 | +0.0000 | win_joint | win_joint |
| 9 | 146 | p0_time_rescaled | historical | 0.861 | 0.861 | 0.637 | +0.224 | +0.224 | -0.8252 | +0.0000 | win_joint | win_joint |
| 9 | 147 | p0_interp | historical | 0.476 | 0.476 | 0.322 | +0.154 | +0.154 | +0.0774 | +0.0000 | win_fidelity_only | win_fidelity_only |
| 9 | 148 | p1_time_rescaled | historical | 0.920 | 0.920 | 0.752 | +0.168 | +0.168 | -0.7499 | +0.0000 | win_joint | win_joint |
| 9 | 149 | p2_time_rescaled | historical | 0.483 | 0.483 | 0.215 | +0.268 | +0.268 | -0.1618 | +0.0000 | win_joint | win_joint |
| 9 | 150 | p2_perturb_0.1 | historical | 0.020 | 0.020 | 0.005 | +0.016 | +0.016 | -0.3023 | +0.0000 | tie | tie |
| 9 | 151 | p1_perturb_0.1 | historical | 0.779 | 0.779 | 0.772 | +0.006 | +0.006 | -0.5968 | +0.0000 | tie | tie |
| 9 | 152 | p0_interp | historical | 0.428 | 0.428 | 0.168 | +0.259 | +0.259 | -0.2575 | +0.0000 | win_joint | win_joint |
| 9 | 153 | nested_K | historical | 0.912 | 0.912 | 0.787 | +0.124 | +0.124 | -0.5589 | +0.0000 | win_joint | win_joint |
| 12 | 142 | p2_interp | historical | 0.182 | 0.182 | 0.198 | -0.016 | -0.016 | -0.3396 | +0.0000 | tie | tie |
| 12 | 143 | p1_time_rescaled | historical | 0.691 | 0.691 | 0.478 | +0.213 | +0.213 | -1.3840 | +0.0000 | win_joint | win_joint |
| 12 | 144 | p1_perturb_0.1 | historical | 0.607 | 0.607 | 0.445 | +0.162 | +0.162 | -1.1627 | +0.0000 | win_joint | win_joint |
| 12 | 145 | p1_interp | historical | 0.469 | 0.469 | 0.098 | +0.371 | +0.371 | -0.2875 | +0.0000 | win_joint | win_joint |
| 12 | 146 | fresh_ramp | historical | 0.274 | 0.274 | 0.248 | +0.026 | +0.026 | -2.0476 | +0.0000 | tie | tie |
| 12 | 147 | p0_time_rescaled | historical | 0.269 | 0.269 | 0.248 | +0.022 | +0.022 | +0.2700 | +0.0000 | tie | tie |
| 12 | 148 | p1_interp | historical | 0.148 | 0.148 | 0.146 | +0.002 | +0.002 | +0.1132 | +0.0000 | tie | tie |
| 12 | 149 | p1_perturb_0.1 | historical | 0.232 | 0.232 | 0.265 | -0.033 | -0.033 | -0.7620 | +0.0000 | tie | tie |
| 12 | 150 | p1_interp | historical | 0.598 | 0.598 | 0.264 | +0.334 | +0.334 | -0.8534 | +0.0000 | win_joint | win_joint |
| 12 | 151 | p0_time_rescaled | historical | 0.473 | 0.473 | 0.228 | +0.246 | +0.246 | -0.9960 | +0.0000 | win_joint | win_joint |
| 12 | 152 | p1_perturb_0.1 | historical | 0.082 | 0.082 | 0.118 | -0.036 | -0.036 | -0.4065 | +0.0000 | tie | tie |
| 12 | 153 | p1_interp | historical | 0.437 | 0.437 | 0.424 | +0.013 | +0.013 | -0.6357 | +0.0000 | tie | tie |

## Cost

Optimizer wall 1090.7 s of 4200. Starts 25344, of which 432 are the extra L=6 KM draws. L=6 KM_box states whose label is one of those draws: 0. Logged energy evaluations 1994365, L=1 grid evaluations 18432, capped starts 2.
