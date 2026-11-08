# Historical B=0.1 KM diverse L=1 lineages

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, the QAOA-depth diagnosis, the energy-legal selection diagnostic, the uniform box initialization, and the scale-normalized restarts are not this result.

KM is the interleaved shared-gamma circuit with the round-4 starts and two repeats. KM_div keeps that L=6 vector and adds lineages grown from optimized L=1 states the energy beam dropped, then selects the lowest true energy. The continuation column is equal search. The diverse column is not.

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
| 6 | 300 | 9.3e-14 | 6.1e-14 | 2.3e-14 | 0 | 4.0e-14 | True |
| 9 | 300 | 5.1e-13 | 1.3e-13 | 7.1e-14 | 0 | 2.4e-13 | True |
| 12 | 300 | 2.7e-13 | 3.2e-14 | 8.9e-14 | 0 | 1.3e-13 | True |

## Equal depth, equal search, continuation KM versus QAOA repeats 0 and 1

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 9 (9 / 0) | 0 | 3 | +0.206 [+0.028, +0.368] | +0.221 | 0.000488 |
| KM | 6 | 3 | 12 | 10 (10 / 0) | 0 | 2 | +0.208 [+0.105, +0.423] | +0.260 | 0.00635 |
| KM | 9 | 6 | 12 | 8 (7 / 1) | 2 | 2 | +0.137 [-0.034, +0.173] | +0.074 | 0.388 |
| KM | 9 | 3 | 12 | 7 (7 / 0) | 1 | 4 | +0.149 [-0.013, +0.256] | +0.124 | 0.388 |
| KM | 12 | 6 | 12 | 4 (3 / 1) | 1 | 7 | +0.015 [-0.067, +0.126] | +0.033 | 1 |
| KM | 12 | 3 | 12 | 2 (2 / 0) | 0 | 10 | +0.027 [+0.014, +0.040] | +0.037 | 0.00635 |

## Equal depth, diverse-inclusive KM at L=6, unequal search

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM_div | 6 | 6 | 12 | 9 (9 / 0) | 0 | 3 | +0.206 [+0.028, +0.368] | +0.221 | 0.000488 |
| KM_div | 9 | 6 | 12 | 8 (7 / 1) | 2 | 2 | +0.137 [-0.034, +0.173] | +0.074 | 0.388 |
| KM_div | 12 | 6 | 12 | 4 (3 / 1) | 1 | 7 | +0.015 [-0.067, +0.126] | +0.033 | 1 |

## Energy drop

An N is deeper when the median energy drop exceeds 0.01 or at least 4 instances drop by more than 0.01. The drop is continuation KM energy minus diverse-inclusive energy.

| N | n | median drop | n drop > 0.01 | deeper | continuation pass | diverse pass |
|---|---|---|---|---|---|---|
| 6 | 12 | 0.0000 | 0 | False | True | True |
| 9 | 12 | 0.0000 | 0 | False | False | False |
| 12 | 12 | 0.0000 | 0 | False | False | False |

N=6: Continuation KM clears the historical comparator on this sample. The diverse column says whether a lower energy keeps that clearance.
N=9: The diverse L=1 lineages did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten on this sample.
N=12: The diverse L=1 lineages did not lower the KM energy under the frozen 0.01 rule. The continuation minimum was not beaten on this sample.

## Secondary equal parameter count, 12 parameters, not the verdict

| N | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|
| 6 | 4 (2) | 6 | -0.018 [-0.120, +0.064] |
| 9 | 0 (0) | 9 | -0.154 [-0.173, -0.077] |
| 12 | 0 (0) | 7 | -0.103 [-0.138, -0.062] |

## Verdict

Continuation column: **partial**.
Diverse-inclusive column: **partial**.
The verdict of this round is the diverse-inclusive column. Equal depth is the comparison. Equal search is the continuation column only.
- N=6: continuation pass, diverse pass, deeper False, on 12 complete instances.
- N=9: continuation does not pass, diverse does not pass, deeper False, on 12 complete instances.
- N=12: continuation does not pass, diverse does not pass, deeper False, on 12 complete instances.

## L=6 instances

| N | k | continuation label | diverse label | F continuation | F diverse | F QAOA | dF continuation | dF diverse | E continuation-QAOA | energy drop | continuation | diverse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 166 | p0_interp | historical | 0.999 | 0.999 | 0.996 | +0.003 | +0.003 | -0.0213 | +0.0000 | tie | tie |
| 6 | 167 | p2_time_rescaled | historical | 0.982 | 0.982 | 0.831 | +0.151 | +0.151 | -0.2875 | +0.0000 | win_joint | win_joint |
| 6 | 168 | nested_K | historical | 0.951 | 0.951 | 0.527 | +0.424 | +0.424 | -0.4152 | +0.0000 | win_joint | win_joint |
| 6 | 169 | fresh_ramp | historical | 0.324 | 0.324 | 0.298 | +0.026 | +0.026 | -0.0892 | +0.0000 | tie | tie |
| 6 | 170 | fresh_ramp | historical | 0.999 | 0.999 | 0.445 | +0.555 | +0.555 | -0.4149 | +0.0000 | win_joint | win_joint |
| 6 | 171 | p1_interp | historical | 0.994 | 0.994 | 0.972 | +0.022 | +0.022 | -0.1691 | +0.0000 | win_joint | win_joint |
| 6 | 172 | p0_perturb_0.1 | historical | 0.958 | 0.958 | 0.801 | +0.157 | +0.157 | -0.3980 | +0.0000 | win_joint | win_joint |
| 6 | 173 | p0_time_rescaled | historical | 0.972 | 0.972 | 0.636 | +0.336 | +0.336 | -0.6973 | +0.0000 | win_joint | win_joint |
| 6 | 174 | p0_interp | historical | 0.764 | 0.764 | 0.364 | +0.399 | +0.399 | -0.3549 | +0.0000 | win_joint | win_joint |
| 6 | 175 | p0_perturb_0.1 | historical | 0.319 | 0.319 | 0.289 | +0.030 | +0.030 | -0.0003 | +0.0000 | tie | tie |
| 6 | 176 | p2_time_rescaled | historical | 0.884 | 0.884 | 0.593 | +0.291 | +0.291 | -0.3835 | +0.0000 | win_joint | win_joint |
| 6 | 177 | p1_perturb_0.3 | historical | 0.984 | 0.984 | 0.729 | +0.254 | +0.254 | -0.6889 | +0.0000 | win_joint | win_joint |
| 9 | 166 | p2_perturb_0.1 | historical | 0.462 | 0.462 | 0.295 | +0.167 | +0.167 | +0.2574 | +0.0000 | win_fidelity_only | win_fidelity_only |
| 9 | 167 | p0_perturb_0.1 | historical | 0.163 | 0.163 | 0.178 | -0.016 | -0.016 | +0.3572 | +0.0000 | tie | tie |
| 9 | 168 | nested_K | historical | 0.776 | 0.776 | 0.636 | +0.140 | +0.140 | -0.7030 | +0.0000 | win_joint | win_joint |
| 9 | 169 | fresh_ramp_jitter | historical | 0.830 | 0.830 | 0.779 | +0.051 | +0.051 | -0.2039 | +0.0000 | win_joint | win_joint |
| 9 | 170 | p0_time_rescaled | historical | 0.892 | 0.892 | 0.758 | +0.134 | +0.134 | -0.4681 | +0.0000 | win_joint | win_joint |
| 9 | 171 | fresh_ramp | historical | 0.084 | 0.084 | 0.254 | -0.170 | -0.170 | -0.0090 | +0.0000 | loss_fidelity_only | loss_fidelity_only |
| 9 | 172 | p2_interp | historical | 0.842 | 0.842 | 0.684 | +0.157 | +0.157 | -0.9333 | +0.0000 | win_joint | win_joint |
| 9 | 173 | fresh_ramp | historical | 0.684 | 0.684 | 0.505 | +0.179 | +0.179 | -0.4340 | +0.0000 | win_joint | win_joint |
| 9 | 174 | p0_perturb_0.1 | historical | 0.936 | 0.936 | 0.743 | +0.193 | +0.193 | -1.4653 | +0.0000 | win_joint | win_joint |
| 9 | 175 | p1_perturb_0.1 | historical | 0.066 | 0.066 | 0.117 | -0.052 | -0.052 | -0.1368 | +0.0000 | tie | tie |
| 9 | 176 | fresh_ramp_jitter | historical | 0.198 | 0.198 | 0.286 | -0.088 | -0.088 | +0.1927 | +0.0000 | loss_joint | loss_joint |
| 9 | 177 | p2_time_rescaled | historical | 0.919 | 0.919 | 0.734 | +0.186 | +0.186 | -0.7312 | +0.0000 | win_joint | win_joint |
| 12 | 166 | fresh_ramp_jitter | historical | 0.564 | 0.564 | 0.336 | +0.228 | +0.228 | -0.4663 | +0.0000 | win_joint | win_joint |
| 12 | 167 | p2_interp | historical | 0.073 | 0.073 | 0.036 | +0.037 | +0.037 | +0.3715 | +0.0000 | tie | tie |
| 12 | 168 | p1_interp | historical | 0.737 | 0.737 | 0.595 | +0.142 | +0.142 | -1.1349 | +0.0000 | win_joint | win_joint |
| 12 | 169 | p1_interp | historical | 0.256 | 0.256 | 0.225 | +0.030 | +0.030 | -0.3490 | +0.0000 | tie | tie |
| 12 | 170 | p0_time_rescaled | historical | 0.141 | 0.141 | 0.216 | -0.075 | -0.075 | +0.5393 | +0.0000 | tie | tie |
| 12 | 171 | p1_time_rescaled | historical | 0.426 | 0.426 | 0.316 | +0.111 | +0.111 | +0.3287 | +0.0000 | win_fidelity_only | win_fidelity_only |
| 12 | 172 | p0_time_rescaled | historical | 0.154 | 0.154 | 0.232 | -0.078 | -0.078 | -0.2177 | +0.0000 | loss_fidelity_only | loss_fidelity_only |
| 12 | 173 | fresh_ramp_jitter | historical | 0.120 | 0.120 | 0.178 | -0.059 | -0.059 | -0.4856 | +0.0000 | tie | tie |
| 12 | 174 | p0_time_rescaled | historical | 0.405 | 0.405 | 0.247 | +0.157 | +0.157 | -0.4302 | +0.0000 | win_joint | win_joint |
| 12 | 175 | p0_time_rescaled | historical | 0.010 | 0.010 | 0.010 | -0.000 | -0.000 | -0.5260 | +0.0000 | tie_both_failed | tie_both_failed |
| 12 | 176 | fresh_ramp | historical | 0.147 | 0.147 | 0.163 | -0.016 | -0.016 | +0.3797 | +0.0000 | tie | tie |
| 12 | 177 | p2_interp | historical | 0.070 | 0.070 | 0.149 | -0.079 | -0.079 | -0.6585 | +0.0000 | tie | tie |

## Lineage gaps, descriptive

A start recovers the continuation energy when it is at most 1e-8 above it. The nearest other gap is descriptive. It is not a second threshold.

| N | L=6 lineage starts | strictly lower | recovered | nearest gap above 1e-8 |
|---|---|---|---|---|
| 6 | 284 | 0 | 0 | 0.0404 |
| 9 | 284 | 0 | 4 | 0.1614 |
| 12 | 280 | 0 | 11 | 1.1224 |

## Cost

Optimizer wall 1308.3 s of 4200. Starts 29152, of which 4240 are diverse-lineage starts. Lineages opened 212. L=6 KM_div states whose label is a lineage child: 0. Logged energy evaluations 2466200, L=1 grid evaluations 18432, capped starts 12.
