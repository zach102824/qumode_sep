# Historical B=0.1 energy-legal selection diagnostic

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, the repulsion escape, and the QAOA-depth diagnosis are not this result.

The claim is energy-only interleaved shared-gamma KM against equal-depth QAOA, both on the round-4 starts and repeats 0 and 1. The diagnostic state is the highest subspace overlap among L=6 KM starts whose energy is at most the QAOA energy plus 1e-06. That column is not the verdict. The unrestricted maximum overlap has no pass rule.

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
| 6 | 312 | 2.0e-13 | 7.4e-14 | 1.6e-14 | 0 | 1.1e-13 | True |
| 9 | 312 | 2.5e-13 | 6.4e-14 | 4.4e-14 | 0 | 1.2e-13 | True |
| 12 | 312 | 5.0e-13 | 2.7e-13 | 1.7e-13 | 0 | 2.4e-13 | True |

## Equal depth, energy-only, the claim

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 11 (11 / 0) | 0 | 1 | +0.202 [+0.043, +0.385] | +0.271 | 0.00635 |
| KM | 6 | 3 | 12 | 9 (9 / 0) | 1 | 2 | +0.170 [+0.032, +0.251] | +0.170 | 0.146 |
| KM | 9 | 6 | 12 | 3 (3 / 0) | 4 | 5 | -0.048 [-0.072, +0.021] | -0.001 | 0.388 |
| KM | 9 | 3 | 12 | 3 (3 / 0) | 1 | 8 | +0.004 [-0.028, +0.095] | +0.040 | 0.774 |
| KM | 12 | 6 | 12 | 4 (4 / 0) | 0 | 8 | -0.007 [-0.018, +0.109] | +0.046 | 0.774 |
| KM | 12 | 3 | 12 | 3 (3 / 0) | 0 | 9 | +0.017 [+0.007, +0.088] | +0.048 | 0.00635 |

## Diagnostic column at L=6, same starts, not the verdict

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p | n differ |
|---|---|---|---|---|---|---|---|---|---|---|
| KM_fsel | 6 | 6 | 12 | 11 (11 / 0) | 0 | 1 | +0.202 [+0.043, +0.385] | +0.273 | 0.00635 | 2 |
| KM_fsel | 9 | 6 | 12 | 3 (3 / 0) | 4 | 5 | -0.048 [-0.072, +0.021] | +0.009 | 0.388 | 1 |
| KM_fsel | 12 | 6 | 12 | 4 (4 / 0) | 0 | 8 | -0.007 [-0.018, +0.109] | +0.046 | 0.774 | 3 |

## Diagnosis

A size is selection-limited when the energy-only column does not pass and the diagnostic column does. The diagnostic state differs when its subspace overlap exceeds the energy-only overlap by more than 1e-6 or its energy differs by more than 1e-8.

| N | n | claim pass | diagnostic pass | selection_limited | n differ | median F gain | median pool |
|---|---|---|---|---|---|---|---|
| 6 | 12 | True | True | False | 2 | +0.0000 | 20.0 |
| 9 | 12 | False | False | False | 1 | +0.0000 | 20.0 |
| 12 | 12 | False | False | False | 3 | +0.0000 | 20.5 |

N=6: Energy-only KM clears the historical comparator on this sample. The diagnostic state also clears it.
N=9: A different energy-legal KM state was available. It still does not clear the comparator.
N=12: A different energy-legal KM state was available. It still does not clear the comparator.

## Unrestricted maximum overlap, not a comparator

| N | n differ | median F gain | median energy gap |
|---|---|---|---|
| 6 | 3 | +0.0000 | +0.0000 |
| 9 | 5 | +0.0000 | +0.0000 |
| 12 | 6 | +0.0000 | +0.0000 |

## Secondary equal parameter count, 12 parameters, not the verdict

| N | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|
| 6 | 2 (1) | 7 | -0.054 [-0.103, +0.005] |
| 9 | 0 (0) | 5 | -0.065 [-0.169, -0.032] |
| 12 | 0 (0) | 8 | -0.093 [-0.139, -0.061] |

## Verdict

Energy-only claim: **partial**.
Energy-legal diagnostic, not the verdict: **partial**.
- N=6: energy-only pass, diagnostic pass, selection_limited False, on 12 complete instances.
- N=9: energy-only does not pass, diagnostic does not pass, selection_limited False, on 12 complete instances.
- N=12: energy-only does not pass, diagnostic does not pass, selection_limited False, on 12 complete instances.

## L=6 instances

| N | k | KM label | fsel label | F KM | F fsel | F QAOA | dF claim | dF fsel | E KM-QAOA | pool | differ | claim | fsel | F max | E gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 130 | p2_perturb_0.1 | p0_interp | 0.970 | 0.970 | 0.779 | +0.191 | +0.191 | -0.3237 | 20 | False | win_joint | win_joint | 0.970 | +0.0000 |
| 6 | 131 | p2_interp | p2_interp | 0.994 | 0.994 | 0.949 | +0.045 | +0.045 | -0.1106 | 20 | False | win_joint | win_joint | 0.994 | +0.0000 |
| 6 | 132 | p0_interp | p0_interp | 0.968 | 0.968 | 0.216 | +0.752 | +0.752 | -0.1843 | 20 | False | win_joint | win_joint | 0.968 | +0.0000 |
| 6 | 133 | nested_K | fresh_ramp_jitter | 0.978 | 0.978 | 0.937 | +0.041 | +0.041 | -0.1484 | 24 | True | win_joint | win_joint | 0.978 | +0.0064 |
| 6 | 134 | fresh_ramp_jitter | fresh_ramp_jitter | 0.447 | 0.447 | 0.473 | -0.026 | -0.026 | +0.3433 | 0 | False | tie | tie | 0.661 | +0.8583 |
| 6 | 135 | p1_perturb_0.1 | p1_perturb_0.1 | 1.000 | 1.000 | 0.843 | +0.156 | +0.156 | -0.4375 | 25 | False | win_joint | win_joint | 1.000 | +0.0000 |
| 6 | 136 | p1_interp | p1_interp | 0.860 | 0.860 | 0.646 | +0.214 | +0.214 | -0.1833 | 17 | False | win_joint | win_joint | 0.860 | +0.0000 |
| 6 | 137 | fresh_ramp | fresh_ramp_jitter | 0.868 | 0.888 | 0.022 | +0.846 | +0.865 | -0.1382 | 24 | True | win_joint | win_joint | 0.888 | +0.0099 |
| 6 | 138 | fresh_ramp | fresh_ramp_jitter | 0.957 | 0.957 | 0.702 | +0.255 | +0.255 | -0.2122 | 18 | False | win_joint | win_joint | 0.957 | +0.0000 |
| 6 | 139 | p0_perturb_0.1 | p0_perturb_0.1 | 0.718 | 0.718 | 0.474 | +0.244 | +0.244 | -0.1039 | 10 | False | win_joint | win_joint | 0.718 | +0.0000 |
| 6 | 140 | p2_time_rescaled | p1_interp | 0.976 | 0.976 | 0.957 | +0.019 | +0.019 | -0.0729 | 15 | False | win_joint | win_joint | 0.976 | +0.0000 |
| 6 | 141 | p0_time_rescaled | nested_K | 0.939 | 0.939 | 0.423 | +0.516 | +0.516 | -0.4170 | 22 | False | win_joint | win_joint | 0.939 | +0.0000 |
| 9 | 130 | p2_interp | fresh_ramp_jitter | 0.932 | 0.932 | 0.896 | +0.036 | +0.036 | -0.3507 | 21 | False | win_joint | win_joint | 0.932 | +0.0000 |
| 9 | 131 | fresh_ramp_jitter | p2_interp | 0.002 | 0.002 | 0.026 | -0.024 | -0.024 | -0.1658 | 18 | False | tie | tie | 0.044 | +12.0854 |
| 9 | 132 | p0_perturb_0.1 | p1_interp | 0.008 | 0.008 | 0.178 | -0.170 | -0.170 | -0.7143 | 23 | False | loss_fidelity_only | loss_fidelity_only | 0.040 | +12.6181 |
| 9 | 133 | p0_perturb_0.1 | p0_perturb_0.1 | 0.239 | 0.239 | 0.320 | -0.081 | -0.081 | +0.5522 | 0 | False | loss_joint | loss_joint | 0.239 | +0.0000 |
| 9 | 134 | p1_interp | p1_interp | 0.163 | 0.163 | 0.241 | -0.077 | -0.077 | -0.0589 | 23 | False | loss_fidelity_only | loss_fidelity_only | 0.163 | +0.0000 |
| 9 | 135 | p0_perturb_0.1 | p0_perturb_0.1 | 0.036 | 0.036 | 0.095 | -0.059 | -0.059 | +0.2650 | 0 | False | tie | tie | 0.051 | +10.8650 |
| 9 | 136 | p1_time_rescaled | fresh_ramp | 0.423 | 0.543 | 0.216 | +0.207 | +0.327 | -0.3483 | 23 | True | win_joint | win_joint | 0.543 | +0.0085 |
| 9 | 137 | p1_time_rescaled | p0_interp | 0.062 | 0.062 | 0.103 | -0.041 | -0.041 | -0.1974 | 22 | False | tie | tie | 0.062 | +0.0000 |
| 9 | 138 | p2_perturb_0.1 | p1_perturb_0.1 | 0.817 | 0.817 | 0.508 | +0.309 | +0.309 | -0.6971 | 19 | False | win_joint | win_joint | 0.817 | +0.0000 |
| 9 | 139 | p0_interp | p0_perturb_0.1 | 0.130 | 0.130 | 0.125 | +0.005 | +0.005 | -0.6323 | 22 | False | tie | tie | 0.130 | +0.0000 |
| 9 | 140 | p2_perturb_0.1 | nested_K | 0.031 | 0.031 | 0.085 | -0.054 | -0.054 | -0.1183 | 19 | False | tie | tie | 0.037 | +8.8518 |
| 9 | 141 | p2_time_rescaled | p2_time_rescaled | 0.340 | 0.340 | 0.407 | -0.067 | -0.067 | +0.7262 | 0 | False | loss_joint | loss_joint | 0.340 | +0.0000 |
| 12 | 130 | p2_interp | p0_time_rescaled | 0.579 | 0.579 | 0.330 | +0.249 | +0.249 | -1.5569 | 20 | True | win_joint | win_joint | 0.579 | +0.0000 |
| 12 | 131 | p2_time_rescaled | p0_interp | 0.469 | 0.469 | 0.377 | +0.092 | +0.092 | -0.2766 | 23 | False | win_joint | win_joint | 0.469 | +0.0000 |
| 12 | 132 | p1_interp | fresh_ramp_jitter | 0.075 | 0.075 | 0.083 | -0.008 | -0.008 | -0.3198 | 23 | False | tie | tie | 0.117 | +1.1613 |
| 12 | 133 | p2_time_rescaled | p2_time_rescaled | 0.128 | 0.128 | 0.183 | -0.055 | -0.055 | +0.9012 | 0 | False | tie | tie | 0.128 | +0.0000 |
| 12 | 134 | fresh_ramp | p1_time_rescaled | 0.401 | 0.401 | 0.276 | +0.125 | +0.125 | -0.1865 | 21 | False | win_joint | win_joint | 0.401 | +0.0000 |
| 12 | 135 | p0_perturb_0.1 | p1_perturb_0.1 | 0.598 | 0.598 | 0.358 | +0.240 | +0.240 | -1.6284 | 22 | True | win_joint | win_joint | 0.598 | +0.0000 |
| 12 | 136 | nested_K | nested_K | 0.117 | 0.117 | 0.124 | -0.007 | -0.007 | +0.4478 | 0 | False | tie | tie | 0.117 | +0.0000 |
| 12 | 137 | nested_K | p2_time_rescaled | 0.571 | 0.571 | 0.585 | -0.014 | -0.014 | -0.5252 | 21 | False | tie | tie | 0.571 | +0.0000 |
| 12 | 138 | fresh_ramp | fresh_ramp | 0.040 | 0.040 | 0.063 | -0.023 | -0.023 | +0.1985 | 0 | False | tie | tie | 0.047 | +0.0023 |
| 12 | 139 | nested_K | fresh_ramp_jitter | 0.229 | 0.229 | 0.228 | +0.001 | +0.001 | -0.2584 | 22 | True | tie | tie | 0.229 | +0.0000 |
| 12 | 140 | nested_K | nested_K | 0.097 | 0.097 | 0.109 | -0.012 | -0.012 | +0.3342 | 0 | False | tie | tie | 0.097 | +0.0000 |
| 12 | 141 | fresh_ramp | fresh_ramp | 0.187 | 0.187 | 0.224 | -0.037 | -0.037 | +0.5811 | 0 | False | tie | tie | 0.187 | +0.0000 |

## Cost

Optimizer wall 1046.0 s of 4200. Starts 24912. Logged energy evaluations 1985526, L=1 grid evaluations 18432, capped starts 34.
