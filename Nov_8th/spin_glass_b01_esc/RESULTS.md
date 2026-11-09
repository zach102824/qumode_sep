# Historical B=0.1 repulsion escape

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The packing rounds, the energy-only union, the quarter-pi restarts, and the classical popcount initialization are not this result.

KM keeps the round-4 starts on the interleaved triples, then repels from the energy-best state. QAOA uses the round-4 start list and does not receive the escape. Equal search is not claimed.

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
| 6 | 288 | 2.4e-13 | 2.3e-13 | 2.5e-14 | 0 | 1.2e-13 | True |
| 9 | 288 | 7.8e-13 | 3.8e-13 | 8.2e-14 | 0 | 3.9e-13 | True |
| 12 | 288 | 2.8e-13 | 2.0e-14 | 2.1e-13 | 0 | 1.5e-13 | True |

## Equal depth, primary QAOA repeats 0 and 1

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM | 6 | 6 | 12 | 11 (11 / 0) | 1 | 0 | +0.134 [+0.014, +0.292] | +0.120 | 0.00635 |
| KM | 6 | 3 | 12 | 10 (9 / 1) | 1 | 1 | +0.219 [+0.102, +0.412] | +0.224 | 0.0386 |
| KM | 9 | 6 | 12 | 7 (7 / 0) | 1 | 4 | +0.214 [-0.051, +0.363] | +0.160 | 0.388 |
| KM | 9 | 3 | 12 | 7 (7 / 0) | 0 | 5 | +0.099 [-0.008, +0.283] | +0.133 | 0.146 |
| KM | 12 | 6 | 12 | 4 (3 / 1) | 1 | 7 | +0.016 [-0.039, +0.180] | +0.078 | 0.774 |
| KM | 12 | 3 | 12 | 3 (3 / 0) | 0 | 9 | +0.013 [-0.001, +0.104] | +0.064 | 0.146 |

## Secondary equal parameter count, 12 parameters, not the verdict

| N | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|
| 6 | 5 (1) | 6 | -0.016 [-0.041, +0.051] |
| 9 | 1 (0) | 5 | -0.058 [-0.138, -0.028] |
| 12 | 1 (0) | 3 | -0.041 [-0.118, -0.027] |

## Verdict

Overall: **partial**.
- N=6: pass on 12 complete instances.
- N=9: does not pass on 12 complete instances.
- N=12: does not pass on 12 complete instances.

## L=6 instances

| N | k | label | F KM | F QAOA | dF | E KM - E QAOA | outcome |
|---|---|---|---|---|---|---|---|
| 6 | 106 | polish_swap | 0.993 | 0.979 | +0.014 | -0.0823 | win_joint |
| 6 | 107 | polish_swap | 0.995 | 0.980 | +0.015 | -0.0693 | win_joint |
| 6 | 108 | p0_interp | 0.958 | 0.569 | +0.389 | -0.7032 | win_joint |
| 6 | 109 | nested_K | 0.996 | 0.990 | +0.006 | -0.0211 | win_joint |
| 6 | 110 | p2_interp | 0.984 | 0.903 | +0.081 | -0.2898 | win_joint |
| 6 | 111 | p2_interp | 0.999 | 0.972 | +0.026 | -0.0994 | win_joint |
| 6 | 112 | nested_K | 0.960 | 0.617 | +0.343 | -0.0115 | win_joint |
| 6 | 113 | p1_perturb_0.1 | 0.948 | 0.760 | +0.187 | -0.2947 | win_joint |
| 6 | 114 | p2_interp | 0.971 | 0.730 | +0.242 | -0.5665 | win_joint |
| 6 | 115 | nested_K | 0.556 | 0.109 | +0.447 | -0.1279 | win_joint |
| 6 | 116 | p1_interp | 0.805 | 0.565 | +0.239 | -0.0945 | win_joint |
| 6 | 117 | nested_K | 0.013 | 0.558 | -0.545 | -0.1900 | loss_fidelity_only |
| 9 | 106 | p1_interp | 0.363 | 0.326 | +0.037 | -0.3648 | tie |
| 9 | 107 | fresh_ramp_jitter | 0.475 | 0.253 | +0.222 | -0.0410 | win_joint |
| 9 | 108 | p0_interp | 0.028 | 0.100 | -0.072 | +0.4009 | tie |
| 9 | 109 | fresh_ramp | 0.956 | 0.750 | +0.206 | -0.9151 | win_joint |
| 9 | 110 | nested_K | 0.936 | 0.519 | +0.417 | -0.9789 | win_joint |
| 9 | 111 | p2_time_rescaled | 0.284 | 0.329 | -0.045 | +0.1364 | tie |
| 9 | 112 | p1_perturb_0.1 | 0.441 | 0.208 | +0.233 | -0.4495 | win_joint |
| 9 | 113 | nested_K | 0.879 | 0.518 | +0.360 | -0.9193 | win_joint |
| 9 | 114 | p1_time_rescaled | 0.644 | 0.273 | +0.371 | -0.5500 | win_joint |
| 9 | 115 | polish_signflip | 0.953 | 0.587 | +0.366 | -1.0448 | win_joint |
| 9 | 116 | p2_interp | 0.288 | 0.345 | -0.057 | +0.5374 | tie |
| 9 | 117 | polish_signflip | 0.155 | 0.277 | -0.122 | +0.0216 | loss_joint |
| 12 | 106 | p0_time_rescaled | 0.185 | 0.153 | +0.033 | +0.1883 | tie |
| 12 | 107 | fresh_ramp | 0.025 | 0.013 | +0.013 | +0.2714 | tie |
| 12 | 108 | polish_signflip | 0.042 | 0.121 | -0.078 | +0.4701 | tie |
| 12 | 109 | p2_interp | 0.099 | 0.080 | +0.019 | -0.0658 | tie |
| 12 | 110 | p2_time_rescaled | 0.028 | 0.040 | -0.013 | -0.3077 | tie |
| 12 | 111 | p1_perturb_0.1 | 0.066 | 0.089 | -0.023 | +0.8886 | tie |
| 12 | 112 | nested_K | 0.001 | 0.055 | -0.054 | +0.1821 | tie |
| 12 | 113 | p1_time_rescaled | 0.252 | 0.043 | +0.209 | +0.0456 | win_fidelity_only |
| 12 | 114 | p0_time_rescaled | 0.841 | 0.690 | +0.151 | -1.0901 | win_joint |
| 12 | 115 | fresh_ramp | 0.649 | 0.416 | +0.233 | -0.1202 | win_joint |
| 12 | 116 | fresh_ramp_jitter | 0.774 | 0.237 | +0.537 | -1.3606 | win_joint |
| 12 | 117 | p1_time_rescaled | 0.311 | 0.404 | -0.093 | -0.1422 | loss_fidelity_only |

## Cost

Optimizer wall 939.0 s of 4200. Starts 27504, of which 2592 are repulsion or polish. L=6 energy-selected KM states whose label is an escape: 5. Logged energy evaluations 2142548, L=1 grid evaluations 18432, capped starts 43.
