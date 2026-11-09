# Historical B=0.1 quarter-pi shell restarts

Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. The earlier packing rounds and the energy-only union are not this result.

QAOA uses the round-4 start list. KM_pack and KM_fid add four quarter-pi shell restarts at each depth. Equal search is not claimed.

Primary repeats incomplete: []. Never started: [].

| N | primary complete |
|---|---|
| 6 | 12/12 |
| 9 | 12/12 |
| 12 | 12/12 |

Each component has 3+3L parameters (21 at L=6). The union keeps one of those vectors. QAOA has 2L (12 at L=6).

## Independent replay

| N | states | max rel dE | max abs dF | max abs dE0 | n_sub mismatches | max norm err | pass |
|---|---|---|---|---|---|---|---|
| 6 | 432 | 8.9e-13 | 5.3e-13 | 3.6e-14 | 0 | 4.4e-13 | True |
| 9 | 432 | 3.1e-13 | 9.2e-14 | 8.9e-14 | 0 | 1.5e-13 | True |
| 12 | 432 | 8.7e-13 | 2.3e-13 | 1.1e-13 | 0 | 4.4e-13 | True |

## Equal depth, primary QAOA repeats 0 and 1

| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|---|---|---|
| KM_union | 6 | 6 | 12 | 11 (11 / 0) | 1 | 0 | +0.122 [+0.047, +0.358] | +0.207 | 0.00635 |
| KM_union | 6 | 3 | 12 | 10 (10 / 0) | 1 | 1 | +0.321 [+0.187, +0.515] | +0.326 | 0.00635 |
| KM_union | 9 | 6 | 12 | 6 (6 / 0) | 3 | 3 | +0.073 [-0.057, +0.191] | +0.112 | 0.774 |
| KM_union | 9 | 3 | 12 | 8 (8 / 0) | 0 | 4 | +0.117 [+0.009, +0.231] | +0.174 | 0.146 |
| KM_union | 12 | 6 | 12 | 7 (6 / 1) | 2 | 3 | +0.087 [-0.025, +0.230] | +0.094 | 0.388 |
| KM_union | 12 | 3 | 12 | 2 (2 / 0) | 0 | 10 | +0.028 [-0.000, +0.070] | +0.051 | 0.146 |
| | | | | | | | | | |
| KM_pack | 6 | 6 | 12 | 10 (10 / 0) | 2 | 0 | +0.099 [+0.026, +0.270] | +0.119 | 0.0386 |
| KM_pack | 6 | 3 | 12 | 8 (8 / 0) | 2 | 2 | +0.195 [-0.006, +0.321] | +0.173 | 0.146 |
| KM_pack | 9 | 6 | 12 | 4 (4 / 0) | 5 | 3 | -0.005 [-0.094, +0.119] | +0.018 | 1 |
| KM_pack | 9 | 3 | 12 | 7 (7 / 0) | 0 | 5 | +0.095 [-0.011, +0.117] | +0.088 | 0.774 |
| KM_pack | 12 | 6 | 12 | 4 (3 / 1) | 2 | 6 | +0.047 [-0.025, +0.166] | +0.074 | 0.388 |
| KM_pack | 12 | 3 | 12 | 2 (2 / 0) | 0 | 10 | +0.028 [-0.000, +0.070] | +0.051 | 0.146 |
| | | | | | | | | | |
| KM_fid | 6 | 6 | 12 | 11 (11 / 0) | 1 | 0 | +0.122 [+0.047, +0.358] | +0.186 | 0.00635 |
| KM_fid | 6 | 3 | 12 | 10 (10 / 0) | 1 | 1 | +0.321 [+0.135, +0.515] | +0.316 | 0.00635 |
| KM_fid | 9 | 6 | 12 | 6 (6 / 0) | 2 | 4 | +0.070 [-0.016, +0.180] | +0.121 | 0.388 |
| KM_fid | 9 | 3 | 12 | 8 (8 / 0) | 0 | 4 | +0.116 [+0.014, +0.210] | +0.174 | 0.146 |
| KM_fid | 12 | 6 | 12 | 5 (5 / 0) | 2 | 5 | +0.060 [-0.029, +0.103] | +0.040 | 0.388 |
| KM_fid | 12 | 3 | 12 | 4 (4 / 0) | 0 | 8 | +0.051 [-0.003, +0.095] | +0.054 | 0.388 |
| | | | | | | | | | |

## Secondary equal parameter count, 12 parameters, not the verdict

| N | wins (joint) | losses | median dF [95% CI] |
|---|---|---|---|
| 6 | 8 (5) | 4 | +0.051 [-0.008, +0.185] |
| 9 | 1 (1) | 7 | -0.075 [-0.146, +0.006] |
| 12 | 0 (0) | 7 | -0.089 [-0.134, -0.052] |

## Verdict

Overall: **partial**.
- N=6: pass on 12 complete instances.
- N=9: does not pass on 12 complete instances.
- N=12: does not pass on 12 complete instances.

## L=6 union instances

| N | k | chosen | label | F pack | F Fiedler | F union | F QAOA | dF union | E union - E QAOA | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 80 | KM_fid | p1_interp | 0.958 | 0.969 | 0.969 | 0.855 | +0.114 | -0.2969 | win_joint |
| 6 | 81 | KM_fid | p1_time_rescaled | 0.694 | 0.694 | 0.694 | 0.445 | +0.249 | -0.0620 | win_joint |
| 6 | 82 | KM_pack | p2_perturb_0.1 | 0.996 | 0.988 | 0.996 | 0.901 | +0.095 | -0.2725 | win_joint |
| 6 | 83 | KM_pack | fresh_ramp_jitter | 1.000 | 1.000 | 1.000 | 0.964 | +0.036 | -0.1532 | win_joint |
| 6 | 84 | KM_fid | qpi_shell_2 | 0.884 | 0.881 | 0.881 | 0.593 | +0.288 | -0.1741 | win_joint |
| 6 | 85 | KM_pack | qpi_shell_0 | 0.995 | 0.995 | 0.995 | 0.866 | +0.129 | -0.3766 | win_joint |
| 6 | 86 | KM_fid | nested_K_fid | 0.996 | 0.997 | 0.997 | 0.981 | +0.017 | -0.0756 | win_joint |
| 6 | 87 | KM_pack | p1_perturb_0.3 | 0.884 | 0.642 | 0.884 | 0.058 | +0.826 | -0.3794 | win_joint |
| 6 | 88 | KM_fid | p2_time_rescaled | 0.001 | 0.998 | 0.998 | 0.411 | +0.587 | -0.1939 | win_joint |
| 6 | 89 | KM_pack | p0_interp | 0.999 | 0.999 | 0.999 | 0.941 | +0.058 | -0.2366 | win_joint |
| 6 | 90 | KM_fid | p2_time_rescaled | 0.046 | 0.099 | 0.099 | 0.445 | -0.346 | -0.0820 | loss_fidelity_only |
| 6 | 91 | KM_fid | p1_interp | 0.999 | 0.999 | 0.999 | 0.572 | +0.427 | -0.1590 | win_joint |
| 9 | 80 | KM_fid | nested_K_fid | 0.164 | 0.990 | 0.990 | 0.259 | +0.731 | -1.5651 | win_joint |
| 9 | 81 | KM_fid | p2_perturb_0.1 | 0.268 | 0.422 | 0.422 | 0.241 | +0.181 | -0.1424 | win_joint |
| 9 | 82 | KM_fid | p2_perturb_0.1 | 0.308 | 0.372 | 0.372 | 0.383 | -0.011 | +0.5089 | tie |
| 9 | 83 | KM_pack | p0_time_rescaled | 0.455 | 0.419 | 0.455 | 0.254 | +0.201 | -0.7639 | win_joint |
| 9 | 84 | KM_pack | p0_interp | 0.115 | 0.156 | 0.115 | 0.241 | -0.126 | +0.2051 | loss_joint |
| 9 | 85 | KM_pack | p1_perturb_0.1 | 0.026 | 0.024 | 0.026 | 0.046 | -0.020 | -0.4514 | tie |
| 9 | 86 | KM_fid | fresh_ramp_jitter | 0.504 | 0.506 | 0.506 | 0.328 | +0.178 | -0.6018 | win_joint |
| 9 | 87 | KM_fid | p1_perturb_0.1 | 0.803 | 0.876 | 0.876 | 0.740 | +0.136 | -0.1951 | win_joint |
| 9 | 88 | KM_pack | fresh_ramp_jitter | 0.511 | 0.605 | 0.511 | 0.604 | -0.093 | -0.1992 | loss_fidelity_only |
| 9 | 89 | KM_fid | p0_interp | 0.905 | 0.915 | 0.915 | 0.656 | +0.260 | -1.2183 | win_joint |
| 9 | 90 | KM_pack | qpi_gamma_shell0 | 0.004 | 0.020 | 0.004 | 0.109 | -0.105 | -0.4776 | loss_fidelity_only |
| 9 | 91 | KM_pack | p1_perturb_0.1 | 0.357 | 0.351 | 0.357 | 0.346 | +0.010 | -0.2770 | tie |
| 12 | 80 | KM_fid | fresh_ramp | 0.422 | 0.571 | 0.571 | 0.371 | +0.200 | -1.4337 | win_joint |
| 12 | 81 | KM_pack | qpi_gamma_shell0 | 0.231 | 0.258 | 0.231 | 0.188 | +0.043 | -0.1833 | tie |
| 12 | 82 | KM_pack | p2_time_rescaled | 0.594 | 0.301 | 0.594 | 0.251 | +0.343 | -1.0371 | win_joint |
| 12 | 83 | KM_pack | p2_time_rescaled | 0.322 | 0.263 | 0.322 | 0.252 | +0.070 | +0.1487 | win_fidelity_only |
| 12 | 84 | KM_fid | p1_time_rescaled | 0.303 | 0.375 | 0.375 | 0.271 | +0.103 | -0.6852 | win_joint |
| 12 | 85 | KM_pack | fresh_ramp_jitter | 0.715 | 0.588 | 0.715 | 0.387 | +0.328 | -1.7810 | win_joint |
| 12 | 86 | KM_pack | p1_interp | 0.172 | 0.172 | 0.172 | 0.337 | -0.165 | -0.7481 | loss_fidelity_only |
| 12 | 87 | KM_fid | p1_time_rescaled | 0.132 | 0.112 | 0.112 | 0.219 | -0.107 | +0.5727 | loss_joint |
| 12 | 88 | KM_pack | qpi_shell_0 | 0.602 | 0.415 | 0.602 | 0.341 | +0.261 | -0.9333 | win_joint |
| 12 | 89 | KM_pack | fresh_ramp_jitter | 0.173 | 0.173 | 0.173 | 0.188 | -0.015 | -0.9017 | tie |
| 12 | 90 | KM_fid | qpi_shell_1 | 0.185 | 0.224 | 0.224 | 0.121 | +0.103 | -0.0624 | win_joint |
| 12 | 91 | KM_pack | p0_perturb_0.1 | 0.053 | 0.046 | 0.053 | 0.089 | -0.035 | +0.1676 | tie |

The L=6 union state itself is a quarter-pi start on 6 of 36 instances. A later depth can also descend from an earlier quarter-pi state.

## Cost

Optimizer wall 2422.3 s of 4200. Starts 46944, of which 3456 are quarter-pi restarts. Logged energy evaluations 4739971, L=1 grid evaluations 32256, capped starts 194.
