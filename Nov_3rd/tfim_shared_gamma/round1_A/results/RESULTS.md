# Round 1 direction A — shared-gamma F_K versus QAOA

Generator `G = Hc + thetaK * sK * F_K` with `F_K = sum_{j<k} K_jk Z_j Z_k`, `K = offdiag(J^2)`, `sK` the upper-triangle RMS match to `J`, and the coefficient of `Hc` fixed at 1. `gamma` and `thetaK` are one scalar each at every depth. Each layer has shell controls `(c1, c2, c3)`.

## Targets

| N | B | E0 | gap | gap in P=R=+1 | S (bits) | P | R | residual | H hash |
|---|---|---|---|---|---|---|---|---|---|
| 12 | -0.3449 | -8.615128 | 4.331e-01 | 4.331015e-01 | 1.013684 | 1.0000 | 1.0000 | 2.06e-14 | `61ae4877f594` |
| 6 | -0.1704 | -3.842972 | 4.984e-04 | 7.051256e-01 | 1.000897 | 1.0000 | 1.0000 | 3.40e-15 | `a5a93f80aabf` |

B is the half-chain entropy maximum on the contract grid, refined to 1e-4. It is this run's numerical realization, not a published Pagano field. The ground state is the full-space vector. `gap` is the full-space gap; `gap in P=R=+1` is the gap to the next state that shares the vacuum's quantum numbers. A tiny full-space gap can be a symmetry partner outside that sector.
N=12 entropy is flat: 3 coarse-grid fields sit within 1e-4 bits of the maximum (B from -0.350 to -0.340). The frozen B is the argmax under the stated tie rule, not evidence of a sharp peak.
N=6 entropy is flat: 7 coarse-grid fields sit within 1e-4 bits of the maximum (B from -0.185 to -0.155). The frozen B is the argmax under the stated tie rule, not evidence of a sharp peak.

## Depth table

Energy is the objective. Fidelity `F` and `eta = (E-Emax)/(E0-Emax)` are diagnostics. Negative `dE` means family A has lower energy than independent-angle QAOA at the same depth.

| N | L | method | params | E | E-E0 | F | eta | theta | gamma | evals | starts | init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1 | family_A | 5 | -3.055731 | 7.8724e-01 | 0.3373 | 0.9357 | -0.5978 | -0.5520 | 1601 | 24 | screen_top |
| 6 | 1 | native_D | 4 | -2.618209 | 1.2248e+00 | 0.2034 | 0.9000 |  | -0.2730 | 829 | 24 | screen_top |
| 6 | 1 | qaoa | 2 | -2.590795 | 1.2522e+00 | 0.2018 | 0.8977 |  | -0.2622 | 1069 | 22 | fresh_0 |
| 6 | 1 | random_K | 5 | -2.625379 | 1.2176e+00 | 0.2073 | 0.9005 | 0.0971 | -0.2724 | 1592 | 25 | warm_native_theta0 |
| 6 | 1 | refl_K | 5 | -2.957092 | 8.8588e-01 | 0.3571 | 0.9276 | 0.8031 | 0.2625 | 1639 | 25 | fresh_6 |
| 6 | 1 | shared_qaoa | 2 | -2.590795 | 1.2522e+00 | 0.2018 | 0.8977 |  | 0.2622 | 734 | 23 | screen_rand_0 |
| 6 | 2 | family_A | 8 | -3.265773 | 5.7720e-01 | 0.6138 | 0.9529 | 0.1487 | 0.9956 | 922 | 18 | warm_native_theta0 |
| 6 | 2 | native_D | 7 | -3.215277 | 6.2770e-01 | 0.5840 | 0.9487 |  | 1.0904 | 731 | 15 | fresh_2 |
| 6 | 2 | qaoa | 4 | -2.953025 | 8.8995e-01 | 0.2952 | 0.9273 |  | -0.2244 | 453 | 15 | pagano_off_0.1 |
| 6 | 2 | random_K | 8 | -3.217495 | 6.2548e-01 | 0.5855 | 0.9489 | -0.0132 | 1.0881 | 968 | 18 | warm_native_theta0 |
| 6 | 2 | refl_K | 8 | -3.237424 | 6.0555e-01 | 0.5786 | 0.9505 | -0.0524 | 1.1055 | 1530 | 18 | warm_native_theta0 |
| 6 | 2 | shared_qaoa | 3 | -2.711726 | 1.1312e+00 | 0.2300 | 0.9076 |  | 0.2224 | 335 | 16 | beta_shift_0.1 |
| 6 | 3 | family_A | 11 | -3.809890 | 3.3083e-02 | 0.9831 | 0.9973 | -0.2258 | 1.2929 | 1271 | 14 | interp_noise_0.15 |
| 6 | 3 | native_D | 10 | -3.756038 | 8.6934e-02 | 0.9568 | 0.9929 |  | 1.0675 | 846 | 13 | interp_noise_0.05 |
| 6 | 3 | qaoa | 6 | -3.161918 | 6.8105e-01 | 0.3809 | 0.9444 |  | -0.2029 | 410 | 10 | interp_linear |
| 6 | 3 | random_K | 11 | -3.765166 | 7.7807e-02 | 0.9586 | 0.9936 | 0.0297 | 1.0731 | 1139 | 14 | interp_linear |
| 6 | 3 | refl_K | 11 | -3.792449 | 5.0523e-02 | 0.9782 | 0.9959 | 0.0985 | 1.0622 | 1222 | 14 | warm_native_theta0 |
| 6 | 3 | shared_qaoa | 4 | -2.944447 | 8.9853e-01 | 0.3025 | 0.9266 |  | 0.2518 | 345 | 12 | fresh_7 |
| 6 | 4 | family_A | 14 | -3.795358 | 4.7614e-02 | 0.9827 | 0.9961 | -0.0537 | 1.1083 | 1232 | 13 | warm_native_theta0 |
| 6 | 4 | native_D | 13 | -3.788041 | 5.4932e-02 | 0.9811 | 0.9955 |  | 1.0588 | 960 | 12 | interp_noise_0.05 |
| 6 | 4 | qaoa | 8 | -3.314442 | 5.2853e-01 | 0.4829 | 0.9568 |  | -0.2254 | 433 | 10 | interp_cubic |
| 6 | 4 | random_K | 14 | -3.789144 | 5.3829e-02 | 0.9812 | 0.9956 | 0.0120 | 1.0603 | 1233 | 14 | warm_native_theta0 |
| 6 | 4 | refl_K | 14 | -3.797963 | 4.5010e-02 | 0.9806 | 0.9963 | 0.1018 | 1.0833 | 1311 | 13 | interp_noise_0.05 |
| 6 | 4 | shared_qaoa | 5 | -3.055509 | 7.8746e-01 | 0.3500 | 0.9357 |  | 0.2373 | 413 | 12 | interp_gamma_-0.05 |
| 6 | 5 | family_A | 17 | -3.831709 | 1.1263e-02 | 0.9922 | 0.9991 | -0.1047 | 1.1827 | 1338 | 14 | interp_cubic |
| 6 | 5 | native_D | 16 | -3.830656 | 1.2316e-02 | 0.9936 | 0.9990 |  | 1.0921 | 1205 | 13 | interp_noise_0.15 |
| 6 | 5 | qaoa | 10 | -3.425517 | 4.1746e-01 | 0.5598 | 0.9659 |  | -0.2207 | 674 | 10 | interp_cubic |
| 6 | 5 | random_K | 17 | -3.831354 | 1.1618e-02 | 0.9945 | 0.9991 | -0.0080 | 1.0912 | 1357 | 14 | warm_native_theta0 |
| 6 | 5 | refl_K | 17 | -3.837161 | 5.8114e-03 | 0.9971 | 0.9995 | 0.0604 | 1.1085 | 1266 | 13 | warm_native_theta0 |
| 6 | 5 | shared_qaoa | 6 | -3.157115 | 6.8586e-01 | 0.3975 | 0.9440 |  | 0.2410 | 452 | 12 | fresh_1 |
| 6 | 6 | family_A | 20 | -3.841711 | 1.2611e-03 | 0.9995 | 0.9999 | -0.0783 | 1.1531 | 1328 | 14 | interp_cubic |
| 6 | 6 | native_D | 19 | -3.841181 | 1.7911e-03 | 0.9990 | 0.9999 |  | 1.0704 | 1249 | 13 | interp_raw_cubic |
| 6 | 6 | qaoa | 12 | -3.514643 | 3.2833e-01 | 0.6754 | 0.9732 |  | -0.2524 | 711 | 10 | dup_edge |
| 6 | 6 | random_K | 20 | -3.841545 | 1.4272e-03 | 0.9992 | 0.9999 | -0.0010 | 1.0698 | 1326 | 14 | warm_native_theta0 |
| 6 | 6 | refl_K | 20 | -3.841891 | 1.0816e-03 | 0.9995 | 0.9999 | 0.0343 | 1.0874 | 1329 | 14 | interp_raw_cubic |
| 6 | 6 | shared_qaoa | 7 | -3.222347 | 6.2063e-01 | 0.4212 | 0.9493 |  | -0.2247 | 609 | 12 | qaoa_mean_gamma |
| 6 | 7 | qaoa | 14 | -3.577838 | 2.6513e-01 | 0.7521 | 0.9783 |  | -0.1934 | 785 | 10 | interp_cubic |
| 6 | 8 | qaoa | 16 | -3.636051 | 2.0692e-01 | 0.8171 | 0.9831 |  | -0.2150 | 879 | 10 | interp_cubic |
| 6 | 9 | qaoa | 18 | -3.730715 | 1.1226e-01 | 0.9259 | 0.9908 |  | -0.2816 | 918 | 10 | dup_edge |
| 6 | 10 | qaoa | 20 | -3.779975 | 6.2997e-02 | 0.9477 | 0.9949 |  | -0.2088 | 959 | 10 | interp_cubic |
| 12 | 1 | family_A | 5 | -7.455723 | 1.1594e+00 | 0.1384 | 0.9642 | -0.5274 | -0.3728 | 1599 | 25 | fresh_2 |
| 12 | 1 | native_D | 4 | -6.845986 | 1.7691e+00 | 0.0817 | 0.9454 |  | 0.1988 | 879 | 23 | screen_rand_1 |
| 12 | 1 | qaoa | 2 | -6.823821 | 1.7913e+00 | 0.0814 | 0.9447 |  | 0.1961 | 1145 | 22 | screen_top |
| 12 | 1 | random_K | 5 | -6.878647 | 1.7365e+00 | 0.0850 | 0.9464 | 0.1719 | 0.2007 | 1619 | 24 | screen_top |
| 12 | 1 | refl_K | 5 | -6.887092 | 1.7280e+00 | 0.0768 | 0.9467 | 0.2322 | 0.1835 | 1706 | 25 | screen_top |
| 12 | 1 | shared_qaoa | 2 | -6.823821 | 1.7913e+00 | 0.0814 | 0.9447 |  | 0.1961 | 814 | 23 | qaoa_mean_gamma |
| 12 | 2 | family_A | 8 | -7.561321 | 1.0538e+00 | 0.1635 | 0.9675 | -0.5735 | -0.2816 | 1009 | 18 | dup_c1_shift_0.0 |
| 12 | 2 | native_D | 7 | -7.042993 | 1.5721e+00 | 0.0953 | 0.9515 |  | 0.1547 | 684 | 15 | dup_gamma_0.05 |
| 12 | 2 | qaoa | 4 | -7.436077 | 1.1791e+00 | 0.1340 | 0.9636 |  | 0.1715 | 389 | 15 | pagano_off_0.1 |
| 12 | 2 | random_K | 8 | -7.066493 | 1.5486e+00 | 0.0981 | 0.9522 | 0.1317 | 0.1523 | 1101 | 18 | dup_c1_shift_0.0 |
| 12 | 2 | refl_K | 8 | -7.061795 | 1.5533e+00 | 0.0904 | 0.9521 | 0.1392 | 0.1449 | 1172 | 18 | dup_c1_shift_0.0 |
| 12 | 2 | shared_qaoa | 3 | -7.018391 | 1.5967e+00 | 0.0953 | 0.9507 |  | 0.1660 | 393 | 16 | gamma_shift_0.1 |
| 12 | 3 | family_A | 11 | -7.924530 | 6.9060e-01 | 0.2633 | 0.9787 | -0.4868 | 0.3508 | 1200 | 13 | warm_shared_theta0 |
| 12 | 3 | native_D | 10 | -7.523008 | 1.0921e+00 | 0.1475 | 0.9663 |  | -0.2043 | 913 | 13 | fresh_1 |
| 12 | 3 | qaoa | 6 | -7.689216 | 9.2591e-01 | 0.1713 | 0.9714 |  | 0.1462 | 398 | 10 | dup_edge |
| 12 | 3 | random_K | 11 | -7.536357 | 1.0788e+00 | 0.1501 | 0.9667 | 0.0603 | 0.2040 | 1194 | 13 | interp_linear |
| 12 | 3 | refl_K | 11 | -7.530452 | 1.0847e+00 | 0.1402 | 0.9665 | 0.0511 | 0.2002 | 1221 | 13 | warm_shared_theta0 |
| 12 | 3 | shared_qaoa | 4 | -7.431972 | 1.1832e+00 | 0.1388 | 0.9635 |  | 0.1891 | 327 | 12 | fresh_4 |
| 12 | 4 | family_A | 14 | -8.017000 | 5.9813e-01 | 0.3014 | 0.9815 | -0.5048 | 0.3141 | 1261 | 13 | interp_cubic |
| 12 | 4 | native_D | 13 | -7.635959 | 9.7917e-01 | 0.1659 | 0.9698 |  | 0.1839 | 1126 | 13 | warm_shared_linear |
| 12 | 4 | qaoa | 8 | -7.857014 | 7.5811e-01 | 0.2161 | 0.9766 |  | 0.1385 | 540 | 10 | interp_cubic |
| 12 | 4 | random_K | 14 | -7.649348 | 9.6578e-01 | 0.1686 | 0.9702 | 0.0559 | 0.1834 | 1241 | 13 | warm_native_theta0 |
| 12 | 4 | refl_K | 14 | -7.642407 | 9.7272e-01 | 0.1589 | 0.9700 | 0.0417 | 0.1809 | 1205 | 13 | warm_native_theta0 |
| 12 | 4 | shared_qaoa | 5 | -7.573406 | 1.0417e+00 | 0.1570 | 0.9679 |  | 0.1737 | 341 | 12 | qaoa_mean_gamma |
| 12 | 5 | family_A | 17 | -8.113736 | 5.0139e-01 | 0.3869 | 0.9845 | -0.5701 | 0.3414 | 1250 | 13 | interp_cubic |
| 12 | 5 | native_D | 16 | -7.754597 | 8.6053e-01 | 0.2022 | 0.9735 |  | 0.2088 | 1033 | 12 | interp_cubic |
| 12 | 5 | qaoa | 10 | -7.980547 | 6.3458e-01 | 0.2592 | 0.9804 |  | 0.1329 | 660 | 10 | dup_edge |
| 12 | 5 | random_K | 17 | -7.766198 | 8.4893e-01 | 0.2039 | 0.9738 | 0.0443 | 0.2077 | 1282 | 13 | warm_native_theta0 |
| 12 | 5 | refl_K | 17 | -7.759290 | 8.5584e-01 | 0.1927 | 0.9736 | 0.0290 | 0.2055 | 1272 | 13 | warm_native_theta0 |
| 12 | 5 | shared_qaoa | 6 | -7.676042 | 9.3909e-01 | 0.1723 | 0.9710 |  | 0.1662 | 408 | 12 | interp_gamma_0.05 |
| 12 | 6 | family_A | 20 | -8.179768 | 4.3536e-01 | 0.4432 | 0.9866 | -0.6298 | 0.3218 | 1215 | 13 | interp_cubic |
| 12 | 6 | native_D | 19 | -7.854089 | 7.6104e-01 | 0.2247 | 0.9765 |  | 0.1924 | 1195 | 13 | interp_noise_0.05 |
| 12 | 6 | qaoa | 12 | -8.060380 | 5.5475e-01 | 0.2955 | 0.9829 |  | 0.1205 | 657 | 10 | interp_cubic |
| 12 | 6 | random_K | 20 | -7.863239 | 7.5189e-01 | 0.2259 | 0.9768 | 0.0376 | 0.1911 | 1334 | 14 | warm_native_theta0 |
| 12 | 6 | refl_K | 20 | -7.855584 | 7.5954e-01 | 0.2179 | 0.9766 | 0.0156 | 0.1905 | 1358 | 14 | warm_native_theta0 |
| 12 | 6 | shared_qaoa | 7 | -7.739123 | 8.7601e-01 | 0.1886 | 0.9730 |  | 0.1621 | 499 | 12 | qaoa_mean_gamma |
| 12 | 7 | qaoa | 14 | -8.127633 | 4.8750e-01 | 0.3342 | 0.9850 |  | 0.1207 | 694 | 10 | interp_cubic |
| 12 | 8 | qaoa | 16 | -8.180760 | 4.3437e-01 | 0.3736 | 0.9866 |  | 0.1167 | 888 | 10 | interp_cubic |
| 12 | 9 | qaoa | 18 | -8.225992 | 3.8914e-01 | 0.4129 | 0.9880 |  | 0.1139 | 779 | 10 | dup_edge |
| 12 | 10 | qaoa | 20 | -8.266761 | 3.4837e-01 | 0.4532 | 0.9893 |  | 0.1114 | 806 | 10 | interp_cubic |

## Family A versus controls

| N | L | dE vs QAOA | dF vs QAOA | both | dE vs D | dE vs random K | dE vs refl K | QAOA L+1 E | both vs L+1 | matched L | matched E |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1 | -4.6494e-01 | 0.1355 | True | -4.3752e-01 | -4.3035e-01 | -9.8639e-02 | -2.9530e+00 | True | 3 | -3.1619e+00 |
| 6 | 2 | -3.1275e-01 | 0.3186 | True | -5.0495e-02 | -4.8278e-02 | -2.8349e-02 | -3.1619e+00 | True | 4 | -3.3144e+00 |
| 6 | 3 | -6.4797e-01 | 0.6021 | True | -5.3852e-02 | -4.4724e-02 | -1.7441e-02 | -3.3144e+00 | True | 6 | -3.5146e+00 |
| 6 | 4 | -4.8092e-01 | 0.4999 | True | -7.3173e-03 | -6.2142e-03 | 2.6048e-03 | -3.4255e+00 | True | 7 | -3.5778e+00 |
| 6 | 5 | -4.0619e-01 | 0.4324 | True | -1.0533e-03 | -3.5544e-04 | 5.4517e-03 | -3.5146e+00 | True | 9 | -3.7307e+00 |
| 6 | 6 | -3.2707e-01 | 0.3241 | True | -5.2998e-04 | -1.6610e-04 | 1.7953e-04 | -3.5778e+00 | True | 10 | -3.7800e+00 |
| 12 | 1 | -6.3190e-01 | 0.0570 | True | -6.0974e-01 | -5.7708e-01 | -5.6863e-01 | -7.4361e+00 | True | 3 | -7.6892e+00 |
| 12 | 2 | -1.2524e-01 | 0.0295 | True | -5.1833e-01 | -4.9483e-01 | -4.9953e-01 | -7.6892e+00 | False | 4 | -7.8570e+00 |
| 12 | 3 | -2.3531e-01 | 0.0920 | True | -4.0152e-01 | -3.8817e-01 | -3.9408e-01 | -7.8570e+00 | True | 6 | -8.0604e+00 |
| 12 | 4 | -1.5999e-01 | 0.0853 | True | -3.8104e-01 | -3.6765e-01 | -3.7459e-01 | -7.9805e+00 | True | 7 | -8.1276e+00 |
| 12 | 5 | -1.3319e-01 | 0.1276 | True | -3.5914e-01 | -3.4754e-01 | -3.5445e-01 | -8.0604e+00 | True | 9 | -8.2260e+00 |
| 12 | 6 | -1.1939e-01 | 0.1477 | True | -3.2568e-01 | -3.1653e-01 | -3.2418e-01 | -8.1276e+00 | True | 10 | -8.2668e+00 |

## Transfer N=6 to N=12

Primary disclosed scale is `kappa` (gamma_12 = gamma_6 * kappa_6 / kappa_12). Theta is unchanged because each size RMS-matches its own K to its own J. Shell controls are unchanged. Raw gamma and an RMS rescaling are sensitivity checks, not a selection on the N=12 ground state. This is not the retrained N=12 column.

| source L | method | scale | E | E-E0 | F | eta | primary |
|---|---|---|---|---|---|---|---|
| 1 | qaoa | raw | -6.577026 | 2.0381e+00 | 0.0885 | 0.9371 | False |
| 1 | qaoa | kappa | -6.817807 | 1.7973e+00 | 0.0799 | 0.9446 | True |
| 1 | qaoa | rms | -5.598747 | 3.0164e+00 | 0.0828 | 0.9069 | False |
| 1 | shared_qaoa | raw | -6.577026 | 2.0381e+00 | 0.0885 | 0.9371 | False |
| 1 | shared_qaoa | kappa | -6.817807 | 1.7973e+00 | 0.0799 | 0.9446 | True |
| 1 | shared_qaoa | rms | -5.598747 | 3.0164e+00 | 0.0828 | 0.9069 | False |
| 1 | native_D | raw | -6.544123 | 2.0710e+00 | 0.0869 | 0.9361 | False |
| 1 | native_D | kappa | -6.836972 | 1.7782e+00 | 0.0802 | 0.9451 | True |
| 1 | native_D | rms | -5.435331 | 3.1798e+00 | 0.0778 | 0.9019 | False |
| 1 | family_A | raw | -6.928815 | 1.6863e+00 | 0.1474 | 0.9480 | False |
| 1 | family_A | kappa | -7.391997 | 1.2231e+00 | 0.1481 | 0.9623 | True |
| 1 | family_A | rms | -5.214806 | 3.4003e+00 | 0.0874 | 0.8951 | False |
| 1 | random_K | raw | -6.597548 | 2.0176e+00 | 0.0890 | 0.9378 | False |
| 1 | random_K | kappa | -6.858659 | 1.7565e+00 | 0.0817 | 0.9458 | True |
| 1 | random_K | rms | -5.528985 | 3.0861e+00 | 0.0796 | 0.9048 | False |
| 1 | refl_K | raw | -5.093165 | 3.5220e+00 | 0.0420 | 0.8913 | False |
| 1 | refl_K | kappa | -6.295089 | 2.3200e+00 | 0.0602 | 0.9284 | True |
| 1 | refl_K | rms | -3.289522 | 5.3256e+00 | 0.0159 | 0.8357 | False |
| 2 | qaoa | raw | -7.029424 | 1.5857e+00 | 0.1362 | 0.9511 | False |
| 2 | qaoa | kappa | -7.422235 | 1.1929e+00 | 0.1282 | 0.9632 | True |
| 2 | qaoa | rms | -4.955246 | 3.6599e+00 | 0.0930 | 0.8871 | False |
| 2 | shared_qaoa | raw | -6.866898 | 1.7482e+00 | 0.1026 | 0.9461 | False |
| 2 | shared_qaoa | kappa | -7.013226 | 1.6019e+00 | 0.0925 | 0.9506 | True |
| 2 | shared_qaoa | rms | -6.286360 | 2.3288e+00 | 0.0993 | 0.9282 | False |
| 2 | native_D | raw | -1.480522 | 7.1346e+00 | 0.0136 | 0.7799 | False |
| 2 | native_D | kappa | -0.173467 | 8.4417e+00 | 0.0054 | 0.7396 | True |
| 2 | native_D | rms | -0.557020 | 8.0581e+00 | 0.0013 | 0.7514 | False |
| 2 | family_A | raw | -1.349621 | 7.2655e+00 | 0.0164 | 0.7759 | False |
| 2 | family_A | kappa | -0.089537 | 8.5256e+00 | 0.0067 | 0.7370 | True |
| 2 | family_A | rms | -0.387224 | 8.2279e+00 | 0.0073 | 0.7462 | False |
| 2 | random_K | raw | -1.472413 | 7.1427e+00 | 0.0129 | 0.7797 | False |
| 2 | random_K | kappa | -0.180562 | 8.4346e+00 | 0.0057 | 0.7398 | True |
| 2 | random_K | rms | -0.529807 | 8.0853e+00 | 0.0017 | 0.7506 | False |
| 2 | refl_K | raw | -1.311130 | 7.3040e+00 | 0.0077 | 0.7747 | False |
| 2 | refl_K | kappa | -0.156902 | 8.4582e+00 | 0.0082 | 0.7391 | True |
| 2 | refl_K | rms | -0.373916 | 8.2412e+00 | 0.0015 | 0.7458 | False |
| 3 | qaoa | raw | -7.335348 | 1.2798e+00 | 0.1743 | 0.9605 | False |
| 3 | qaoa | kappa | -7.684014 | 9.3111e-01 | 0.1708 | 0.9713 | True |
| 3 | qaoa | rms | -5.871873 | 2.7433e+00 | 0.1066 | 0.9154 | False |
| 3 | shared_qaoa | raw | -7.028185 | 1.5869e+00 | 0.1413 | 0.9510 | False |
| 3 | shared_qaoa | kappa | -7.418654 | 1.1965e+00 | 0.1327 | 0.9631 | True |
| 3 | shared_qaoa | rms | -5.023520 | 3.5916e+00 | 0.0948 | 0.8892 | False |
| 3 | native_D | raw | -1.714136 | 6.9010e+00 | 0.0145 | 0.7871 | False |
| 3 | native_D | kappa | 1.228502 | 9.8436e+00 | 0.0027 | 0.6963 | True |
| 3 | native_D | rms | -0.226824 | 8.3883e+00 | 0.0012 | 0.7412 | False |
| 3 | family_A | raw | -1.456796 | 7.1583e+00 | 0.0207 | 0.7792 | False |
| 3 | family_A | kappa | -0.661040 | 7.9541e+00 | 0.0029 | 0.7546 | True |
| 3 | family_A | rms | -0.379789 | 8.2353e+00 | 0.0022 | 0.7459 | False |
| 3 | random_K | raw | -1.864331 | 6.7508e+00 | 0.0150 | 0.7917 | False |
| 3 | random_K | kappa | 1.370159 | 9.9853e+00 | 0.0026 | 0.6920 | True |
| 3 | random_K | rms | -0.506589 | 8.1085e+00 | 0.0010 | 0.7499 | False |
| 3 | refl_K | raw | -1.491560 | 7.1236e+00 | 0.0086 | 0.7802 | False |
| 3 | refl_K | kappa | 0.703299 | 9.3184e+00 | 0.0015 | 0.7125 | True |
| 3 | refl_K | rms | -0.398333 | 8.2168e+00 | 0.0004 | 0.7465 | False |
| 4 | qaoa | raw | -6.900996 | 1.7141e+00 | 0.1896 | 0.9471 | False |
| 4 | qaoa | kappa | -7.812793 | 8.0234e-01 | 0.2344 | 0.9752 | True |
| 4 | qaoa | rms | -4.302578 | 4.3125e+00 | 0.0486 | 0.8670 | False |
| 4 | shared_qaoa | raw | -7.148279 | 1.4668e+00 | 0.1653 | 0.9547 | False |
| 4 | shared_qaoa | kappa | -7.569025 | 1.0461e+00 | 0.1550 | 0.9677 | True |
| 4 | shared_qaoa | rms | -5.726012 | 2.8891e+00 | 0.1218 | 0.9109 | False |
| 4 | native_D | raw | -1.471171 | 7.1440e+00 | 0.0132 | 0.7796 | False |
| 4 | native_D | kappa | 0.512739 | 9.1279e+00 | 0.0048 | 0.7184 | True |
| 4 | native_D | rms | -0.210500 | 8.4046e+00 | 0.0026 | 0.7407 | False |
| 4 | family_A | raw | -1.769326 | 6.8458e+00 | 0.0192 | 0.7888 | False |
| 4 | family_A | kappa | 0.129560 | 8.7447e+00 | 0.0077 | 0.7302 | True |
| 4 | family_A | rms | -0.437515 | 8.1776e+00 | 0.0049 | 0.7477 | False |
| 4 | random_K | raw | -1.524659 | 7.0905e+00 | 0.0137 | 0.7813 | False |
| 4 | random_K | kappa | 0.545428 | 9.1606e+00 | 0.0052 | 0.7174 | True |
| 4 | random_K | rms | -0.254323 | 8.3608e+00 | 0.0021 | 0.7421 | False |
| 4 | refl_K | raw | -1.438371 | 7.1768e+00 | 0.0204 | 0.7786 | False |
| 4 | refl_K | kappa | -0.199643 | 8.4155e+00 | 0.0042 | 0.7404 | True |
| 4 | refl_K | rms | -0.338777 | 8.2764e+00 | 0.0003 | 0.7447 | False |
| 5 | qaoa | raw | -6.753436 | 1.8617e+00 | 0.2144 | 0.9426 | False |
| 5 | qaoa | kappa | -7.875452 | 7.3968e-01 | 0.2864 | 0.9772 | True |
| 5 | qaoa | rms | -3.341938 | 5.2732e+00 | 0.0398 | 0.8373 | False |
| 5 | shared_qaoa | raw | -7.283966 | 1.3312e+00 | 0.1857 | 0.9589 | False |
| 5 | shared_qaoa | kappa | -7.665164 | 9.4996e-01 | 0.1778 | 0.9707 | True |
| 5 | shared_qaoa | rms | -5.979033 | 2.6361e+00 | 0.1241 | 0.9187 | False |
| 5 | native_D | raw | -1.577397 | 7.0377e+00 | 0.0233 | 0.7829 | False |
| 5 | native_D | kappa | 0.229533 | 8.8447e+00 | 0.0052 | 0.7271 | True |
| 5 | native_D | rms | -0.578919 | 8.0362e+00 | 0.0024 | 0.7521 | False |
| 5 | family_A | raw | -1.956105 | 6.6590e+00 | 0.0536 | 0.7946 | False |
| 5 | family_A | kappa | 0.730152 | 9.3453e+00 | 0.0006 | 0.7117 | True |
| 5 | family_A | rms | -0.647039 | 7.9681e+00 | 0.0005 | 0.7542 | False |
| 5 | random_K | raw | -1.514351 | 7.1008e+00 | 0.0215 | 0.7809 | False |
| 5 | random_K | kappa | 0.250917 | 8.8660e+00 | 0.0054 | 0.7265 | True |
| 5 | random_K | rms | -0.521420 | 8.0937e+00 | 0.0031 | 0.7503 | False |
| 5 | refl_K | raw | -1.369074 | 7.2461e+00 | 0.0255 | 0.7765 | False |
| 5 | refl_K | kappa | 0.106162 | 8.7213e+00 | 0.0010 | 0.7310 | True |
| 5 | refl_K | rms | -0.416986 | 8.1981e+00 | 0.0017 | 0.7471 | False |
| 6 | qaoa | raw | -5.929263 | 2.6859e+00 | 0.2371 | 0.9171 | False |
| 6 | qaoa | kappa | -7.710558 | 9.0457e-01 | 0.3328 | 0.9721 | True |
| 6 | qaoa | rms | -0.931567 | 7.6836e+00 | 0.0321 | 0.7630 | False |
| 6 | shared_qaoa | raw | -7.447739 | 1.1674e+00 | 0.1941 | 0.9640 | False |
| 6 | shared_qaoa | kappa | -7.732439 | 8.8269e-01 | 0.1899 | 0.9728 | True |
| 6 | shared_qaoa | rms | -6.207646 | 2.4075e+00 | 0.1203 | 0.9257 | False |
| 6 | native_D | raw | -1.668584 | 6.9465e+00 | 0.0325 | 0.7857 | False |
| 6 | native_D | kappa | 0.480962 | 9.0961e+00 | 0.0010 | 0.7194 | True |
| 6 | native_D | rms | -0.352697 | 8.2624e+00 | 0.0040 | 0.7451 | False |
| 6 | family_A | raw | -1.860869 | 6.7543e+00 | 0.0175 | 0.7916 | False |
| 6 | family_A | kappa | -0.188686 | 8.4264e+00 | 0.0033 | 0.7401 | True |
| 6 | family_A | rms | -0.214704 | 8.4004e+00 | 0.0015 | 0.7409 | False |
| 6 | random_K | raw | -1.866661 | 6.7485e+00 | 0.0313 | 0.7918 | False |
| 6 | random_K | kappa | 0.539820 | 9.1549e+00 | 0.0028 | 0.7176 | True |
| 6 | random_K | rms | -0.599667 | 8.0155e+00 | 0.0036 | 0.7527 | False |
| 6 | refl_K | raw | -1.725672 | 6.8895e+00 | 0.0228 | 0.7875 | False |
| 6 | refl_K | kappa | 0.179854 | 8.7950e+00 | 0.0005 | 0.7287 | True |
| 6 | refl_K | rms | -0.524780 | 8.0903e+00 | 0.0028 | 0.7504 | False |
| 7 | qaoa | raw | -5.923375 | 2.6918e+00 | 0.2716 | 0.9170 | False |
| 7 | qaoa | kappa | -7.740245 | 8.7488e-01 | 0.3663 | 0.9730 | True |
| 7 | qaoa | rms | -0.853374 | 7.7618e+00 | 0.0356 | 0.7606 | False |
| 8 | qaoa | raw | -6.160338 | 2.4548e+00 | 0.2767 | 0.9243 | False |
| 8 | qaoa | kappa | -7.800123 | 8.1501e-01 | 0.4061 | 0.9749 | True |
| 8 | qaoa | rms | -1.998140 | 6.6170e+00 | 0.0355 | 0.7959 | False |
| 9 | qaoa | raw | -4.903644 | 3.7115e+00 | 0.2305 | 0.8855 | False |
| 9 | qaoa | kappa | -7.166776 | 1.4484e+00 | 0.4396 | 0.9553 | True |
| 9 | qaoa | rms | -1.027671 | 7.5875e+00 | 0.0260 | 0.7659 | False |
| 10 | qaoa | raw | -3.236209 | 5.3789e+00 | 0.2061 | 0.8341 | False |
| 10 | qaoa | kappa | -6.671115 | 1.9440e+00 | 0.4033 | 0.9400 | True |
| 10 | qaoa | rms | 1.200764 | 9.8159e+00 | 0.0175 | 0.6972 | False |

## Gamma-ramp heuristic

An effective gamma ramp from thetaK<0 was a design heuristic, not a result. The numbers below are the test: the optimized shared thetaK, the correlation of the normalized feature with Hc, and the energy gaps versus thetaK=0 and versus equal-RMS controls. A negative thetaK is consistent with the heuristic only when those controls lose.

## Failures and skips

None recorded.

## Reading

Equal depth: family A is better on both energy and fidelity in 12 of 12 computed cells. That comparison gives A more parameters (2+3L versus 2L).
QAOA at L+1: the equal-depth both-metric edge also beats QAOA at L+1 in 11 cells. It does not at N=12 L=2.
Matched parameter count, QAOA depth ceil((2+3L)/2): both-metric edge survives at N=6 L=3 (QAOA L=6), N=6 L=4 (QAOA L=7), N=6 L=5 (QAOA L=9), N=6 L=6 (QAOA L=10). It does not survive at N=6 L=1 (QAOA L=3), N=6 L=2 (QAOA L=4), N=12 L=1 (QAOA L=3), N=12 L=2 (QAOA L=4), N=12 L=3 (QAOA L=6), N=12 L=4 (QAOA L=7), N=12 L=5 (QAOA L=9), N=12 L=6 (QAOA L=10). Every listed cell has its matched QAOA depth.
The N=6 matched-parameter wins are already present in native D, which fixes thetaK=0 and uses fewer parameters than family A. They are evidence for the nonlinear shell under a shared gamma, not for the physical K matrix. At N=12 native D loses to equal-depth QAOA, and family A loses to parameter-matched QAOA at every depth that was checked.
Controls: at N=12 the physical K feature is ahead of native D (thetaK=0), equal-RMS random K, and equal-RMS reflection-symmetric K on energy at every computed depth, with optimized thetaK negative. At N=6 that gap is large at L=1 and is gone by L=4, where a reflection-symmetric random K matches or beats physical K. Native D alone already beats equal-depth QAOA at N=6 for L>=2 and does not beat equal-depth QAOA at N=12. The N=12 equal-depth edge is the K feature, not the nonlinear shell by itself.
The negative thetaK at N=12 is consistent with the effective-gamma-ramp heuristic and, at that size, the physical matrix beats both equal-RMS controls. The heuristic is not a derivation: thetaK changes sign across N=6 depths, and a reflection-symmetric random matrix can do as well once the shell is expressive.
Restart standard deviations are large because they include failed starts. They are not an error bar on the reported best energy. QAOA rescues with extra fresh starts did not lower the QAOA best energies. Family A used more evaluations than QAOA because it has more parameters and more starts; QAOA was not cut off by the shared cap.
N=6 to N=12 transfer with frozen angles is a different result from retraining. The predeclared kappa rescaling carries L=1 family A to an energy close to the retrained L=1 result. For L>=2 every disclosed gamma scale (raw, kappa, rms) leaves the nonlinear shell far from the ground state, while transferred QAOA with the same kappa rule stays near its retrained energy. A shared shell schedule fit at N=6 does not transfer.
Fidelity along the optimized family-A schedules is not monotone at every depth. Best-fidelity starts were recorded and were not allowed to replace best-energy angles.
This is not a sealed win. Equal-depth gains that die under parameter matching, or that a random reflection-symmetric K can copy, are not a demonstrated advantage of this feature.

## Provenance

- git: `34da47bf05123701684d755fa52c5e33e104bbde`
- python 3.12.3 numpy 2.5.4 scipy 1.18.1
- wall seconds, last process: 30.81507423700009
- timing: Initial sweep, including validation and L=1..7, took 157.5s. The resume that added matched-depth QAOA L=8..10 took 30.8s. Per-job seconds are the optimization times and were not repeated on resume.
- validation passed: True

