# Round 1 direction D — shared-gamma native shell vs QAOA

No robust win. Equal-depth differences, where present, did not clear every gate on both sizes.

Training minimized energy only. Fidelity is a post-hoc diagnostic of the best-energy state.
Family D uses G = Hc, one gamma shared by every layer, and per-layer shell coefficients (c1, c2, c3).
Split gamma and per-layer theta were not optimized.

## Frozen fields

| N | B | hex | entropy (bits) | E0 | gap | Emax | GS residual | P+ weight | R+ weight | H sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | -0.170800 | `-0x1.5dcc63f141207p-3` | 1.000897 | -3.84334934 | 5.054e-04 | 8.39993164 | 2.65e-15 | 1.00000000 | 1.00000000 | `682c538790d1` |
| 12 | -0.344400 | `-0x1.60aa64c2f837cp-2` | 1.013685 | -8.61320246 | 1.613e-03 | 23.79999935 | 1.30e-14 | 1.00000000 | 1.00000000 | `42e67c0a3e6a` |

B maximizes the half-chain entropy of the full-space ground state. It is this run's numerical realization, not a published Pagano field. Alpha = 1.1, J0 = 1, open chain, J_ij = |i-j|^{-1.1}.
The gap column is the full-space splitting to the P=-1, R=-1 partner. Both circuits stay in P=+1, R=+1, which contains the ground state. The gap to the next state in that sector is larger and is given in the notes.

## Best energy and the fidelity of that state

| N | L | method | params | E | E-E0 | F | eta | median E | starts | nfev | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1 | qaoa | 2 | -2.59257981 | 1.250770e+00 | 0.202151 | 0.8978 | -2.592580 | 24/24 | 318 | grid_21059 |
| 6 | 1 | shared_beta | 2 | -2.59257981 | 1.250770e+00 | 0.202151 | 0.8978 | -2.592580 | 48/48 | 496 | identical_to_qaoa_L1 |
| 6 | 1 | nov3_D | 4 | -2.61994807 | 1.223401e+00 | 0.203705 | 0.9001 | -2.619948 | 83/83 | 1913 | screen_798 |
| 6 | 1 | drop_c2 | 3 | -2.59809228 | 1.245257e+00 | 0.202123 | 0.8983 | -2.598092 | 51/51 | 730 | qaoa_c3_grid |
| 6 | 1 | drop_c3 | 3 | -2.61978166 | 1.223568e+00 | 0.203591 | 0.9001 | -2.619782 | 53/53 | 990 | qaoa_c2_grid |
| 6 | 2 | qaoa | 4 | -2.95449461 | 8.888547e-01 | 0.295563 | 0.9274 | -2.954495 | 44/44 | 1109 | shift_0.3 |
| 6 | 2 | shared_beta | 3 | -2.71339135 | 1.129958e+00 | 0.230341 | 0.9077 | -2.713391 | 23/23 | 506 | perturb_0.15_0 |
| 6 | 2 | nov3_D | 7 | -3.23010038 | 6.132490e-01 | 0.551612 | 0.9499 | -2.739831 | 170/170 | 8618 | unbounded_blitz:near_14 |
| 6 | 2 | drop_c2 | 5 | -3.17539033 | 6.679590e-01 | 0.503261 | 0.9454 | -2.735739 | 32/32 | 1156 | random_2 |
| 6 | 2 | drop_c3 | 5 | -3.22637637 | 6.169730e-01 | 0.546708 | 0.9496 | -2.654232 | 69/69 | 2515 | adiabatic_random:adiab_2.50_0.80_0.00 |
| 6 | 3 | qaoa | 6 | -3.16317125 | 6.801781e-01 | 0.381265 | 0.9444 | -3.142960 | 31/31 | 1242 | perturb_0.1_1 |
| 6 | 3 | shared_beta | 4 | -2.94595011 | 8.973992e-01 | 0.302870 | 0.9267 | -2.945950 | 21/21 | 590 | gamma_shift_-0.1 |
| 6 | 3 | nov3_D | 10 | -3.75610825 | 8.724109e-02 | 0.956677 | 0.9929 | -3.074318 | 22/22 | 1664 | perturb_0.15_0 |
| 6 | 3 | drop_c2 | 7 | -3.71918142 | 1.241679e-01 | 0.935355 | 0.9899 | -2.978268 | 23/23 | 1187 | random_3 |
| 6 | 3 | drop_c3 | 7 | -3.10276870 | 7.405806e-01 | 0.528483 | 0.9395 | -3.044824 | 23/23 | 1103 | random_3 |
| 6 | 4 | qaoa | 8 | -3.31537929 | 5.279700e-01 | 0.483193 | 0.9569 | -3.293363 | 32/32 | 2030 | perturb_0.2_1 |
| 6 | 4 | shared_beta | 5 | -3.05692730 | 7.864220e-01 | 0.350331 | 0.9358 | -3.056927 | 21/21 | 783 | random_7 |
| 6 | 4 | nov3_D | 13 | -3.79368000 | 4.966933e-02 | 0.982329 | 0.9959 | -3.200926 | 63/63 | 6447 | adiabatic_random:adiab_0.30_0.40_0.00 |
| 6 | 4 | drop_c2 | 9 | -3.76851993 | 7.482940e-02 | 0.973127 | 0.9939 | -3.086566 | 61/61 | 3927 | adiabatic_random:adiab_0.30_0.80_0.00 |
| 6 | 4 | drop_c3 | 9 | -3.27838187 | 5.649675e-01 | 0.652062 | 0.9539 | -2.904831 | 61/61 | 4582 | adiabatic_random:adiab_1.50_0.15_0.00 |
| 6 | 5 | qaoa | 10 | -3.43731806 | 4.060313e-01 | 0.552303 | 0.9668 | -3.369907 | 32/32 | 2656 | random_2 |
| 6 | 5 | shared_beta | 6 | -3.15838884 | 6.849605e-01 | 0.397758 | 0.9441 | -2.219533 | 21/21 | 990 | perturb_0.15_1 |
| 6 | 5 | nov3_D | 16 | -3.83157660 | 1.177274e-02 | 0.995272 | 0.9990 | -3.501492 | 23/23 | 4168 | perturb_0.03_0 |
| 6 | 5 | drop_c2 | 11 | -3.82861693 | 1.473241e-02 | 0.991071 | 0.9988 | -3.791575 | 24/24 | 2307 | perturb_0.03_1 |
| 6 | 5 | drop_c3 | 11 | -3.38208884 | 4.612605e-01 | 0.724487 | 0.9623 | -3.008134 | 24/24 | 1988 | projected_D |
| 6 | 6 | qaoa | 12 | -3.53592534 | 3.074240e-01 | 0.647668 | 0.9749 | -3.256784 | 249/249 | 38071 | unbounded_blitz:wide_32 |
| 6 | 6 | shared_beta | 7 | -3.22353364 | 6.198157e-01 | 0.421543 | 0.9494 | -3.137371 | 21/21 | 1325 | perturb_0.15_1 |
| 6 | 6 | nov3_D | 19 | -3.84189918 | 1.450162e-03 | 0.999147 | 0.9999 | -3.536865 | 160/160 | 36697 | unbounded_blitz:keep |
| 6 | 6 | drop_c2 | 13 | -3.83269190 | 1.065744e-02 | 0.992133 | 0.9991 | -3.265462 | 24/24 | 2732 | gamma_shift_0.1 |
| 6 | 6 | drop_c3 | 13 | -3.61041758 | 2.329318e-01 | 0.869257 | 0.9810 | -3.324190 | 24/24 | 2334 | perturb_0.15_0 |
| 6 | 7 | qaoa | 14 | -3.58404732 | 2.593020e-01 | 0.706785 | 0.9788 | -3.077475 | 93/93 | 15921 | adiabatic_random:rescue_random_5 |
| 6 | 8 | qaoa | 16 | -3.63665800 | 2.066913e-01 | 0.817244 | 0.9831 | -3.602770 | 32/32 | 6135 | perturb_0.05_1 |
| 12 | 1 | qaoa | 2 | -6.81887631 | 1.794326e+00 | 0.080928 | 0.9446 | -6.818876 | 24/24 | 431 | grid_11708 |
| 12 | 1 | shared_beta | 2 | -6.81887631 | 1.794326e+00 | 0.080928 | 0.9446 | -6.818876 | 36/36 | 601 | identical_to_qaoa_L1 |
| 12 | 1 | nov3_D | 4 | -6.84106788 | 1.772135e+00 | 0.081276 | 0.9453 | -6.841068 | 71/71 | 1959 | qaoa_shell_grid |
| 12 | 1 | drop_c2 | 3 | -6.82937587 | 1.783827e+00 | 0.080978 | 0.9450 | -6.829376 | 38/38 | 670 | qaoa_c3_grid |
| 12 | 1 | drop_c3 | 3 | -6.84029861 | 1.772904e+00 | 0.081248 | 0.9453 | -6.840299 | 40/40 | 938 | screen_602 |
| 12 | 2 | qaoa | 4 | -7.43165148 | 1.181551e+00 | 0.133349 | 0.9635 | -7.431651 | 36/36 | 1202 | perturb_0.05_1 |
| 12 | 2 | shared_beta | 3 | -7.01364568 | 1.599557e+00 | 0.094784 | 0.9507 | -7.013646 | 23/23 | 734 | random_3 |
| 12 | 2 | nov3_D | 7 | -7.03822535 | 1.574977e+00 | 0.094784 | 0.9514 | -7.038225 | 33/33 | 1822 | perturb_0.03_1 |
| 12 | 2 | drop_c2 | 5 | -7.03737844 | 1.575824e+00 | 0.094925 | 0.9514 | -7.037378 | 32/32 | 1301 | perturb_0.08_0 |
| 12 | 2 | drop_c3 | 5 | -7.02457935 | 1.588623e+00 | 0.094026 | 0.9510 | -7.024579 | 32/32 | 1480 | duplicate_4 |
| 12 | 3 | qaoa | 6 | -7.68512767 | 9.280748e-01 | 0.170536 | 0.9714 | -7.685128 | 31/31 | 1641 | perturb_0.05_2 |
| 12 | 3 | shared_beta | 4 | -7.42752772 | 1.185675e+00 | 0.138166 | 0.9634 | -7.427528 | 21/21 | 811 | random_6 |
| 12 | 3 | nov3_D | 10 | -7.51866170 | 1.094541e+00 | 0.146840 | 0.9662 | -7.518662 | 22/22 | 2014 | qaoa_ratio_c1 |
| 12 | 3 | drop_c2 | 7 | -7.44527473 | 1.167928e+00 | 0.138916 | 0.9640 | -7.445275 | 23/23 | 1314 | projected_D |
| 12 | 3 | drop_c3 | 7 | -7.51749482 | 1.095708e+00 | 0.146383 | 0.9662 | -7.517495 | 23/23 | 1802 | perturb_0.08_0 |
| 12 | 4 | qaoa | 8 | -7.85316959 | 7.600329e-01 | 0.215279 | 0.9766 | -7.784943 | 32/32 | 2372 | perturb_0.1_1 |
| 12 | 4 | shared_beta | 5 | -7.56914148 | 1.044061e+00 | 0.156353 | 0.9678 | -7.569141 | 21/21 | 951 | perturb_0.08_0 |
| 12 | 4 | nov3_D | 13 | -7.63178236 | 9.814201e-01 | 0.165173 | 0.9697 | -7.631782 | 23/23 | 3319 | perturb_0.08_0 |
| 12 | 4 | drop_c2 | 9 | -7.58451212 | 1.028690e+00 | 0.159515 | 0.9683 | -7.584512 | 24/24 | 1912 | gamma_shift_0.1 |
| 12 | 4 | drop_c3 | 9 | -7.62531635 | 9.878861e-01 | 0.162975 | 0.9695 | -7.625316 | 24/24 | 2087 | qaoa_c1_only |
| 12 | 5 | qaoa | 10 | -7.97690777 | 6.362947e-01 | 0.258378 | 0.9804 | -7.924059 | 32/32 | 3311 | perturb_0.05_1 |
| 12 | 5 | shared_beta | 6 | -7.67192204 | 9.412804e-01 | 0.171607 | 0.9710 | -7.671922 | 21/21 | 932 | qaoa_ratio_rescale |
| 12 | 5 | nov3_D | 16 | -7.75058903 | 8.626134e-01 | 0.201389 | 0.9734 | -7.750589 | 23/23 | 4994 | random_5 |
| 12 | 5 | drop_c2 | 11 | -7.69431011 | 9.188924e-01 | 0.179986 | 0.9717 | -7.694310 | 24/24 | 2970 | interp_linear_alt |
| 12 | 5 | drop_c3 | 11 | -7.74235560 | 8.708469e-01 | 0.202632 | 0.9731 | -7.742356 | 24/24 | 2715 | interp_cubic |
| 12 | 6 | qaoa | 12 | -8.05689981 | 5.563027e-01 | 0.294677 | 0.9828 | -8.040319 | 32/32 | 3905 | fourier_5 |
| 12 | 6 | shared_beta | 7 | -7.73511385 | 8.780886e-01 | 0.187842 | 0.9729 | -7.735114 | 21/21 | 1492 | gamma_shift_-0.1 |
| 12 | 6 | nov3_D | 19 | -7.85022212 | 7.629803e-01 | 0.223923 | 0.9765 | -7.850222 | 23/23 | 5465 | perturb_0.3_1 |
| 12 | 6 | drop_c2 | 13 | -7.76913312 | 8.440693e-01 | 0.197952 | 0.9740 | -7.769133 | 24/24 | 2912 | perturb_0.08_0 |
| 12 | 6 | drop_c3 | 13 | -7.84701280 | 7.661897e-01 | 0.221562 | 0.9764 | -7.847013 | 24/24 | 3516 | perturb_0.3_0 |
| 12 | 7 | qaoa | 14 | -8.12428283 | 4.889196e-01 | 0.333286 | 0.9849 | -8.048227 | 32/32 | 6065 | perturb_0.1_1 |
| 12 | 8 | qaoa | 16 | -8.17753342 | 4.356690e-01 | 0.372760 | 0.9866 | -8.078029 | 32/32 | 8419 | perturb_0.02_1 |

## Effect sizes versus equal-depth QAOA

Negative energy difference means the method has lower energy than QAOA. Positive fidelity difference means higher fidelity. These columns are measurements. The verdict section applies the gates.

| N | L | method | E - E_QAOA | F - F_QAOA | fraction of QAOA residual closed |
|---|---|---|---|---|---|
| 6 | 1 | drop_c2 | -5.512469e-03 | -2.821565e-05 | 0.0044 |
| 6 | 1 | drop_c3 | -2.720185e-02 | 1.439540e-03 | 0.0217 |
| 6 | 1 | nov3_D | -2.736826e-02 | 1.553515e-03 | 0.0219 |
| 6 | 1 | shared_beta | 0.000000e+00 | 0.000000e+00 | 0.0000 |
| 6 | 2 | drop_c2 | -2.208957e-01 | 2.076974e-01 | 0.2485 |
| 6 | 2 | drop_c3 | -2.718818e-01 | 2.511449e-01 | 0.3059 |
| 6 | 2 | nov3_D | -2.756058e-01 | 2.560485e-01 | 0.3101 |
| 6 | 2 | shared_beta | 2.411033e-01 | -6.522198e-02 | -0.2713 |
| 6 | 3 | drop_c2 | -5.560102e-01 | 5.540901e-01 | 0.8174 |
| 6 | 3 | drop_c3 | 6.040255e-02 | 1.472181e-01 | -0.0888 |
| 6 | 3 | nov3_D | -5.929370e-01 | 5.754120e-01 | 0.8717 |
| 6 | 3 | shared_beta | 2.172211e-01 | -7.839479e-02 | -0.3194 |
| 6 | 4 | drop_c2 | -4.531406e-01 | 4.899341e-01 | 0.8583 |
| 6 | 4 | drop_c3 | 3.699742e-02 | 1.688692e-01 | -0.0701 |
| 6 | 4 | nov3_D | -4.783007e-01 | 4.991364e-01 | 0.9059 |
| 6 | 4 | shared_beta | 2.584520e-01 | -1.328621e-01 | -0.4895 |
| 6 | 5 | drop_c2 | -3.912989e-01 | 4.387679e-01 | 0.9637 |
| 6 | 5 | drop_c3 | 5.522923e-02 | 1.721843e-01 | -0.1360 |
| 6 | 5 | nov3_D | -3.942585e-01 | 4.429689e-01 | 0.9710 |
| 6 | 5 | shared_beta | 2.789292e-01 | -1.545450e-01 | -0.6870 |
| 6 | 6 | drop_c2 | -2.967666e-01 | 3.444642e-01 | 0.9653 |
| 6 | 6 | drop_c3 | -7.449224e-02 | 2.215891e-01 | 0.2423 |
| 6 | 6 | nov3_D | -3.059738e-01 | 3.514790e-01 | 0.9953 |
| 6 | 6 | shared_beta | 3.123917e-01 | -2.261253e-01 | -1.0162 |
| 12 | 1 | drop_c2 | -1.049956e-02 | 5.029688e-05 | 0.0059 |
| 12 | 1 | drop_c3 | -2.142230e-02 | 3.206453e-04 | 0.0119 |
| 12 | 1 | nov3_D | -2.219156e-02 | 3.484480e-04 | 0.0124 |
| 12 | 1 | shared_beta | 0.000000e+00 | 0.000000e+00 | 0.0000 |
| 12 | 2 | drop_c2 | 3.942730e-01 | -3.842356e-02 | -0.3337 |
| 12 | 2 | drop_c3 | 4.070721e-01 | -3.932294e-02 | -0.3445 |
| 12 | 2 | nov3_D | 3.934261e-01 | -3.856468e-02 | -0.3330 |
| 12 | 2 | shared_beta | 4.180058e-01 | -3.856411e-02 | -0.3538 |
| 12 | 3 | drop_c2 | 2.398529e-01 | -3.162065e-02 | -0.2584 |
| 12 | 3 | drop_c3 | 1.676328e-01 | -2.415335e-02 | -0.1806 |
| 12 | 3 | nov3_D | 1.664660e-01 | -2.369671e-02 | -0.1794 |
| 12 | 3 | shared_beta | 2.575999e-01 | -3.237031e-02 | -0.2776 |
| 12 | 4 | drop_c2 | 2.686575e-01 | -5.576455e-02 | -0.3535 |
| 12 | 4 | drop_c3 | 2.278532e-01 | -5.230459e-02 | -0.2998 |
| 12 | 4 | nov3_D | 2.213872e-01 | -5.010663e-02 | -0.2913 |
| 12 | 4 | shared_beta | 2.840281e-01 | -5.892666e-02 | -0.3737 |
| 12 | 5 | drop_c2 | 2.825977e-01 | -7.839194e-02 | -0.4441 |
| 12 | 5 | drop_c3 | 2.345522e-01 | -5.574599e-02 | -0.3686 |
| 12 | 5 | nov3_D | 2.263187e-01 | -5.698830e-02 | -0.3557 |
| 12 | 5 | shared_beta | 3.049857e-01 | -8.677109e-02 | -0.4793 |
| 12 | 6 | drop_c2 | 2.877667e-01 | -9.672505e-02 | -0.5173 |
| 12 | 6 | drop_c3 | 2.098870e-01 | -7.311501e-02 | -0.3773 |
| 12 | 6 | nov3_D | 2.066777e-01 | -7.075325e-02 | -0.3715 |
| 12 | 6 | shared_beta | 3.217860e-01 | -1.068341e-01 | -0.5784 |

## Measured reading

Program success requires the pre-set gates at two or more depths on both N=6 and N=12. That did not happen. N=12 native D is behind equal-depth QAOA from L=2 through L=6 on energy and on fidelity. Extra adiabatic seeds, differential evolution, and an unbounded random blitz did not reverse that ordering.

The cell flag `robust` is only the best-of-restarts gate. It is not a claim that a typical random start wins, and it is not program success. Compare the median energy with QAOA's best energy:

| N | L | best D | median D | best QAOA | median D minus best QAOA |
|---|---|---|---|---|---|
| 6 | 1 | -2.619948 | -2.619948 | -2.592580 | -0.027368 |
| 6 | 2 | -3.230100 | -2.739831 | -2.954495 | 0.214664 |
| 6 | 3 | -3.756108 | -3.074318 | -3.163171 | 0.088853 |
| 6 | 4 | -3.793680 | -3.200926 | -3.315379 | 0.114453 |
| 6 | 5 | -3.831577 | -3.501492 | -3.437318 | -0.064174 |
| 6 | 6 | -3.841899 | -3.536865 | -3.535925 | -0.000939 |
| 12 | 1 | -6.841068 | -6.841068 | -6.818876 | -0.022192 |
| 12 | 2 | -7.038225 | -7.038225 | -7.431651 | 0.393426 |
| 12 | 3 | -7.518662 | -7.518662 | -7.685128 | 0.166466 |
| 12 | 4 | -7.631782 | -7.631782 | -7.853170 | 0.221387 |
| 12 | 5 | -7.750589 | -7.750589 | -7.976908 | 0.226319 |
| 12 | 6 | -7.850222 | -7.850222 | -8.056900 | 0.206678 |

A positive last column means the median D start is worse than the best QAOA energy. At N=6 the best shell energy is ahead of equal-depth QAOA for L>=2, and L=3,4,5 also clear the matched-depth and ablation gates. Dropping c3 removes most of that N=6 gain. Dropping c2 does not. Zero-shot transfer of those N=6 shell angles onto N=12 collapses; kappa rescaling of the cost angle does not repair it. Shared-beta is the same ansatz as QAOA at L=1 and is behind independent-angle QAOA at every L>1.

## Verdict

Program success: **False**.

| N | L | equal-depth ahead | robust | why |
|---|---|---|---|---|
| 6 | 1 | True | False | energy improvement 0.0273683 does not exceed 0.0625385; fidelity improvement 0.00155352 < 0.01; D does not beat QAOA L+1 (-2.9544946) by 0.001; D does not beat matched QAOA L=2 (-2.9544946) by 0.001 |
| 6 | 2 | True | False | D does not beat matched QAOA L=4 (-3.3153793) by 0.001 |
| 6 | 3 | True | True | passed every gate |
| 6 | 4 | True | True | passed every gate |
| 6 | 5 | True | True | passed every gate |
| 6 | 6 | True | False | missing matched-parameter QAOA at L=10 |
| 6 | 7 | None | False | missing D or QAOA |
| 6 | 8 | None | False | missing D or QAOA |
| 12 | 1 | True | False | energy improvement 0.0221916 does not exceed 0.0897163; fidelity improvement 0.000348448 < 0.01; D does not beat QAOA L+1 (-7.4316515) by 0.001; D does not beat matched QAOA L=2 (-7.4316515) by 0.001 |
| 12 | 2 | False | False | energy improvement -0.393426 does not exceed 0.0590775; fidelity improvement -0.0385647 < 0.01; D does not beat QAOA L+1 (-7.6851277) by 0.001; D does not beat matched QAOA L=4 (-7.8531696) by 0.001 |
| 12 | 3 | False | False | energy improvement -0.166466 does not exceed 0.0464037; fidelity improvement -0.0236967 < 0.01; D does not beat QAOA L+1 (-7.8531696) by 0.001; D does not beat matched QAOA L=5 (-7.9769078) by 0.001 |
| 12 | 4 | False | False | energy improvement -0.221387 does not exceed 0.0380016; fidelity improvement -0.0501066 < 0.01; D does not beat QAOA L+1 (-7.9769078) by 0.001; D does not beat matched QAOA L=7 (-8.1242828) by 0.001 |
| 12 | 5 | False | False | energy improvement -0.226319 does not exceed 0.0318147; fidelity improvement -0.0569883 < 0.01; D does not beat QAOA L+1 (-8.0568998) by 0.001; D does not beat matched QAOA L=8 (-8.1775334) by 0.001 |
| 12 | 6 | False | False | energy improvement -0.206678 does not exceed 0.0278151; fidelity improvement -0.0707533 < 0.01; D does not beat QAOA L+1 (-8.1242828) by 0.001; missing matched-parameter QAOA at L=10 |
| 12 | 7 | None | False | missing D or QAOA |
| 12 | 8 | None | False | missing D or QAOA |

## Zero-shot N=6 to N=12

Angles trained at N=6 are evaluated on the N=12 Hamiltonian with no further training. `kappa` multiplies only cost angles by kappa(6)/kappa(12). Shell coefficients and mixer betas are copied. This rescaling is a J-only hypothesis, not a fitted transfer.

| L | method | kind | E | E-E0 | F |
|---|---|---|---|---|---|
| 1 | qaoa | raw | -6.573506 | 2.039696e+00 | 0.088034 |
| 1 | qaoa | kappa_cost | -6.812669 | 1.800534e+00 | 0.079442 |
| 1 | shared_beta | raw | -6.573506 | 2.039696e+00 | 0.088034 |
| 1 | shared_beta | kappa_cost | -6.812669 | 1.800534e+00 | 0.079442 |
| 1 | nov3_D | raw | -6.540932 | 2.072270e+00 | 0.086434 |
| 1 | nov3_D | kappa_cost | -6.831963 | 1.781239e+00 | 0.079718 |
| 1 | drop_c2 | raw | -6.580710 | 2.032493e+00 | 0.087699 |
| 1 | drop_c2 | kappa_cost | -6.823202 | 1.790001e+00 | 0.079215 |
| 1 | drop_c3 | raw | -6.540607 | 2.072595e+00 | 0.086405 |
| 1 | drop_c3 | kappa_cost | -6.832246 | 1.780957e+00 | 0.079751 |
| 2 | qaoa | raw | -7.027241 | 1.585962e+00 | 0.135644 |
| 2 | qaoa | kappa_cost | -7.417763 | 1.195439e+00 | 0.127550 |
| 2 | shared_beta | raw | -6.863363 | 1.749840e+00 | 0.102120 |
| 2 | shared_beta | kappa_cost | -7.008363 | 1.604840e+00 | 0.091973 |
| 2 | nov3_D | raw | -0.553073 | 8.060129e+00 | 0.008057 |
| 2 | nov3_D | kappa_cost | -0.393199 | 8.220003e+00 | 0.010097 |
| 2 | drop_c2 | raw | -1.101174 | 7.512029e+00 | 0.003697 |
| 2 | drop_c2 | kappa_cost | -0.714049 | 7.899154e+00 | 0.002136 |
| 2 | drop_c3 | raw | -0.443731 | 8.169471e+00 | 0.006060 |
| 2 | drop_c3 | kappa_cost | -0.685425 | 7.927778e+00 | 0.000621 |
| 3 | qaoa | raw | -7.333641 | 1.279561e+00 | 0.173786 |
| 3 | qaoa | kappa_cost | -7.679910 | 9.332920e-01 | 0.170029 |
| 3 | shared_beta | raw | -7.025867 | 1.587336e+00 | 0.140683 |
| 3 | shared_beta | kappa_cost | -7.414089 | 1.199114e+00 | 0.132023 |
| 3 | nov3_D | raw | -1.714150 | 6.899052e+00 | 0.014517 |
| 3 | nov3_D | kappa_cost | 1.228739 | 9.841941e+00 | 0.002718 |
| 3 | drop_c2 | raw | -1.904182 | 6.709020e+00 | 0.021291 |
| 3 | drop_c2 | kappa_cost | 0.808691 | 9.421893e+00 | 0.003814 |
| 3 | drop_c3 | raw | -1.511114 | 7.102089e+00 | 0.000859 |
| 3 | drop_c3 | kappa_cost | 1.037088 | 9.650291e+00 | 0.006990 |
| 4 | qaoa | raw | -6.900690 | 1.712513e+00 | 0.189398 |
| 4 | qaoa | kappa_cost | -7.809499 | 8.037032e-01 | 0.233518 |
| 4 | shared_beta | raw | -7.146169 | 1.467034e+00 | 0.164713 |
| 4 | shared_beta | kappa_cost | -7.564546 | 1.048656e+00 | 0.154269 |
| 4 | nov3_D | raw | -1.572698 | 7.040504e+00 | 0.019593 |
| 4 | nov3_D | kappa_cost | 0.101276 | 8.714479e+00 | 0.008292 |
| 4 | drop_c2 | raw | -1.553525 | 7.059677e+00 | 0.039575 |
| 4 | drop_c2 | kappa_cost | 0.505750 | 9.118952e+00 | 0.000735 |
| 4 | drop_c3 | raw | 0.091126 | 8.704328e+00 | 0.004246 |
| 4 | drop_c3 | kappa_cost | 0.008925 | 8.622128e+00 | 0.000878 |
| 5 | qaoa | raw | -7.093064 | 1.520139e+00 | 0.288860 |
| 5 | qaoa | kappa_cost | -7.799932 | 8.132703e-01 | 0.254079 |
| 5 | shared_beta | raw | -7.282138 | 1.331065e+00 | 0.185093 |
| 5 | shared_beta | kappa_cost | -7.661023 | 9.521791e-01 | 0.176945 |
| 5 | nov3_D | raw | -1.069382 | 7.543820e+00 | 0.013547 |
| 5 | nov3_D | kappa_cost | 0.960366 | 9.573568e+00 | 0.003300 |
| 5 | drop_c2 | raw | -1.774909 | 6.838293e+00 | 0.017012 |
| 5 | drop_c2 | kappa_cost | 0.516186 | 9.129388e+00 | 0.000716 |
| 5 | drop_c3 | raw | -1.465290 | 7.147913e+00 | 0.021210 |
| 5 | drop_c3 | kappa_cost | 0.019668 | 8.632871e+00 | 0.006630 |
| 6 | qaoa | raw | -4.854438 | 3.758765e+00 | 0.277324 |
| 6 | qaoa | kappa_cost | -6.947168 | 1.666034e+00 | 0.283545 |
| 6 | shared_beta | raw | -7.445749 | 1.167453e+00 | 0.193531 |
| 6 | shared_beta | kappa_cost | -7.728453 | 8.847491e-01 | 0.189081 |
| 6 | nov3_D | raw | -1.871253 | 6.741950e+00 | 0.031562 |
| 6 | nov3_D | kappa_cost | 0.561119 | 9.174322e+00 | 0.002719 |
| 6 | drop_c2 | raw | -1.852813 | 6.760389e+00 | 0.013788 |
| 6 | drop_c2 | kappa_cost | 0.413413 | 9.026615e+00 | 0.000016 |
| 6 | drop_c3 | raw | -0.569821 | 8.043382e+00 | 0.005354 |
| 6 | drop_c3 | kappa_cost | -0.547898 | 8.065304e+00 | 0.000263 |
| 7 | qaoa | raw | -5.130226 | 3.482977e+00 | 0.258801 |
| 7 | qaoa | kappa_cost | -7.229691 | 1.383511e+00 | 0.309215 |
| 8 | qaoa | raw | -6.161115 | 2.452087e+00 | 0.277152 |
| 8 | qaoa | kappa_cost | -7.797432 | 8.157703e-01 | 0.405061 |

## Notes

- Site 0 is MSB and the left end of the open chain. Z|0>=+|0>, Z|1>=-|1>.
- Shell phases use the contract formula. c2=c3=0 reproduces shared-gamma QAOA with c1=beta.
- QAOA L=2 primary seed uses beta2=beta1+0.2*|B| and gamma2=gamma1+0.2. The beta offset is the paper's -0.2 converted by beta_ours=beta_paper*B.
- L=3 schedules are linear interpolations of normalized layer time. Later depths use a natural cubic spline. The optimizer is not constrained to stay smooth.
- Zero shell does not insert an identity layer: two zero-shell layers at gamma equal one layer at 2 gamma.
- Kappa transfer scales cost angles only. It is not used to train N=12.
- source_sha256=6b7007ba2e53545e1cdc528b07e83afeb26b1df53592b409dd67bf104c694777
- budget_min=80.0
- kappa6=2.789189 kappa12=3.935676 cost-angle scale kappa6/kappa12=0.708694
- Rescue pass added adiabatic seeds, random local searches, and differential evolution.
- A new vector replaced the published one only when its energy was lower.
- Fidelity was not used to accept a vector.
- The first excited state is the opposite-parity, opposite-reflection partner. Both QAOA and the shell circuit stay in P=+1, R=+1, which contains the ground state. The accessible gap is the gap to the next state in that sector.
- N=6 full-space gap 5.053821e-04, accessible P=R=+1 gap 7.047690e-01.
- N=12 full-space gap 1.613234e-03, accessible P=R=+1 gap 4.337981e-01.
- The tiny full-space gap is the splitting to the inaccessible P=-1, R=-1 partner.
- Unbounded blitz: wide Gaussian starts and perturbations of the incumbent, accepted only on lower energy.
