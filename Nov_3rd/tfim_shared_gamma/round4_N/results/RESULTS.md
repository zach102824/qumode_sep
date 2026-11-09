# Round 4 N — pre-registered reductions

R_M is the main new fit: sublattice shells, G = Hc + thetaM Mhat, raw P = 3L+2. R_full (P = 3L+3), block-K (P = 3L+3), and contiguous N_red (P = 3L+3) are the pre-registered comparisons. A within-threshold result would support a parameter reduction only. It is not an advantage and not a seal. No parameter count was discounted after the fit.

M is the block-mean projection of Hinter (max abs 0.0e+00). With rM = RMS(M)/RMS(J) = 0.803619, Hc + thetaM Mhat = Hintra + (Hinter - M) + (1 + thetaM/rM) M holds to 2.1e-14. The source multiply-scale RMS(J)/RMS(M) is 1.244371, matching the stored source constant. Full K is outside the span of J and M (residual fraction 0.287), and raw K differs from its block mean (relative 0.601). Hc stays at coefficient 1, so this is not a scale gauge. M on a computational basis state depends only on the cavity Hamming weights.

A shift of any one shell coefficient by π is a global phase. Sending gamma and the shell to their negatives conjugates the state and does not leave the same vector. Gamma and theta were not wrapped. Stored source angles, replayed in this simulator and not imported as code, match the published energies through 1e-12, and the L5 and L6 sublattice+M states overlap the stored vectors.

R_M is worse than the fresh R_full best energy by more than 0.03 at both L5 and L6. That is the pre-registered kill for this reduction. L5: best dE +0.1794, median dE +0.5447, dF -0.2181. L6: best dE +0.1398, median dE -7.0839, dF -0.1284. The L6 median gap is not evidence for dropping K: R_full's median is pulled by failed starts, and the best-energy gap is the kill.

Against the replayed source R_full state, which this fresh fit did not fully recover, L5 best dE +0.2022, dF -0.1625; L6 best dE +0.1628, dF -0.1053. That comparison is labeled and is not a second pass/fail rule.

Block-K versus fresh R_full: L4 best dE +0.0611 (within 0.01: False), median no worse: False, dF -0.1049; L6 best dE +0.0204 (within 0.01: False), median no worse: True, dF -0.0101. Replacing full K by its block mean is a different generator at the same raw P. It does not cancel under the fixed Hc coefficient.

N_red versus the hash-checked round-3 family N (P = 3L+4): L4 best dE +0.0625 (within 0.01: False), dF -0.0986; L5 best dE +0.0021 (within 0.01: True), dF -0.0105; L6 best dE +0.0108 (within 0.01: False), dF -0.0285. A pass would be a parameter reduction only. These cells do not open an advantage.

Matched depth is ceil(P/2). For R_M that is p = 3,4,6,7,9,10 at L1..6, and the next depth is one higher, plus p=12. P=20 versus p=10 is a rounded count, not an exact match and not an exhaustive optimum. Equal-depth gains sit on a separate ledger from the parameter-matched ledger. The published p=12 energy edge of about 0.004 over the source L6 state is not treated as noise, and a seed spread is not the uncertainty of a reported energy. The success column is the L-BFGS flag. Medians, IQRs, and CDFs use every finished start.

Fresh R_full L5 has raw P=18, the same count as QAOA p9. Its energy is -0.0492 relative to that QAOA state and its fidelity difference is +0.2449. Against p10 the energy difference is -0.0083. Fresh R_full L6 has raw P=21. Against QAOA p11 (P=22) the energy difference is -0.0003 and the fidelity difference is +0.1481. Against p12 the energy difference is +0.0269 and the fidelity difference is +0.1183. p12 is lower in energy and lower in fidelity. It does not have both error at most 0.292 and fidelity at least 0.610, and the fresh L6 state does not dominate p12. No seal.

Manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Packing `4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717`.
Optimizer time summed over the finished jobs is 719 s. Report assembly took 7 s and did not refit those jobs.

## Replay of stored source angles

These rows reuse the stored angles in this simulator. They are a baseline check, not a new fit.

- A_cont_L1: dE 8.438e-14, dF -6.106e-16
- A_cont_L2: dE 5.951e-14, dF -5.829e-16
- A_cont_L3: dE 1.243e-14, dF 2.776e-16
- A_cont_L4: dE 1.599e-13, dF -4.885e-15
- A_cont_L5: dE 1.350e-13, dF -5.551e-15
- A_cont_L6: dE 5.329e-15, dF 4.441e-16
- A_sub_L1: dE 3.553e-14, dF 2.498e-16
- A_sub_L2: dE 3.819e-14, dF -3.608e-16
- A_sub_L3: dE -1.510e-14, dF 1.832e-15
- A_sub_L4: dE -3.730e-14, dF 2.220e-15
- A_sub_L5: dE 2.487e-14, dF 1.110e-16
- A_sub_L6: dE -8.349e-14, dF 3.830e-15
- A_sub_M_L1: dE 3.197e-14, dF -7.494e-16
- A_sub_M_L2: dE 5.329e-14, dF -1.776e-15
- A_sub_M_L3: dE -1.599e-14, dF 7.216e-16
- A_sub_M_L4: dE 8.349e-14, dF -4.496e-15
- A_sub_M_L5: dE -1.332e-13, dF 9.548e-15, overlap 1.000000000000
- A_sub_M_L6: dE -1.421e-14, dF 1.554e-15, overlap 1.000000000000
- qaoa_L1: dE 9.770e-15, dF 2.776e-16
- qaoa_L2: dE 1.066e-14, dF 3.608e-16
- qaoa_L3: dE 1.155e-14, dF 3.053e-16
- qaoa_L4: dE 2.576e-14, dF 8.327e-17
- qaoa_L5: dE 2.043e-14, dF 2.220e-16
- qaoa_L6: dE 1.776e-15, dF 1.110e-16

| arm | L | P | E | error | F | best F not used | var | median | IQR | success | failed | evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N_red | 1 | 6 | -7.59901 | 1.0142e+00 | 0.1820 | 0.1820 | 1.458e+00 | -7.5990 | 0.000 | 10 | 0 | 409 |
| N_red | 2 | 9 | -7.65241 | 9.6080e-01 | 0.1987 | 0.1987 | 1.373e+00 | -7.6495 | 3.054 | 7 | 0 | 844 |
| N_red | 3 | 12 | -8.00339 | 6.0981e-01 | 0.3172 | 0.3172 | 7.729e-01 | -7.8416 | 0.323 | 6 | 0 | 1127 |
| N_red | 4 | 15 | -8.07677 | 5.3643e-01 | 0.4797 | 0.4797 | 1.538e+00 | -8.0596 | 5.430 | 14 | 0 | 3856 |
| N_red | 5 | 18 | -8.23095 | 3.8226e-01 | 0.5617 | 0.5617 | 6.932e-01 | -8.2216 | 0.213 | 14 | 0 | 4175 |
| N_red | 6 | 21 | -8.26289 | 3.5032e-01 | 0.5853 | 0.5853 | 5.879e-01 | -8.1627 | 7.253 | 8 | 6 | 6298 |
| R_M | 1 | 5 | -6.85148 | 1.7617e+00 | 0.0657 | 0.1255 | 2.624e+00 | -6.1096 | 6.486 | 18 | 0 | 666 |
| R_M | 2 | 8 | -7.32790 | 1.2853e+00 | 0.2384 | 0.2384 | 5.687e+00 | -7.2796 | 0.560 | 15 | 0 | 1191 |
| R_M | 3 | 11 | -7.97372 | 6.3949e-01 | 0.4243 | 0.4243 | 1.640e+00 | -3.9472 | 7.273 | 14 | 0 | 2154 |
| R_M | 4 | 14 | -8.06086 | 5.5235e-01 | 0.4382 | 0.4382 | 1.149e+00 | -0.8071 | 7.384 | 14 | 0 | 3464 |
| R_M | 5 | 17 | -8.09270 | 5.2051e-01 | 0.4389 | 0.5026 | 9.818e-01 | -7.6084 | 6.374 | 10 | 4 | 4779 |
| R_M | 6 | 20 | -8.15864 | 4.5456e-01 | 0.5046 | 0.5304 | 9.009e-01 | -8.0371 | 7.122 | 6 | 8 | 5648 |
| R_blockK | 1 | 6 | -7.02728 | 1.5859e+00 | 0.2342 | 0.2342 | 7.707e+00 | -6.9848 | 4.815 | 9 | 1 | 583 |
| R_blockK | 2 | 9 | -7.95402 | 6.5919e-01 | 0.5217 | 0.5217 | 1.900e+00 | -7.7983 | 6.501 | 11 | 0 | 1580 |
| R_blockK | 3 | 12 | -8.12963 | 4.8357e-01 | 0.5032 | 0.5304 | 9.728e-01 | -7.9809 | 5.594 | 6 | 0 | 1708 |
| R_blockK | 4 | 15 | -8.17581 | 4.3739e-01 | 0.5000 | 0.5398 | 6.574e-01 | -7.8940 | 7.050 | 4 | 6 | 4346 |
| R_blockK | 5 | 18 | -8.17811 | 4.3510e-01 | 0.4998 | 0.5738 | 6.481e-01 | -8.1213 | 0.147 | 0 | 4 | 1800 |
| R_blockK | 6 | 21 | -8.27809 | 3.3512e-01 | 0.6229 | 0.7319 | 6.144e-01 | -8.1185 | 5.451 | 2 | 8 | 4437 |
| R_full | 1 | 6 | -7.67878 | 9.3442e-01 | 0.2375 | 0.2679 | 1.570e+00 | -7.6788 | 5.185 | 14 | 0 | 842 |
| R_full | 2 | 9 | -8.06734 | 5.4587e-01 | 0.4280 | 0.5139 | 1.230e+00 | -8.0524 | 0.602 | 11 | 0 | 1204 |
| R_full | 3 | 12 | -8.11115 | 5.0206e-01 | 0.4331 | 0.5362 | 7.738e-01 | -7.7750 | 5.680 | 9 | 1 | 2251 |
| R_full | 4 | 15 | -8.23694 | 3.7626e-01 | 0.6049 | 0.6049 | 7.235e-01 | -7.9695 | 0.468 | 8 | 2 | 3556 |
| R_full | 5 | 18 | -8.27209 | 3.4111e-01 | 0.6569 | 0.6569 | 7.149e-01 | -8.1531 | 0.210 | 4 | 6 | 4495 |
| R_full | 6 | 21 | -8.29847 | 3.1473e-01 | 0.6330 | 0.6330 | 5.349e-01 | -0.9532 | 5.409 | 3 | 7 | 4452 |
| qaoa | 1 | 2 | -6.81888 | 1.7943e+00 | 0.0809 | 0.0809 | 3.098e+00 | -6.8189 | 6.710 | 64 | 0 | 883 |
| qaoa | 2 | 4 | -7.43165 | 1.1816e+00 | 0.1333 | 0.1333 | 1.954e+00 | -3.9108 | 7.037 | 60 | 6 | 1590 |
| qaoa | 3 | 6 | -7.68513 | 9.2807e-01 | 0.1705 | 0.1705 | 1.227e+00 | -0.5219 | 5.500 | 57 | 2 | 2411 |
| qaoa | 4 | 8 | -7.85317 | 7.6003e-01 | 0.2153 | 0.2153 | 1.074e+00 | -0.5508 | 2.619 | 83 | 4 | 4418 |
| qaoa | 5 | 10 | -7.97691 | 6.3629e-01 | 0.2584 | 0.2584 | 8.326e-01 | -0.6229 | 6.275 | 64 | 2 | 4809 |
| qaoa | 6 | 12 | -8.05690 | 5.5630e-01 | 0.2947 | 0.2947 | 6.822e-01 | -0.6982 | 3.805 | 66 | 1 | 6328 |
| qaoa | 7 | 14 | -8.12428 | 4.8892e-01 | 0.3333 | 0.3333 | 5.916e-01 | -0.7690 | 3.554 | 46 | 2 | 6328 |
| qaoa | 8 | 16 | -8.17753 | 4.3567e-01 | 0.3728 | 0.3728 | 5.600e-01 | -0.7800 | 0.107 | 43 | 3 | 6328 |
| qaoa | 9 | 18 | -8.22287 | 3.9033e-01 | 0.4120 | 0.4120 | 5.340e-01 | -0.8808 | 0.148 | 34 | 5 | 6328 |
| qaoa | 10 | 20 | -8.26375 | 3.4945e-01 | 0.4523 | 0.4523 | 5.303e-01 | -0.9343 | 0.913 | 30 | 6 | 6319 |
| qaoa | 11 | 22 | -8.29822 | 3.1499e-01 | 0.4849 | 0.4849 | 4.552e-01 | -0.9500 | 0.150 | 36 | 4 | 6328 |
| qaoa | 12 | 24 | -8.32541 | 2.8779e-01 | 0.5147 | 0.5147 | 4.211e-01 | -1.0268 | 2.887 | 24 | 6 | 6328 |

## Equal depth, next depth, matched depth, and p=12

Matched p is ceil(P/2). That is a rounded parameter count, not an exact match and not an exhaustive optimum. A both-metric cell is energy and fidelity together on the energy-selected states.

- N_red L1 P=6 matched p=3: both vs L True, L+1 True, matched False, next False, p12 False
- N_red L2 P=9 matched p=5: both vs L True, L+1 False, matched False, next False, p12 False
- N_red L3 P=12 matched p=6: both vs L True, L+1 True, matched False, next False, p12 False
- N_red L4 P=15 matched p=8: both vs L True, L+1 True, matched False, next False, p12 False
- N_red L5 P=18 matched p=9: both vs L True, L+1 True, matched True, next False, p12 False
- N_red L6 P=21 matched p=11: both vs L True, L+1 True, matched False, next False, p12 False
- R_M L1 P=5 matched p=3: both vs L False, L+1 False, matched False, next False, p12 False
- R_M L2 P=8 matched p=4: both vs L False, L+1 False, matched False, next False, p12 False
- R_M L3 P=11 matched p=6: both vs L True, L+1 True, matched False, next False, p12 False
- R_M L4 P=14 matched p=7: both vs L True, L+1 True, matched False, next False, p12 False
- R_M L5 P=17 matched p=9: both vs L True, L+1 True, matched False, next False, p12 False
- R_M L6 P=20 matched p=10: both vs L True, L+1 True, matched False, next False, p12 False
- R_blockK L1 P=6 matched p=3: both vs L True, L+1 False, matched False, next False, p12 False
- R_blockK L2 P=9 matched p=5: both vs L True, L+1 True, matched False, next False, p12 False
- R_blockK L3 P=12 matched p=6: both vs L True, L+1 True, matched True, next True, p12 False
- R_blockK L4 P=15 matched p=8: both vs L True, L+1 True, matched False, next False, p12 False
- R_blockK L5 P=18 matched p=9: both vs L True, L+1 True, matched False, next False, p12 False
- R_blockK L6 P=21 matched p=11: both vs L True, L+1 True, matched False, next False, p12 False
- R_full L1 P=6 matched p=3: both vs L True, L+1 True, matched False, next False, p12 False
- R_full L2 P=9 matched p=5: both vs L True, L+1 True, matched True, next True, p12 False
- R_full L3 P=12 matched p=6: both vs L True, L+1 True, matched True, next False, p12 False
- R_full L4 P=15 matched p=8: both vs L True, L+1 True, matched True, next True, p12 False
- R_full L5 P=18 matched p=9: both vs L True, L+1 True, matched True, next True, p12 False
- R_full L6 P=21 matched p=11: both vs L True, L+1 True, matched True, next False, p12 False

N=6 smoke only: R_M L1 E=-3.26278 F=0.5478. Not a claim.

