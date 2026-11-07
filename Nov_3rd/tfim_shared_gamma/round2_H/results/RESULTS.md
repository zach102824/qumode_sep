# Round 2 family H — shared-gamma homotopy of full-K A

Legal candidates use one shared gamma and one shared theta_K. The independent-gamma homotopy stage is an initializer only. It is not a certified ceiling and it is not a legal ansatz.

## Frozen target

- Manifest SHA256 `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`
- Hc diagonal SHA256 (little-endian float64) `d650ce798d622a3fe37ef3e7f86f2619f36eadf7acf0422aab821a539e216353`
- N=12 B=-0.3444 full-space gap=0.0016132338724013096 accessible P=R=+1 gap=0.4337981341960315
- E0=-8.6132024589353 Emax=23.799999349755723 entropy_bits=1.013685161866968 GS residual=1.6623090606507287e-14
- FK RMS scale rms(J)/rms(K)=0.3687169636309484 (full upper triangle, including zeros).
- The full-space gap is the split to the P=-1, R=-1 partner. The accessible gap is larger. Fidelity uses the full-space ground state.

## Results

| method | L | P | E | E-E0 | F | median E | starts | nfev | tag |
|---|---|---|---|---|---|---|---|---|---|
| H_plain | 1 | 5 | -7.451338 | 1.161864e+00 | 0.1377 | -7.451338 | 23/23 | 1600 | screen_237 |
| qaoa | 1 | 2 | -6.818876 | 1.794326e+00 | 0.0809 | -6.818876 | 114/114 | 11574 | grid_4471 |
| thetaK0 | 1 | 4 | -6.841068 | 1.772135e+00 | 0.0813 | -6.841068 | 29/29 | 1146 | screen |
| H_homotopy | 2 | 8 | -7.557203 | 1.055999e+00 | 0.1629 | -7.557203 | 5/5 | 1884 | from_theta0 |
| H_plain | 2 | 8 | -7.557203 | 1.055999e+00 | 0.1629 | -7.557203 | 15/15 | 864 | perturb_0.12 |
| qaoa | 2 | 4 | -7.431651 | 1.181551e+00 | 0.1333 | -7.431651 | 190/190 | 6252 | duplicate |
| randomK | 2 | 8 | -7.040671 | 1.572531e+00 | 0.0962 | -7.040671 | 5/5 | 411 | random_3 |
| thetaK0 | 2 | 7 | -7.038225 | 1.574977e+00 | 0.0948 | -7.038225 | 12/12 | 658 | dup_1 |
| H_homotopy | 3 | 11 | -7.920826 | 6.923768e-01 | 0.2625 | -7.920826 | 5/5 | 2256 | random_1 |
| H_plain | 3 | 11 | -7.920826 | 6.923768e-01 | 0.2625 | -7.920826 | 10/10 | 1131 | random_2 |
| qaoa | 3 | 6 | -7.685128 | 9.280748e-01 | 0.1705 | -7.685128 | 252/252 | 13270 | perturb_0.08 |
| thetaK0 | 3 | 10 | -7.518662 | 1.094541e+00 | 0.1468 | -7.518662 | 8/8 | 560 | perturb_0.12 |
| H_homotopy | 4 | 14 | -8.013502 | 5.997005e-01 | 0.3006 | -8.013502 | 5/5 | 2436 | from_theta0 |
| H_plain | 4 | 14 | -8.013502 | 5.997005e-01 | 0.3006 | -8.013502 | 10/10 | 1611 | perturb_0.04 |
| qaoa | 4 | 8 | -7.853170 | 7.600329e-01 | 0.2153 | -7.853170 | 318/318 | 22899 | perturb_0.03 |
| randomK | 4 | 14 | -7.634357 | 9.788454e-01 | 0.1674 | -7.226892 | 5/5 | 651 | qaoa_c1 |
| thetaK0 | 4 | 13 | -7.631782 | 9.814201e-01 | 0.1652 | -7.631782 | 8/8 | 1152 | perturb_0.04 |
| H_homotopy | 5 | 17 | -8.110540 | 5.026628e-01 | 0.3861 | -8.110540 | 5/5 | 2864 | from_theta0 |
| H_plain | 5 | 17 | -8.110540 | 5.026628e-01 | 0.3861 | -8.110540 | 10/10 | 1865 | perturb_0.04 |
| qaoa | 5 | 10 | -7.976908 | 6.362947e-01 | 0.2584 | -7.928892 | 223/223 | 21198 | perturb_0.2 |
| thetaK0 | 5 | 16 | -7.750589 | 8.626134e-01 | 0.2014 | -7.750522 | 8/8 | 1185 | perturb_0.12 |
| H_homotopy | 6 | 20 | -8.176766 | 4.364363e-01 | 0.4426 | -8.176766 | 5/5 | 3219 | split_plain |
| H_plain | 6 | 20 | -8.176766 | 4.364363e-01 | 0.4426 | -8.159717 | 10/10 | 2052 | perturb_0.04 |
| qaoa | 6 | 12 | -8.056900 | 5.563027e-01 | 0.2947 | -8.023739 | 296/296 | 35949 | fourier_1 |
| randomK | 6 | 20 | -7.857520 | 7.556823e-01 | 0.2308 | -7.640513 | 5/5 | 733 | qaoa_c1 |
| thetaK0 | 6 | 19 | -7.850222 | 7.629803e-01 | 0.2239 | -7.847003 | 8/8 | 1363 | interp_cubic |
| qaoa | 7 | 14 | -8.124283 | 4.889196e-01 | 0.3333 | -8.102757 | 217/217 | 24879 | perturb_0.2 |
| qaoa | 8 | 16 | -8.177533 | 4.356690e-01 | 0.3728 | -8.161295 | 12/12 | 1407 | interp_linear |
| qaoa | 9 | 18 | -8.222874 | 3.903283e-01 | 0.4120 | -8.202885 | 114/114 | 14387 | perturb_0.03 |
| qaoa | 10 | 20 | -8.263750 | 3.494523e-01 | 0.4523 | -8.243550 | 124/124 | 16724 | fourier_2 |

## Family H versus QAOA

Candidate is H_homotopy for L>1 and H_plain for L=1, where the homotopy is vacuous. H_plain is the equally budgeted shared-gamma bootstrap, reported separately. Negative energy difference means the row has lower energy than that QAOA depth.

| L | candidate | E_c - E_qaoa(L) | F_c - F_qaoa(L) | vs L+1 | vs ceil(P/2) | plain - qaoa(L) | theta0 - qaoa(L) | median candidate - best QAOA(L) |
|---|---|---|---|---|---|---|---|---|
| 1 | H_plain | -0.6325 | 0.0568 | -0.0197 | 0.2338 | -0.6325 | -0.0222 | -0.6325 |
| 2 | H_homotopy | -0.1256 | 0.0295 | 0.1279 | 0.2960 | -0.1256 | 0.3934 | -0.1256 |
| 3 | H_homotopy | -0.2357 | 0.0920 | -0.0677 | 0.1361 | -0.2357 | 0.1665 | -0.2357 |
| 4 | H_homotopy | -0.1603 | 0.0853 | -0.0366 | 0.1108 | -0.1603 | 0.2214 | -0.1603 |
| 5 | H_homotopy | -0.1336 | 0.1278 | -0.0536 | 0.1123 | -0.1336 | 0.2263 | -0.1336 |
| 6 | H_homotopy | -0.1199 | 0.1480 | -0.0525 | 0.0870 | -0.1199 | 0.2067 | -0.1199 |

Equal-L differences are one ledger. Parameter-matched QAOA is p=ceil((3L+2)/2). A lower energy at equal L is not by itself a quantum-advantage claim, and a loss at matched parameter count is not a reason to hide the equal-L numbers.

## Measured reading

Homotopy did not beat the plain shared-gamma bootstrap. At every N=12 depth the legal homotopy energy matches H_plain. The independent-gamma stage reached a lower relaxed energy and the exact projection back to one gamma did not keep that gain. Those relaxed numbers are initializers. They are not legal candidates and they are not a certified ceiling: a failed or successful local relaxed search does not prove what shared gamma can or cannot reach.

Against equal-depth QAOA, the legal full-K ansatz has lower energy and higher fidelity at L=1 through L=6. The margin survives the QAOA rescue top-ups, which added many starts without lowering the best QAOA energy. At L=6 the plain median (-8.160) is still below the best equal-depth QAOA energy (-8.057).

Against p=ceil(P/2) the ordering reverses: that QAOA depth has lower energy at every L from 1 to 6. Several equal-L wins also remain ahead of QAOA at p=L+1 (L=1,3,4,5,6). L=2 does not. Beating p=L, or even p=L+1, is not a seal under the stricter matched-parameter ledger.

theta_K=0 stays well behind full-K and behind equal-depth QAOA for L>=2, in line with round-1 native D on this target. One frozen Gaussian K with the same RMS, and a smaller start budget, stays near theta_K=0 rather than near FK. That is evidence about this draw, not a full random-matrix survey.

Legal angles are one gamma and one theta_K. From H_plain:

- L=1: gamma=0.372973, theta_K=-0.527440
- L=2: gamma=0.281932, theta_K=-0.573737
- L=3: gamma=0.351049, theta_K=-0.486939
- L=4: gamma=0.314303, theta_K=-0.504938
- L=5: gamma=0.341585, theta_K=-0.570251
- L=6: gamma=0.321957, theta_K=-0.630066

| L | P | H energy error | QAOA L | QAOA L+1 | QAOA ceil(P/2) | H F | QAOA L F |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 1.1619 | 1.7943 | 1.1816 | 0.9281 | 0.1377 | 0.0809 |
| 2 | 8 | 1.0560 | 1.1816 | 0.9281 | 0.7600 | 0.1629 | 0.1333 |
| 3 | 11 | 0.6924 | 0.9281 | 0.7600 | 0.5563 | 0.2625 | 0.1705 |
| 4 | 14 | 0.5997 | 0.7600 | 0.6363 | 0.4889 | 0.3006 | 0.2153 |
| 5 | 17 | 0.5027 | 0.6363 | 0.5563 | 0.3903 | 0.3861 | 0.2584 |
| 6 | 20 | 0.4364 | 0.5563 | 0.4889 | 0.3495 | 0.4426 | 0.2947 |

## Notes

- Manifest SHA256 14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9
- Hc diagonal SHA256 little-endian float64 d650ce798d622a3fe37ef3e7f86f2619f36eadf7acf0422aab821a539e216353
- FK scale rms_J/rms_K = 0.368716963631 (full upper triangle, zeros included).
- Homotopy independent-gamma stages are initializers. They are not a certified ceiling and not legal candidates.
- QAOA beta2 = beta1 + 0.2*|B| converts the paper offset because beta multiplies sum X and B is not in the mixer.
- N6 is a smoke test only. N12 is the comparison.
