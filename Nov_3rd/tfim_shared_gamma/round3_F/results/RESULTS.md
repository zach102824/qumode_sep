# Round 3 family F — tied polynomial of Hc

Legal generator `G = Hc + theta_K (FK + Q4) / rK = Hc + theta_K / (2 rK) (Hc^2 - c0 I)`. One shared gamma and one shared theta_K. This is a function of Hc. Free theta_Q is a separate diagnostic and is not f(Hc).

## Target and support

- Manifest SHA256 `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`
- Hc diagonal SHA256 `d650ce798d622a3fe37ef3e7f86f2619f36eadf7acf0422aab821a539e216353`
- N=12 B=-0.3444 E0=-8.6132024589353 full-space gap=0.0016132338724013096 accessible gap=0.4337981341960315
- rK=2.712107 c0=14.829437 two-body supports=66 four-body supports=495
- The 495 four-body supports are charged. Being a polynomial of Hc is not a promise that the hardware cost equals two-body A.
- Full-space gap is the P=-1, R=-1 partner. Fidelity uses the full-space ground state.

## Results

| method | L | P | E | E-E0 | F | Var(H) | median | IQR | nfev | tag |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 5 | -7.451338 | 1.1619e+00 | 0.1377 | 1.6092 | -7.4513 | 0.0000 | 1600 | screen_237 |
| F | 1 | 5 | -6.913216 | 1.7000e+00 | 0.0892 | 3.2651 | -6.9132 | 0.0000 | 666 | restart_2 |
| qaoa | 1 | 2 | -6.818876 | 1.7943e+00 | 0.0809 | 3.0975 | -6.8189 | 0.0000 | 11574 | grid_4471 |
| A | 2 | 8 | -7.557203 | 1.0560e+00 | 0.1629 | 1.4946 | -7.5572 | 0.0000 | 864 | perturb_0.12 |
| F | 2 | 8 | -7.194232 | 1.4190e+00 | 0.1123 | 2.9029 | -7.1942 | 2.4683 | 1499 | duplicate |
| freeQ4 | 2 | 9 | -7.571677 | 1.0415e+00 | 0.1653 | 1.5423 | -5.9793 | 6.1053 | 671 | restart_3 |
| qaoa | 2 | 4 | -7.431651 | 1.1816e+00 | 0.1333 | 1.9544 | -7.4317 | 0.1849 | 6252 | duplicate |
| sameG | 2 | 5 | -7.549985 | 1.0632e+00 | 0.1489 | 1.5211 | -7.5500 | 0.0000 | 319 | from_F |
| shuffle4 | 2 | 8 | -7.155002 | 1.4582e+00 | 0.1066 | 2.6913 | -7.1550 | 3.4548 | 452 | from_F |
| thetaK0 | 2 | 7 | -7.038225 | 1.5750e+00 | 0.0948 | 2.5019 | -7.0382 | 0.0000 | 340 | restart_4 |
| A | 3 | 11 | -7.920826 | 6.9238e-01 | 0.2625 | 0.9085 | -7.9208 | 0.2647 | 1131 | random_2 |
| F | 3 | 11 | -7.598665 | 1.0145e+00 | 0.1664 | 1.6641 | -7.5987 | 4.4922 | 2506 | restart_2 |
| qaoa | 3 | 6 | -7.685128 | 9.2807e-01 | 0.1705 | 1.2273 | -7.6851 | 0.1174 | 13270 | perturb_0.08 |
| A | 4 | 14 | -8.013502 | 5.9970e-01 | 0.3006 | 0.7294 | -8.0135 | 0.0000 | 1611 | perturb_0.04 |
| F | 4 | 14 | -7.724650 | 8.8855e-01 | 0.1990 | 1.4976 | -7.7247 | 6.4012 | 3797 | qaoa_warm |
| freeQ4 | 4 | 15 | -8.021058 | 5.9214e-01 | 0.3040 | 0.7311 | -7.9820 | 6.5574 | 1904 | from_F |
| qaoa | 4 | 8 | -7.853170 | 7.6003e-01 | 0.2153 | 1.0743 | -7.8532 | 0.1556 | 22899 | perturb_0.03 |
| sameG | 4 | 9 | -7.954793 | 6.5841e-01 | 0.2509 | 0.8076 | -7.8063 | 0.0606 | 906 | from_qaoa |
| shuffle4 | 4 | 14 | -7.660514 | 9.5269e-01 | 0.1755 | 1.3658 | -0.5665 | 5.8743 | 1207 | restart_3 |
| thetaK0 | 4 | 13 | -7.631782 | 9.8142e-01 | 0.1652 | 1.2808 | -3.2084 | 2.5917 | 736 | restart_4 |
| A | 5 | 17 | -8.110540 | 5.0266e-01 | 0.3861 | 0.7459 | -8.1105 | 0.0003 | 1865 | perturb_0.04 |
| F | 5 | 17 | -7.824292 | 7.8891e-01 | 0.2347 | 1.4771 | -7.3496 | 6.7678 | 5599 | perturb |
| qaoa | 5 | 10 | -7.976908 | 6.3629e-01 | 0.2584 | 0.8326 | -7.9289 | 0.2257 | 21198 | perturb_0.2 |
| A | 6 | 20 | -8.176766 | 4.3644e-01 | 0.4426 | 0.7029 | -8.1597 | 5.4262 | 2052 | perturb_0.04 |
| F | 6 | 20 | -7.932767 | 6.8044e-01 | 0.3110 | 1.4984 | -7.8295 | 1.8331 | 7507 | linear |
| freeQ4 | 6 | 21 | -8.180863 | 4.3234e-01 | 0.4404 | 0.6693 | -8.1493 | 4.8906 | 2606 | from_F |
| qaoa | 6 | 12 | -8.056900 | 5.5630e-01 | 0.2947 | 0.6822 | -8.0237 | 0.1957 | 35949 | fourier_1 |
| sameG | 6 | 13 | -8.101947 | 5.1126e-01 | 0.3246 | 0.7529 | -8.0123 | 5.1211 | 1233 | from_F |
| shuffle4 | 6 | 20 | -7.866537 | 7.4667e-01 | 0.2331 | 1.0565 | -0.6852 | 2.9751 | 3146 | from_F |
| thetaK0 | 6 | 19 | -7.850222 | 7.6298e-01 | 0.2239 | 1.0307 | -4.8930 | 6.0432 | 1652 | restart_2 |
| qaoa | 7 | 14 | -8.124283 | 4.8892e-01 | 0.3333 | 0.5916 | -8.1028 | 0.1512 | 24879 | perturb_0.2 |
| qaoa | 8 | 16 | -8.177533 | 4.3567e-01 | 0.3728 | 0.5600 | -8.1613 | 5.4541 | 1407 | interp_linear |
| qaoa | 9 | 18 | -8.222874 | 3.9033e-01 | 0.4120 | 0.5340 | -8.2029 | 0.1883 | 14387 | perturb_0.03 |
| qaoa | 10 | 20 | -8.263750 | 3.4945e-01 | 0.4523 | 0.5303 | -8.2436 | 0.1973 | 16724 | fourier_2 |

## Ledgers

Negative `E_F - E_other` means F has lower energy. Equal depth and matched depth are separate.

| L | E_F-E_A | E_F-E_Q(L) | F_F-F_Q(L) | E_F-E_Q(L+1) | E_F-E_Q(match) | F_F-F_Q(match) |
|---|---|---|---|---|---|---|
| 1 | 0.5381 | -0.0943 | 0.0083 | 0.5184 | 0.7719 | -0.0813 |
| 2 | 0.3630 | 0.2374 | -0.0211 | 0.4909 | 0.6589 | -0.1030 |
| 3 | 0.3222 | 0.0865 | -0.0041 | 0.2545 | 0.4582 | -0.1283 |
| 4 | 0.2889 | 0.1285 | -0.0162 | 0.2523 | 0.3996 | -0.1342 |
| 5 | 0.2862 | 0.1526 | -0.0236 | 0.2326 | 0.3986 | -0.1773 |
| 6 | 0.2440 | 0.1241 | 0.0163 | 0.1915 | 0.3310 | -0.1413 |

A is the round-2 full-K replay (FK only, same pair coefficients and same P). Shuffle4 permutes the 495 four-body coefficients with seed 31108 and keeps FK. thetaK0 is G=Hc. sameG and freeQ4 are diagnostics. freeQ4 is not f(Hc).

Polishing F from the stored A angles did not lower F's energy at any depth. The tied optimum is a different point, with a smaller |theta_K| than A.

