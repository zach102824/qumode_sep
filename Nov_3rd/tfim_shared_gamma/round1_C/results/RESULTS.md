# Round 1 direction C — shared star generator vs QAOA

The four-body star is the cubic monomial suggested by a formal Schrieffer-Wolff expansion of a transverse field around a uniform classical bound built from J alone. That motivation is a hypothesis. This run does not solve a Schrieffer-Wolff effective Hamiltonian, does not claim the series is controlled at the entropy peak, and does not fix theta_C to a perturbative value. Theta is a free shared energy-fit coefficient in front of an explicitly normalized J-polynomial feature.

Target: open chain `H = sum_{i<j} |i-j|^{-1.1} Z_i Z_j + B sum_i X_i`, `J0=1`.
This is an ideal power-law model. `B` is the entropy-maximizing value found on the contract grid, not a published Pagano field.
Training minimized energy only. Fidelity was not used to choose angles, features, or the transfer map.

## Frozen fields

| N | B | E0 | gap | Emax | S (nats) | ⟨P⟩ | ⟨R⟩ | P=+1 weight | GS residual | H hash |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | -0.1721 | -3.844581 | 5.286e-04 | 8.400426 | 0.6938 | 1.000000 | 1.000000 | 1.000000 | 4.79e-15 | `ace4b037e792` |
| 12 | -0.3444 | -8.613202 | 1.613e-03 | 23.799999 | 0.7026 | 1.000000 | 1.000000 | 1.000000 | 1.57e-14 | `50d78163c348` |

Endpoint expansions: {"6": [], "12": []}

## Feature structure

Normalization: each of `F_K` and `F_star` is rescaled to the Pauli HS norm of `Hc`. Duplicate 4-qubit masks sum all center contributions (four per mask).

- N=6: star masks 15 (full 15), HS fraction retained 1.0000, max centers/mask 4, odd masks 0, full-space two-body R² 1.110e-16, intra-cavity Hc HS fraction 0.708.
  - Single-flip residual ratio after removing 1, Hc, F_K: 7.725274058162e-13 (R² 1.0). Néel cat: both Néel strings carry the same Hc and the same F_star, so the feature is only a phase there.
  - GS-probability-weighted residual ratio (diagnostic, not used in training): 0.26073333901692847 (R² 0.9320181259250835).
- N=12: star masks 90 (full 495), HS fraction retained 0.8385, max centers/mask 4, odd masks 0, full-space two-body R² 2.220e-16, intra-cavity Hc HS fraction 0.598.
  - Single-flip residual ratio after removing 1, Hc, F_K: 0.7447570167740778 (R² 0.44533698596577564). Néel cat: both Néel strings carry the same Hc and the same F_star, so the feature is only a phase there.
  - GS-probability-weighted residual ratio (diagnostic, not used in training): 0.3339163573703222 (R² 0.8884998662805352).

## Optimizer

L-BFGS-B, maxiter 80, maxfun 700, ftol 1e-14, gtol 1e-08. Starts: 18 at L=1 (grid-initialized) and 12 at L>1, identical across methods. QAOA L=2 uses the Pagano shift after converting β_paper·B into our β. Candidate L=2 duplicates shells and shifts only c1; gamma stays one scalar. L≥3 interpolates shells (linear at L=3, cubic after) and copies gamma and theta. D, A, and C each replace up to two random starts with the same-depth simpler model (shared-linear into D, D into A, A and D into C) with every new coefficient set to 0. The number of starts does not change. If a non-QAOA energy beats QAOA by more than 1e-5, QAOA gets 12 extra rescue starts, including a projection of that depth's c1 schedule onto beta.

## Results

| N | L | method | params | E−E0 | F | η | median E−E0 | ΔE vs QAOA | ΔF vs QAOA | ΔE vs QAOA L+1 | ΔF vs QAOA L+1 | grad | nfev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1 | qaoa | 2 | 1.24620 | 0.2032 | 0.8982 | 1.24620 | — | — | — | — | 2.49e-07 | 213 |
| 6 | 1 | shared_linear | 2 | 1.24620 | 0.2032 | 0.8982 | 1.24620 | 0.00000 | 0.0000 | 0.36089 | -0.0935 | 4.09e-09 | 203 |
| 6 | 1 | D | 4 | 1.21898 | 0.2048 | 0.9005 | 1.21898 | -0.02722 | 0.0016 | 0.33367 | -0.0920 | 1.72e-07 | 451 |
| 6 | 1 | A | 5 | 0.78299 | 0.3388 | 0.9361 | 0.78299 | -0.46320 | 0.1356 | -0.10231 | 0.0421 | 2.21e-08 | 591 |
| 6 | 1 | C | 6 | 0.78078 | 0.3421 | 0.9362 | 0.78078 | -0.46542 | 0.1389 | -0.10453 | 0.0454 | 7.09e-08 | 689 |
| 6 | 2 | qaoa | 4 | 0.88531 | 0.2967 | 0.9277 | 1.22963 | — | — | — | — | 7.30e-08 | 368 |
| 6 | 2 | shared_linear | 3 | 1.12577 | 0.2314 | 0.9081 | 1.12577 | 0.24047 | -0.0653 | 0.44844 | -0.1509 | 4.10e-08 | 290 |
| 6 | 2 | D | 7 | 1.06600 | 0.5407 | 0.9129 | 1.09933 | 0.18069 | 0.2440 | 0.38867 | 0.1584 | 2.59e-07 | 475 |
| 6 | 2 | A | 8 | 0.66773 | 0.4260 | 0.9455 | 0.67354 | -0.21758 | 0.1293 | -0.00960 | 0.0437 | 1.50e-07 | 598 |
| 6 | 2 | C | 9 | 0.66632 | 0.4255 | 0.9456 | 0.66632 | -0.21898 | 0.1288 | -0.01101 | 0.0432 | 1.88e-07 | 761 |
| 6 | 3 | qaoa | 6 | 0.67733 | 0.3823 | 0.9447 | 1.17778 | — | — | — | — | 1.32e-07 | 401 |
| 6 | 3 | shared_linear | 4 | 0.89374 | 0.3041 | 0.9270 | 0.89374 | 0.21641 | -0.0783 | 0.36759 | -0.1802 | 6.28e-10 | 302 |
| 6 | 3 | D | 10 | 0.73636 | 0.3975 | 0.9399 | 1.20984 | 0.05903 | 0.0152 | 0.21021 | -0.0867 | 1.17e-06 | 698 |
| 6 | 3 | A | 11 | 0.37582 | 0.6730 | 0.9693 | 0.43822 | -0.30151 | 0.2907 | -0.15033 | 0.1888 | 3.63e-07 | 924 |
| 6 | 3 | C | 12 | 0.31535 | 0.8172 | 0.9742 | 0.41984 | -0.36198 | 0.4348 | -0.21080 | 0.3329 | 3.02e-07 | 1173 |
| 6 | 4 | qaoa | 8 | 0.52615 | 0.4843 | 0.9570 | 0.90220 | — | — | — | — | 8.31e-08 | 436 |
| 6 | 4 | shared_linear | 5 | 0.78304 | 0.3515 | 0.9361 | 0.78304 | 0.25689 | -0.1327 | 0.36763 | -0.2097 | 3.74e-09 | 393 |
| 6 | 4 | D | 13 | 0.39244 | 0.7067 | 0.9680 | 0.63997 | -0.13371 | 0.2224 | -0.02297 | 0.1454 | 5.07e-07 | 1113 |
| 6 | 4 | A | 14 | 0.01818 | 0.9906 | 0.9985 | 0.21659 | -0.50797 | 0.5063 | -0.39723 | 0.4293 | 7.02e-04 | 1137 |
| 6 | 4 | C | 15 | 0.01729 | 0.9913 | 0.9986 | 0.80398 | -0.50886 | 0.5071 | -0.39812 | 0.4301 | 2.31e-06 | 1356 |
| 6 | 5 | qaoa | 10 | 0.41541 | 0.5613 | 0.9661 | 0.97689 | — | — | — | — | 3.62e-08 | 628 |
| 6 | 5 | shared_linear | 6 | 0.68205 | 0.3987 | 0.9443 | 0.68205 | 0.26664 | -0.1625 | 0.35547 | -0.2773 | 4.34e-09 | 406 |
| 6 | 5 | D | 16 | 0.14697 | 0.9082 | 0.9880 | 0.42555 | -0.26844 | 0.3470 | -0.17960 | 0.2322 | 1.45e-06 | 1180 |
| 6 | 5 | A | 17 | 0.02290 | 0.9848 | 0.9981 | 0.31713 | -0.39251 | 0.4236 | -0.30368 | 0.3088 | 2.49e-06 | 1398 |
| 6 | 5 | C | 18 | 0.02263 | 0.9853 | 0.9982 | 0.53132 | -0.39278 | 0.4240 | -0.30395 | 0.3092 | 3.81e-06 | 1563 |
| 6 | 6 | qaoa | 12 | 0.32658 | 0.6760 | 0.9733 | 0.63321 | — | — | — | — | 2.78e-07 | 832 |
| 6 | 6 | shared_linear | 7 | 0.61718 | 0.4226 | 0.9496 | 0.61718 | 0.29061 | -0.2534 | 0.35349 | -0.3304 | 8.15e-08 | 594 |
| 6 | 6 | D | 19 | 0.13082 | 0.9278 | 0.9893 | 0.40932 | -0.19576 | 0.2518 | -0.13287 | 0.1748 | 2.60e-06 | 1229 |
| 6 | 6 | A | 20 | 0.00381 | 0.9984 | 0.9997 | 0.17552 | -0.32277 | 0.3223 | -0.25989 | 0.2454 | 1.28e-03 | 1637 |
| 6 | 6 | C | 21 | 0.00339 | 0.9986 | 0.9997 | 0.29286 | -0.32318 | 0.3226 | -0.26030 | 0.2456 | 4.54e-03 | 1625 |
| 6 | 7 | qaoa | 14 | 0.26369 | 0.7530 | 0.9785 | 0.69858 | — | — | — | — | 3.07e-07 | 871 |
| 12 | 1 | qaoa | 2 | 1.79433 | 0.0809 | 0.9446 | 1.79433 | — | — | — | — | 2.14e-08 | 234 |
| 12 | 1 | shared_linear | 2 | 1.79433 | 0.0809 | 0.9446 | 8.34203 | 0.00000 | -0.0000 | 0.61278 | -0.0524 | 4.75e-10 | 255 |
| 12 | 1 | D | 4 | 1.77213 | 0.0813 | 0.9453 | 4.70470 | -0.02219 | 0.0003 | 0.59058 | -0.0521 | 5.62e-07 | 571 |
| 12 | 1 | A | 5 | 1.16186 | 0.1377 | 0.9642 | 1.16186 | -0.63246 | 0.0568 | -0.01969 | 0.0044 | 3.86e-08 | 619 |
| 12 | 1 | C | 6 | 1.15748 | 0.1372 | 0.9643 | 1.15748 | -0.63685 | 0.0563 | -0.02407 | 0.0039 | 7.92e-08 | 714 |
| 12 | 1 | C_unrestricted | 6 | 1.16032 | 0.1375 | 0.9642 | 1.16032 | -0.63401 | 0.0566 | -0.02123 | 0.0042 | 4.73e-07 | 717 |
| 12 | 2 | qaoa | 4 | 1.18155 | 0.1333 | 0.9635 | 4.48040 | — | — | — | — | 4.17e-07 | 204 |
| 12 | 2 | shared_linear | 3 | 1.59956 | 0.0948 | 0.9507 | 1.59956 | 0.41801 | -0.0386 | 0.67148 | -0.0758 | 1.88e-07 | 272 |
| 12 | 2 | D | 7 | 1.57498 | 0.0948 | 0.9514 | 1.62688 | 0.39343 | -0.0386 | 0.64690 | -0.0758 | 1.41e-06 | 470 |
| 12 | 2 | A | 8 | 1.05600 | 0.1629 | 0.9674 | 1.05600 | -0.12555 | 0.0295 | 0.12792 | -0.0077 | 1.75e-06 | 716 |
| 12 | 2 | C | 9 | 1.02874 | 0.1640 | 0.9683 | 1.02874 | -0.15281 | 0.0307 | 0.10067 | -0.0065 | 2.01e-06 | 780 |
| 12 | 2 | C_unrestricted | 9 | 1.03437 | 0.1633 | 0.9681 | 1.03437 | -0.14718 | 0.0299 | 0.10629 | -0.0072 | 6.26e-07 | 730 |
| 12 | 3 | qaoa | 6 | 0.92807 | 0.1705 | 0.9714 | 3.16766 | — | — | — | — | 7.76e-07 | 390 |
| 12 | 3 | shared_linear | 4 | 1.18567 | 0.1382 | 0.9634 | 1.18567 | 0.25760 | -0.0324 | 0.42564 | -0.0771 | 7.82e-07 | 378 |
| 12 | 3 | D | 10 | 1.09454 | 0.1468 | 0.9662 | 1.09454 | 0.16647 | -0.0237 | 0.33451 | -0.0684 | 3.73e-06 | 885 |
| 12 | 3 | A | 11 | 0.69238 | 0.2625 | 0.9786 | 0.72751 | -0.23570 | 0.0920 | -0.06766 | 0.0472 | 5.55e-07 | 1077 |
| 12 | 3 | C | 12 | 0.67511 | 0.2664 | 0.9792 | 0.75955 | -0.25297 | 0.0958 | -0.08492 | 0.0511 | 1.76e-06 | 1075 |
| 12 | 3 | C_unrestricted | 12 | 0.68186 | 0.2660 | 0.9790 | 0.74861 | -0.24621 | 0.0955 | -0.07817 | 0.0507 | 1.24e-06 | 1062 |
| 12 | 4 | qaoa | 8 | 0.76003 | 0.2153 | 0.9766 | 4.60390 | — | — | — | — | 7.70e-07 | 523 |
| 12 | 4 | shared_linear | 5 | 1.04406 | 0.1564 | 0.9678 | 4.47637 | 0.28403 | -0.0589 | 0.40777 | -0.1020 | 1.22e-08 | 338 |
| 12 | 4 | D | 13 | 0.98142 | 0.1652 | 0.9697 | 1.19803 | 0.22139 | -0.0501 | 0.34513 | -0.0932 | 2.50e-05 | 1033 |
| 12 | 4 | A | 14 | 0.59970 | 0.3006 | 0.9815 | 0.59970 | -0.16033 | 0.0853 | -0.03659 | 0.0422 | 1.31e-05 | 1152 |
| 12 | 4 | C | 15 | 0.58454 | 0.3081 | 0.9820 | 0.58454 | -0.17549 | 0.0928 | -0.05175 | 0.0497 | 1.98e-05 | 1133 |
| 12 | 4 | C_unrestricted | 15 | 0.58539 | 0.3053 | 0.9819 | 0.58539 | -0.17464 | 0.0901 | -0.05090 | 0.0470 | 2.32e-06 | 1141 |
| 12 | 5 | qaoa | 10 | 0.63629 | 0.2584 | 0.9804 | 2.21260 | — | — | — | — | 1.19e-06 | 776 |
| 12 | 5 | shared_linear | 6 | 0.94128 | 0.1716 | 0.9710 | 4.00963 | 0.30499 | -0.0868 | 0.38498 | -0.1231 | 9.52e-07 | 407 |
| 12 | 5 | D | 16 | 0.86261 | 0.2014 | 0.9734 | 4.29219 | 0.22632 | -0.0570 | 0.30631 | -0.0933 | 2.94e-06 | 1209 |
| 12 | 5 | A | 17 | 0.50266 | 0.3862 | 0.9845 | 0.51391 | -0.13363 | 0.1278 | -0.05364 | 0.0915 | 6.63e-04 | 1155 |
| 12 | 5 | C | 18 | 0.50092 | 0.3849 | 0.9845 | 0.50262 | -0.13537 | 0.1265 | -0.05538 | 0.0902 | 9.93e-05 | 1107 |
| 12 | 6 | qaoa | 12 | 0.55630 | 0.2947 | 0.9828 | 0.72735 | — | — | — | — | 6.87e-07 | 736 |
| 12 | 6 | shared_linear | 7 | 0.87809 | 0.1878 | 0.9729 | 0.87809 | 0.32179 | -0.1068 | 0.38917 | -0.1454 | 6.16e-07 | 509 |
| 12 | 6 | D | 19 | 0.76298 | 0.2239 | 0.9765 | 2.74508 | 0.20668 | -0.0708 | 0.27406 | -0.1094 | 1.34e-05 | 1297 |
| 12 | 6 | A | 20 | 0.43644 | 0.4426 | 0.9865 | 0.47648 | -0.11987 | 0.1479 | -0.05248 | 0.1093 | 9.14e-04 | 1151 |
| 12 | 6 | C | 21 | 0.42242 | 0.4450 | 0.9870 | 4.07392 | -0.13388 | 0.1503 | -0.06650 | 0.1117 | 6.98e-07 | 1266 |
| 12 | 7 | qaoa | 14 | 0.48892 | 0.3333 | 0.9849 | 0.93459 | — | — | — | — | 7.70e-07 | 893 |

## What beat same-depth QAOA on both energy and fidelity

These rows are equal-depth only. Most of them do not survive the parameter-matched QAOA comparison in the following section.

- N=6 L=1 D: ΔE=-0.02722, ΔF=0.0016, params 4 vs QAOA 2.
- N=6 L=1 A: ΔE=-0.46320, ΔF=0.1356, params 5 vs QAOA 2.
- N=6 L=1 C: ΔE=-0.46542, ΔF=0.1389, params 6 vs QAOA 2.
- N=6 L=2 A: ΔE=-0.21758, ΔF=0.1293, params 8 vs QAOA 4.
- N=6 L=2 C: ΔE=-0.21898, ΔF=0.1288, params 9 vs QAOA 4.
- N=6 L=3 A: ΔE=-0.30151, ΔF=0.2907, params 11 vs QAOA 6.
- N=6 L=3 C: ΔE=-0.36198, ΔF=0.4348, params 12 vs QAOA 6.
- N=6 L=4 D: ΔE=-0.13371, ΔF=0.2224, params 13 vs QAOA 8.
- N=6 L=4 A: ΔE=-0.50797, ΔF=0.5063, params 14 vs QAOA 8.
- N=6 L=4 C: ΔE=-0.50886, ΔF=0.5071, params 15 vs QAOA 8.
- N=6 L=5 D: ΔE=-0.26844, ΔF=0.3470, params 16 vs QAOA 10.
- N=6 L=5 A: ΔE=-0.39251, ΔF=0.4236, params 17 vs QAOA 10.
- N=6 L=5 C: ΔE=-0.39278, ΔF=0.4240, params 18 vs QAOA 10.
- N=6 L=6 D: ΔE=-0.19576, ΔF=0.2518, params 19 vs QAOA 12.
- N=6 L=6 A: ΔE=-0.32277, ΔF=0.3223, params 20 vs QAOA 12.
- N=6 L=6 C: ΔE=-0.32318, ΔF=0.3226, params 21 vs QAOA 12.
- N=12 L=1 A: ΔE=-0.63246, ΔF=0.0568, params 5 vs QAOA 2.
- N=12 L=1 C: ΔE=-0.63685, ΔF=0.0563, params 6 vs QAOA 2.
- N=12 L=1 C_unrestricted: ΔE=-0.63401, ΔF=0.0566, params 6 vs QAOA 2.
- N=12 L=2 A: ΔE=-0.12555, ΔF=0.0295, params 8 vs QAOA 4.
- N=12 L=2 C: ΔE=-0.15281, ΔF=0.0307, params 9 vs QAOA 4.
- N=12 L=2 C_unrestricted: ΔE=-0.14718, ΔF=0.0299, params 9 vs QAOA 4.
- N=12 L=3 A: ΔE=-0.23570, ΔF=0.0920, params 11 vs QAOA 6.
- N=12 L=3 C: ΔE=-0.25297, ΔF=0.0958, params 12 vs QAOA 6.
- N=12 L=3 C_unrestricted: ΔE=-0.24621, ΔF=0.0955, params 12 vs QAOA 6.
- N=12 L=4 A: ΔE=-0.16033, ΔF=0.0853, params 14 vs QAOA 8.
- N=12 L=4 C: ΔE=-0.17549, ΔF=0.0928, params 15 vs QAOA 8.
- N=12 L=4 C_unrestricted: ΔE=-0.17464, ΔF=0.0901, params 15 vs QAOA 8.
- N=12 L=5 A: ΔE=-0.13363, ΔF=0.1278, params 17 vs QAOA 10.
- N=12 L=5 C: ΔE=-0.13537, ΔF=0.1265, params 18 vs QAOA 10.
- N=12 L=6 A: ΔE=-0.11987, ΔF=0.1479, params 20 vs QAOA 12.
- N=12 L=6 C: ΔE=-0.13388, ΔF=0.1503, params 21 vs QAOA 12.

## Cells that also beat QAOA at L+1 on both

- N=6 L=1 A: ΔE(L+1)=-0.10231, ΔF(L+1)=0.0421.
- N=6 L=1 C: ΔE(L+1)=-0.10453, ΔF(L+1)=0.0454.
- N=6 L=2 A: ΔE(L+1)=-0.00960, ΔF(L+1)=0.0437.
- N=6 L=2 C: ΔE(L+1)=-0.01101, ΔF(L+1)=0.0432.
- N=6 L=3 A: ΔE(L+1)=-0.15033, ΔF(L+1)=0.1888.
- N=6 L=3 C: ΔE(L+1)=-0.21080, ΔF(L+1)=0.3329.
- N=6 L=4 D: ΔE(L+1)=-0.02297, ΔF(L+1)=0.1454.
- N=6 L=4 A: ΔE(L+1)=-0.39723, ΔF(L+1)=0.4293.
- N=6 L=4 C: ΔE(L+1)=-0.39812, ΔF(L+1)=0.4301.
- N=6 L=5 D: ΔE(L+1)=-0.17960, ΔF(L+1)=0.2322.
- N=6 L=5 A: ΔE(L+1)=-0.30368, ΔF(L+1)=0.3088.
- N=6 L=5 C: ΔE(L+1)=-0.30395, ΔF(L+1)=0.3092.
- N=6 L=6 D: ΔE(L+1)=-0.13287, ΔF(L+1)=0.1748.
- N=6 L=6 A: ΔE(L+1)=-0.25989, ΔF(L+1)=0.2454.
- N=6 L=6 C: ΔE(L+1)=-0.26030, ΔF(L+1)=0.2456.
- N=12 L=1 A: ΔE(L+1)=-0.01969, ΔF(L+1)=0.0044.
- N=12 L=1 C: ΔE(L+1)=-0.02407, ΔF(L+1)=0.0039.
- N=12 L=1 C_unrestricted: ΔE(L+1)=-0.02123, ΔF(L+1)=0.0042.
- N=12 L=3 A: ΔE(L+1)=-0.06766, ΔF(L+1)=0.0472.
- N=12 L=3 C: ΔE(L+1)=-0.08492, ΔF(L+1)=0.0511.
- N=12 L=3 C_unrestricted: ΔE(L+1)=-0.07817, ΔF(L+1)=0.0507.
- N=12 L=4 A: ΔE(L+1)=-0.03659, ΔF(L+1)=0.0422.
- N=12 L=4 C: ΔE(L+1)=-0.05175, ΔF(L+1)=0.0497.
- N=12 L=4 C_unrestricted: ΔE(L+1)=-0.05090, ΔF(L+1)=0.0470.
- N=12 L=5 A: ΔE(L+1)=-0.05364, ΔF(L+1)=0.0915.
- N=12 L=5 C: ΔE(L+1)=-0.05538, ΔF(L+1)=0.0902.
- N=12 L=6 A: ΔE(L+1)=-0.05248, ΔF(L+1)=0.1093.
- N=12 L=6 C: ΔE(L+1)=-0.06650, ΔF(L+1)=0.1117.

Parameter-matched QAOA depth is `ceil(n_params/2)`. Where that depth was not run, the table leaves the L+1 columns as the only extra-depth check and does not invent a matched-parameter win.

## How to read this

Same-depth wins are not parameter-matched wins. QAOA at depth ceil(n_params/2) is the matched-parameter comparator when that depth was actually run (through L=7). A negative ΔE is a lower energy than that QAOA. Fidelity is only the best-energy state.

| N | candidate | L | params | E−E0 | F | matched QAOA L | QAOA E−E0 | QAOA F | both better? |
|---|---|---|---|---|---|---|---|---|---|
| 6 | D | 1 | 4 | 1.21898 | 0.2048 | 2 | 0.88531 | 0.2967 | no |
| 6 | A | 1 | 5 | 0.78299 | 0.3388 | 3 | 0.67733 | 0.3823 | no |
| 6 | C | 1 | 6 | 0.78078 | 0.3421 | 3 | 0.67733 | 0.3823 | no |
| 6 | D | 2 | 7 | 1.06600 | 0.5407 | 4 | 0.52615 | 0.4843 | no |
| 6 | A | 2 | 8 | 0.66773 | 0.4260 | 4 | 0.52615 | 0.4843 | no |
| 6 | C | 2 | 9 | 0.66632 | 0.4255 | 5 | 0.41541 | 0.5613 | no |
| 6 | D | 3 | 10 | 0.73636 | 0.3975 | 5 | 0.41541 | 0.5613 | no |
| 6 | A | 3 | 11 | 0.37582 | 0.6730 | 6 | 0.32658 | 0.6760 | no |
| 6 | C | 3 | 12 | 0.31535 | 0.8172 | 6 | 0.32658 | 0.6760 | yes |
| 6 | D | 4 | 13 | 0.39244 | 0.7067 | 7 | 0.26369 | 0.7530 | no |
| 6 | A | 4 | 14 | 0.01818 | 0.9906 | 7 | 0.26369 | 0.7530 | yes |
| 6 | C | 4 | 15 | 0.01729 | 0.9913 | 8 | not run | not run | no |
| 6 | D | 5 | 16 | 0.14697 | 0.9082 | 8 | not run | not run | no |
| 6 | A | 5 | 17 | 0.02290 | 0.9848 | 9 | not run | not run | no |
| 6 | C | 5 | 18 | 0.02263 | 0.9853 | 9 | not run | not run | no |
| 6 | D | 6 | 19 | 0.13082 | 0.9278 | 10 | not run | not run | no |
| 6 | A | 6 | 20 | 0.00381 | 0.9984 | 10 | not run | not run | no |
| 6 | C | 6 | 21 | 0.00339 | 0.9986 | 11 | not run | not run | no |
| 12 | D | 1 | 4 | 1.77213 | 0.0813 | 2 | 1.18155 | 0.1333 | no |
| 12 | A | 1 | 5 | 1.16186 | 0.1377 | 3 | 0.92807 | 0.1705 | no |
| 12 | C | 1 | 6 | 1.15748 | 0.1372 | 3 | 0.92807 | 0.1705 | no |
| 12 | D | 2 | 7 | 1.57498 | 0.0948 | 4 | 0.76003 | 0.2153 | no |
| 12 | A | 2 | 8 | 1.05600 | 0.1629 | 4 | 0.76003 | 0.2153 | no |
| 12 | C | 2 | 9 | 1.02874 | 0.1640 | 5 | 0.63629 | 0.2584 | no |
| 12 | D | 3 | 10 | 1.09454 | 0.1468 | 5 | 0.63629 | 0.2584 | no |
| 12 | A | 3 | 11 | 0.69238 | 0.2625 | 6 | 0.55630 | 0.2947 | no |
| 12 | C | 3 | 12 | 0.67511 | 0.2664 | 6 | 0.55630 | 0.2947 | no |
| 12 | D | 4 | 13 | 0.98142 | 0.1652 | 7 | 0.48892 | 0.3333 | no |
| 12 | A | 4 | 14 | 0.59970 | 0.3006 | 7 | 0.48892 | 0.3333 | no |
| 12 | C | 4 | 15 | 0.58454 | 0.3081 | 8 | not run | not run | no |
| 12 | D | 5 | 16 | 0.86261 | 0.2014 | 8 | not run | not run | no |
| 12 | A | 5 | 17 | 0.50266 | 0.3862 | 9 | not run | not run | no |
| 12 | C | 5 | 18 | 0.50092 | 0.3849 | 9 | not run | not run | no |
| 12 | D | 6 | 19 | 0.76298 | 0.2239 | 10 | not run | not run | no |
| 12 | A | 6 | 20 | 0.43644 | 0.4426 | 10 | not run | not run | no |
| 12 | C | 6 | 21 | 0.42242 | 0.4450 | 11 | not run | not run | no |

Parameter-matched wins on both energy and fidelity (best energy state):
- N=6 C L=3 versus QAOA L=6.
- N=6 A L=4 versus QAOA L=7.

Family C versus Family A (theta_C free versus theta_C fixed at 0). Small gaps mean the star is not carrying the result.

| N | L | A E−E0 | C E−E0 | C−A energy | A F | C F | C theta_C | C best start |
|---|---|---|---|---|---|---|---|---|
| 6 | 1 | 0.78299 | 0.78078 | -0.00222 | 0.3388 | 0.3421 | 0.0145 | gc1_top3 |
| 6 | 2 | 0.66773 | 0.66632 | -0.00141 | 0.4260 | 0.4255 | -0.0160 | random6 |
| 6 | 3 | 0.37582 | 0.31535 | -0.06047 | 0.6730 | 0.8172 | -0.0682 | random6 |
| 6 | 4 | 0.01818 | 0.01729 | -0.00089 | 0.9906 | 0.9913 | 0.0008 | nested_A_thetaC0 |
| 6 | 5 | 0.02290 | 0.02263 | -0.00027 | 0.9848 | 0.9853 | 0.0005 | nested_A_thetaC0 |
| 6 | 6 | 0.00381 | 0.00339 | -0.00041 | 0.9984 | 0.9986 | 0.0008 | nested_A_thetaC0 |
| 12 | 1 | 1.16186 | 1.15748 | -0.00439 | 0.1377 | 0.1372 | -0.0214 | random11 |
| 12 | 2 | 1.05600 | 1.02874 | -0.02725 | 0.1629 | 0.1640 | -0.0701 | perturb_std0.08 |
| 12 | 3 | 0.69238 | 0.67511 | -0.01727 | 0.2625 | 0.2664 | -0.0235 | nested_D_theta0 |
| 12 | 4 | 0.59970 | 0.58454 | -0.01516 | 0.3006 | 0.3081 | -0.0407 | nested_A_thetaC0 |
| 12 | 5 | 0.50266 | 0.50092 | -0.00174 | 0.3862 | 0.3849 | -0.0064 | nested_A_thetaC0 |
| 12 | 6 | 0.43644 | 0.42242 | -0.01402 | 0.4426 | 0.4450 | -0.0290 | bootstrap:shell cubic interpolation with verifie |

Medians are over the multi-start energies, not over fidelities. A best-energy win whose median energy is worse than QAOA's best energy is a single-basin result.

N=6 operator check: full-space regression of F_star on all two-body Z strings plus identity has R² 1.110e-16 (orthogonal, not a two-body rewrite). On Néel single flips, R² against 1+Hc+F_K is 1.0 (residual ratio 7.725274058162e-13, n=12). On single, pair, and cavity-block flips together, residual ratio is 0.7020437979848281 (R² 0.5071345057110381). GS-weighted residual ratio is 0.26073333901692847 (R² 0.9320181259250835); that weight is diagnostic only.
N=12 operator check: full-space regression of F_star on all two-body Z strings plus identity has R² 2.220e-16 (orthogonal, not a two-body rewrite). On Néel single flips, R² against 1+Hc+F_K is 0.44533698596577564 (residual ratio 0.7447570167740778, n=24). On single, pair, and cavity-block flips together, residual ratio is 0.6892857234493955 (R² 0.5248851914488435). GS-weighted residual ratio is 0.3339163573703222 (R² 0.8884998662805352); that weight is diagnostic only.

Frozen transfer uses no N=12 training. Primary map only:
- A: L1 F=0.147 (retrained 0.13772612103894352); L2 F=0.184 (retrained 0.1628707494505047); L3 F=0.313 (retrained 0.2625065416794494); L4 F=0.001 (retrained 0.30060410160792556); L5 F=0.000 (retrained 0.38616319745486716); L6 F=0.000 (retrained 0.44261583385211767)
- C: L1 F=0.148 (retrained 0.1372122198885242); L2 F=0.184 (retrained 0.1640181618122937); L3 F=0.131 (retrained 0.266372611310075); L4 F=0.001 (retrained 0.30810092245142); L5 F=0.000 (retrained 0.384861591844735); L6 F=0.000 (retrained 0.44493273233326913)
- qaoa: L1 F=0.079 (retrained 0.08092758618268102); L2 F=0.127 (retrained 0.1333485917093669); L3 F=0.170 (retrained 0.17053640541315773); L4 F=0.233 (retrained 0.2152791718091483); L5 F=0.286 (retrained 0.2583777810792306); L6 F=0.331 (retrained 0.2946766418659755)
- D: L1 F=0.080 (retrained 0.08127603452836552); L2 F=0.005 (retrained 0.09478391396017076); L3 F=0.168 (retrained 0.14683970444345096); L4 F=0.132 (retrained 0.1651725243079838); L5 F=0.022 (retrained 0.20138998484137305); L6 F=0.028 (retrained 0.2238614786268904)

Large |theta_K| with a near-zero theta_C means the energy fit wants the two-body K feature, not the four-body star. That does not confirm the Schrieffer-Wolff expansion; the entropy-peak field is outside the regime where that expansion would be controlled, and theta_C was trained rather than fixed.

The entropy maximum is flat. On the coarse grid the best N=6 entropies differ by about 1e-5 nats between B=-0.17 and B=-0.175, and the frozen value sits near ln 2. B is the argmax on the 1e-4 refinement, not a sharply resolved critical field. The gap quoted above is from a dense diagonalization at that B. Coarse-scan gaps at N=12 are not used; a two-eigenvalue sparse solve did not resolve the first excited state.

Points whose analytic gradient norm exceeded 1e-3 were continued with L-BFGS-B (maxiter 400, maxfun 4000). That continuation is recorded as cell['polish'] when it changed the saved point, and its evaluations are included in nfev_sum.

## Frozen N=6 to N=12 transfer

Primary map, fixed before looking at N=12 fidelity: shell controls copied, theta copied as a dimensionless HS weight, gamma multiplied by κ_6/κ_12. Two pre-specified sensitivities are also reported: unscaled gamma, and theta rescaled so the raw monomial prefactor matches. Neither map is chosen after seeing the score, and neither is used to start N=12 training.

| N6 L | method | mode | E−E0 | F | retrained E−E0 | retrained F |
|---|---|---|---|---|---|---|
| 1 | qaoa | primary_kappa_theta_copy | 1.80084 | 0.0794 | 1.79433 | 0.0809 |
| 1 | qaoa | unscaled_gamma | 2.03741 | 0.0880 | 1.79433 | 0.0809 |
| 1 | qaoa | alt_raw_prefactor | 1.80084 | 0.0794 | 1.79433 | 0.0809 |
| 1 | shared_linear | primary_kappa_theta_copy | 1.80084 | 0.0794 | 1.79433 | 0.0809 |
| 1 | shared_linear | unscaled_gamma | 2.03741 | 0.0880 | 1.79433 | 0.0809 |
| 1 | shared_linear | alt_raw_prefactor | 1.80084 | 0.0794 | 1.79433 | 0.0809 |
| 1 | D | primary_kappa_theta_copy | 1.78137 | 0.0797 | 1.77213 | 0.0813 |
| 1 | D | unscaled_gamma | 2.06934 | 0.0865 | 1.77213 | 0.0813 |
| 1 | D | alt_raw_prefactor | 1.78137 | 0.0797 | 1.77213 | 0.0813 |
| 1 | A | primary_kappa_theta_copy | 1.22532 | 0.1472 | 1.16186 | 0.1377 |
| 1 | A | unscaled_gamma | 1.68151 | 0.1469 | 1.16186 | 0.1377 |
| 1 | A | alt_raw_prefactor | 2.58837 | 0.1334 | 1.16186 | 0.1377 |
| 1 | C | primary_kappa_theta_copy | 1.24296 | 0.1483 | 1.15748 | 0.1372 |
| 1 | C | unscaled_gamma | 1.71480 | 0.1480 | 1.15748 | 0.1372 |
| 1 | C | alt_raw_prefactor | 2.68020 | 0.1331 | 1.15748 | 0.1372 |
| 2 | qaoa | primary_kappa_theta_copy | 1.19548 | 0.1275 | 1.18155 | 0.1333 |
| 2 | qaoa | unscaled_gamma | 1.58234 | 0.1358 | 1.18155 | 0.1333 |
| 2 | qaoa | alt_raw_prefactor | 1.19548 | 0.1275 | 1.18155 | 0.1333 |
| 2 | shared_linear | primary_kappa_theta_copy | 1.60502 | 0.0919 | 1.59956 | 0.0948 |
| 2 | shared_linear | unscaled_gamma | 1.74791 | 0.1021 | 1.59956 | 0.0948 |
| 2 | shared_linear | alt_raw_prefactor | 1.60502 | 0.0919 | 1.59956 | 0.0948 |
| 2 | D | primary_kappa_theta_copy | 8.44002 | 0.0050 | 1.57498 | 0.0948 |
| 2 | D | unscaled_gamma | 8.38183 | 0.0002 | 1.57498 | 0.0948 |
| 2 | D | alt_raw_prefactor | 8.44002 | 0.0050 | 1.57498 | 0.0948 |
| 2 | A | primary_kappa_theta_copy | 1.13262 | 0.1840 | 1.05600 | 0.1629 |
| 2 | A | unscaled_gamma | 1.40869 | 0.1835 | 1.05600 | 0.1629 |
| 2 | A | alt_raw_prefactor | 3.60641 | 0.1573 | 1.05600 | 0.1629 |
| 2 | C | primary_kappa_theta_copy | 1.11706 | 0.1839 | 1.02874 | 0.1640 |
| 2 | C | unscaled_gamma | 1.39214 | 0.1829 | 1.02874 | 0.1640 |
| 2 | C | alt_raw_prefactor | 3.54460 | 0.1590 | 1.02874 | 0.1640 |
| 3 | qaoa | primary_kappa_theta_copy | 0.93332 | 0.1699 | 0.92807 | 0.1705 |
| 3 | qaoa | unscaled_gamma | 1.27541 | 0.1741 | 0.92807 | 0.1705 |
| 3 | qaoa | alt_raw_prefactor | 0.93332 | 0.1699 | 0.92807 | 0.1705 |
| 3 | shared_linear | primary_kappa_theta_copy | 1.19926 | 0.1320 | 1.18567 | 0.1382 |
| 3 | shared_linear | unscaled_gamma | 1.58403 | 0.1408 | 1.18567 | 0.1382 |
| 3 | shared_linear | alt_raw_prefactor | 1.19926 | 0.1320 | 1.18567 | 0.1382 |
| 3 | D | primary_kappa_theta_copy | 1.43081 | 0.1683 | 1.09454 | 0.1468 |
| 3 | D | unscaled_gamma | 2.30984 | 0.1652 | 1.09454 | 0.1468 |
| 3 | D | alt_raw_prefactor | 1.43081 | 0.1683 | 1.09454 | 0.1468 |
| 3 | A | primary_kappa_theta_copy | 0.89861 | 0.3127 | 0.69238 | 0.2625 |
| 3 | A | unscaled_gamma | 1.71308 | 0.2815 | 0.69238 | 0.2625 |
| 3 | A | alt_raw_prefactor | 1.27388 | 0.3028 | 0.69238 | 0.2625 |
| 3 | C | primary_kappa_theta_copy | 6.29298 | 0.1313 | 0.67511 | 0.2664 |
| 3 | C | unscaled_gamma | 7.33561 | 0.0656 | 0.67511 | 0.2664 |
| 3 | C | alt_raw_prefactor | 8.57836 | 0.0295 | 0.67511 | 0.2664 |
| 4 | qaoa | primary_kappa_theta_copy | 0.80284 | 0.2335 | 0.76003 | 0.2153 |
| 4 | qaoa | unscaled_gamma | 1.70706 | 0.1900 | 0.76003 | 0.2153 |
| 4 | qaoa | alt_raw_prefactor | 0.80284 | 0.2335 | 0.76003 | 0.2153 |
| 4 | shared_linear | primary_kappa_theta_copy | 1.04899 | 0.1542 | 1.04406 | 0.1564 |
| 4 | shared_linear | unscaled_gamma | 1.46387 | 0.1649 | 1.04406 | 0.1564 |
| 4 | shared_linear | alt_raw_prefactor | 1.04899 | 0.1542 | 1.04406 | 0.1564 |
| 4 | D | primary_kappa_theta_copy | 4.89849 | 0.1322 | 0.98142 | 0.1652 |
| 4 | D | unscaled_gamma | 4.85411 | 0.1512 | 0.98142 | 0.1652 |
| 4 | D | alt_raw_prefactor | 4.89849 | 0.1322 | 0.98142 | 0.1652 |
| 4 | A | primary_kappa_theta_copy | 8.85689 | 0.0006 | 0.59970 | 0.3006 |
| 4 | A | unscaled_gamma | 8.84602 | 0.0025 | 0.59970 | 0.3006 |
| 4 | A | alt_raw_prefactor | 8.07288 | 0.0010 | 0.59970 | 0.3006 |
| 4 | C | primary_kappa_theta_copy | 8.86615 | 0.0007 | 0.58454 | 0.3081 |
| 4 | C | unscaled_gamma | 8.85610 | 0.0024 | 0.58454 | 0.3081 |
| 4 | C | alt_raw_prefactor | 8.08324 | 0.0010 | 0.58454 | 0.3081 |
| 5 | qaoa | primary_kappa_theta_copy | 0.74118 | 0.2856 | 0.63629 | 0.2584 |
| 5 | qaoa | unscaled_gamma | 1.85999 | 0.2150 | 0.63629 | 0.2584 |
| 5 | qaoa | alt_raw_prefactor | 0.74118 | 0.2856 | 0.63629 | 0.2584 |
| 5 | shared_linear | primary_kappa_theta_copy | 0.95218 | 0.1768 | 0.94128 | 0.1716 |
| 5 | shared_linear | unscaled_gamma | 1.32714 | 0.1853 | 0.94128 | 0.1716 |
| 5 | shared_linear | alt_raw_prefactor | 0.95218 | 0.1768 | 0.94128 | 0.1716 |
| 5 | D | primary_kappa_theta_copy | 8.77588 | 0.0219 | 0.86261 | 0.2014 |
| 5 | D | unscaled_gamma | 6.84108 | 0.0530 | 0.86261 | 0.2014 |
| 5 | D | alt_raw_prefactor | 8.77588 | 0.0219 | 0.86261 | 0.2014 |
| 5 | A | primary_kappa_theta_copy | 8.91776 | 0.0004 | 0.50266 | 0.3862 |
| 5 | A | unscaled_gamma | 8.78586 | 0.0025 | 0.50266 | 0.3862 |
| 5 | A | alt_raw_prefactor | 8.32019 | 0.0007 | 0.50266 | 0.3862 |
| 5 | C | primary_kappa_theta_copy | 8.92028 | 0.0004 | 0.50092 | 0.3849 |
| 5 | C | unscaled_gamma | 8.85572 | 0.0018 | 0.50092 | 0.3849 |
| 5 | C | alt_raw_prefactor | 8.28864 | 0.0004 | 0.50092 | 0.3849 |
| 6 | qaoa | primary_kappa_theta_copy | 0.90074 | 0.3314 | 0.55630 | 0.2947 |
| 6 | qaoa | unscaled_gamma | 2.66676 | 0.2389 | 0.55630 | 0.2947 |
| 6 | qaoa | alt_raw_prefactor | 0.90074 | 0.3314 | 0.55630 | 0.2947 |
| 6 | shared_linear | primary_kappa_theta_copy | 0.88471 | 0.1890 | 0.87809 | 0.1878 |
| 6 | shared_linear | unscaled_gamma | 1.16429 | 0.1938 | 0.87809 | 0.1878 |
| 6 | shared_linear | alt_raw_prefactor | 0.88471 | 0.1890 | 0.87809 | 0.1878 |
| 6 | D | primary_kappa_theta_copy | 7.69383 | 0.0279 | 0.76299 | 0.2239 |
| 6 | D | unscaled_gamma | 6.35663 | 0.0997 | 0.76299 | 0.2239 |
| 6 | D | alt_raw_prefactor | 7.69383 | 0.0279 | 0.76299 | 0.2239 |
| 6 | A | primary_kappa_theta_copy | 8.72507 | 0.0002 | 0.43644 | 0.4426 |
| 6 | A | unscaled_gamma | 8.94984 | 0.0049 | 0.43644 | 0.4426 |
| 6 | A | alt_raw_prefactor | 8.12624 | 0.0022 | 0.43644 | 0.4426 |
| 6 | C | primary_kappa_theta_copy | 8.70422 | 0.0001 | 0.42242 | 0.4449 |
| 6 | C | unscaled_gamma | 8.74736 | 0.0037 | 0.42242 | 0.4449 |
| 6 | C | alt_raw_prefactor | 8.10066 | 0.0012 | 0.42242 | 0.4449 |

## Failures and incomplete cells

No skipped cells were recorded.

## Provenance

- code hash `54b06879f9481c04511a4e0160dd6b466b9e4069d4474f509a306627c06ed176`
- wall time 119.39040104700007 s
- python 3.12.3 numpy 2.5.4 scipy 1.18.1
- host Linux-6.12.113+-x86_64-with-glibc2.39
- Artifacts: `results/cells/`, `results/states/`, `results/targets/`, `results/features/`, `results/figures/`, `results/comparison.json`, `results/transfer.json`, `results/cells_all.json`.

