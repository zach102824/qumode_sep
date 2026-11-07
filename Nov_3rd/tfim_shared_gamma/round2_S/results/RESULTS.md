# Round 2 family S

Gap correction: on the frozen N=12 target the dense full-space gap is 1.613234e-03 and the P=R=+1 gap is 4.337981e-01. Round 1's N=12 figure 0.433 was the accessible gap. The partner at the full gap has P=R=-1. Fidelity uses the full-space ground state, which is in P=R=+1.

Family S trains one gamma, one thetaIn and one thetaK. thetaIn=1 is inter-K A, Hc plus a normalized inter-only FK, not the original full-K A. Both controls are in the table.

Equal depth, both energy and fidelity versus standard QAOA: 5 of 6 N=12 depths. That ledger gives S 3L+3 parameters against QAOA's 2L, and the intra phases are level-resolved SNAP rather than the shared Hamming shell.

Parameter-matched standard QAOA, depth ceil((3L+3)/2): ahead at none; not ahead at L1, L2, L3, L4, L5, L6.

Versus QAOA at L+1, the both-metric edge fails at L1, L2, L3, L4, L5 and still holds at L6. That depth still loses the matched-depth energy comparison.

same-G QAOA uses independent gamma, a linear mixer, and the same shared thetaIn and thetaK. It is a feature-versus-shell diagnostic. It is not standard QAOA and not a rigorous ceiling; a local optimizer miss would not prove that shared gamma cannot win.

thetaIn is the trained ratio of intra couplings to the fixed inter Hc coefficient. Near 1, family S has collapsed to inter-K A. A large value is an intra-heavy profile, not a free scale gauge. The reflection-symmetric shuffled intra control loses to the physical intra pattern. Full-K A is a separate control; thetaIn=1 does not recover it.

The c2 commutation claim is false. One XX and one ZZ anticommute when they share a single site, so the intra-cavity sum of XX does not commute with intra ZZ. XXX does commute with intra ZZ (two overlaps) and anticommutes with an inter ZZ that touches the cavity once.

Seed medians include failed starts and are not an error bar on the best energy. Best and median are both stored. No seal is claimed.

## Frozen spectra

| N | B | E0 | full gap | accessible gap | P | R | residual |
|---|---|---|---|---|---|---|---|
| 12 | -0.3444 | -8.613202 | 1.613234e-03 | 4.337981e-01 | 1.000 | 1.000 | 1.70e-14 |
| 6 | -0.1708 | -3.843349 | 5.053821e-04 | 7.047690e-01 | 1.000 | 1.000 | 3.80e-15 |

Manifest SHA-256 `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`.

## N=12 depth table

| L | method | P | E | E-E0 | F | median | best | thetaIn | thetaK | evals |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | family_S | 6 | -7.27078 | 1.3424e+00 | 0.1095 | -7.2708 | -7.2708 | 0.748 | -0.434 | 1031 |
| 1 | full_K_A | 5 | -7.45134 | 1.1619e+00 | 0.1377 | -7.4513 | -7.4513 |  | -0.527 | 727 |
| 1 | inter_K_A | 5 | -7.24663 | 1.3666e+00 | 0.1062 | -7.2466 | -7.2466 |  | -0.414 | 602 |
| 1 | qaoa | 2 | -6.81888 | 1.7943e+00 | 0.0809 | -6.8189 | -6.8189 |  |  | 2194 |
| 1 | same_G_qaoa | 4 | -7.25597 | 1.3572e+00 | 0.1086 | -7.2560 | -7.2560 | 0.762 | -0.429 | 1315 |
| 1 | shuffled_intra | 6 | -6.78560 | 1.8276e+00 | 0.0637 | -6.7856 | -6.7856 | 0.556 | -0.415 | 999 |
| 1 | thetaK0 | 5 | -7.03881 | 1.5744e+00 | 0.0899 | -7.0388 | -7.0388 | 2.289 |  | 769 |
| 2 | family_S | 9 | -7.38115 | 1.2321e+00 | 0.1240 | -7.3811 | -7.3811 | 0.987 | -0.443 | 970 |
| 2 | full_K_A | 8 | -7.55720 | 1.0560e+00 | 0.1629 | -7.5222 | -7.5572 |  | -0.574 | 809 |
| 2 | inter_K_A | 8 | -7.38112 | 1.2321e+00 | 0.1240 | -7.3811 | -7.3811 |  | -0.442 | 820 |
| 2 | qaoa | 4 | -7.43165 | 1.1816e+00 | 0.1333 | -0.5726 | -7.4317 |  |  | 1011 |
| 2 | same_G_qaoa | 6 | -7.66932 | 9.4388e-01 | 0.2156 | -7.6344 | -7.6693 | 2.288 | -0.439 | 1274 |
| 2 | shuffled_intra | 9 | -6.88527 | 1.7279e+00 | 0.0733 | -6.8232 | -6.8853 | 1.378 | -0.416 | 860 |
| 2 | thetaK0 | 8 | -7.15210 | 1.4611e+00 | 0.1006 | -7.1521 | -7.1521 | 2.377 |  | 977 |
| 3 | family_S | 12 | -7.83604 | 7.7716e-01 | 0.2441 | -7.8106 | -7.8360 | 1.660 | -0.432 | 887 |
| 3 | full_K_A | 11 | -7.92083 | 6.9238e-01 | 0.2625 | -7.9208 | -7.9208 |  | -0.487 | 830 |
| 3 | inter_K_A | 11 | -7.76932 | 8.4389e-01 | 0.2145 | -7.7408 | -7.7693 |  | -0.432 | 814 |
| 3 | qaoa | 6 | -7.68513 | 9.2807e-01 | 0.1705 | -0.4553 | -7.6851 |  |  | 3553 |
| 3 | same_G_qaoa | 8 | -7.89903 | 7.1417e-01 | 0.2404 | -0.5999 | -7.8990 | 1.217 | -0.391 | 1250 |
| 3 | shuffled_intra | 12 | -7.08315 | 1.5301e+00 | 0.0816 | -6.9725 | -7.0831 | 0.961 | -0.414 | 877 |
| 3 | thetaK0 | 11 | -7.66628 | 9.4693e-01 | 0.2020 | -7.5643 | -7.6663 | 3.669 |  | 832 |
| 4 | family_S | 15 | -7.91670 | 6.9650e-01 | 0.2674 | -3.9230 | -7.9167 | 1.423 | -0.448 | 939 |
| 4 | full_K_A | 14 | -8.01350 | 5.9970e-01 | 0.3006 | -7.8742 | -8.0135 |  | -0.505 | 910 |
| 4 | inter_K_A | 14 | -7.88059 | 7.3261e-01 | 0.2488 | -7.4624 | -7.8806 |  | -0.445 | 892 |
| 4 | qaoa | 8 | -7.85317 | 7.6003e-01 | 0.2153 | -0.5587 | -7.8532 |  |  | 4367 |
| 4 | same_G_qaoa | 10 | -7.98692 | 6.2628e-01 | 0.2768 | -7.4859 | -7.9869 | 1.126 | -0.338 | 1511 |
| 4 | shuffled_intra | 15 | -7.13490 | 1.4783e+00 | 0.0892 | -7.0079 | -7.1349 | 0.863 | -0.404 | 913 |
| 4 | thetaK0 | 14 | -7.72617 | 8.8703e-01 | 0.2079 | -7.6357 | -7.7262 | 2.401 |  | 892 |
| 5 | family_S | 18 | -8.00314 | 6.1007e-01 | 0.3193 | -3.5563 | -8.0031 | 1.223 | -0.459 | 887 |
| 5 | full_K_A | 17 | -8.11054 | 5.0266e-01 | 0.3861 | -0.8829 | -8.1105 |  | -0.570 | 868 |
| 5 | inter_K_A | 17 | -7.99051 | 6.2269e-01 | 0.3089 | -0.8581 | -7.9905 |  | -0.461 | 885 |
| 5 | qaoa | 10 | -7.97691 | 6.3629e-01 | 0.2584 | -0.6201 | -7.9769 |  |  | 3880 |
| 5 | same_G_qaoa | 12 | -8.01304 | 6.0017e-01 | 0.2918 | -0.7206 | -8.0130 | 1.085 | -0.346 | 1484 |
| 5 | shuffled_intra | 18 | -7.15210 | 1.4611e+00 | 0.0858 | -7.1085 | -7.1521 | 0.615 | -0.426 | 916 |
| 5 | thetaK0 | 17 | -7.94181 | 6.7139e-01 | 0.4343 | -7.6241 | -7.9418 | 8.281 |  | 1019 |
| 6 | family_S | 21 | -8.24719 | 3.6602e-01 | 0.5591 | -7.9425 | -8.2472 | 3.827 | -0.391 | 993 |
| 6 | full_K_A | 20 | -8.17677 | 4.3644e-01 | 0.4427 | -8.0878 | -8.1768 |  | -0.630 | 887 |
| 6 | inter_K_A | 20 | -8.05632 | 5.5688e-01 | 0.3412 | -7.9036 | -8.0563 |  | -0.469 | 821 |
| 6 | qaoa | 12 | -8.05690 | 5.5630e-01 | 0.2947 | -0.7228 | -8.0569 |  |  | 4552 |
| 6 | same_G_qaoa | 14 | -8.15799 | 4.5521e-01 | 0.3841 | -6.7766 | -8.1580 | 1.577 | -0.416 | 1548 |
| 6 | shuffled_intra | 21 | -7.23894 | 1.3743e+00 | 0.1083 | -7.0724 | -7.2389 | 0.737 | -0.408 | 870 |
| 6 | thetaK0 | 20 | -8.04746 | 5.6575e-01 | 0.4424 | -3.8183 | -8.0475 | 7.355 |  | 1010 |
| 7 | qaoa | 14 | -8.12428 | 4.8892e-01 | 0.3333 | -0.7752 | -8.1243 |  |  | 911 |
| 8 | qaoa | 16 | -8.17753 | 4.3567e-01 | 0.3728 | -0.8187 | -8.1775 |  |  | 975 |
| 9 | qaoa | 18 | -8.22287 | 3.9033e-01 | 0.4120 | -0.8412 | -8.2229 |  |  | 1176 |
| 10 | qaoa | 20 | -8.26375 | 3.4945e-01 | 0.4523 | -0.8989 | -8.2638 |  |  | 1169 |
| 11 | qaoa | 22 | -8.29822 | 3.1499e-01 | 0.4849 | -0.9458 | -8.2982 |  |  | 1193 |

## Candidate versus baselines

| L | dE QAOA | dF QAOA | both L | both L+1 | both matched | dE fullK | dE interK | dE theta0 | dE shuf | dE sameG |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -4.5190e-01 | 2.8601e-02 | True | False | False | 1.8056e-01 | -2.4153e-02 | -2.3197e-01 | -4.8518e-01 | -1.4815e-02 |
| 2 | 5.0505e-02 | -9.3316e-03 | False | False | False | 1.7606e-01 | -2.7125e-05 | -2.2905e-01 | -4.9588e-01 | 2.8818e-01 |
| 3 | -1.5091e-01 | 7.3537e-02 | True | False | False | 8.4787e-02 | -6.6721e-02 | -1.6976e-01 | -7.5289e-01 | 6.2989e-02 |
| 4 | -6.3531e-02 | 5.2081e-02 | True | False | False | 9.6802e-02 | -3.6105e-02 | -1.9053e-01 | -7.8180e-01 | 7.0220e-02 |
| 5 | -2.6230e-02 | 6.0915e-02 | True | False | False | 1.0740e-01 | -1.2629e-02 | -6.1323e-02 | -8.5103e-01 | 9.8996e-03 |
| 6 | -1.9029e-01 | 2.6441e-01 | True | True | False | -7.0420e-02 | -1.9087e-01 | -1.9973e-01 | -1.0082e+00 | -8.9197e-02 |

Negative dE means family S has lower energy than that baseline.

## Hardware ledger

All generators here are two-body. N=12 has 12 intra bonds and 54 inter bonds, and 6 cavity pairs. Each layer applies one shared-gamma inter diagonal (Hc_inter and FK_inter are one compiled inter profile once thetaK is fixed), plus one level-resolved 8-phase diagonal on each cavity for the intra part, plus one Hamming shell shared across cavities. The intra phases are not that shell: equal Hamming weight states in a cavity do not share an intra energy. This is not claimed to cost the same as native Hc.

