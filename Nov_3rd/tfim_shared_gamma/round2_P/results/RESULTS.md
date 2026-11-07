# Round 2 family P — cavity-parity cross term

Candidate `G = Hc + theta_K F_K + theta_P F_P` with one shared gamma and one shared theta vector. `F_P = sum_{a<b} Jbar_ab P_a P_b`, `P_a` the product of the three Z's in cavity a, `Jbar_ab` the mean of the original 3x3 block of J. This is a J-polynomial feature. It is not a derived effective Hamiltonian.

Manifest SHA256 `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Every row below uses this file. No new entropy scan.

## Spectrum

| N | B | E0 | full gap | sector gap | S (nats) | GS in P=R=+1 | hc hash |
|---|---|---|---|---|---|---|---|
| 6 | -0.1708 | -3.843349 | 5.053821e-04 | 7.047690e-01 | 0.6938 | True | `78555cdca184` |
| 12 | -0.3444 | -8.613202 | 1.613234e-03 | 4.337981e-01 | 0.7026 | True | `eef806253f93` |

The full gap is E1-E0 of the unrestricted Hamiltonian. The sector gap is the gap inside P=+1, R=+1. Fidelity uses the full-space ground state.

## Normalization and algebra

- N=6: RMS(Hc)=0.646109, scale_K=0.528837, scale_P=1.693092 (nonzero-only diagnostic scale 1.693092, not used). Six-body nonzeros 1 / 1. Shuffle identity: True. GS-weighted R² of F_P on 1+Hc+F_K: 0.8498225532151266 (diagnostic only).
- N=12: RMS(Hc)=0.474013, scale_K=0.368717, scale_P=20.631932 (nonzero-only diagnostic scale 1.662568, not used). Six-body nonzeros 6 / 924. Shuffle identity: False. GS-weighted R² of F_P on 1+Hc+F_K: 0.7354196145966743 (diagnostic only).

Checked algebra: P_a = ZZZ anticommutes with XXX on that cavity (three overlaps). P_a P_b has weight 6, so it preserves global parity. The claim that c2 sum XX commutes with every intra-cavity ZZ is not used; it is false in general.

## Hardware ledger

Hc already couples every cavity pair, so PaPb does not add a new pair. It changes the phase pattern inside each two-cavity diagonal from a ZZ polynomial to a pattern that includes the 6-qubit parity. Treating that as the same pair-gate cost as native Hc is an assumption, not a measured hardware result. QAOA applies L independent two-body cost unitaries and L linear mixers. Those are different ledgers from parameter count.

## Results

Equal-layer comparisons are reported as equal-layer results. They are not a quantum-advantage claim. Parameter-matched QAOA is p=ceil((3L+3)/2) for the primary ansatz.

| N | L | method | P | E−E0 | F | median E−E0 | vs QAOA L ΔE | vs QAOA L+1 ΔE | vs matched ΔE | thetaP or thetaS | grad | nfev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1 | qaoa | 2 | 1.25077 | 0.2022 | 1.25077 | — | — | — | — | 5.06e-09 | 148 |
| 6 | 1 | A | 5 | 0.78624 | 0.3377 | 0.78624 | -0.46453 | — | — | — | 2.47e-07 | 396 |
| 6 | 1 | P | 6 | 0.78624 | 0.3377 | 0.78624 | -0.46453 | — | — | -0.0018 | 5.35e-09 | 493 |
| 6 | 1 | Sblock | 6 | 0.60183 | 0.4206 | 0.60183 | -0.64894 | — | — | 0.3146 | 2.85e-07 | 482 |
| 6 | 1 | Pshuffle | 6 | 0.78624 | 0.3377 | 0.78624 | -0.46453 | — | — | -0.0018 | 2.44e-07 | 474 |
| 6 | 1 | P_thetaK0 | 5 | 1.22297 | 0.2038 | 1.22297 | -0.02780 | — | — | -0.0453 | 1.19e-07 | 435 |
| 6 | 1 | qaoa_sameG | 4 | 0.80549 | 0.3318 | 0.80549 | -0.44528 | — | — | 0.0022 | 4.68e-08 | 406 |
| 12 | 1 | qaoa | 2 | 1.79433 | 0.0809 | 1.79433 | — | — | — | — | 1.20e-08 | 903 |
| 12 | 1 | A | 5 | 1.16186 | 0.1377 | 1.16186 | -0.63246 | -0.01969 | 0.23379 | — | 6.90e-08 | 437 |
| 12 | 1 | P | 6 | 1.16183 | 0.1377 | 1.16183 | -0.63249 | -0.01972 | 0.23376 | -0.0004 | 1.08e-06 | 581 |
| 12 | 1 | Sblock | 6 | 1.14151 | 0.1461 | 1.14151 | -0.65281 | -0.04004 | 0.21344 | 0.1257 | 4.88e-08 | 494 |
| 12 | 1 | Pshuffle | 6 | 1.16183 | 0.1377 | 1.16183 | -0.63249 | -0.01972 | 0.23376 | -0.0004 | 5.73e-07 | 555 |
| 12 | 1 | P_thetaK0 | 5 | 1.77208 | 0.0813 | 1.77208 | -0.02224 | 0.59053 | 0.84401 | -0.0009 | 9.54e-09 | 508 |
| 12 | 1 | qaoa_sameG | 4 | 1.17579 | 0.1370 | 1.17579 | -0.61853 | -0.00576 | 0.24772 | 0.0000 | 4.12e-07 | 499 |
| 12 | 1 | P_indgamma | 6 | 1.16183 | 0.1377 | 5.20361 | -0.63249 | -0.01972 | 0.23376 | -0.0004 | 1.08e-06 | 547 |
| 12 | 2 | qaoa | 4 | 1.18155 | 0.1333 | 1.18155 | — | — | — | — | 5.85e-09 | 1006 |
| 12 | 2 | A | 8 | 1.05600 | 0.1629 | 1.05600 | -0.12555 | 0.12792 | 0.41970 | — | 7.00e-07 | 460 |
| 12 | 2 | P | 9 | 1.04803 | 0.1626 | 1.04803 | -0.13352 | 0.11996 | 0.41174 | 0.0125 | 2.89e-07 | 513 |
| 12 | 2 | Sblock | 9 | 1.01469 | 0.1829 | 1.01517 | -0.16686 | 0.08661 | 0.37839 | 0.1919 | 9.45e-07 | 561 |
| 12 | 2 | Pshuffle | 9 | 1.04808 | 0.1626 | 1.04808 | -0.13347 | 0.12001 | 0.41179 | 0.0124 | 2.08e-06 | 567 |
| 12 | 2 | P_thetaK0 | 8 | 1.57447 | 0.0948 | 1.57447 | 0.39292 | 0.64640 | 0.93818 | 0.0059 | 4.82e-07 | 492 |
| 12 | 2 | qaoa_sameG | 6 | 0.76353 | 0.2308 | 2.06778 | -0.41802 | -0.16454 | 0.12724 | 0.0012 | 1.42e-06 | 403 |
| 12 | 2 | P_indgamma | 10 | 0.71614 | 0.2403 | 0.71614 | -0.46541 | -0.21193 | 0.07985 | -0.0002 | 3.02e-06 | 672 |
| 12 | 3 | qaoa | 6 | 0.92807 | 0.1705 | 0.92807 | — | — | — | — | 1.62e-07 | 1646 |
| 12 | 3 | A | 11 | 0.69238 | 0.2625 | 0.76264 | -0.23570 | -0.06766 | 0.13607 | — | 8.52e-07 | 620 |
| 12 | 3 | P | 12 | 0.69238 | 0.2625 | 0.69247 | -0.23570 | -0.06766 | 0.13607 | 0.0000 | 7.35e-06 | 628 |
| 12 | 3 | Sblock | 12 | 0.66265 | 0.2853 | 0.71110 | -0.26543 | -0.09739 | 0.10635 | 0.1933 | 3.27e-06 | 670 |
| 12 | 3 | Pshuffle | 12 | 0.69238 | 0.2625 | 0.77651 | -0.23570 | -0.06766 | 0.13607 | 0.0000 | 5.70e-06 | 582 |
| 12 | 3 | P_thetaK0 | 11 | 1.09345 | 0.1470 | 1.09345 | 0.16537 | 0.33341 | 0.53714 | 0.0017 | 6.01e-06 | 676 |
| 12 | 3 | qaoa_sameG | 8 | 0.55432 | 0.3115 | 0.55432 | -0.37375 | -0.20571 | -0.00198 | 0.0003 | 3.13e-07 | 515 |
| 12 | 3 | P_indgamma | 14 | 0.50097 | 0.3577 | 0.50098 | -0.42710 | -0.25906 | -0.05533 | -0.0001 | 1.55e-05 | 716 |
| 12 | 4 | qaoa | 8 | 0.76003 | 0.2153 | 4.29654 | — | — | — | — | 8.17e-07 | 2177 |
| 12 | 4 | A | 14 | 0.59970 | 0.3006 | 0.60197 | -0.16033 | -0.03659 | 0.16403 | — | 1.63e-05 | 655 |
| 12 | 4 | P | 15 | 0.59750 | 0.3013 | 0.64986 | -0.16253 | -0.03879 | 0.16184 | 0.0031 | 4.26e-05 | 689 |
| 12 | 4 | Sblock | 15 | 0.57538 | 0.3285 | 0.57539 | -0.18465 | -0.06091 | 0.13971 | 0.1932 | 9.09e-06 | 664 |
| 12 | 4 | Pshuffle | 15 | 0.59743 | 0.3014 | 0.67399 | -0.16261 | -0.03887 | 0.16176 | 0.0031 | 5.31e-06 | 725 |
| 12 | 4 | P_thetaK0 | 14 | 0.97887 | 0.1658 | 0.97887 | 0.21884 | 0.34258 | 0.54320 | 0.0032 | 1.82e-05 | 723 |
| 12 | 4 | qaoa_sameG | 10 | 0.44931 | 0.3805 | 0.44932 | -0.31073 | -0.18699 | 0.01364 | 0.0003 | 5.65e-06 | 684 |
| 12 | 4 | P_indgamma | 18 | 0.37718 | 0.4637 | 0.39662 | -0.38285 | -0.25911 | -0.05849 | 0.0002 | 5.11e-06 | 742 |
| 12 | 5 | qaoa | 10 | 0.63629 | 0.2584 | 0.70562 | — | — | — | — | 6.89e-07 | 2482 |
| 12 | 5 | A | 17 | 0.50266 | 0.3861 | 0.50270 | -0.13363 | -0.05364 | 0.11233 | — | 4.13e-06 | 713 |
| 12 | 5 | P | 18 | 0.50264 | 0.3861 | 0.50903 | -0.13365 | -0.05366 | 0.11231 | 0.0002 | 1.82e-04 | 665 |
| 12 | 5 | Sblock | 18 | 0.47210 | 0.4335 | 0.47211 | -0.16419 | -0.08420 | 0.08178 | 0.1904 | 6.21e-06 | 732 |
| 12 | 5 | Pshuffle | 18 | 0.50263 | 0.3861 | 0.50807 | -0.13366 | -0.05367 | 0.11230 | 0.0003 | 1.51e-04 | 649 |
| 12 | 5 | P_thetaK0 | 17 | 0.86090 | 0.2010 | 0.86117 | 0.22461 | 0.30460 | 0.47058 | 0.0057 | 3.38e-06 | 729 |
| 12 | 5 | qaoa_sameG | 12 | 0.35723 | 0.4545 | 0.35724 | -0.27906 | -0.19907 | -0.03310 | 0.0002 | 3.74e-05 | 705 |
| 12 | 5 | P_indgamma | 22 | 0.28018 | 0.5811 | 0.28213 | -0.35612 | -0.27612 | -0.11015 | 0.0002 | 1.29e-05 | 976 |
| 12 | 6 | qaoa | 12 | 0.55630 | 0.2947 | 0.57289 | — | — | — | — | 1.03e-06 | 2958 |
| 12 | 6 | A | 20 | 0.43644 | 0.4426 | 0.43644 | -0.11987 | -0.05248 | 0.12145 | — | 2.80e-06 | 746 |
| 12 | 6 | P | 21 | 0.43507 | 0.4449 | 0.43570 | -0.12123 | -0.05385 | 0.12009 | 0.0023 | 8.10e-04 | 676 |
| 12 | 6 | Sblock | 21 | 0.40231 | 0.5061 | 0.40231 | -0.15399 | -0.08661 | 0.08732 | 0.2052 | 1.93e-06 | 754 |
| 12 | 6 | Pshuffle | 21 | 0.43491 | 0.4455 | 0.43506 | -0.12140 | -0.05401 | 0.11992 | 0.0024 | 1.67e-04 | 678 |
| 12 | 6 | P_thetaK0 | 20 | 0.76233 | 0.2241 | 0.76385 | 0.20603 | 0.27341 | 0.44735 | 0.0023 | 2.91e-06 | 886 |
| 12 | 6 | qaoa_sameG | 14 | 0.30291 | 0.5104 | 0.30829 | -0.25339 | -0.18601 | -0.01208 | -0.0000 | 2.09e-05 | 758 |
| 12 | 6 | P_indgamma | 26 | 0.20883 | 0.6940 | 0.21425 | -0.34748 | -0.28009 | -0.10616 | 0.0002 | 1.33e-05 | 916 |
| 12 | 7 | qaoa | 14 | 0.48892 | 0.3333 | 0.54796 | — | — | — | — | 7.71e-07 | 2979 |
| 12 | 8 | qaoa | 16 | 0.43567 | 0.3728 | 0.61569 | — | — | — | — | 7.25e-07 | 556 |
| 12 | 9 | qaoa | 18 | 0.39033 | 0.4120 | 0.51154 | — | — | — | — | 2.03e-06 | 610 |
| 12 | 10 | qaoa | 20 | 0.34945 | 0.4523 | 0.36632 | — | — | — | — | 2.66e-06 | 613 |
| 12 | 11 | qaoa | 22 | 0.31499 | 0.4849 | 3.97941 | — | — | — | — | 3.11e-06 | 595 |

## Reading

N12 L=1: P error 1.1618 F 0.138 thetaP -0.0004 (median error 1.1618); QAOA L error 1.7943 F 0.081; A error 1.1619 F 0.138; Sblock error 1.1415 F 0.146 thetaS 0.1257; shuffle error 1.1618 F 0.138; thetaK0 error 1.7721 F 0.081; QAOA L+1 error 1.1816 F 0.133; QAOA p=3 error 0.9281 F 0.171
N12 L=2: P error 1.0480 F 0.163 thetaP 0.0125 (median error 1.0480); QAOA L error 1.1816 F 0.133; A error 1.0560 F 0.163; Sblock error 1.0147 F 0.183 thetaS 0.1919; shuffle error 1.0481 F 0.163; thetaK0 error 1.5745 F 0.095; QAOA L+1 error 0.9281 F 0.171; QAOA p=5 error 0.6363 F 0.258
N12 L=3: P error 0.6924 F 0.263 thetaP 0.0000 (median error 0.6925); QAOA L error 0.9281 F 0.171; A error 0.6924 F 0.263; Sblock error 0.6626 F 0.285 thetaS 0.1933; shuffle error 0.6924 F 0.263; thetaK0 error 1.0934 F 0.147; QAOA L+1 error 0.7600 F 0.215; QAOA p=6 error 0.5563 F 0.295
N12 L=4: P error 0.5975 F 0.301 thetaP 0.0031 (median error 0.6499); QAOA L error 0.7600 F 0.215; A error 0.5997 F 0.301; Sblock error 0.5754 F 0.328 thetaS 0.1932; shuffle error 0.5974 F 0.301; thetaK0 error 0.9789 F 0.166; QAOA L+1 error 0.6363 F 0.258; QAOA p=8 error 0.4357 F 0.373
N12 L=5: P error 0.5026 F 0.386 thetaP 0.0002 (median error 0.5090); QAOA L error 0.6363 F 0.258; A error 0.5027 F 0.386; Sblock error 0.4721 F 0.433 thetaS 0.1904; shuffle error 0.5026 F 0.386; thetaK0 error 0.8609 F 0.201; QAOA L+1 error 0.5563 F 0.295; QAOA p=9 error 0.3903 F 0.412
N12 L=6: P error 0.4351 F 0.445 thetaP 0.0023 (median error 0.4357); QAOA L error 0.5563 F 0.295; A error 0.4364 F 0.443; Sblock error 0.4023 F 0.506 thetaS 0.2052; shuffle error 0.4349 F 0.445; thetaK0 error 0.7623 F 0.224; QAOA L+1 error 0.4889 F 0.333; QAOA p=11 error 0.3150 F 0.485

Equal-layer both-metric depths for P vs standard QAOA at N=12: [1, 2, 3, 4, 5, 6].
Parameter-matched both-metric depths for P: none.
thetaP stays near 0 at every N=12 depth, and P tracks full-K A. The SaSb control, with thetaS around 0.2, beats P at every N=12 depth, so the six-body parity is not what produces the equal-layer gap. Shuffled Jbar tracks P, which is expected when thetaP is unused. thetaK=0 loses the equal-layer gain. Same-G QAOA (independent gamma, linear mixer, shared theta) beats shared-gamma P from L=2 onward; that is a feature-versus-shell diagnostic, not a redefinition of standard QAOA. The independent-gamma shell rows are one local optimization of a relaxed ansatz. They are not a certified ceiling and are not used to discard or to certify the legal candidate.


## Failures

None.

## Provenance

- code hash `0d4e7860059cf85417f1753444000a3e8d73ed5605a7bcfefe5cc40a211f8a67`
- manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`
- wall 153.11988468100026 s
- numpy 2.5.4 scipy 1.18.1
- Artifacts: `results/cells`, `results/states`, `results/figures`, `results/comparison.json`.

