# N=6 fixed-field confirmation

Target `6f43239b5e9d7fd19727e9805d98eb43c958e58b9d45dbaa50ad98baf84d5fb8`, packing `d67626a133e8e66ecd85f1dff78d2d9fec5aada79d6f08503952f4a0b29bd201`. B = -0.3444, not -0.1708. E0 = -4.117124992097, full gap 2.828121e-02.

| L | best error | F | median | q25 | q75 | equal-L QAOA | matched QAOA | both equal | both matched |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.254639 | 0.7270 | 0.2729 | 0.2546 | 0.2729 | 0.711513075899179 | p3 0.32636761424525407 | True | True |
| 2 | 0.072514 | 0.9307 | 0.0726 | 0.0725 | 0.0726 | 0.45888983159039975 | p5 0.19696690658440552 | True | True |
| 3 | 0.014812 | 0.9920 | 0.0633 | 0.0363 | 0.0633 | 0.32636761424525407 | p6 0.15100218114605468 | True | True |
| 4 | 0.012216 | 0.9945 | 0.0307 | 0.0198 | 0.0308 | 0.26088595494251 | p8 0.08205434283390556 | True | True |
| 5 | 0.007210 | 0.9967 | 0.0174 | 0.0100 | 0.0189 | 0.19696690658440552 | p9 0.06260249032469645 | True | True |
| 6 | 0.004384 | 0.9977 | 0.0126 | 0.0080 | 0.0132 | 0.15100218114605468 | p11 0.018832172672096803 | True | True |

Candidate optimizer evaluations 56040 of the ceiling 100000. QAOA optimizer evaluations 400000 of the frozen floor 400000. Shortfall 0.

The J, K, and M coefficient vectors on the 15 pair slots have rank 3, so no feature was removed. P stays 3L+3.

A dense-matrix energy of the stored L5 and L6 best vectors matches the logged energy to 0, and the state hashes match. Both states have norm 1 and +1 parity weight 1. The L6 energy error 0.00438 is above the ground state by more than the comparison tolerance, so this is not a numerical tie with the ground state.

Against the strongest recorded p=12 state (error 0.009243, fidelity 0.994558, 664 finished starts, floor of 400000 optimizer evaluations met), the L6 best is lower in energy and higher in fidelity. Sixteen of 48 L6 chains meet both inequalities. The Wilson interval is [0.217, 0.475]. The L6 median error is 0.01259, which is above that p=12 energy, so the median does not beat it. Seven of 48 L5 chains beat p=12 on both metrics, interval [0.072, 0.272]. The L5 and L6 bests also beat their matched depths, p=9 and p=11, on both metrics.

The extra QAOA starts tagged center are σ=0.05 perturbations of the running best, not a separate layer-center interpolant. The ascending ladder itself uses the supplement endpoint grid: linear into p=3 and a natural cubic after that. Gamma is not rescaled. The p=2 beta step is the paper offset 0.2 multiplied by B = -0.3444.

Eta on these states is about 0.999 and is not fidelity. The published 94–96% figures are eta on a different target. The N=12 result is not used as a seed and is not treated as evidence for this size. The historical N=6 field -0.1708 is a different Hamiltonian. QAOA reached its frozen 400000-evaluation floor, so that comparison is not short of the preregistered floor. No fairness claim is made beyond these recorded depths, and no global optimum is claimed.

Wall 440.6 s.
