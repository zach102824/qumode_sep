# N=9 fixed-field confirmation

Target `220d4b40e897df0c8fa8ca0185dff7495c6e8ba62625134a05cbaf76876f3828`, packing `5ae567c9d89189c2fd495161a676056bd7aa09b4f043fdbe25ad5b82470dcfdc`. B = -0.3444. E0 = -6.337088433486, full gap 7.807622e-03. The full gap is measured. The first excited state is the opposite-parity partner.

| L | best error | F | median | q25 | q75 | equal-L QAOA | matched QAOA | both equal | both matched |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.755146 | 0.3074 | 0.7551 | 0.7551 | 1.5401 | 1.2214876606272558 | p3 0.5986040060064113 | True | False |
| 2 | 0.693985 | 0.3498 | 0.6940 | 0.6940 | 1.4182 | 0.7914639403398969 | p5 0.38646734706252506 | True | False |
| 3 | 0.454719 | 0.5077 | 0.4547 | 0.4547 | 0.9199 | 0.5986040060064113 | p6 0.3366824656771943 | True | False |
| 4 | 0.403337 | 0.5469 | 0.4033 | 0.4033 | 0.7474 | 0.48033837159329273 | p8 0.25526853272868166 | True | False |
| 5 | 0.325644 | 0.6025 | 0.3648 | 0.3648 | 0.3899 | 0.38646734706252506 | p9 0.22328823015703048 | True | False |
| 6 | 0.232740 | 0.6330 | 0.3223 | 0.3223 | 0.3667 | 0.3366824656771943 | p11 0.17033323937929268 | True | False |

Candidate optimizer evaluations 45654 of the ceiling 100000. QAOA optimizer evaluations 182616 of the frozen floor 182616. Shortfall 0.

J, K, and M on the 36 pair slots have rank 3, so all three features stay and P remains 3L+3. The measured full gap is 0.007808. That partner has parity −1. The next +1 state is 0.453 above the ground state. Odd N was not assumed to be degenerate.

A dense-matrix energy of the stored L6 best matches the log to 2e-15, the state hash matches, and the parity weight is 1. The energy error 0.233 is not a numerical tie with the ground state.

At equal layer count the L6 best beats QAOA at p=6 on both energy and fidelity, and 35 of 48 chains do. The Wilson interval is [0.590, 0.834]. At the parameter-matched depth p=11, and at p=12, none of the 48 chains does. The strongest p=12 state has error 0.152058 and fidelity 0.730618. The candidate L6 best has error 0.232740 and fidelity 0.633043. QAOA is lower in energy and higher in fidelity there. The same split holds at L5 against p=9 and p=12. Equal-layer and parameter-count comparisons are different ledgers. This slice does not retune the field or the packing.

The QAOA floor was set, after the candidate ledger closed and before QAOA started, at 4 × 45654 = 182616. That floor was met. Eta on the L6 best is 0.989 and is not fidelity. The N=6 and N=12 results are not evidence for this size. N=15 and N=18 were not run.

Wall 396.4 s.
