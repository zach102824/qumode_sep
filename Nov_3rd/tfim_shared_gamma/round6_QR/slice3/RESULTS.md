# Round 6 QR slice 3

p9 then p10. Tau stays 1e-9. Allowances were not reduced. Both depths stopped on the cap. The stored QAOA energies did not move, so the candidate both-metric counts stay 29/48 at L5 and 32/48 at L6 against p9, p10, p11, and p12. Those two success sets are not inferred from the marginals: the L5 set and the L6 set share 29 chain indices. A failed QAOA improvement is not itself candidate dominance; the logged candidate states still have lower energy and higher fidelity than these QAOA states. No seal. The corrected source total 52927 did not change the allowances. The 1000 and 384 cost gaps stay as reconciled notes, not a new budget.

| p | E | F | new | allowed | short | completed | partial |
|---|---|---|---|---|---|---|---|
| 9 | -8.222874 | 0.4120 | 133076 | 133076 | 0 | 1436 | 1 |
| | stride4_L6_v1 dE +2.228083e-01 dF -4.903635e-01 candidate_both True | | | | | | |
| | stride4_L5_D_arm0 dE +1.622817e-01 dF -4.778077e-01 candidate_both True | | | | | | |
| | stride4_L6_v2_arm2 dE +2.228083e-01 dF -4.903636e-01 candidate_both True | | | | | | |
| | KS_L6 dE +1.066495e-01 dF -1.919095e-01 candidate_both True | | | | | | |
| 10 | -8.263750 | 0.4523 | 138662 | 138662 | 0 | 1355 | 1 |
| | stride4_L6_v1 dE +1.819323e-01 dF -4.500635e-01 candidate_both True | | | | | | |
| | stride4_L5_D_arm0 dE +1.214057e-01 dF -4.375076e-01 candidate_both True | | | | | | |
| | stride4_L6_v2_arm2 dE +1.819323e-01 dF -4.500635e-01 candidate_both True | | | | | | |
| | KS_L6 dE +6.577350e-02 dF -1.516094e-01 candidate_both True | | | | | | |
