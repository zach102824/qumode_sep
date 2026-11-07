# Round 6 QR slice 2

Tau stays 1e-9. v1 is C's arm0 state, the fidelity this branch froze. v2 is D's arm2 family-best. Same-record replay residuals are about 1e-15. The 3.27e-8 fidelity gap is the cross-record difference, so the two are not declared the same state. Positive energy deltas mean QAOA is higher than the comparison state. None of these rows is a both-metric win. p7, p12, and p11 each reached the frozen remainder, including one cap-truncated partial start. p9, p10, and p13 were not run. C's 53927 versus D's 52927, and 124788 versus 124404, stay unresolved and did not change the allowances. No seal: L5 rescue, the L5 distribution, raw-ledger reconciliation, and the hardware ledger are still open.

| p | E | F | new | allowed | short | completed | partial |
|---|---|---|---|---|---|---|---|
| 7 | -8.124283 | 0.3333 | 415 | 415 | 0 | 5 | 1 |
| | v1_stride4_L6 dE +3.213997e-01 dF -5.690843e-01 dominates False | | | | | | |
| | v2_stride4_L6 dE +3.213997e-01 dF -5.690844e-01 dominates False | | | | | | |
| | v1_KS_L6 dE +2.052408e-01 dF -2.706303e-01 dominates False | | | | | | |
| 12 | -8.325408 | 0.5147 | 108516 | 108516 | 0 | 868 | 1 |
| | v1_stride4_L6 dE +1.202750e-01 dF -3.876516e-01 dominates False | | | | | | |
| | v2_stride4_L6 dE +1.202750e-01 dF -3.876516e-01 dominates False | | | | | | |
| | v1_KS_L6 dE +4.116125e-03 dF -8.919753e-02 dominates False | | | | | | |
| 11 | -8.298216 | 0.4849 | 95146 | 95146 | 0 | 784 | 1 |
| | v1_stride4_L6 dE +1.474669e-01 dF -4.175191e-01 dominates False | | | | | | |
| | v2_stride4_L6 dE +1.474669e-01 dF -4.175192e-01 dominates False | | | | | | |
| | v1_KS_L6 dE +3.130809e-02 dF -1.190651e-01 dominates False | | | | | | |
