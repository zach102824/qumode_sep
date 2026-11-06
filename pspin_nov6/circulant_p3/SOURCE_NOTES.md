# Sources checked for the circulant ring

The round-2 brief is an Opus hypothesis. Nothing in it was treated as a citation.

No paper was found, or used, as the source of this specific two-window Hamiltonian. The polynomials 1+x+x^3 and 1+x^2+x^3 do divide x^7+1 over GF(2); that algebraic fact is standard and was not used as a substitute for the rank test. The satisfiability matrix was row-reduced directly for every N from 6 to 14.

Pagano et al., arXiv:1906.02700, was opened in round 1 (HTML at https://arxiv.org/html/1906.02700). The same bootstrap was reused: L=1 grid, L=2 seeds with β decreased and γ increased, higher depth by linear (new depth 3) or cubic interpolation in normalized layer time, then a local energy minimization. The shift values used here are 0.05, 0.1, and 0.2 on this model's generators. Their supplement does not discuss this ring.

Round-1 collective-p=3 numbers cited in the comparison come from `pspin_nov6/collective_p3/results/qaoa_round1.csv` in this branch, not from an external paper.
