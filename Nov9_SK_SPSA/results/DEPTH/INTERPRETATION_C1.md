# DEPTH C1 — charged L1-continuation is not adopted

This batch asked whether L>1 fade is mainly a cold-start failure. The test was ordinary C0 SPSA with every query inside Q=1024·L: arm A is a cold depth-L fit; arm B spends 1024 queries on L=1, pads the energy-selected point into depth L, and continues under the remaining budget. Seeds 10036–10043 only. Energy selects. F is post hoc.

**Reject continuation.** The preregistered gate is N15 L3 Nov3, n=8, bootstrap 95% CI of mean(E_cont − E_cold). That CI is [−0.028, +0.061] and the mean is +0.022 (continuation higher energy). It is not entirely below 0. Continuation was lower energy on 1/8 seeds. Fidelity moved the same way (mean ΔF −0.027, 2/8 continuation wins) and is not in the gate.

The rest of the matrix agrees. QAOA continuation lost energy 0/8 in every cell, and the mean ΔE CI lies entirely above 0 at both N and both L. Nov3 continuation lost 0/8 at L=2 (both N) with CIs above 0; at L=3 the CIs include 0 and still do not clear the gate. Energy-better-of cold/continuation therefore selected cold on 62 of 64 circuit×N×L×seed cells (the two exceptions are Nov3 L=3).

QAOA extra layers with γ=β=0 are identity, checked before the fits. So B is a true warm start on QAOA, and it still lost. The failure is the charged split: L1 consumes 1024 queries that cold L spends at the evaluation depth, and the continuation remainder has a smaller K (123 vs 251 at L=2; 251 vs 379 at L=3). Nov3 cannot add an identity layer under shared γ, which was frozen, not repaired.

Under the energy-better of the two arms, QAOA beats Nov3 on energy at every cell (mean Δ(E−E0)/N of Nov3−QAOA CI entirely above 0). Nov3 has higher post hoc F at N12; at N15 the F CI includes 0. Energy remains the selection rule.

128/128 fits ok. Charged 327680. Outside-cap optimizer queries 0. All 16 B* instances accepted (odd E1, even GS, resid < 1e-9). M2 vs ref spot-check at N12/N15 L3 passed. N3COND 772224 and LEXACT SPSA 2560 are unchanged. Candidate development only. GR queue not patched. Frozen v1 not retuned. No further continuation split in this report.
