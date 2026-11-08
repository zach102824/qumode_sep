# Overriding circuit constraint

Recorded from the user message of Nov 8, 21:09 Asia/Shanghai. It supersedes the earlier permission to try per-layer gamma.

The circuit is the Nov 3 shared-gamma sequence and nothing else:

- Cavity X frame. Start from |0>^N.
- Repeat UcX = W D W, then the per-cavity popcount shellSNAP.
- Final W, and evaluate the original target in that frame.
- One gamma, one thetaK, and one thetaM, shared by every layer.
- Three shell controls per layer, the same three numbers on every cavity.
- D = exp[-i gamma (Hc + thetaK FKhat + thetaM FMhat)].
- Fixed interleaved grouping `[a, a+N/3, a+2N/3]`.
- FK is the fixed pair feature J @ J. FM is the fixed block-mean feature on that grouping.

Not allowed: a gamma of its own on any layer, extra gates, extra channels, a different head, or a looser sharing pattern. Initialization, the optimizer, and data-derived couplings are allowed only when they do not add a continuous control or a gate. The |J|-max regrouping changes the support of the shell operators and of FMhat. That change is uncertain under this constraint, so the fixed grouping stays.

## What stopped

Round 5 had already finished KM_gl on k=28..43 (exit 0, 13:06 UTC) when this constraint arrived. No KM_gl or KM_pack process was killed because none was running. No further fit of either variant will be started. Round 5's "confirmed" verdict is void. See `Nov_8th/spin_glass_r5/SCOPE.md`.

## What continues

The original shared-gamma KM family, against the same equal-depth independent-layer QAOA, energy-only selection, fidelity only after selection. Round 6 tests extra initial points inside that family. It does not add a parameter.
