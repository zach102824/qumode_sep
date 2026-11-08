# Round 5 scope after the Nov 8 constraint

The confirmatory family in `PREREG.json` is KM_gl. `PREREG.json` is left unchanged so its hash still matches the run that already finished. The claim in that file is void.

KM_gl is out of scope. It is a different parametrization: layer ell applies `a_ell * Hc + b * Khat + c * Mhat` with a separate `a_ell`. The original circuit uses one gamma for every layer, `D = exp[-i gamma (Hc + thetaK FKhat + thetaM FMhat)]`.

What was stopped: nothing was still running. The optimizer, the main-simulator evaluation, the independent replay, and `summarize.py` had all exited before this note. Their files are kept. The replay checked KM_gl states along with the others; passing replay does not make KM_gl a valid method.

What remains usable from this directory:

- `KM` and `qaoa` selected states. Same Hamiltonian, same interleaved packing, same optimizer budget as round 4, fresh instances k=28..43, two repeats. These were fit as controls, under a protocol frozen before the fits. They were not the preregistered claim, so they are reference numbers, not a confirmation of KM.
- At L=6, best of two repeats, from `RESULTS.md` (KM versus QAOA):

| N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |
|---|---|---|---|---|---|---|
| 6 | 15 (15 / 0) | 0 | 1 | +0.165 [+0.060, +0.282] | +0.179 | 3.05e-05 |
| 9 | 9 (8 / 1) | 3 | 4 | +0.112 [-0.016, +0.238] | +0.104 | 0.073 |
| 12 | 7 (6 / 1) | 1 | 8 | +0.049 [-0.000, +0.090] | +0.090 | 0.035 |

The N=9 interval includes 0. The N=12 interval touches 0. This is a clearer N=6 result than round 3 and a positive but not confirmed shift at N=9 and N=12. It is not a global-optimum claim.

KM_pack was not fit in this round. It is not being confirmed. It changes which sites sit in each shell triple.
