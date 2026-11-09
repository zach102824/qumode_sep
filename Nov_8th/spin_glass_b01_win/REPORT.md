# Historical B=0.1 window confirmation is partial

Protocol `95ec3265` before any fit. Running marker `e6cef2fb`. PREREG SHA-256 `1946d1a2463f4d5af9e39973ea2d3dc51414010004e2cdc0baa549cd936d1ced`.

This is the historical author field `B = 0.1`. It is not a critical-field result. `B_cri` stays withheld. The Hamiltonian is the published full spin glass and was not changed. The circuit is the original interleaved shared-gamma KM: one gamma, one thetaK, one thetaM, three shell angles per layer, 21 parameters at L=6. QAOA has 12 parameters at L=6. Both use the round-4 start list. There is no second packing and no extra start.

Maximin stays partial. On k=190..201 that packing missed N=12 while interleaved KM cleared it. Ferromagnetic-sum packing stays partial. This round does not refit either packing and does not rebrand those N=6 or N=9 passes.

## Rule

k=202..225 was split before the fit into block A (202..213) and block B (214..225). All 24 instances at each of N=6, 9, and 12 finished. A size clears only when the pool of 24 meets the equal-depth win rule and both blocks have a positive median fidelity difference and at least as many wins as losses. A block interval is not a pass. Earlier windows are not in this bootstrap.

## Verdict

Overall **partial**. N=6 clears. N=9 does not. N=12 does not.

| sample | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | role |
|---|---|---|---|---|---|---|
| pool | 6 | 21 (20 / 1) | 0 | 3 | +0.253 [+0.138, +0.369] | claim |
| pool | 9 | 11 (10 / 1) | 4 | 9 | +0.019 [-0.039, +0.189] | claim |
| pool | 12 | 10 (8 / 2) | 2 | 12 | +0.037 [-0.025, +0.151] | claim |
| block A | 6 | 10 (9 / 1) | 0 | 2 | +0.247 [+0.029, +0.407] | control |
| block B | 6 | 11 (11 / 0) | 0 | 1 | +0.256 [+0.162, +0.362] | control |
| block A | 9 | 4 (4 / 0) | 3 | 5 | -0.038 [-0.069, +0.132] | control |
| block B | 9 | 7 (6 / 1) | 1 | 4 | +0.129 [-0.003, +0.273] | control |
| block A | 12 | 3 (3 / 0) | 1 | 8 | +0.021 [-0.045, +0.072] | control |
| block B | 12 | 7 (5 / 2) | 1 | 4 | +0.120 [-0.038, +0.256] | control |

N=6 is window-stable: both block medians are positive and neither block has a loss. The one fidelity-only win is k=212. The ties are k=203, k=208, and k=220. The selected L=6 state at k=219 hit the evaluation cap and has fidelity 0.998.

N=9 fails the pool: the interval crosses 0, with 4 losses (k=205 and k=208 are fidelity-only, so KM energy is lower there; k=210 and k=225 are joint). The blocks disagree. Block A has median -0.038. Block B has median +0.129 and its interval still crosses 0. Block B is not a result.

N=12 fails the pool: 10 wins, 2 losses, 12 ties, and the interval crosses 0. The two losses are joint, k=213 and k=223. The fidelity-only wins are k=219 and k=221. Both block medians are positive, so the frozen sign check calls N=12 window-stable, and that is not enough to clear. Neither block interval clears 0. The N=12 pass of interleaved KM on k=190..201 does not replicate on this fresh sample of 24. That earlier pass stays one window. It is not confirmed here, and it is not evidence for maximin or for any other packing.

Several N=12 ties have a large energy gap and a small fidelity gap. k=204 is lower by 1.898 with dF +0.027. The equal-depth fidelity claim does not follow from a lower energy.

## Equal parameter count

KM at L=3 against QAOA at L=6 is 12 parameters on the pool and is not the verdict. N=6 has 10 wins (5 joint) and 12 losses, median -0.006 [-0.082, +0.118]. N=9 has 0 wins and 10 losses, median -0.049 [-0.142, -0.023]. N=12 has 0 wins and 11 losses, median -0.080 [-0.101, -0.048].

## Replay and cost

Independent replay passed at all three N. The largest energy error is 9.3e-13 relative, the largest fidelity error is 1.7e-13, and there is no ground-space mismatch.

Optimizer wall 2105.0 s of 21600. Starts 49824. Logged energy evaluations 4000342. L=1 grid evaluations 36864. Capped starts 58. k=100 and k=101 stay unused. The next unused k is 226.
