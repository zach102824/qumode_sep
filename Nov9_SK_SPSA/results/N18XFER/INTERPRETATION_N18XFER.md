# N18XFER — UNSTABLE. Not adoption. GR unpatched.

Same PR116 N=15 L=1 concentration/transfer protocol at N=18. Not an architecture change and not a GR retune. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10648–10663**. Train **10648–10655**, test **10656–10663**, split frozen before fits. Original shared-Uc Nov3 vs QAOA p=1. Energy selects on independent fits. F post hoc. Mean and median of train parameter vectors frozen before any test SPSA or transfer energy. Both aggregators locked; primary label uses mean; median was not selected on this test and was not switched from PR116's test-median result.

**Label UNSTABLE.** Test shared-mean Nov3−QAOA CI is entirely >0 on n=8 (not entirely <0). Not a GR write. Does not revive L2/L3. Shared-mean beating shared-mean QAOA at N=15 does not replace per-instance SPSA and does not scale to N=18 under this protocol.

| contrast | mean ΔE | CI | wins | mean E a / b | mean F a / b |
|---|---|---|---|---|---|
| 1 train independent Nov3−QAOA | −0.356 | [−0.401, −0.306] | 8/8 | −16.99 / −16.64 | 0.629 / 0.495 |
| 2a test shared-mean Nov3−QAOA (primary) | +1.826 | [+1.729, +1.970] | 0/8 | −15.03 / −16.85 | 0.222 / 0.422 |
| 2b test shared-median Nov3−QAOA | +2.226 | [+2.091, +2.384] | 0/8 | −15.03 / −17.25 | 0.222 / 0.516 |
| 3 test ceiling independent Nov3−QAOA | −0.310 | [−0.343, −0.273] | 8/8 | −17.60 / −17.29 | 0.644 / 0.526 |
| 4 transfer gap mean Nov3 vs independent Nov3 | +2.572 | [+2.405, +2.759] | 0/8 | −15.03 / −17.60 | 0.222 / 0.644 |

Train independent and test ceiling both recover the usual L=1 Nov3 energy edge (ΔE/N −0.0198 train, −0.0172 test ceiling). That is the matching-opportunity ceiling. Shared-mean Nov3 does **not** beat shared-mean QAOA at N=18: CI entirely >0, 0/8.

Shared-mean does not replace per-instance SPSA. Contrast 4: shared-mean Nov3 is worse than independently optimized Nov3 by +2.57 (mean E −15.03 vs −17.60). Shared-median Nov3 is the same collapse (mean E −15.03); median was frozen alongside mean and was not chosen on test. At N=18 the SNAP s1 train sample is 4/4 sign-split, so both mean and median s1 sit near 0.

**Train concentration (n=8 caveat).** Descriptive vs PR116 N=15 train stds (different seeds; not a paired test; not used to choose aggregator).

| coord | N=18 mean | N=18 std | N=18 range | PR116 N=15 std |
|---|---|---|---|---|
| Nov3 γ | −0.001 | 0.285 | [−0.327, +0.301] | 0.252 |
| Nov3 θ_K | −0.390 | 0.053 | [−0.442, −0.277] | 0.051 |
| Nov3 θ_M | −0.008 | 0.016 | [−0.024, +0.020] | 0.023 |
| Nov3 s1 | −0.005 | 0.930 | [−0.895, +0.910] | 0.806 |
| Nov3 s2 | +0.070 | 0.090 | [−0.072, +0.199] | 0.104 |
| Nov3 s3 | −0.151 | 0.289 | [−0.618, +0.132] | 0.234 |
| QAOA γ | −0.169 | 0.196 | [−0.291, +0.313] | 0.261 |
| QAOA β | +0.295 | 0.278 | [−0.392, +0.402] | 0.402 |

θ_K and θ_M still concentrate (std 0.053 / 0.016, comparable to PR116 0.051 / 0.023). s1 does not: four train vectors cluster near −0.87 and four near +0.85, so the coordinate-wise mean and the coordinate-wise median both land near 0. Averaging a bimodal SNAP coordinate is why the transferred Nov3 state is weak. QAOA β is less split than at N=15 (std 0.278 vs 0.402); transferred QAOA remains a usable mixer (shared-mean E −16.85 vs independent −17.29). The mean-vs-mean sign flip relative to PR116 is a failed transferred Nov3, not a better transferred QAOA.

32/32 independent fits ok. 32/32 transfer evals ok (1 query each). B* rejects 0. Charged 32800. Recipe hash matched. Seeds 10600–15 (PR116), 10616–31, 10632–47, 10664–79, 10552–59, hole 10560–67 unused. No L2+, no L3, no GR patch, no candidacy, no final.
