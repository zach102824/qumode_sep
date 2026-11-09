# L1XFER — TRANSFER_MEAN_NEGATIVE_CI. Not adoption. GR unpatched.

Exploratory L=1 parameter concentration and transfer, not an architecture change and not a GR retune. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10600–10615**. Train **10600–10607**, test **10608–10615**, split frozen before fits. Original shared-Uc Nov3 vs QAOA p=1. Energy selects on independent fits. F post hoc. Mean and median of train parameter vectors frozen before any test SPSA or transfer energy. Both aggregators locked; primary label uses mean; median was not selected on test.

**Label TRANSFER_MEAN_NEGATIVE_CI.** Test shared-mean Nov3−QAOA CI entirely <0 on n=8. Not a GR write. Does not revive L2/L3.

| contrast | mean ΔE | CI | wins | mean F a / b |
|---|---|---|---|---|
| 1 train independent Nov3−QAOA | −0.293 | [−0.323, −0.263] | 8/8 | 0.668 / 0.550 |
| 2a test shared-mean Nov3−QAOA (primary) | −0.764 | [−0.863, −0.663] | 8/8 | 0.368 / 0.263 |
| 2b test shared-median Nov3−QAOA | −0.350 | [−0.414, −0.296] | 8/8 | 0.648 / 0.503 |
| 3 test ceiling independent Nov3−QAOA | −0.303 | [−0.362, −0.258] | 8/8 | 0.665 / 0.536 |
| 4 transfer gap mean Nov3 vs independent Nov3 | +1.199 | [+1.054, +1.354] | 0/8 | 0.368 / 0.665 |

Train independent and test ceiling both recover the usual L=1 Nov3 energy edge (ΔE/N about −0.020). Shared-mean Nov3 beats shared-mean QAOA, which is the preregistered fair transfer contrast.

That mean-vs-mean ΔE is **not** a claim that a single shared Nov3 vector replaces per-instance SPSA. Contrast 4: shared-mean Nov3 is worse than independently optimized Nov3 by +1.20 (mean E −12.27 vs −13.47). Shared-median Nov3 sits near the independent ceiling (mean E −13.45 vs −13.47). Median was frozen alongside mean and was not chosen on test.

**Train concentration (n=8 caveat).**

| coord | mean | std | range |
|---|---|---|---|
| Nov3 γ | −0.142 | 0.252 | [−0.335, +0.263] |
| Nov3 θ_K | −0.369 | 0.051 | [−0.441, −0.294] |
| Nov3 θ_M | −0.001 | 0.023 | [−0.034, +0.037] |
| Nov3 s1 | −0.407 | 0.806 | [−0.859, +0.907] |
| Nov3 s2 | +0.009 | 0.104 | [−0.150, +0.217] |
| Nov3 s3 | −0.027 | 0.234 | [−0.392, +0.165] |
| QAOA γ | +0.046 | 0.261 | [−0.277, +0.287] |
| QAOA β | −0.092 | 0.401 | [−0.394, +0.404] |

θ_K and θ_M concentrate. s1 and QAOA β do not (wide, apparently bimodal). Averaging those coordinates is why the shared mean is a weaker Nov3 state and a weak QAOA mixer; the inflated mean-vs-mean |ΔE| is partly a bad transferred QAOA, not a better transferred Nov3.

32/32 independent fits ok. 32/32 transfer evals ok (1 query each). B* rejects 0. Charged 32800. Recipe hash matched. Seeds 10552–59 not reused (PR112 collision stands). Hole 10560–67 unused. No L2+, no L3, no GR patch, no candidacy, no final.
