# N18TWO — TWO_TEMPLATE_TRANSFER_NEGATIVE_CI. Not adoption. GR unpatched.

Two-template bimodal cluster-mean transfer at N=18 L=1. Not an unsigned-mean copy at another N, not a post-hoc of PR119/PR121 test sets, not L2/L3. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10728–10743**. Train **10728–10735**, test **10736–10743**, split frozen before fits. Original shared-Uc Nov3 vs QAOA p=1. Energy selects on independent fits. F post hoc.

From train only: Nov3 partitioned by sign(s1) into cluster+ (n=5) and cluster− (n=3); QAOA partitioned by sign(β) into cluster+ (n=3) and cluster− (n=5). All four clusters nonempty, so no method reduced to a single template. Two-template energy = min of that method's two charged frozen-template energies (energy-only; not F; not test-time SPSA). Unsigned mean and s1-only sign-aligned mean frozen and evaluated as secondary replication only. Primary vs secondary was not switched after seeing test.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.** Test better-of-two Nov3 minus better-of-two QAOA CI entirely <0 on n=8. Not a GR write. Does not revive L2/L3. Does not replace per-instance SPSA.

| contrast | mean ΔE | CI | wins | mean E a / b |
|---|---|---|---|---|
| primary two-template Nov3 − two-template QAOA | −0.398 | [−0.449, −0.342] | 8/8 | −15.21 / −14.81 |
| 1 train independent Nov3−QAOA | −0.330 | [−0.347, −0.314] | 8/8 | −17.18 / −16.85 |
| 2 test ceiling independent Nov3−QAOA | −0.411 | [−0.504, −0.336] | 8/8 | −15.28 / −14.87 |
| 3 two-template Nov3 vs independent QAOA | −0.339 | [−0.390, −0.288] | 8/8 | −15.21 / −14.87 |
| 4 transfer gap two-template Nov3 vs independent Nov3 | +0.073 | [−0.002, +0.195] | 3/8 | −15.21 / −15.28 |
| 5a Nov3 cluster+ vs unsigned QAOA | −2.371 | [−2.603, −2.098] | 8/8 | −15.20 / −12.83 |
| 5b Nov3 cluster− vs unsigned QAOA | −2.345 | [−2.571, −2.079] | 8/8 | −15.18 / −12.83 |
| 5c Nov3 cluster+ vs independent QAOA | −0.335 | [−0.379, −0.286] | 8/8 | −15.20 / −14.87 |
| 5d Nov3 cluster− vs independent QAOA | −0.309 | [−0.373, −0.244] | 8/8 | −15.18 / −14.87 |
| 6a unsigned-mean Nov3 − unsigned QAOA (replication) | −0.059 | [−0.067, −0.051] | 8/8 | −12.89 / −12.83 |
| 6b sign-aligned Nov3 − unsigned QAOA (replication) | −0.785 | [−0.884, −0.695] | 8/8 | −13.62 / −12.83 |

Per-instance L=1 Nov3 still wins on train (ΔE/N −0.0183) and on the test ceiling (ΔE/N −0.0229). Two-template Nov3 beats two-template QAOA, which is the preregistered primary gate. Two-template Nov3 also beats independently fitted test QAOA (contrast 3).

That does **not** replace per-instance SPSA. Contrast 4: two-template Nov3 is slightly worse than independently optimized Nov3 (mean E −15.21 vs −15.28; gap +0.073; CI includes 0; 3/8). The gap is far smaller than PR121's sign-aligned gap (+1.779). Energy-min among two frozen templates is extra matching opportunity relative to a single shared vector; QAOA received the same two-template opportunity.

On this test draw the Nov3 two-template chose cluster+ on 7/8 seeds and cluster− on 1/8. QAOA chose the same pattern. Each single Nov3 cluster mean already sits near the independent Nov3 ceiling (−15.20 / −15.18 vs −15.28) because each cluster concentrates γ with s1 (cluster+ γ mean +0.267 std 0.019; cluster− γ mean −0.254 std 0.027). Unsigned mean still mixes those families (γ std 0.270, s1 std 0.897) and collapses (E −12.89). Unsigned QAOA also collapses (E −12.83) because β is bimodal (mean −0.099, std 0.402); the two β clusters themselves are tight (std 0.008 / 0.005).

**Train concentration (n=8 caveat).** Nov3 s1 5/3, QAOA β 3/5. θ_K std 0.028 and θ_M std 0.025 still concentrate. After s1-only alignment s1 std 0.031, but γ remains mixed (std 0.270), which is why sign-aligned Nov3 (−13.62) does not match either cluster mean.

Unsigned-mean and sign-aligned replications on this new draw do not reopen PR119 and do not license unsigned-mean copies at other N.

32/32 independent fits ok. 56/56 transfer evals ok (1 query each). B* rejects 0. Charged 32824. Recipe hash matched. Seeds 10712–27 (R34), 10744–59 (FT), 10680–95, 10648–63, 10600–15, 10624–31, 10560–67 unused. No L2+, no L3, no GR patch, no candidacy, no final.
