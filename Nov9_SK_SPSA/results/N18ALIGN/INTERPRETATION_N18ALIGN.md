# N18ALIGN — TRANSFER_MEAN_NEGATIVE_CI. Not adoption. GR unpatched.

Mechanistic confirmatory follow-up to PR119, not an unsigned-mean copy at another N and not a post-hoc of PR119's test set. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10680–10695**. Train **10680–10687**, test **10688–10695**, split frozen before fits. Original shared-Uc Nov3 vs QAOA p=1. Energy selects on independent fits. F post hoc.

No covariant discrete gauge is documented in the saved Nov3 recipe. Sign-alignment therefore flips **only s1** when s1<0; γ, θ_K, θ_M, s2, s3 are unchanged. Larger-cluster: partition by sign(s1); this train was 5/3, so the strictly larger cluster is s1>0 (n=5). Mean and median of the unsigned train vectors were also frozen; unsigned primary aggregator remains mean. Primary vs secondary was not switched after seeing test.

**Label TRANSFER_MEAN_NEGATIVE_CI.** Test sign-aligned-mean Nov3 minus unsigned shared-mean QAOA CI entirely <0 on n=8. Not a GR write. Does not revive L2/L3. Does not replace per-instance SPSA.

| contrast | mean ΔE | CI | wins | mean E a / b | mean F a / b |
|---|---|---|---|---|---|
| 1 train independent Nov3−QAOA | −0.314 | [−0.346, −0.274] | 8/8 | −17.70 / −17.39 | 0.670 / 0.552 |
| 2 primary sign-aligned Nov3 − unsigned QAOA | −0.969 | [−1.027, −0.916] | 8/8 | −13.81 / −12.85 | 0.301 / 0.199 |
| 2b larger-cluster Nov3 − unsigned QAOA | −2.629 | [−2.844, −2.494] | 8/8 | −15.47 / −12.85 | 0.582 / 0.199 |
| 2c unsigned-mean Nov3 − unsigned QAOA (PR119 replication) | −0.349 | [−0.363, −0.335] | 8/8 | −13.19 / −12.85 | 0.229 / 0.199 |
| 3 test ceiling independent Nov3−QAOA | −0.360 | [−0.472, −0.275] | 8/8 | −15.59 / −15.23 | 0.597 / 0.481 |
| 4 transfer gap sign-aligned vs independent Nov3 | +1.779 | [+1.484, +2.234] | 0/8 | −13.81 / −15.59 | 0.301 / 0.597 |

Per-instance L=1 Nov3 still wins on train (ΔE/N −0.0175) and on the test ceiling (ΔE/N −0.0200). Sign-aligned-mean Nov3 beats unsigned-mean QAOA, which is the preregistered primary gate.

That ΔE is **not** a claim that a single shared Nov3 vector replaces per-instance SPSA. Contrast 4: sign-aligned Nov3 is worse than independently optimized Nov3 by +1.78 (mean E −13.81 vs −15.59). Unsigned shared-mean QAOA itself is a weak mixer here (mean E −12.85 vs independent QAOA −15.23) because train β is bimodal (mean −0.001, std 0.422). Part of the primary |ΔE| is a collapsed transferred QAOA, as in PR116.

Larger-cluster Nov3 (raw mean of the five s1>0 vectors) sits near the independent Nov3 ceiling (−15.47 vs −15.59). Sign-aligned mean of all eight does not, because flipping only s1 still averages γ/s2/s3 across the two s1-sign families. Those other coordinates co-vary with s1 even without a documented covariant gauge.

**Train concentration (n=8 caveat).** This draw is 5/3, not PR119's 4/4. Unsigned s1 mean +0.226 std 0.884; after s1-only alignment s1 mean +0.857 std 0.036 (n_flipped=3). θ_K std 0.057 and θ_M std 0.029 still concentrate. Larger-cluster (s1>0, n=5) also concentrates γ (std 0.024 vs unsigned 0.264).

Unsigned-mean Nov3 vs unsigned-mean QAOA is a secondary PR119-protocol replication on a **new** seed band. It is entirely <0 here (CI [−0.363, −0.335]) because this train is 5/3 rather than 4/4 **and** because transferred QAOA collapsed. It does not reopen PR119's unsigned primary gate and does not license unsigned-mean copies at other N.

32/32 independent fits ok. 32/32 transfer evals ok (1 query each). B* rejects 0. Charged 32800. Recipe hash matched. Seeds 10648–63 (PR119) not reused. Hole 10560–67 unused. FT 10664–79 and 10696–10711 unused. No L2+, no L3, no GR patch, no candidacy, no final.
