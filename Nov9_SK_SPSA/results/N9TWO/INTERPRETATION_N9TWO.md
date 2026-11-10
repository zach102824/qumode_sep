# N9TWO — TWO_TEMPLATE_TRANSFER_NEGATIVE_CI. Not adoption. GR unpatched.

Independent-stream confirmatory two-template bimodal cluster-mean transfer at N=9 L=1. Same protocol as PR126/PR125/PR123 and FT PR122/PR124. Not a new untested size (N6/N21 not opened), not an unsigned-mean copy, not a post-hoc of prior test sets, not L2/L3. Does not copy FT PR122 numbers and does not reuse 10744–59. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10840–10855**. Train **10840–10847**, test **10848–10855**, split frozen before fits. Original shared-Uc Nov3 vs QAOA p=1. Energy selects on independent fits. F post hoc.

From train only: Nov3 partitioned by sign(s1) into cluster+ (n=2) and cluster− (n=6); QAOA partitioned by sign(β) into cluster+ (n=5) and cluster− (n=3). All four clusters nonempty, so no method reduced to a single template and no empty cluster invented a template. Two-template energy = min of that method's two charged frozen-template energies (energy-only; not F; not test-time SPSA). Unsigned mean and s1-only sign-aligned mean frozen and evaluated as secondary replication only. Primary vs secondary was not switched after seeing test. Transfer-gap CI including 0 vs entirely >0 is a descriptive of contrast 4 and does not change the primary gate.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.** Test better-of-two Nov3 minus better-of-two QAOA CI entirely <0 on n=8. Not a GR write. Does not revive L2/L3. Does not replace per-instance SPSA. After this N=9 confirmatory, do not open another two-template size in this stream without a new parent steer.

| contrast | mean ΔE | CI | wins | mean E a / b |
|---|---|---|---|---|
| primary two-template Nov3 − two-template QAOA | −0.183 | [−0.270, −0.096] | 8/8 | −6.35 / −6.17 |
| 1 train independent Nov3−QAOA | −0.224 | [−0.275, −0.186] | 8/8 | −5.59 / −5.37 |
| 2 test ceiling independent Nov3−QAOA | −0.258 | [−0.283, −0.231] | 8/8 | −6.47 / −6.22 |
| 3 two-template Nov3 vs independent QAOA | −0.133 | [−0.219, −0.041] | 5/8 | −6.35 / −6.22 |
| 4 transfer gap two-template Nov3 vs independent Nov3 | +0.124 | [+0.046, +0.210] | 0/8 | −6.35 / −6.47 |
| 5a Nov3 cluster+ vs unsigned QAOA | −1.367 | [−1.586, −1.218] | 8/8 | −6.28 / −4.91 |
| 5b Nov3 cluster− vs unsigned QAOA | −1.433 | [−1.637, −1.301] | 8/8 | −6.35 / −4.91 |
| 5c Nov3 cluster+ vs independent QAOA | −0.065 | [−0.164, +0.066] | 5/8 | −6.28 / −6.22 |
| 5d Nov3 cluster− vs independent QAOA | −0.130 | [−0.216, −0.040] | 5/8 | −6.35 / −6.22 |
| 6a unsigned-mean Nov3 − unsigned QAOA (replication) | −0.680 | [−0.788, −0.582] | 8/8 | −5.59 / −4.91 |
| 6b sign-aligned Nov3 − unsigned QAOA (replication) | +1.770 | [+1.459, +2.047] | 0/8 | −3.14 / −4.91 |

Per-instance L=1 Nov3 still wins on train (ΔE/N −0.0249) and on the test ceiling (ΔE/N −0.0286). Two-template Nov3 beats two-template QAOA, which is the preregistered primary gate. Two-template Nov3 also beats independently fitted test QAOA (contrast 3; CI entirely <0, 5/8).

That does **not** replace per-instance SPSA. Contrast 4: two-template Nov3 is worse than independently optimized Nov3 (mean E −6.35 vs −6.47; gap +0.124; **CI entirely >0**; 0/8). This is the descriptive asked of this confirmatory stream: unlike FT PR122, where the transfer-gap CI included 0, this N=9 draw's two-template Nov3 does not sit at the independent Nov3 ceiling. That matches the qualitative N=12 pattern (PR126 and FT PR127) rather than FT PR122's including-0 gap. That descriptive does not switch the primary, does not copy FT PR122, and does not license a new two-template size.

On this test draw the Nov3 two-template chose cluster− on 7/8 seeds (the majority train cluster) and cluster+ on 1/8. QAOA chose cluster− on 6/8 and cluster+ on 2/8. Each Nov3 cluster concentrates γ with s1 (cluster+ γ mean +0.392 std 0.036, n=2; cluster− γ mean −0.368 std 0.053, n=6). Unsigned mean still mixes those families (γ std 0.355, s1 std 0.813). Sign-aligned Nov3 collapses (E −3.14) because the s1-only flip leaves γ mixed (std 0.355). Unsigned QAOA also collapses (E −4.91) because β is bimodal; the two β clusters themselves are tight (std 0.005 / 0.019).

**Train concentration (n=8 caveat).** Nov3 s1 2/6, QAOA β 5/3. θ_K std 0.040 and θ_M std 0.047 still concentrate. After s1-only alignment s1 std 0.045, but γ remains mixed.

Unsigned-mean and sign-aligned replications on this new draw do not reopen PR119 and do not license unsigned-mean copies at other N.

32/32 independent fits ok. 56/56 transfer evals ok (1 query each). B* rejects 0. Charged 32824. Recipe hash matched. Seeds 10792–10807 (PR126), 10760–75 (PR125), 10728–43 (PR123), 10744–59 (FT PR122), 10776–91 (FT PR124), 10808–23 (FT PR127), 10824–39 (R35), 10856–71 (FT this hour), 10712–27 (R34), 10680–95, 10648–63, 10600–15, 10624–31, 10560–67 unused. No L2+, no L3, no GR patch, no candidacy, no final. No N6/N21.
