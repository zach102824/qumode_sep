# N15TWO — TWO_TEMPLATE_TRANSFER_NEGATIVE_CI. Not adoption. GR unpatched.

Two-template bimodal cluster-mean transfer at N=15 L=1. Completes the two-template ladder at the next untested size after PR123 N=18. Same protocol as PR123; not an unsigned-mean copy at another N, not a post-hoc of PR116/PR119/PR121/PR123 test sets, not L2/L3. Does not copy sibling FT PR122 N=9 and does not claim cross-stream independence from it. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10760–10775**. Train **10760–10767**, test **10768–10775**, split frozen before fits. Original shared-Uc Nov3 vs QAOA p=1. Energy selects on independent fits. F post hoc.

From train only: Nov3 partitioned by sign(s1) into cluster+ (n=4) and cluster− (n=4); QAOA partitioned by sign(β) into cluster+ (n=1) and cluster− (n=7). All four clusters nonempty, so no method reduced to a single template and no empty cluster invented a template. QAOA cluster+ is a singleton (the one train β>0 vector). Two-template energy = min of that method's two charged frozen-template energies (energy-only; not F; not test-time SPSA). Unsigned mean and s1-only sign-aligned mean frozen and evaluated as secondary replication only. Primary vs secondary was not switched after seeing test.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.** Test better-of-two Nov3 minus better-of-two QAOA CI entirely <0 on n=8. Not a GR write. Does not revive L2/L3. Does not replace per-instance SPSA.

| contrast | mean ΔE | CI | wins | mean E a / b |
|---|---|---|---|---|
| primary two-template Nov3 − two-template QAOA | −0.388 | [−0.457, −0.333] | 8/8 | −12.50 / −12.12 |
| 1 train independent Nov3−QAOA | −0.331 | [−0.398, −0.273] | 8/8 | −12.11 / −11.78 |
| 2 test ceiling independent Nov3−QAOA | −0.324 | [−0.388, −0.265] | 8/8 | −12.52 / −12.19 |
| 3 two-template Nov3 vs independent QAOA | −0.308 | [−0.364, −0.254] | 8/8 | −12.50 / −12.19 |
| 4 transfer gap two-template Nov3 vs independent Nov3 | +0.016 | [−0.016, +0.046] | 3/8 | −12.50 / −12.52 |
| 5a Nov3 cluster+ vs unsigned QAOA | −0.584 | [−0.814, −0.382] | 8/8 | −12.42 / −11.84 |
| 5b Nov3 cluster− vs unsigned QAOA | −0.614 | [−0.752, −0.503] | 8/8 | −12.45 / −11.84 |
| 5c Nov3 cluster+ vs independent QAOA | −0.230 | [−0.330, −0.119] | 7/8 | −12.42 / −12.19 |
| 5d Nov3 cluster− vs independent QAOA | −0.260 | [−0.317, −0.189] | 8/8 | −12.45 / −12.19 |
| 6a unsigned-mean Nov3 − unsigned QAOA (replication) | +1.659 | [+1.571, +1.741] | 0/8 | −10.18 / −11.84 |
| 6b sign-aligned Nov3 − unsigned QAOA (replication) | +1.328 | [+1.232, +1.419] | 0/8 | −10.51 / −11.84 |

Per-instance L=1 Nov3 still wins on train (ΔE/N −0.0221) and on the test ceiling (ΔE/N −0.0216). Two-template Nov3 beats two-template QAOA, which is the preregistered primary gate. Two-template Nov3 also beats independently fitted test QAOA (contrast 3).

That does **not** replace per-instance SPSA. Contrast 4: two-template Nov3 is slightly worse than independently optimized Nov3 (mean E −12.50 vs −12.52; gap +0.016; CI includes 0; 3/8). Energy-min among two frozen templates is extra matching opportunity relative to a single shared vector; QAOA received the same two-template opportunity.

On this test draw the Nov3 two-template chose cluster+ on 4/8 seeds and cluster− on 4/8. QAOA chose cluster− on 6/8 and cluster+ on 2/8. Each single Nov3 cluster mean sits near the independent Nov3 ceiling (−12.42 / −12.45 vs −12.52) because each cluster concentrates γ with s1 (cluster+ γ mean +0.331 std 0.077; cluster− γ mean −0.281 std 0.013). Unsigned mean still mixes those families (γ std 0.331, s1 std 0.927) and collapses (E −10.18). Sign-aligned Nov3 also collapses (E −10.51) because the s1-only flip leaves γ mixed (std 0.331). Unsigned QAOA is milder than unsigned Nov3 (−11.84 vs independent −12.19) because β is 1/7 rather than 4/4; the majority β− cluster is tight (std 0.004) while β+ is a singleton.

**Train concentration (n=8 caveat).** Nov3 s1 4/4, QAOA β 1/7. θ_K std 0.046 and θ_M std 0.020 still concentrate. After s1-only alignment s1 std 0.028, but γ remains mixed (std 0.331), which is why sign-aligned Nov3 does not match either cluster mean.

Unsigned-mean and sign-aligned replications on this new draw do not reopen PR119, do not post-hoc-gate PR116's N=15 unsigned-mean test set, and do not license unsigned-mean copies at other N.

32/32 independent fits ok. 56/56 transfer evals ok (1 query each). B* rejects 0. Charged 32824. Recipe hash matched. Seeds 10728–43 (PR123), 10744–59 (FT PR122), 10776–91 (FT next), 10712–27 (R34), 10680–95, 10648–63, 10600–15, 10624–31, 10560–67 unused. No L2+, no L3, no GR patch, no candidacy, no final.
