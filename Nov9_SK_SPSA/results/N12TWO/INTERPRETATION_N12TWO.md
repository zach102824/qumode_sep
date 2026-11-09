# N12TWO — TWO_TEMPLATE_TRANSFER_NEGATIVE_CI. Not adoption. GR unpatched.

Independent-stream confirmatory two-template bimodal cluster-mean transfer at N=12 L=1. Same protocol as PR125 / FT PR124. Not a new untested size, not an unsigned-mean copy, not a post-hoc of PR116/PR119/PR121/PR123/PR125 or FT PR124 test sets, not L2/L3. Does not copy FT PR124 numbers and does not reuse 10776–91. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10792–10807**. Train **10792–10799**, test **10800–10807**, split frozen before fits. Original shared-Uc Nov3 vs QAOA p=1. Energy selects on independent fits. F post hoc.

From train only: Nov3 partitioned by sign(s1) into cluster+ (n=5) and cluster− (n=3); QAOA partitioned by sign(β) into cluster+ (n=5) and cluster− (n=3). All four clusters nonempty, so no method reduced to a single template and no empty cluster invented a template. Two-template energy = min of that method's two charged frozen-template energies (energy-only; not F; not test-time SPSA). Unsigned mean and s1-only sign-aligned mean frozen and evaluated as secondary replication only. Primary vs secondary was not switched after seeing test. Transfer-gap CI entirely >0 is a descriptive of contrast 4 and does not change the primary gate.

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.** Test better-of-two Nov3 minus better-of-two QAOA CI entirely <0 on n=8. Not a GR write. Does not revive L2/L3. Does not replace per-instance SPSA.

| contrast | mean ΔE | CI | wins | mean E a / b |
|---|---|---|---|---|
| primary two-template Nov3 − two-template QAOA | −0.306 | [−0.383, −0.242] | 8/8 | −9.93 / −9.62 |
| 1 train independent Nov3−QAOA | −0.267 | [−0.312, −0.223] | 8/8 | −9.69 / −9.42 |
| 2 test ceiling independent Nov3−QAOA | −0.286 | [−0.335, −0.241] | 8/8 | −9.96 / −9.67 |
| 3 two-template Nov3 vs independent QAOA | −0.259 | [−0.301, −0.207] | 8/8 | −9.93 / −9.67 |
| 4 transfer gap two-template Nov3 vs independent Nov3 | +0.028 | [+0.006, +0.050] | 2/8 | −9.93 / −9.96 |
| 5a Nov3 cluster+ vs unsigned QAOA | −1.694 | [−1.957, −1.460] | 8/8 | −9.88 / −8.18 |
| 5b Nov3 cluster− vs unsigned QAOA | −1.713 | [−1.920, −1.525] | 8/8 | −9.89 / −8.18 |
| 5c Nov3 cluster+ vs independent QAOA | −0.205 | [−0.283, −0.110] | 7/8 | −9.88 / −9.67 |
| 5d Nov3 cluster− vs independent QAOA | −0.223 | [−0.266, −0.177] | 8/8 | −9.89 / −9.67 |
| 6a unsigned-mean Nov3 − unsigned QAOA (replication) | −0.079 | [−0.097, −0.063] | 8/8 | −8.26 / −8.18 |
| 6b sign-aligned Nov3 − unsigned QAOA (replication) | −0.765 | [−0.836, −0.695] | 8/8 | −8.95 / −8.18 |

Per-instance L=1 Nov3 still wins on train (ΔE/N −0.0223) and on the test ceiling (ΔE/N −0.0239). Two-template Nov3 beats two-template QAOA, which is the preregistered primary gate. Two-template Nov3 also beats independently fitted test QAOA (contrast 3).

That does **not** replace per-instance SPSA. Contrast 4: two-template Nov3 is worse than independently optimized Nov3 (mean E −9.93 vs −9.96; gap +0.028; **CI entirely >0**; 2/8). This is the descriptive asked of this confirmatory stream: unlike PR123/PR125, where the transfer-gap CI included 0, this N=12 draw's two-template Nov3 does not sit at the independent Nov3 ceiling. That descriptive does not switch the primary.

On this test draw the Nov3 two-template chose cluster+ on 4/8 seeds and cluster− on 4/8. QAOA chose cluster+ on 6/8 and cluster− on 2/8. Each single Nov3 cluster mean sits near but not at the independent Nov3 ceiling (−9.88 / −9.89 vs −9.96) because each cluster concentrates γ with s1 (cluster+ γ mean +0.325 std 0.038; cluster− γ mean −0.286 std 0.052). Unsigned mean still mixes those families (γ std 0.319, s1 std 0.898) and collapses (E −8.26). Unsigned QAOA also collapses (E −8.18) because β is bimodal (mean +0.104, std 0.403); the two β clusters themselves are tight (std 0.005 / 0.010).

**Train concentration (n=8 caveat).** Nov3 s1 5/3, QAOA β 5/3. θ_K std 0.042 and θ_M std 0.049 still concentrate. After s1-only alignment s1 std 0.028, but γ remains mixed (std 0.319), which is why sign-aligned Nov3 (−8.95) does not match either cluster mean.

Unsigned-mean and sign-aligned replications on this new draw do not reopen PR119 and do not license unsigned-mean copies at other N.

32/32 independent fits ok. 56/56 transfer evals ok (1 query each). B* rejects 0. Charged 32824. Recipe hash matched. Seeds 10776–91 (FT PR124), 10760–75 (PR125), 10728–43 (PR123), 10744–59 (FT PR122), 10808–23 (FT this hour), 10712–27 (R34), 10680–95, 10648–63, 10600–15, 10624–31, 10560–67 unused. No L2+, no L3, no GR patch, no candidacy, no final.
