# STARTS S1 — TQA concentration is a candidate, not a GR patch

S8 is closed. Extra starts starved the TQA trajectory that usually wins. This batch put the same Q=3072 on that one start (S1, K=1531) and, cheaply, on a two-start split (S2, K=763). Historical four-start C0 (S4, K=379) was reused from PR82 after a field-level recipe audit: 16/16 files matched C0 (nstarts=4, K=379, Q=3072, live gain, cat 32/3032/8). S4 was not recharged. Seeds 10048–10055 are read-dev. Energy selects. F post hoc. Candidate development only.

**Adopt S1 as a candidate. Do not patch GR.** The preregistered gate is N15 L3 Nov3, n=8: bootstrap 95% CI of mean(E_S1 − E_S4) entirely below 0 **and** median ΔE < 0. Observed mean −0.038, median −0.033, CI [−0.053, −0.027]. S1 was lower energy on **8/8** seeds. Post hoc F moved with energy (mean ΔF +0.019, 7/8 S1 wins) and is not in the gate.

QAOA S1 also beat S4 on energy (mean ΔE −0.016, CI [−0.030, −0.007], 8/8). S2, not in the gate, beat S4 on Nov3 (mean ΔE −0.023, CI entirely below 0, 8/8) and on QAOA (mean −0.007, CI entirely below 0, 7/8). S2 selected start 0 on all eight Nov3 seeds, so the second start still rarely wins; it mainly costs K. S1 is the stronger concentration.

Under both S1 and S4, QAOA is lower energy than Nov3 (mean Δ(E−E0)/N of Nov3−QAOA CI entirely above 0). Nov3 has higher post hoc F under S1 (mean ΔF +0.029, 6/8). Energy remains the selection rule.

32 new fits ok. Newly charged 98304. Outside-cap 0. S4 49152 not added again. N3COND 772224, LEXACT SPSA 2560, DEPTH C1 327680, STARTS S8 98304 unchanged. No more-starts retry, no continuation, no GR/frozen-v1 patch. Stop.
