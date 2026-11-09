# STARTS S1C — S1 vs S4 confirms on fresh-dev; still not a GR patch

PR83 is candidate-only. This batch froze those S1/S4 recipes (starts.py, C0 SPSA, K=1531 vs 379, TQA start-0) and refit them on eight unused development seeds 10056–10063 at N12 and N15, L=3, Q=3072 inclusive. No second-look tuning. S4 was fitted and charged on these seeds. F is diagnostic only.

**Confirmation passes. Do not patch GR.** The preregistered gate is Nov3 mean(E_S1 − E_S4) bootstrap 95% CI entirely below 0 at **both** sizes.

| size | mean ΔE | median | CI | S1 energy wins |
|---|---|---|---|---|
| N12 | −0.018 | −0.018 | [−0.023, −0.013] | 8/8 |
| N15 | −0.040 | −0.036 | [−0.054, −0.026] | 8/8 |

QAOA S1 also beat S4 at both sizes (8/8). The TQA-concentration improvement vs four-start C0 generalizes off the PR83 read-dev seeds.

**Fair Nov3 vs QAOA is a different claim.** Under S1, Nov3 is still higher energy than QAOA at N15: mean Δ(E−E0)/N = +0.0021, CI [+0.0003, +0.0039], Nov3 lower on 2/8. At N12 the energy CI includes 0 (3/8). Nov3 F is higher under S1 at N15 (mean ΔF +0.041, CI [+0.020, +0.066], 7/8) and mixed-to-higher at N12. Energy remains the selection rule. Higher F does not overturn the energy ranking.

64/64 fits ok. B* rejects 0. Charged 196608. Outside-cap 0. Frozen file hashes matched. N3COND 772224, LEXACT 2560, DEPTH C1 327680, STARTS S8 98304, STARTS S1 98304 unchanged. No S8 retry, no continuation, no N18, no GR/frozen-v1 patch. Stop.
