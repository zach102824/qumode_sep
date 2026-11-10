# N18GEOM — GEOM_TRANSFER_NEGATIVE_CI. Not adoption. GR unpatched.

SK-geometry transfer at N=18 L=1. Freeze train-mean θ_K and θ_M (concentrated even when γ/s1 are bimodal), then per-instance v4 center-4 SPSA of the remaining original-Nov3 coordinates (γ, s1, s2, s3) by projecting those two coordinates in every energy call. Tests whether PR119's unsigned-all collapse is only from averaging the bimodal slice, without two-template min-selection and without dropping per-instance SPSA.

Not two-template. Not unsigned-mean of the full vector. Not sign-aligned as primary. Not L2/L3. Not a GR patch. Frozen v4 center-4 (`alloc.py` sha256 `de595de592f33a1d` / PR79 `347ac23369db`) on exclusive unread seeds **10872–10887**. Train **10872–10879**, test **10880–10887**, split frozen before fits. Freeze committed before any test look. Original shared-Uc Nov3 vs QAOA p=1. Energy selects. F post hoc. packing(18)=6.

Ideas considered this turn: (picked) freeze θ_K, θ_M then SPSA (γ,s1,s2,s3); (held) more aggressive freeze of θ_K, θ_M, s2, s3 with SPSA of only (γ,s1). s3 does not concentrate on this train (std 0.287), so the held freeze remains not justified. No second batch this turn.

**Label GEOM_TRANSFER_NEGATIVE_CI.** Test geom-Nov3 minus independently-fitted QAOA CI entirely <0 on n=8. Not a GR write. Does not revive L2/L3. Does not replace per-instance SPSA (geom still SPSA's four free coordinates per instance). Does not reopen PR119. Does not reopen two-template.

| contrast | mean ΔE | CI | wins | mean E a / b |
|---|---|---|---|---|
| primary geom Nov3 − independent QAOA | −0.356 | [−0.416, −0.302] | 8/8 | −17.14 / −16.78 |
| 1 test ceiling independent Nov3−QAOA | −0.346 | [−0.416, −0.284] | 8/8 | −17.13 / −16.78 |
| 2 transfer gap geom Nov3 vs independent Nov3 | −0.010 | [−0.020, −0.001] | 6/8 | −17.14 / −17.13 |

Per-instance L=1 Nov3 still wins the test ceiling (ΔE/N −0.0192). Geometry-frozen Nov3 also beats independently fitted test QAOA, which is the preregistered primary gate (ΔE/N −0.0198; 8/8).

Contrast 2 is descriptive only and does not switch the primary. On this draw the transfer gap is slightly negative (geom mean E −17.14 vs independent −17.13; CI entirely <0; 6/8). That is the opposite sign of the two-template gaps (those were ≥0). Magnitude is small (~3% of the QAOA edge) on n=8; it does not license dropping independent SPSA, does not license the held dim-2 freeze, and is not adoption. Geometry freeze still runs per-instance SPSA of (γ,s1,s2,s3); it is a constrained energy-map projection, not a frozen full vector.

**Train concentration (n=8 caveat).** θ_K mean −0.402 std 0.035; θ_M mean −0.025 std 0.067. γ std 0.229, s1 std 0.810 still mixed: s1 6/2 (six +, two −), with γ sign tracking s1. s2 std 0.091; s3 std 0.287 does not concentrate. This is the PR119 failure mode (bimodal s1/γ, concentrated geometry) with the bimodal slice left to per-instance SPSA.

32/32 fits ok. B* rejects 0. Audit tk/tm freeze passed on all 8 geom aux vectors. Charged 32768. Recipe hash matched. Seeds 10888–10903 (other stream), 10856–71, 10840–55 (PR129), 10824–39 (R35), 10808–23 (FT PR127), 10792–10807 (PR126), 10776–91 (FT PR124), 10760–75 (PR125), 10744–59 (FT PR122), 10728–43 (PR123), 10712–27 (R34), 10680–95, 10648–63, 10600–15, 10624–31 unused-hold, 10560–67 unused hole unused. No L2+, no L3, no GR patch, no candidacy, no final. No N6/N21. Two-template remains closed.
