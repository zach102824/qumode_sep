# FT v4 center-4 L=2 — GR candidacy note (not a patch)

This is a dossier, not a GR write. PR77 / GR live queue is **not patched** here. FT holdout 10144–10151 and local R19 10184–10191 are in flight; validation 11000–11031 and final 20000–20999 stay unread. No extra N18 L=2 Q=2048 copy. No L3. No Nov9/SYK/AF/N21.

A later owner may adopt this recipe into GR **only after** those in-flight bands close. This file does not do that.

## Recipe (frozen, copied, not invented)

| field | value |
|---|---|
| FT PR | #79 `cursor/n3-init-restart-3851` |
| HEAD | `347ac23369db1251c902638994b4fdc65a019710` |
| `alloc.py` sha256 | `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` |
| bytes | 12143 |
| schedule | four equal SPSA blocks, Q=qL, L=2 ⇒ Q=2048, T=250 |
| gains | c=0.1, A=25, clip 2σ, a0=0.5/median(\|g\|/σ) |
| starts | center-4; Nov3 start 0: γ=0.25, θ=0, shells (−0.5,0,0)/layer; QAOA start 0: (γ,β)=(0.25,0.25)/layer |
| selection | lowest energy among init / last / Polyak / mid; F post hoc |

LE `Nov9_SK_SPSA/experiments/FTREP/alloc.py` is byte-identical to that hash. No schedule tweak on the N12/N15 replicate (PR #92) or the N18 scaling check (PR #93).

## Disjoint seed bands (no overlap)

| band | seeds | role | N | charged |
|---|---|---|---|---|
| FT train | 10018–10025 | FT PR79 development (not re-run here) | 15 | FT stream |
| LE replicate | 10152–10159 | FTREP PR #92 | 12 and 15 | 65536 |
| LE N18 scale | 10168–10175 | FTREP N18 PR #93 | 18 | 32768 |

Held out of this dossier: FT holdout **10144–10151**, local R19 **10184–10191**, local ≤10167 except the two LE bands above, GR 20000+, validation 11000–11031, final 20000–20999.

## Table 1 — FT original (seeds 10018–25, not re-run)

L=2 Q=2048, mean Nov3−QAOA ΔE, bootstrap 95% CI as reported by FT PR79.

| N | mean ΔE | CI | Nov3 lower |
|---|---|---|---|
| 15 | −0.095 | [−0.173, −0.011] | 6/8 |

## Table 2 — LE replicate (seeds 10152–59, PR #92)

Same recipe on LE m2 + `bstar_sparse`. Label **FT-L2 REPLICATED on LE**.

| N | mean ΔE | med | CI | tails | Nov3 lower | start0 Nov3 |
|---|---|---|---|---|---|---|
| 12 | −0.107 | −0.113 | [−0.188, −0.030] | [−0.314, +0.045] | 6/8 | 2/8 |
| 15 | −0.071 | −0.041 | [−0.136, −0.013] | [−0.235, +0.023] | 5/8 | 5/8 |

## Table 3 — LE N=18 scale (seeds 10168–75, PR #93)

Same recipe. Label **N18-L2 SCALED**.

| N | mean ΔE | med | CI | tails | Nov3 lower | start0 Nov3 |
|---|---|---|---|---|---|---|
| 18 | −0.109 | −0.127 | [−0.179, −0.036] | [−0.276, +0.046] | 6/8 | 5/8 |

All three mean-ΔE CIs lie entirely below 0. Sign is stable across the three disjoint bands. Magnitude at N=18 matches N=12, not a fade from N=15. Two of eight N=18 seeds still favor QAOA on energy (10172, 10174).

## Why this is only a candidate

LE S1JOINT / ALT / warm-start never produced a fair Nov3 energy win vs QAOA at L≥2. The portable piece is this frozen v4 center-4 allocator, not those closed axes.

Still **not** a GR patch: FT holdout and local R19 are live, this dossier does not touch PR77, and a later follow-up will assign the next numerical batch. Combined research total after PR #93: **2757248**. Stop.
