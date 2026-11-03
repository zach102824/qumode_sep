**L=2 ALL_CORE under this lock is likely impossible** (nested capacity, not a missing seed). Treat Round 2 as a **lucky-n15** hunt: maximize eval n15 R drop, accept ALL_CORE as a lottery ticket, do not spend compute re-running ratio69_w9.

## Why ALL_CORE is probably closed

Three stacked ceilings, all on the same locked head:

1. **Even with n15 in train**, dense 803-start search on a disjoint n15 pack hit **train R=0.9803** (reopen bar 0.978 never hit, `capacity_proven=true`). That same point **eval’d at n15 R=0.9965**. Best honest scale1000 n15 with n15 in the mix is **0.983** (free-φ) / **0.984** (minimax). L=1 n15=0.984 ≈ L=2: the extra layer buys ~0.001.
2. **train{6,9} can only be worse.** Best n15 so far is **trust69 ρ=0.05 → 0.987**, 0.007 above 0.98 and 0.004 above the n15-in-train eval floor.
3. **Pareto lock inside 8 parameters.** ratio69_w9 s0 **BEATs n6/9/12** (0.958/0.976/0.979) by growing the residual (`c2_rms=0.042`) and **kills n15** (0.999). Tiny residual **helps n15** (0.987) and **loses n6/n9**. ALL_CORE needs both; they point opposite ways.

Smoking gun from s0: n15 **B90** 217.5 vs QAOA 219.5 (tail moved) while **Med** 75.98 vs 76.09 (median did not). Mean-log on {6,9} is the wrong functional for eval R.

Honest expectation: n15 **0.984–0.990**. P(single-seed n15≤0.98) ~15%; P(ALL_CORE) ≲10%. The only existence proof that eval can beat dense-train is free-φ (eval 0.983 vs dense-eval 0.997): a *more transferable* point, not a deeper train minimum.

---

## Ranked Round 2 — maximize n15 drop / lucky BEAT

Do **not** add ratio69_w9 seeds. s0 vs s1 already mapped that basin (n15 0.999 / 1.002). `p=1` rule-scaled is dead. Free-φ under train{6,9} will overfit n9. n12/n15 stay out of the loss.

### 1. `n9med_trust` — n9-only median in a tiny dual-anchored ball

**Claim:** trust ρ=0.05 is already the train69 n15 champ (0.987) and was under-tried (1 seed, n6 still in the loss, mean-ish holdout). Drop n6 from the gradient, match Med, seed the small ball. n6 is a kill-guard only (still legal under train{6,9}).

| | |
|---|---|
| **Frozen** | γ, c1 at QAOA dual (ρ=0 = exact QAOA) |
| **Free** | c2, c3 per layer (4 params), projected onto global ℓ2 ball **ρ ∈ {0.03, 0.05, 0.08}** |
| **Loss** | soft-median (temp 0.05) of **inv_p** on **n9 only**, divided by the fixed QAOA n9 Med (this *is* R, not mean log-ratio). λ=0; the ball is the regularizer |
| **Opt** | Adam 5e-4 × 800, then LBFGS 16×8. Batch 64 on n9. Early-stop: **30% n9 holdout**, checkpoint on true median inv_p of holdout |
| **Seeds** | **0–7 at ρ=0.05 first** (8× ~45s), then 0–3 at 0.03 and 0.08 |

**Kill**
- After Adam: n9 holdout median worse than QAOA (logR>0) → dead seed.
- n6 eval R **< 0.96** → residual too large, n15 will repeat ratio69 (0.999). Stop that ρ.
- After 8 seeds at ρ=0.05: if holdout has not beaten trust69 ρ=0.05’s holdout (−0.0084), stop the sweep.

**Why n15 can move when ratio69_w9 failed.** ratio69 freed γ,c1, put n6 in the mean-log, and selected the checkpoint with the best n9 *mean*. That is anti-correlated with n15 once the residual is large (s0: n9 0.976 / n15 0.999 vs ρ=0.05: n9 1.002 / n15 0.987). Same 4 knobs, right metric, no n6 pull. Expected n15 **0.984–0.990**. This is the highest-leverage untried axis on the only direction that already transferred.

---

### 2. `shrink_p_hedge` — mild c2,c3 shrink, **fixed** p ∈ (0,1), not p=1 and not freely learned from n6 vs n9

**Claim:** at L=2, unscaled shared residual overshoots as n grows (0.958→0.976→0.979→0.999). L=16 `power_p` learned **p=+0.051** and improved n15 0.895→0.883. `rule_scaled` p≡1 killed n15 (0.9995). The untried band is **0 < p < 1**. Do **not** let mean-log n6 vote p into a noisy gauge.

| | |
|---|---|
| **Frozen** | none required. Fallback: freeze γ,c1 if n9 holdout dies |
| **Free** | γ, c1, c2, c3 per layer. **p fixed** in **{0.15, 0.30, 0.45}** (a_n=(8/(n−1))^p, a9=1). p=0.30 ⇒ a6≈1.15, a12≈0.91, a15≈0.84. Plus 2 seeds with learned p ~ N(0.30, 0.15), clip [0, 0.8], penalty on **scaled** c2,c3 |
| **Loss** | soft-median log-ratio, **w6=0.25, w9=1.0**. λ: 1e-2 → 1e-3 on (a_n c2, a_n c3) |
| **Opt** | Adam 1e-3 × 600, LBFGS 12×8, batch 32/n. Early-stop 25% n9 holdout median |
| **Seeds** | 0–3 per fixed p (12 runs) + learned-p seeds 0,1 |

**Kill**
- n9 R worse than ratio69 s0 (0.976) by >0.005 **and** n6 also worse → p too large.
- Learned p pins to 0 or 0.8.
- Implied n15 residual a15·c2_rms **> 0.05** (back in the overshoot regime).

**Why n15 can move when ratio69_w9 failed.** ratio69 used a_n≡1; the two train sizes’ only legal extra degree of freedom is one exponent. Mild shrink keeps the n9-sized residual at n9 and damps it ~15–25% at n15 — enough to fight overshoot, not the p=1 collapse. Expected n15 **0.985–0.995**. Lucky BEAT only if some p lands near the dense basin *and* transfers (the free-φ pattern).

---

### 3. `alpha_ray` — 1D rays from QAOA through known L=2 / distilled L=16 directions

**Claim:** n15-good and n6/n12-good solutions are **different rays from QAOA**. GD on {6,9} cannot sit at the Pareto knee because n15 is invisible. A 1D α scan can. L=16 already knows an n15-transferring dual-break (R=0.883); a 2-layer slice of that is an untried init.

| | |
|---|---|
| **Frozen** | the direction. Only a scalar α, then a tiny orthogonal δ |
| **Free** | α on each ray; after α* is picked, δ in the c2,c3 plane with ‖δ‖≤0.02 |
| **Rays** | (1) QAOA→ratio69_w9 s0 (2) QAOA→trust ρ=0.05 (3) QAOA→trust ρ=0.1 (4) QAOA→L=16 `power_p` s0 **mean-pool** layers 0–7 and 8–15 as the two L=2 layers, rescale ‖(c2,c3)‖ to 0.08 (5) QAOA→L=16 layers {0,15} |
| **Loss** | pick α∈{0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2} on n9 holdout soft-median, **penalize** if n6 logR is >0.02 better than n9 (ratio69 overfit signature). Then polish (α, δ) with the same n9 median loss. **Never** score α on n12/n15 |
| **Opt** | grid is eval-only (~minutes). Polish: Adam 3e-4 × 300 + LBFGS 8 |
| **Seeds** | rays deterministic; polish seeds 0–3 |

**Kill**
- No α∈(0,1) beats α=0 on n9 holdout → dead ray.
- Selected α* has n6 R < 0.96 → too far, n15 will be ≥1.00.
- Distill rays: n9 holdout worse than QAOA by 0.005.

**Why n15 can move when ratio69_w9 failed.** ratio69 committed to one basin and early-stopped on n9 mean. The train69 n15 champ already lives on the trust ray; n12-BEAT lives on the ratio69 ray. α∈(0.3,0.6) on trust, or an L=16-distilled direction that already transfers at depth, is a 1D search in 8-D space and cannot memorize n9. This is the actual lucky-BEAT lottery: a more transferable point than the dense overfit. Expected n15 **0.985–0.992**.

---

## Run order

1. **Idea 1, ρ=0.05, seeds 0–7** (hours, not days). If best n15 still ≥0.990, ALL_CORE is done; write the capacity note.
2. Idea 2 p-hedge in parallel on spare cores (L=2 is cheap).
3. Idea 3 after Idea 1 adds endpoints; include the two L=16 distill rays.

**Stop the shallow hunt** if Idea 1’s best eval n15 is ≥0.986 (no movement vs ρ=0.05 s0) *and* no hedge/ray is ≤0.984. That would confirm train{6,9} cannot reach the 0.983 oracle floor, let alone 0.98.

L=16 is already ALL_CORE; do not spend Round 2 there.
