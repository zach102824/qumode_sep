# Opus: 3 training-only ideas to BEAT QAOA (2026-11-03 ~20:36 Asia/Shanghai)

**Call:** succeeded · model `claude-opus-5-5-medium` · ask mode · ~126s · exit 0  
**Sources:** `explore_OCT29/runs/opus_ideas/brief_20261103_2034_beat_qaoa_12h.txt` · raw `opus_reply_20261103_2034_beat_qaoa_12h.md`  
**Prior:** `OPUS_L12_POLISH_TRANSFER.md`, `OPUS_TRAIN69_BEAT_QAOA.md`  
**Status:** PLAN ONLY — no runs until Zach OKs. Eval L∈{1,2,16} only; gate sequence locked.

## Round 0 gate (must pass first)

Load exact QAOA dual at L∈{1,2,16}, no training → must give R=1.000±0.005 on n∈{6,9,12,15}. L=8 warm UNDER suggests sign/mapping/γ-scaling/sampling mismatch; any beat before this gate is untrusted. Also hash-check train instances vs scale1000 (disjoint).

## Ranked 3 ideas (training-only)

### 1. CORE-RATIO (highest EV Round 1)
- **Train ns:** {6,9,12,15} sealed, 250/n, instance-disjoint from scale1000.
- **Loss:** mean over n of mean_i log(E^θ / E^QAOA) + λ(c₂²+c₃²), λ anneal 1e-1→1e-3.
- **Freeze:** none — γ,c1,c2,c3 all free per layer; no per-n a_n.
- **Opt:** QAOA-warm → Adam (lr 1e-3, 600) → L-BFGS (12); seeds {0,1,2}; shallow: +32 multi-starts.
- **L order:** **L=2 first**, then L=16, then L=1.
- **Success:** R≤0.98 on all CORE; min OK: ≤0.98 on n12 and n15.
- **Wall 8c:** L=1/2 ~0.5–1h; L=16×3seeds ~2–3h.
- **Kill:** train ratio >0.99 on n12/15 after Adam; or held-out n15 R>1 while train ≤0.98 (raise λ).

### 2. RULE-SCALED {9,12}
- **Train ns:** {9,12}; hold out n6/n15; n18 diagnostic.
- **Freeze/scale:** shared γ,c1,c2,c3; dual-breaking scaled \(c_{2,3}^{(n)}=a_n c_{2,3}\) with \(a_n=\kappa\cdot 2/(n-1)\); κ trained; c1,γ unscaled (exact QAOA when c2=c3=0).
- **Opt:** QAOA-warm → Adam 800 → L-BFGS 8; seeds {0,1,2}; shallow κ∈{0.5,1,2} grid.
- **L order:** **L=16 first**, then L=2.
- **Success:** R≤0.98 on n12 (train) and n15 (holdout).
- **Wall:** L=16 ~1.5–2h; L=2 <0.5h.
- **Kill:** n15 R>1 at every κ; or fitted κ→0 (collapses to TIE).

### 3. TRUST-RESIDUAL
- **Train ns:** {9,12,15}.
- **Freeze:** γ and c1 at exact QAOA; train only c2,c3; hard \(\|(c_2,c_3)\|_2\le\rho\), ρ∈{0.02,0.05,0.1,0.2}; pick ρ on n15 val split.
- **Opt:** dual start → projected Adam (5e-4, 600) → L-BFGS in ball; seeds {0,1}; L=16 smoothness across ℓ.
- **L order:** **L=16 first**, then L=2.
- **Success:** R≤0.98 on n12 and n15 at selected ρ.
- **Wall:** L=16 ρ-sweep ~2h; L=2 <0.5h.
- **Kill:** at ρ=0.2 train ratio still >0.985 → frozen γ/c1 too tight; hand off to Idea 1.
- **Caveat:** safest (can't go worse than TIE) but gain may cap ~0.99×.

**Ranking why:** #1 trains on scored sizes (no transfer needed) and covers both depths; #2 cleaner size-generalization story but relies on n15 extrapolate; #3 rarely UNDER but gain capped. Arm C projector only if all three tie at L=16 (last resort — changes head).

## 12h loop sketch (Opus)

| Time | Round | Runs |
|------|-------|------|
| 0:00–0:45 | R0 | Dual identity gate L∈{1,2,16}; train/test hash; timing smoke |
| 0:45–3:30 | R1 | All 3 ideas @ L=2 (2c each) + Idea1 L=16 seed0 (2c) |
| 3:30–7:00 | R2 | L=16 top-2 + revised 3rd; L=1 for R1 winner |
| 7:00–10:30 | R3 | Fill missing depth; confirmation seeds |
| 10:30–12:00 | R4 | Final scale1000 B=1000 champion @ L∈{1,2,16}; RESULTS.md |

**Ask Opus after each trio:** per-n R (+n18), train vs held-out R, ‖c2,c3‖ vs n15 R → generalization vs optimizer failure; one knob change per survivor / kill-or-replace (training-only); if shallow beats but L=16 ties → ask L=2→L=16 schedule interpolate init.
