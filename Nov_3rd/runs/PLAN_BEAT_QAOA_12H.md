# PLAN: Beat QAOA 12h loop (training-only) — 2026-11-03 Asia/Shanghai

**Status:** RUNNING (Zach GO t2552u) — R0 L1/L2 PASS; L16 dual + R1 training.  
**Opus Round-1 ideas:** `OPUS_BEAT_QAOA_12H.md` · raw `explore_OCT29/runs/opus_ideas/opus_reply_20261103_2034_beat_qaoa_12h.md`  
**Deliverables root:** `Nov_3rd/runs/beat_qaoa_12h/`

---

## 1. Goal + locks + allowed axes

**Goal:** Clear beat of fair QAOA (R = Med / Med_QAOA ≤ **0.98**) at **different layer counts**, specifically shallow **L∈{1,2}** and deep **L=16**. Prefer R≤0.98 on all CORE n∈{6,9,12,15}; minimum acceptable clear beat on **n12 and n15**.

**Locked (DO NOT change):**
- Quantum gate sequence / motif / W / shell head definition / hardware layout.
- Motif: Nov1st-final X-sandwich + Hamming-shell SNAP (φ_k k=0..3), W=H^{⊗3}.
- QAOA-warm available: φ_k = −βk (exact dual when c2=c3=0).
- Fair QAOA baseline: paper init β=0.01, γ=−0.01, same L, scale1000 B=1000 Med.

**Allowed (training-only):**
- Train set / sample counts / which n
- Optimizer (Adam, L-BFGS, projected, multi-start)
- Init / warm / polish protocol
- Loss (incl. ratio-to-QAOA, dual-anchor λ, CVaR, etc.)
- Dual-anchor freezes (γ, c1, …)
- Strength scaling a_n / κ
- Early-stop / trust-region ρ / layer smoothness
- Seed sweeps

**Not allowed:** new gates, Arm C densified-projector rebuild (last-resort caveat only if all training ideas tie at L=16).

**Compute:** 8 cores on box.

---

## 2. Eval protocol

| Item | Value |
|------|-------|
| Eval depths | **L∈{1,2,16} only** (no primary L=4/8/12) |
| Eval set | `scale1000` B=1000 |
| CORE n | {6,9,12,15}; n18 diagnostic only |
| Metric | Med cost; R = Med / Med_QAOA (same L, fair paper-init QAOA) |
| BEAT | R ≤ 0.98 |
| TIE | 0.98 < R ≤ 1.02 |
| UNDER | R > 1.02 |
| QAOA ref | paper init β=0.01 γ=−0.01, train/ref from n6 as prior fair protocol |

Before claiming any beat: **Round 0 dual-identity gate** — exact QAOA dual with no training must give R=1.000±0.005 on CORE at each of L∈{1,2,16}. Fix sign/mapping/γ-scaling if not. Hash-check train instances ∉ scale1000.

---

## 3. Round structure (Opus → 3 ideas → run → table → Opus)

```
R0 gate (dual identity + hash + smoke)
  ↓
loop until ~12h or clear beat at shallow AND deep:
  Ask Opus → exactly 3 training-only ideas (ranked)
  Run trio on 8 cores (parallelize by idea × L × seed)
  Write per-round RESULTS table under beat_qaoa_12h/round_k/
  Ask Opus with results → kill/promote + 1 knob change per survivor
  Promote winner; fill missing depth; confirm seeds
  ↓
Final scale1000 champion @ L∈{1,2,16} → RESULTS.md
```

**After each trio, ask Opus:**
1. Per-n R table (CORE + n18), train-ratio vs held-out R, ‖c2,c3‖ vs n15 R — generalization gap vs optimizer limit?
2. Exactly one knob change per surviving idea; which to kill/replace (replacement must stay training-only).
3. If shallow beats but L=16 ties: try interpolate winning L=2 schedule → L=16 init instead of QAOA-warm.

---

## 4. Round 1 ideas (from live Opus 2026-11-03 ~20:36 Asia/Shanghai)

**Highest-EV start:** Idea 1 CORE-RATIO @ L=2, with Idea 1 L=16 seed0 in parallel.

### Idea 1 — CORE-RATIO (rank 1)
| Knob | Setting |
|------|---------|
| Train ns | {6,9,12,15} sealed, 250/n, disjoint from scale1000 |
| Freeze | none — γ,c1,c2,c3 free per layer; no a_n |
| Loss | (1/4) Σ_n mean_i log(E^θ/E^QAOA) + λ(c₂²+c₃²), λ: 1e-1→1e-3 |
| Opt | QAOA-warm → Adam(1e-3,600) → L-BFGS(12); seeds {0,1,2}; shallow +32 multi-starts |
| L first | **L=2**, then L=16, then L=1 |
| Success | R≤0.98 all CORE; min: n12 & n15 ≤0.98 |
| Wall 8c | L=1/2 ~0.5–1h; L=16×3 ~2–3h |
| Kill | train ratio >0.99 on n12/15 after Adam; or holdout n15 R>1 while train ≤0.98 |

### Idea 2 — RULE-SCALED {9,12} (rank 2)
| Knob | Setting |
|------|---------|
| Train ns | {9,12}; holdout n6/n15; n18 diagnostic |
| Freeze/scale | shared γ,c1,c2,c3; \(c_{2,3}^{(n)}=\kappa\cdot\frac{2}{n-1}\,c_{2,3}\); κ trained; c1,γ unscaled |
| Opt | QAOA-warm → Adam(800) → L-BFGS(8); seeds {0,1,2}; shallow κ∈{0.5,1,2} |
| L first | **L=16**, then L=2 |
| Success | R≤0.98 on n12 (train) and n15 (holdout) |
| Wall 8c | L=16 ~1.5–2h; L=2 <0.5h |
| Kill | n15 R>1 at every κ; or κ→0 (TIE collapse) |

### Idea 3 — TRUST-RESIDUAL (rank 3)
| Knob | Setting |
|------|---------|
| Train ns | {9,12,15} |
| Freeze | γ,c1 frozen at QAOA; train c2,c3 only; ‖(c2,c3)‖₂ ≤ ρ, ρ∈{0.02,0.05,0.1,0.2} |
| Opt | dual start → projected Adam(5e-4,600) → L-BFGS in ball; seeds {0,1}; L=16 layer smoothness |
| L first | **L=16**, then L=2 |
| Success | R≤0.98 on n12 and n15 at selected ρ |
| Wall 8c | L=16 ρ-sweep ~2h; L=2 <0.5h |
| Kill | at ρ=0.2 train ratio still >0.985 → hand off to Idea 1 |

### Suggested R1 schedule (8 cores)
| Time | Round | Runs |
|------|-------|------|
| 0:00–0:45 | R0 | Dual identity @ L∈{1,2,16}; train/test hash; timing smoke |
| 0:45–3:30 | R1 | All 3 ideas @ L=2 (2c each) + Idea1 L=16 seed0 (2c) |
| 3:30–7:00 | R2 | L=16 for top-2 + revised 3rd; L=1 for R1 winner |
| 7:00–10:30 | R3 | Fill missing depth; confirmation seeds |
| 10:30–12:00 | R4 | Final scale1000 B=1000 champion @ L∈{1,2,16}; RESULTS.md |

---

## 5. Kill / promote rules

**Promote** an idea if:
- R≤0.98 on n12 and n15 at the tested L, OR
- clear improvement vs warm TIE with train-ratio ≤0.98 and held-out not UNDER.

**Kill** an idea if its kill criterion fires (see §4), or after two rounds it never beats n15.

**Replace** killed idea with Opus-suggested training-only variant (1 knob change preferred).

**Champion** = best R on worst of {n12,n15} among ideas that also don’t UNDER on other CORE n; must eventually cover both shallow (L=1 or 2) and deep (L=16).

**Hard stop:** if R0 dual-identity fails → fix mapping before any polish training.

**Last resort (not Round 1):** only if all three training ideas tie/UNDER at L=16 after R2–R3, ask Opus whether Arm C-style head is required (out of training-only scope — Zach call).

---

## 6. Deliverables path

```
Nov_3rd/runs/beat_qaoa_12h/
  README.md                 # pointer to this PLAN + status
  round_0/                  # dual identity + hash + smoke
    dual_identity.json
    hash_disjoint.json
  round_1/                  # first trio results
    idea1_core_ratio/
    idea2_rule_scaled/
    idea3_trust_residual/
    RESULTS.md
  round_k/ …
  champion/
    params_L{1,2,16}.pt
    results_scale1000.json
  RESULTS.md                # final table L∈{1,2,16} × CORE n
  OPUS_log.md               # links to each round’s brief+reply
```

Supporting docs (already written):
- `Nov_3rd/runs/PLAN_BEAT_QAOA_12H.md` (this file)
- `Nov_3rd/runs/OPUS_BEAT_QAOA_12H.md`
- `explore_OCT29/runs/opus_ideas/brief_20261103_2034_beat_qaoa_12h.txt`
- `explore_OCT29/runs/opus_ideas/opus_reply_20261103_2034_beat_qaoa_12h.md`

---

## Next step

**Started** 2026-11-03 20:42 Asia/Shanghai. See `beat_qaoa_12h/STATUS.md`.
