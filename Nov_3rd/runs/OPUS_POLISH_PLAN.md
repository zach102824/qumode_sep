# OPUS polish plan — Nov_3rd L=1 angles

_Asia/Shanghai 2026-11-03 ~15:16 — Opus 5.5 **medium** (`claude-opus-5-5-medium`)_  
Source: `explore_OCT29/runs/opus_ideas/opus_reply_20261103_1516_nov3rd_polish_plan.md`

## One-liner
At L=1 accept the warm-start **TIE**. Make the training objective agree with Med and keep steps close to warm. **First** audit whether \(n=6\) gradients align with \(n=15\); if not, polishing on \(n=6\) cannot transfer.

## Chosen next steps (in order)
1. **Gradient-transfer audit** at L=1 warm (Idea 1) — &lt;1 h. Cosines of \(g_6\) vs \(g_{15}\) (and \(g_{12}\) vs \(g_{15}\)) on γ / linear-β / non-linear φ parts.  
   - Pass: non-linear \(\cos(g_6,g_{15})\ge0.5\) → keep n=6.  
   - Kill: \(\cos&lt;0.2\) → abandon n=6 for non-linear φ (Idea 5).
2. **Log-ratio loss + proximal nested basis** (Ideas 2+3) at L=1 — few hours.  
   \(\mathcal L=\mathrm{mean}_i\log(E_i(\theta)/E_i^{\mathrm{warm}})\); fit only \(\{\gamma,\beta,c_2\}\) with \(\lambda\|\theta-\theta_{\mathrm{warm}}\|^2\), \(\lambda\in\{1,0.1,0.01\}\), val n=9, gate n=15.  
   - Pass: train Med ≤ warm; \(R\le1.02\) (target \(\le1.00\)) on n12/n15.  
   - Neutral/kill: lock L=1 TIE (warm) and go to grow.
3. **Grow L from warm, polish new layer only** (Idea 4) — E3-style. Confirm warm=QAOA at L=4 first; then L=1→4 with new-layer proximal polish + short all-layer pass.

## Keep vs abandon n=6
- **Keep** n=6×1000 by default for γ and β (QAOA subspace concentrates).  
- **Abandon** n=6 for non-linear φ if Idea 1 kills transfer **or** Ideas 2+3 still UNDER at n15 → switch train to n9+n12 (or n6+n12), n15 sealed gate (Idea 5).

## Confirmed warm table (debug, Asia/Shanghai)
n6/9/12/15 warm Med = 9.779 / 29.279 / 90.033 / **289.893** (all \(R=1.000\) vs QAOA L=1).

## Hard protocol (already in PLAN.md)
Always record warm-only baseline; reject polish if train Med &gt; warm Med; gate \(R\le1.02\); warm is fallback at every L. No Hc-free projector tricks. LBFGS only.
