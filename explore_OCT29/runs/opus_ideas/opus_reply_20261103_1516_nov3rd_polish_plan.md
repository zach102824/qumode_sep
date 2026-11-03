## 1. One-liner for Zach

At L=1, accept the warm-start tie. Spend the polish budget on making the training objective agree with Med, and keep each step close to the warm start. Before tuning any optimizer, measure whether n=6 gradients point the same way as n=15 gradients. If they don't, no amount of polishing on n=6 will transfer.

## 2. Ideas, ranked

**Idea 1 — Gradient-transfer audit at the warm start**
- **Why:** Free polish moved \(\|\Delta\phi\|\approx0.34\) and made results worse from n=12 upward. Even the sealed n9+n12 run (E1) failed at n=15. This looks like the n=6 gradient does not transfer across sizes. At n=6 there are only 2 cavities, so cross-cavity structure is barely represented.
- **Cheapest decisive experiment:** Evaluate at the L=1 warm only, with no optimization. Compute the loss gradient \(g_n\) for n ∈ {6, 9, 12, 15}, using 200–1000 instances each. Split each gradient into its γ part, its β part (along \(\phi_j\propto j\)), and its non-linear part (\(\phi_j\) minus its linear fit in j). Report \(\cos(g_6,g_{15})\) and \(\cos(g_{12},g_{15})\) for each part.
- **Pass:** The non-linear part has \(\cos(g_6,g_{15})\ge0.5\). Keep training on n=6.
- **Kill:** \(\cos(g_6,g_{15})<0.2\) or negative. n=6 cannot drive the extra degrees of freedom; see section 3.

**Idea 2 — Align the objective with Med, and reject anything worse than warm**
- **Why:** On n=6 itself, train Med got worse than warm (≈9.61 → 9.7–10.0). That points to a mismatch between the loss and Med, not to overfitting. Check whether the loss went down while Med went up.
- **Loss to use:** Per-instance log-ratio to warm, \(\mathcal L=\mathrm{mean}_i\log(E_i(\theta)/E_i^{\text{warm}})\). It is scale-free and robust to outliers, so it tracks the median much better than a mean energy does. A smoothed 0.5-quantile is an alternative, and both work fine with L-BFGS.
- **Cheapest decisive experiment:** L=1, 3 seeds, L-BFGS on n=6×1000 with the log-ratio loss. Then evaluate scale1000 on n=9, 12 and 15.
- **Pass:** Train Med ≤ warm Med on all seeds, and R ≤ 1.02 at n=12 and n=15.
- **Kill:** Train Med still above warm. The problem is then the parameterisation, not the objective, so go to Idea 3.

**Idea 3 — Proximal L-BFGS in a nested basis**
- **Why:** This is the trust-region idea, done properly. Write \(\phi_j=-\beta j+\sum_{k\ge2}c_k P_k(j)\) using orthogonal polynomials in j. Let γ, β and \(c_2\) move first and keep higher orders frozen. Add a penalty \(\lambda\|\theta-\theta_{\text{warm}}\|^2\). The QAOA subspace is what transfers; the extra freedom from shell SNAP should only come in a little at a time.
- **Cheapest decisive experiment:** L=1, fit only \(\{\gamma,\beta,c_2\}\), with λ ∈ {1, 0.1, 0.01}. Pick λ using n=9 as a validation set and gate on n=15. That is 3 runs.
- **Pass:** Some λ gives n=15 R ≤ 1.00 and n=12 R ≤ 1.00.
- **Kill:** The best λ only reproduces warm (R=1.000 at every size), meaning \(c_2\) brings nothing at L=1. Stop polishing L=1 and lock the TIE.

**Idea 4 — Grow from warm, polishing only the new layer**
- **Why:** E3 at L=2 passed at about 0.97×. The extra expressivity pays off with depth, not at L=1. At L=1 the dual map may simply be pinned to the QAOA optimum.
- **Method:**
  1. Initialise layer L+1 using QAOA INTERP angles mapped through the dual map. All earlier layers start from the accepted (TIE or BEAT) parameters.
  2. Polish only the new layer, using the Idea 2 loss and the Idea 3 penalty.
  3. Then run one short proximal pass over all layers.
- **Cheapest decisive experiment:** Follow the ladder L=1→4. Before running it, confirm warm=QAOA at L=4.
- **Pass:** R ≤ 0.98 at n=9, 12 and 15 for L=4.
- **Kill:** UNDER at any n after the new-layer-only polish. The ladder STOPs, so fall back to warm-only at that L.

**Idea 5 — Switch the training set (contingency)**
- **Why:** Use this only if Idea 1 kills n=6. E1 shows n9+n12 training can beat at those sizes, but it still failed n=15 at about 1.003–1.017×. That failure margin is small enough for the Idea 2 and Idea 3 constraints to plausibly close.
- **Method:** Train on n9+n12 (or n6+n12) and keep n=15 sealed as the gate. Use the Idea 2 loss plus the Idea 3 penalty.
- **Pass:** n=15 R ≤ 1.00.
- **Kill:** n=15 is still above 1.00, which would mean the drift grows with size. In that case, accept warm-only TIE as the result at that depth.

## 3. Keep n=6, or abandon it?

Keep n=6×1000 by default for γ and β, because the QAOA subspace concentrates across n.

Abandon n=6 for the non-linear \(\phi\) degrees of freedom if either of these happens:
- Idea 1 gives \(\cos(g_6,g_{15})<0.2\) on the non-linear part.
- After Ideas 2 and 3, the best-λ run on n=6 is still UNDER at n=15.

Either result means the n=6 statistics, with only 2 cavities, cannot represent what matters at larger n. In that case, switch to Idea 5.

## 4. Next experiments, in order

1. **Idea 1, gradient audit at L=1 warm.** Under an hour. It decides whether n=6 survives and which parameter directions are safe to move.
2. **Idea 2 combined with Idea 3 at L=1.** Log-ratio loss, fit only \(\{\gamma,\beta,c_2\}\), λ sweep with n=9 validation and n=15 gate. A few hours.
   - If it passes, run the L=4 grow from Idea 4.
   - If it is neutral, lock the L=1 TIE and go straight to the L=4 grow.

Whatever happens, keep warm as the fallback at every L. Polish is accepted only if train Med ≤ warm Med and R ≤ 1.02 at every n.
