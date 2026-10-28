# Opus 5.5 ideas — relaxed U_phase^q + free D(α₀) (NEW design track)

- Model: `claude-opus-5-5-high` via `cursor-agent -p --trust -f --mode ask`
- Generated: 2026-10-28 ~15:36 Asia/Shanghai (wall ≈182s)
- Prompt: `opus_relax_q_prompt.md`
- Ideas only. NEW track — do **not** kill Day2 / old-lock amortization farm.
- Circuit target: `D(α₀) | (SNAP → D → U_phase^{q_ℓ})^L`, q∈{1..10}, U_phase free/fine-tunable (not tied to true cost H).

---

I found no project code in the workspace, so this plan assumes random 3-SAT, Fock levels \(0..2^n-1\) mapped to bitstrings by Gray code, and the same eval set and GNN backbone as Night1.

## 1. Diagnosis

1. **On their own, the new knobs add almost no expressivity.** SNAP and `U_phase` are both diagonal in the same Fock/Gray basis. So `U_phase^{q_ℓ}` followed by the next layer's `SNAP(θ_{ℓ+1})` collapses into one SNAP with phases \(θ_{ℓ+1} + q_ℓ φ\). If SNAP angles are free per layer, any diagonal phase is already reachable, in both the old and the new form. The only real new capacity is `D(α₀)`: starting from vacuum makes the first SNAP a global phase, so the old form lost one layer.
2. **What actually changes is where the formula information lives.** Under the old lock, the GNN had to encode each formula through a few numbers (γ, clause_w) plus the circuit parameters. The map from formula to good parameters was apparently rough, which is why the median was fine and B90 was not. The relaxed form gives the GNN a smoother target to predict: a structured phase table that can carry predicted bit marginals. That is the same idea as warm-start QAOA.
3. **There is a risk of making amortization worse.** If both SNAP angles and `U_phase` are formula-conditioned, the GNN has two redundant routes to the same merged diagonal. One of them is a raw \(2^n\)-dimensional vector, which won't transfer across n and makes memorizing training formulas easy. The n6↔n8 anti-correlation of −0.98 already points to size-specific overfitting.
4. **So the design rule is QAOA's split.** Make the mixer (SNAP, D) global and formula-independent. Put all formula dependence into a structured `U_phase`, with q as the only per-layer strength.
5. **Per-instance BFGS no longer tells you anything on its own.** The relaxed form trivially reaches ≈1.0, as the old one did. The useful ceiling is a **global mixer with per-formula phase tables**, described in §3.

## 2. Spec (locks / free / GNN heads)

Shapes below are for n=8, L=5, m≈34 clauses. Here N is the grid size plus guard levels, about 288.

**Locks (fixed in the spec)**
- The circuit is `D(α₀) | (SNAP(θ_ℓ) → D(α_ℓ) → U_phase^{q_ℓ})` for ℓ=1..L.
- There is **one shared `U_phase` per formula**, used at every layer. A per-layer `U_phase` is allowed only as a ceiling ablation.
- Evaluation always uses **integer q ∈ {1..10}**.
- p(GS) is counted only on the grid. Leaked mass counts as failure, and the leakage penalty stays on.
- Use the same eval set, metric (Med/B90 of 1/p) and baselines as Night1. Report both WalkSATlm (13.99/29.4) and Novelty+ (13.25/28.41).

**Global parameters (learned, not formula-conditioned) in the primary arm**
- `α₀`: one complex number (2 reals). Initialize it small and nonzero (|α₀|≈0.5) so the first SNAP gets gradient.
- `θ`: shape [L, N]. Initialize from Night1's mean SNAP if available, otherwise at 0.
- `α`: shape [L] complex. Initialize from Night1's mean, or as a decreasing ramp.
- `q̃`: shape [L], continuous during training (see q handling). Initialize as a ramp that rounds to about 2, 3, 5, 6, 8.

**GNN head for `U_phase`: per-clause local phase tables (primary)**
- For each clause c, the GNN outputs a table \(t_c\) of 8 phases, one per local assignment of the clause's 3 variables. The phase of a full assignment is the sum of its clauses' table entries: \(φ(x) = Σ_c t_c[x|_c]\).
- Tables are a cost-like prior plus a residual: \(t_c = γ_F·w_c·e_{\text{unsat}} + s·\tanh(Δ_c)\).
  - γ_F (formula scale) and w_c (clause weights) come from GNN readouts.
  - Δ_c is 8 outputs from the clause node. The last layer is zero-initialized, so training starts at a tied cost.
  - s is a learned residual scale, initialized at 0.5 rad.
- Optionally add per-variable fields h_i from variable nodes. These are redundant with the tables but easier to optimize.
- Subtract the mean of φ over the grid as a gauge fix. Do **not** normalize γ_F away: with integer q, γ_F sets the overall magnitude and q sets the ratios between layers.
- Why this form:
  - It is strictly a superset of the old lock (Δ=0 recovers γ·clause_w).
  - It is size-generic.
  - It covers single-variable fields and within-clause pair and triple terms.
  - It cannot cheaply encode an arbitrary \(2^n\) vector.

**q handling (answer to softmax vs. round)**
- Use a continuous relaxation, not a 10-way softmax. Because `U_phase` is diagonal with known phases, fractional powers \(e^{-i\,\tilde q\,φ}\) are exact and cheap, so continuous q̃ is a valid circuit during training.
- Parameterize \(\tilde q = 1 + 9σ(z)\).
- Anneal toward integers in three phases:

| Share of steps | Treatment of q̃ |
|---|---|
| 0–60% | Fully continuous |
| 60–85% | Add penalty \(λ\sin^2(π\tilde q)\), with λ ramped 0→1 (in loss units) |
| 85–100% | Hard rounding with a straight-through gradient; the rest of the model keeps fine-tuning |

- A 10-way softmax with an expected value is just a scalar that costs more. Sampling q, via Gumbel or REINFORCE, is high-variance and unnecessary here.

**Arms that change which parts are formula-conditioned**
- **A2:** q comes from the GNN (global readout → L scalars), with the same annealing.
- **A3:** the GNN also outputs SNAP/D parameters. This tests the redundancy problem from §1; expect it to be worse or unstable.

**Training hyperparameters (all arms)**
- Learning rates: GNN 3e-4 (AdamW), global mixer 3e-3, q logits 1e-2. Gradient clip 1.0.
- Batch of 128 n=8 formulas, 30% of them drawn from a hard-tail buffer (top decile of log 1/p), refreshed every 1k steps.
- Loss: mean(log 1/p) + 1.0·CVaR₂₀%(log 1/p), uncapped.
- Keep H=128.

## 3. Ranked run plan

Run everything in parallel; nothing is gated on a cheap test. The ceilings run alongside the training arms and only inform promotion.

1. **A1 primary:** global mixer, clause-table phase, global q schedule, L=5, n=8, 3 seeds. This is the main bet.
2. **P0 parity:** the new code pinned to the old lock (α₀=0, q≡1, Δ≡0, h≡0, old mixer conditioning). It must reproduce Night1 at about 12.4/84.6, within ±10%. If it doesn't, every other number is suspect.
3. **Global-mixer ceilings**, direct optimization with no GNN, on 64 n=8 training formulas:
   - **C1:** jointly optimize the global θ, α, α₀ and q̃ together with free per-formula tables.
   - **C2:** freeze C1's mixer and optimize tables only on 32 held-out formulas, with L-BFGS from the cost-like init.
   - **C3:** like C2, but only γ_F and h_i per formula.
   - How to read them: if C2 gets ≥90% of formulas to p≥0.9, the GNN only has to predict tables and amortization is feasible. If C2 is good but C3 is poor, the clause residual is what matters. If C2 fails, a formula-conditioned mixer is needed and A3 moves up.
4. **A1-warm:** A1 with the mixer initialized from C1. Freeze the mixer for the first 30% of steps, then unfreeze it at lr 1e-3.
5. **A2:** formula-conditioned q.
6. **A4 minimal relaxation:** Δ≡0, only γ_F and h_i. This ablation tells you which part of the relaxation buys the gain.
7. **A5:** A1 at L=7, to check whether the depth story changes once there is no tied lock.
8. **A3:** GNN-conditioned mixer, to confirm or rule out the redundancy problem.
9. **A6:** per-layer `U_phase`, outside the spec, as an upper bound only.
10. **Distillation (D1)** on the best A1 checkpoint:
    - Freeze the mixer and run tables-only L-BFGS on hard-tail training formulas, starting from the GNN's prediction.
    - Distill with MSE on the tables plus the task loss, for 2 rounds in a DAgger style.
    - Because the mixer is frozen and each optimization starts from the GNN's prediction, the teacher targets are close to unique, which fixes the "multiple optima" problem distillation had under the old lock.
11. **Size transfer:** evaluate promoted runs on n=6 and n=12. Then train on a mix of 50% n8, 25% n6 and 25% n12 with a shared GNN and a **separate global mixer per n**, because grid size differs by n.

**Rules (evaluated every 30 min on the fixed n=8 eval set, integer q)**
- **Kill at 3h** if B90 > 84.6 and Med > 12.4, i.e. no better than the old best on either number.
- **Kill** if B90 is more than 25% worse than the A1 median at the same wall-clock time, or if leakage exceeds 5%.
- **Promote at 6h** if B90 < 50. Promoted runs continue to about 16h.
- **WIN** means Med < 13.99 and B90 < 29.4 on at least 2 of 3 seeds. Report the Novelty+ comparison alongside.
- **Also log:**
  - The amortization gap: GNN 1/p compared with C2 1/p on the same formulas.
  - The fresh-seed hit rate, since the old lock had 0/150.
  - The rounding drop from continuous q̃ to integer q, which should be under 5% on the median.

## 4. Dual-GPU ordered list (~24h)

A 5090 should comfortably fit 4 concurrent n=8 runs, and a global mixer means one matrix exponential per layer per step, so this is cheap. Check memory and throughput on your first launch.

**GPU0 (training arms)**

| Order | Hours | Runs |
|---|---|---|
| 1 | 0–6 | P0 parity, A1 seed 0, A1 seed 1, A1 seed 2 |
| 2 | 6 | Apply kill/promote. Free slots go to A2 (2 seeds) and A4 (1 seed) |
| 3 | 6–16 | Surviving A1 runs plus A2 and A4 |
| 4 | 16–24 | D1 distillation from the best A1, and the mixed-size run (n6/n8/n12) starting from the best A1 |

**GPU1 (ceilings and variants)**

| Order | Hours | Runs |
|---|---|---|
| 1 | 0–4 | C1 joint ceiling, then C2 and C3 on held-out formulas; A3 alongside |
| 2 | 4–14 | A1-warm (2 seeds, mixer from C1), A5 at L=7, A6 per-layer ablation |
| 3 | 14–20 | n=12 evaluation of all promoted checkpoints; C2 at n=12 with a frozen n12 mixer from a short C1-n12 run |
| 4 | 20–24 | Second D1 round, or more seeds of the winning arm if D1 already converged |

**Deliverables at 24h**
- Med/B90 for each arm and seed against WalkSATlm and Novelty+.
- The amortization gap (GNN vs. C2).
- The size of the integer-rounding drop.
- Transfer to n6 and n12.

## 5. What NOT to do

- Don't let the GNN emit raw \(2^n\) phase vectors or per-formula SNAP θ in the main arms. That reopens the memorization and no-size-transfer route.
- Don't make both SNAP and `U_phase` formula-conditioned, because the merged diagonals are redundant (A3 is the only exception, as a test).
- Don't train q with softmax, Gumbel or REINFORCE when exact fractional powers exist. Never evaluate with fractional q.
- Don't start the phase tables from random values; zero-init the residual around the cost.
- Don't drop γ_F by normalizing the phase scale. Without it, q has to carry the magnitude and gets pinned at 1 or 10.
- Don't screen on n=6, because of the anti-correlation.
- Don't re-run the L grid or raise H.
- Don't use per-instance fine-tuning at test time unless you count it in the budget against WalkSAT.
- Don't optimize mean p(GS) or a capped tail loss. B90 is the target.
- Don't use per-layer `U_phase` as the reported spec result.
- Don't stop Day2.

## 6. QAOA comparison (short)

**What transfers**
- **Parameter concentration:** a global mixer plus a global q schedule is the analogue of QAOA's instance-independent angle schedules.
- **Annealing-like ramps:** q increasing and |α_ℓ| decreasing across layers is the natural init.
- **Warm-start QAOA:** the h_i fields and clause residuals act as GNN-predicted biases toward the ground state.
- **Integer-ratio γ lattice:** with γ_F free, discrete q only quantizes the ratios between layer strengths. Ratios over 1..10 approximate smooth ramps closely, so the loss against continuous γ_ℓ is small.

**What doesn't transfer**
- **The mixer isn't a transverse field.** D hops along the 1D Fock ladder, so only Gray-adjacent levels are single bit flips, and its matrix elements grow roughly like \(\sqrt{k}\) up the ladder. Symmetric-hypercube landscape results don't apply.
- **The start state isn't uniform.** A coherent state is Poisson-shaped and narrow, not a uniform superposition, so the adiabatic large-depth argument doesn't hold.
- **SNAP adds diagonal freedom QAOA lacks.** That is exactly what makes a formula-conditioned SNAP redundant with `U_phase`.
- **Periodicity is different.** With learned real-valued phase tables there is no 2π periodicity in γ, so integer q doesn't alias the way it would with an integer cost.

If you want any of this turned into launch configs or scripts, switch to Agent mode.
