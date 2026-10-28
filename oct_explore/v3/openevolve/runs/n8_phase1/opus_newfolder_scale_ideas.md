# Opus 5.5 ideas — new-folder coherent + U_phase^q multi-n scaling

- Model: `claude-opus-5-5-high` via `cursor-agent -p --trust -f --mode ask`
- Generated: 2026-10-28 ~16:04 Asia/Shanghai (wall ≈208s)
- Prompt: `opus_newfolder_scale_prompt.md`
- Ideas only. Brand **new folder** track — do **not** kill/stop Day2 / old-lock farm; do not message qumode_GPU from this call.
- Zach draft edited: freeze GNN from prior data; D(α₀) coherent start; SNAP+D+U_phase^q (q∈1…10); train n=6/8/12; scale-eval larger n.

---

The workspace folder is empty, so this plan is built from your brief alone. Where I've assumed something about the old code, I say so.

The main ideas behind the edits:

- **Per-layer q is the real new lever.** The old lock used one cost angle for the whole depth. Making the angle `q_l·γ` per layer gives a QAOA-style schedule (cost angle that changes from layer to layer), at integer resolution.
- **The coherent start is probably a small gain.** If the old layer order was SNAP→D, then SNAP on the vacuum is just a global phase, and the old circuit already began with a coherent state, `D(α₁)|0⟩`. Adding `D(α₀)` only buys one extra mixing step before the first cost. Keep it, but don't credit wins to it without an ablation.
- **The problem is the tail.** Per-instance BFGS reaches 1/p ≈ 1, but the trained model's n=8 B90 is 84.6. The gap is amortization, not expressivity. Every design choice below is judged on B90 and on how Med grows with n.

## 1. Edited plan (Zach-facing)

1. **New, independent folder `qumode_sep/`.** Copy and freeze only the pieces needed (formula encoding, GNN). Never import from the Day2 tree, and never touch Day2.
2. **Freeze the GNN** (exact spec in §2): bipartite message passing, H=128, 4 rounds, mean aggregation. No more architecture search this campaign.
3. **Ansatz:** `|0⟩ → D(α₀) → [SNAP_l → D(α_l) → U_phase^{q_l}]` for l = 1…L.
   - `U_phase` is one diagonal unitary per formula, the same at every layer.
   - `q_l ∈ {1…10}` is a **global schedule shared by all formulas**, trained as a continuous value then annealed to integers. Only integer-q numbers are reported.
   - Stage A0 uses the exact cost Hamiltonian. Stage A1 adds GNN-predicted clause weights plus per-clause 8-phase residuals, starting from zero so A1 begins exactly at A0.
4. **Train on n = 6, 8 and 12 together**, with n=6 at low weight. Hold out n=10 as an interpolation test and n=14 and 16 as extrapolation tests. The headline is how Med and B90 grow with n compared with WalkSATlm, using one shared mixer whose parameters are defined relative to the size of the space so they carry over to any n.
5. **Add two things Zach's draft lacks:**
   - A **regression anchor**: the new code with q≡1 and α₀=0 must reproduce roughly the Night1 numbers.
   - A **q-oracle diagnostic**: for each instance, find the best integer q schedule with everything else fixed. This decides whether a GNN-predicted q is worth building.

## 2. Spec locks

**U_phase: exact first, then a tunable version gated on evidence.**
- **A0:** `U = exp(−iγ Σ_c unsat_c)`.
  - γ is one shared scalar. Initialise γ≈0.3 so that `q·γ` covers about 0.3 to 3.0, and clamp γ to [0.05, 0.63]. Above 2π/10 the q values start to alias.
- **A1:** cost prior plus GNN residuals:
  - clause weights `w_c ∈ 1 ± 0.25`, via tanh;
  - per-clause 8-entry phase table for the 8 local assignments of a clause, `|δ| ≤ π/8`, L2 penalty 1e-3;
  - optionally a per-formula γ scale of ±20%.
  - All residual heads start at zero.
- **Promotion rule for A1:** only if it improves n=8 validation B90 by at least 15% over A0 **and** does not hurt n=12. The scaling claim must also be stated for A0, because an exact cost is easier to defend and a learned residual is the part most likely to fail at larger n.
- **Never use a free raw 2ⁿ diagonal.** SNAP and U_phase are both diagonal, so between two displacements they merge into one diagonal anyway.

**q: one global schedule, not GNN-predicted (for now).**
- Parameterise as `q̃ = 1 + 9·sigmoid(u)`.
- Initialise with an increasing ramp, `round(linspace(2, 8, L))`.
- Anneal in three phases:
  - 0–50% of steps: q̃ is free.
  - 50–85%: add a penalty `λ(t)·sin²(πq̃)`, with λ ramped from 0 to about the loss scale.
  - At 85%: round q, freeze it, and fine-tune everything else for the remaining 15%.
- Then an **integer polish**: coordinate descent over `q_l ± 1` on the validation set, which is cheap.
- **GNN-predicted q (A2)** only if the q-oracle shows at least a 25% B90 gain over the global schedule at n=8. If it does, train it by supervised distillation from the oracle labels. No Gumbel, no REINFORCE.

**Mixer: one shared mixer, defined relative to the size of the space.**
- **Displacement:** let N = 2ⁿ be the cutoff. Use `α_l = a_l·√N` with a real `a_l ≤ amax_norm`.
  - Phases of α can be dropped: a Fock rotation is diagonal, so the adjacent SNAP absorbs it. For the same reason, make α₀ real.
  - Check what the old `amax = 1.8` meant. If it was in raw units, the old circuit only reached low Fock levels at n=12.
- **SNAP:** a smooth function of the normalised phase-space radius `r = √(k/N)`: Chebyshev in r with K=32 coefficients per layer, plus an explicit linear-in-k term. This makes the same parameters valid at n = 10, 14 and 16.
  - Raw per-level SNAP tables appear only in ceiling runs and in the per-n control ("M-sep").
- **Per-n mixers** exist only as a control, to measure what sharing costs. They can't be the headline, because they can't be applied to an unseen n.

**Multi-n loss.**
- Per-instance loss: `ℓ = log(1/p) − log(Med_WS(n))`, so every size is measured relative to the classical bar.
- Per size: `mean(ℓ) + τ·CVaR₉₀(ℓ)`, with τ ∈ {1.0, 1.6}. CVaR₉₀ is the mean over the worst 10% of instances, which targets the B90 metric directly.
- Size weights: **n6 0.15, n8 0.40, n12 0.45**. Given the −0.98 anti-correlation, n=6 serves as a regulariser only.
- Batch sizes: n6 128, n8 128, n12 32, with gradient accumulation if needed.
- Curriculum: n6+n8 only for the first 20% of steps, then ramp n12 in.

**GNN freeze.**
- Bipartite variable–clause message passing, H=128, rounds=4, residual connections and LayerNorm.
- **Mean aggregation plus degree features, not sum.** Sum aggregation breaks when size and degree change.
- Heads: per clause (w_c and the 8-phase residual) and a graph-level readout (mean+max) for the γ scale.
- The strong-n6 configuration transferred badly because it was trained only on n6, not because of its architecture. Keep its architecture and drop its training recipe.

**Other hyperparameters.**
- Adam with cosine schedule and 100 warmup steps. Mixer lr 1.1e-3, GNN lr 5e-4.
- A0: 2000 steps. A1: 4000 steps.
- Leakage penalty 0.14, applied to a **guard band**: simulate at cutoff 1.25N and penalise population above N.
- Train in complex64; spot-check evaluation in complex128.
- L ∈ {5, 8}, fixed across n for the headline.

**Simulator.**
- For N ≤ 4096: once per cutoff, eigendecompose the truncated generator `a† − a`. Then `D(r) = U·e^{irΛ}·U†` is exact for the truncation, cheap, and trivially differentiable.
- For n ≥ 14: Chebyshev or Taylor expansion of the action of `exp` on the state vector, using the sparse bidiagonal generator. No dense D.

## 3. Folder + code milestones

**Layout**

```
qumode_sep/
├── SPEC.md          # the locks above
├── configs/         # one YAML per stage
├── golden/          # JAX npz fixtures
├── src/qsep/
│   ├── encoding.py  # cost vector, clause tables
│   ├── sim.py       # diagonal ops, eigendecomposition D, expm-action D, guard band
│   ├── mixer.py     # size-normalised parameters
│   ├── uphase.py    # A0/A1 heads
│   ├── qrelax.py
│   ├── gnn.py       # frozen copy
│   ├── losses.py
│   ├── metrics.py   # Med/B90, WS ratios, bootstrap CIs
│   └── data.py      # splits, WS/Novelty+ lookup tables
├── scripts/         # train, ceiling, q_oracle, eval_scaling, tail_report
├── runs/<stage>/<tag>/   # config, checkpoint, metrics.jsonl, git-less hash of src
└── reports/
```

**Milestones**

| Milestone | What | Gate to pass |
|---|---|---|
| M0 | Data and baselines. Reuse the old n6/8/12 splits so the bars stay comparable. Generate n=10, 14, 16 with the **same generator, clause ratio and GS filter**, and compute WalkSATlm and Novelty+ lookups with the same protocol. | n=12 WS Med reproduces about 32.5 |
| M1 | Simulator plus JAX parity at n = 6, 8, 12; check eigendecomposition D against expm-action D; benchmark speed. | State error < 1e-5 (c64); p(GS) relative error < 1e-4; leakage < 1e-4 at chosen α |
| M2 | Regression anchor: q≡1, α₀=0, A1 heads, n=8. | Med within ±10% of 12.4 (or explain the difference) |
| M3 | Per-instance ceilings for the new ansatz: integer q, smooth SNAP vs raw SNAP, n=8 and n=12, L ∈ {5, 8}. | n=8: 10/10 with p ≥ 0.9; smooth-SNAP ceiling ≥ 80% of raw |
| M4 | A0: global only, exact cost, multi-n. | See kill/promote in §4 |
| M5 | A1: GNN residuals. | +15% B90 over A0 at n=8, no n=12 regression |
| M6 | q-oracle plus tail report. Correlate 1/p with the GS Fock index k_GS, the number of ground states, and WS hardness. | Decide A2 or stop |
| M7 | Scaling evaluation (n = 10, 14, 16) and report. | §5 |

## 4. Dual-GPU run order

Never preempt Day2. If Day2 holds one GPU, everything below runs on the free GPU in the same order, and only n=8-sized jobs share Day2's GPU.

| Phase | GPU-A | GPU-B |
|---|---|---|
| 0 | M1 parity (M0's WalkSAT lookups run on CPU in parallel) | — |
| 1 | M2 anchor, then M3 ceilings | M4 A0 sweep: L ∈ {5, 8} × τ ∈ {1.0, 1.6} × 2 seeds = 8 small runs, 2–4 at a time |
| 2 | M5 A1 main: best A0 L, 3 seeds | Ablations on the best A0: learned q vs q≡1, α₀ free vs 0, shared vs per-n mixer; then the M6 q-oracle |
| 3 | A2 (GNN-q distillation) only if the oracle gate passes; otherwise more seeds or warm-start variants of the A1 winner | M7 scaling evaluation (n=16 is the long job) |

**Kill and promote rules** (validation set only; the test set is touched once per promoted configuration):
- **At 25% of steps:** kill if n8 Med > 20 or n12 Med > 60.
- **At 50%:** kill if n8 B90 > 80 (not beating the old lock's trajectory) or Med12/Med8 > 3.0.
- **Promote to the next stage:** integer-q n8 Med ≤ 13, n8 B90 ≤ 50, and n12 Med ≤ 40.
- **Headline candidate:** meets the T2 or T3 criteria in §5 in at least 2 of 3 fresh seeds. Report the median across seeds, never the best seed. Track fresh-start F1 (old lock: 0/150) as a first-class number.

## 5. Scaling metrics / success

**What to report for every n** (n = 6, 8, 10, 12, 14, 16):
- Med and B90 of 1/p(GS) on held-out test sets: at least 300 instances per n, at least 100 for n=16, with bootstrap 95% confidence intervals.
- The ratio to WalkSATlm, `R(n) = Med_Q / Med_WS`, and the same ratio for B90.
- Novelty+ alongside.
- The uniform floor 2ⁿ/#GS as a sanity check.

**Growth rate.**
- Fit log Med against n and compare the per-variable slope with WS.
- WS so far grows about 1.29× per variable (n6→8) and 1.24× per variable (n8→12). Naive extrapolation gives WS n14 ≈ 50 and n16 ≈ 76, but **measure these** in M0; don't use the extrapolations.
- Do the same for B90 (WS grows about 1.35× then 1.27× per variable).

**Success tiers.**
- **T1:** beats the old lock. Integer-q n8 Med < 12.36 and B90 < 60.
- **T2:** classical parity at n=8. Med ≤ 14.0 and B90 ≤ 35.
- **T3 (the goal):**
  - n=12 Med ≤ 32.5 and B90 ≤ 76;
  - Med12/Med8 ≤ 2.3;
  - R(n) flat or falling across n = 10, 14, 16 at **fixed L**.
  - If L has to grow with n, report a depth-adjusted cost (1/p × L ratio) as a separate line; it can't be the headline.

**Interpolation check.** If n=10 is worse than the line between n=8 and n=12, the shared parameterisation is broken. Fix that before believing any n=14 or n=16 number.

## 6. What NOT

- Don't stop, edit, or import from Day2.
- Don't use a free raw 2ⁿ diagonal, formula-conditioned dual diagonals, or Gumbel-q.
- Don't screen or select on n=6 alone.
- Don't report relaxed (continuous) q numbers as results.
- Don't make per-n mixers the headline; they can't extrapolate.
- Don't change the GNN architecture or go to H=256.
- Don't use sum aggregation.
- Don't go past L=8 before the q-oracle. Deeper L didn't help B90 last time.
- Don't use dense D at n ≥ 14, and don't skip the guard band. Truncation artefacts near the cutoff will look like real signal.
- Don't optimise mean 1/p; outliers dominate it. Use log loss plus CVaR.
- Don't use polarity or relabelling tricks that place the ground state at a convenient Fock index. That leaks GS information. If the tail correlates with k_GS, report it honestly.
- Don't compare against WS numbers from a different generator or protocol at n = 10, 14, 16.
- Don't credit improvements to the coherent start without the α₀=0 ablation.
