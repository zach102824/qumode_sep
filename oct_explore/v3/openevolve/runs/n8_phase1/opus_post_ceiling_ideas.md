## 1. Updated diagnosis

The run artifacts and Torch code aren't in this workspace, so this is based only on your notes. Where it depends on how the code works, I say so.

- **The gap is amortization, and most of it is in the tail.** Optimising each instance separately gets about 1.0 / 1.02. The GNN gets about 12.4 / 84.6. So the GNN loses roughly 12× at the median and roughly 80× at B90.
- **The current loss can't see the B90 instances.** The tail term is effectively min(1/p, 33). Any instance with 1/p > 33 gets zero gradient from it. B90 instances sit around 80, so they only get the base term, where they are about 10% of a mean. In practice the recipe optimises the 10–33 band, which is the median. That fits what you've seen: Med beat WalkSAT while B90 stayed at 78–85 across every variant. This is the highest-confidence lever.
- **One caveat on the ceiling needs closing.** BFGS optimised A/TH/g/w all per instance. In the Torch pipeline, A/TH may be global (loaded via `init_th_path`), with only the tied (gamma, clause_w) predicted per formula. If so, the ceiling that matters is shared A/TH plus per-instance g/w, and nobody has measured it. If that ceiling is much worse than 1.0 at B90, part of the bottleneck is the shared parameters, not the GNN.
- **Three unknowns decide the branch:**
  - Is train B90 about equal to test B90 (underfitting / wrong objective), or much lower (generalisation)?
  - Do the GNN outputs actually vary across formulas, or collapse to one compromise schedule?
  - Can a short local refinement from the GNN's prediction reach the per-instance optimum? If yes, distillation is a direct path.
- **12.36 is optimistic.** It was the best of many runs, picked on test. Expect roughly 13–14 on a fresh set. B90 from a few hundred formulas rests on about 20–30 tail instances, so ±20–30% is plausible. The gap between 78 and 85 is probably noise. Use 1024+ fresh val and fresh test formulas.
- **Cold starts fail everywhere** (F1 0/150, H=192 cold). Every run should fine-tune from BEST.

## 2. Answers

**Q1 – What to actually run.** Three blocks, in order:
1. **Diagnostics, 1–2 hours, no GNN training.** An eval pack, plus per-instance optimisation probes on GPU.
2. **Fine-tunes from BEST with an uncapped, normalised tail loss.** Use log-mean-exp of log(1/p) or CVaR over the worst 10–20%, with EMA and early stopping on val. Run a same-loss continuation as the control.
3. **If the gap remains:** basin-consistent teacher distillation, plus hard-instance oversampling.

My guess (not measured) is that the loss change alone moves B90 from about 85 to 50–65. Getting into 29–40 probably needs the teacher and oversampling.

**Q2** – Section 3. The top three are the diagnostic pack, the uncapped tail loss and the amortization probes.

**Q3** – Section 4.

**Q4** – Section 5. Short version: no cold starts, no width or depth changes, no leakpen games, no selecting on test.

## 3. Ranked next experiments (impact × ease)

1. **Diagnostic eval pack** (eval only, about 30 min)
   - Build fresh val and fresh test sets of 1024 formulas each. Touch test once, at the end. Also eval on 1024 train formulas.
   - Eval BEST plus the top 5 Night1 checkpoints (re-ranked on val), with raw and EMA weights.
   - Dump per-instance 1/p, leak, number of ground states (#GS), a unique-GS flag and degree stats. Bootstrap 95% CIs for Med and B90.
   - How to read it:
     - Train B90 ≈ test B90 means the objective is the problem; go to items 2 and 5.
     - Train B90 much lower than test means generalisation; go to items 8 and 4.
     - It also shows whether the B90 instances are mostly unique-GS.

2. **Uncapped, normalised tail loss** (about a 10-line change; fine-tune from BEST)
   - Replace mean(exp(clamp(ℓ, 3.5))) with τ·logmeanexp(ℓ/τ), where ℓ = log(1/p).
     - τ=1 gives log(mean 1/p), the expected number of repeats.
     - τ=0.5 leans harder on the tail.
     - Per-instance weights are a softmax, so they sum to 1. That means no explosion, no clamp and no dead gradient.
   - Alternative: CVaR, the mean ℓ over the worst q·bs instances, with q ∈ {0.1, 0.2}. Needs bs ≥ 128.
   - Minimal-code control: raise the clamp from 3.5 to 5.0 (1/p ≈ 148) with `grad_clip 1.0`.
   - Keep the base term (mean ℓ) for the median.

3. **Amortization probes** (batched per-instance optimisation in Torch, no GNN, about 1 hour)
   - Run on about 512 val formulas, including BEST's worst decile.
   - (a) **Restricted ceiling:** freeze A/TH at BEST and optimise only per-instance (gamma, clause_w).
   - (b) **Warm start from BEST's predictions:** count steps to reach 1/p ≤ 2 and measure how far the parameters move. Few steps and a small move mean distillation (item 6) is a strong bet.
   - (c) **Constant ceiling:** one parameter set shared by all formulas. If BEST is close to this, the GNN isn't conditioning on the formula.
   - (d) **K = 4 / 16 prototypes with oracle assignment:** shows how much specialisation the tail needs.
   - Also compare how much the GNN outputs vary across formulas against how much the warm-started optima vary.

4. **Same-loss continuation, EMA and val early stopping** (existing flags only)
   - BEST with lr 1e-4, cosine schedule, `ema_decay 0.999`, eval every ~5% of steps.
   - Keep the checkpoint with the lowest val B90 subject to val Med ≤ 14.0.
   - This is the control. Without it you can't tell whether a gain from item 2 came from the loss or just from extra training and EMA.

5. **Hard-instance oversampling**
   - Fixed version: 30% of each batch from unique-GS formulas, or from BEST's worst decile. Using #GS as a training label is fine; just never feed it as an input.
   - Adaptive version: keep a per-instance EMA of ℓ and sample in proportion to ℓ, mixed 50/50 with uniform sampling to limit distribution shift.
   - Kill if val Med > 14.5.

6. **Basin-consistent teacher distillation (expert iteration)**
   - Teacher: the per-instance optimum, warm-started from the current GNN output, over the training set. Batched on GPU this should take minutes at n=8.
   - Loss: base + tail + λ·d(θ_GNN, θ*). Use a wrap-aware distance (1 − cos) for angles and phases. Try λ ∈ {0.3, 1.0}, annealed to 0 by 60% of steps so the final objective is p itself.
   - Refresh the teacher from each new parent.
   - This has the highest ceiling of anything here. It ranks 6th only because it needs code and depends on probe 3(b).

7. **Batch size 2–4×, with lr scaled by √k.** Cheap at n=8 with H=128, and it steadies the batch tail statistics. Pair with item 2.

8. **ntrain 2–4×, or fresh formulas every epoch.** Only if item 1 shows a train/test B90 gap. Ground truth at n=8 is cheap: 256-way enumeration.

9. **Model soup.** A uniform weight average of the top 3–5 fine-tunes that share BEST as their init. Cheap, and often trims tail variance.

10. **rounds +1 or +2 at fine-tune time.** Only if message-passing weights are shared across rounds, so BEST loads cleanly. Low priority.

11. **K-prototype head.** The GNN selects or blends among K learned tied schedules, still one (gamma, clause_w) per formula. Only worth building if probe 3(d) shows K ≤ 16 with oracle assignment gets B90 ≲ 30.

## 4. Ordered dual-GPU launch list

**Shared settings for every training run (BASE):**
`--init_th_path <A_lr2e-4_tw0.5_s2> --H 128 --L 5 --rounds/--leakpen/--wd/--ntrain <parent> --lr 1e-4 --lr_schedule cosine --ema_decay 0.999 --grad_clip 1.0 --steps ≈0.5×parent`

- Select on val, never on test.
- Kill a run at 30% of steps if any of these hold:
  - val Med > 16;
  - val B90 is more than 10% worse than BEST's val B90;
  - leak > 0.05 (the ceiling had ≤ 0.023).
- If one run uses well under half of a 5090, pack 3–4 runs per GPU.
- New flags (small code changes): `tail_mode {clamp, lse, cvar}`, `tail_tau`, `cvar_q`, `tail_clamp`, `hard_frac`, `distill_lambda`, `eval_every`.

**Phase 0 (hours 0–2)**
- GPU0: diagnostic eval pack (item 1).
- GPU1: probes 3(a)–(d). Then generate warm-started teacher targets for the full training set.
- Gate: read the train/test gap, the restricted ceiling and the warm-start distance.

**Phase 1 (hours 2–9), seeds 1, 2, 3 for each run**

| GPU0 | GPU1 |
|---|---|
| ctrl: BASE `--tailw 0.5` (old loss) | BASE `--tail_clamp 5.0 --tailw 0.5` |
| BASE `--tail_mode lse --tail_tau 1.0 --tailw 0.5` | BASE `--tail_mode cvar --cvar_q 0.1 --tailw 0.5 --bs 2×parent` |
| BASE `--tail_mode lse --tail_tau 0.5 --tailw 0.5` | same as above with `--cvar_q 0.2` |
| BASE `--tail_mode lse --tail_tau 1.0 --tailw 1.0` | BASE `--tail_mode lse --tail_tau 1.0 --tailw 0.5 --hard_frac 0.3` |

- If runs are slow, drop seed 3 first, then the τ=0.5 and q=0.2 arms.
- Gate at hour 9: the winner W1 has the lowest 3-seed mean val B90 with val Med ≤ 14.0. It must beat the control by more than the CI.

**Phase 2 (hours 9–17)**
- GPU0: distillation from W1's best checkpoint (or from BEST if W1 didn't beat the control). W1 flags plus `--distill_lambda {0.3, 1.0}` (annealed), seeds 1–2. If Phase 0 showed a train/test gap, add a W1 `--ntrain 4×parent` run.
- GPU1: spend about 45 minutes regenerating the teacher from W1's predictions (expert iteration round 2). Then run:
  - W1 with `--hard_frac 0.3`;
  - W1 with adaptive sampling;
  - W1 with `--bs 4×parent --lr 2e-4`.

**Phase 3 (hours 17–24)**
- GPU0: confirmation. Top 2 configs × 5 fresh seeds (11–15), then a single pass on the fresh test set. Report the 5-seed mean and the best run, each with bootstrap CIs, against WalkSATlm 13.99/29.40 and Novelty+ 13.25/28.41.
- GPU1:
  - Soup of the winner's same-parent checkpoints (item 9).
  - rounds +1, if weights are shared across rounds.
  - Passive n6 and n12 eval of the finalists, logging the growth metric Med12/Med8 (WalkSAT is about 2.3). Do not select on it.

## 5. What NOT to do

- **No cold starts or fresh-seed farms.** F1 hit 0/150 and H=192 from cold reached only Med ≈ 88. Always init from BEST or a later winner.
- **No changes to H, and no L or integer-repeat sweeps.** The ceiling is cleared, and L stays the same for train and test.
- **Don't flip leakpen's direction.** The reverse farm collapsed. Keep the parent's value and just monitor leak.
- **Don't raise tailw with the 3.5 clamp still in place.** It only re-weights instances with 1/p < 33, which helps Med, not B90.
- **Don't remove the clamp in favour of a plain mean(1/p).** A few instances with p near 0 will dominate and blow up the gradient. Use log-mean-exp or CVaR.
- **Don't distill onto cold-restart BFGS optima.** Because of symmetries, similar formulas can land in different equivalent basins, and regression averages them into nonsense. Warm-start from GNN outputs only, with an angle-aware distance.
- **Don't select on test or read too much into small differences.** Don't call B90 differences that sit inside the CI, and don't compare single seeds.
- **Don't select on n6, and don't train n12 yet.** The n6/n8 anti-correlation is −0.98. Look at growth metrics only after B90 moves.
- **Keep the headline numbers fixed-cost.** Don't report results that use per-instance test-time optimisation, or inputs that reveal the solution (#GS, brute force). Either one breaks the comparison with WalkSATlm lookups. Use them only as diagnostics or training targets.
- **Don't weight-average independently trained seeds.** Soups only work for fine-tunes that share an init.
