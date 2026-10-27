These are ideas only, no code. Variable names like `leakpen`, `wd`, `tailw`, `init_from` and `lrbase` match flags already in your sibling scripts, so most of these are sweeps rather than new features.

## 1. Diagnosis

At L=5 the 38 is limited by optimization and basin selection, not by GNN capacity or data. The same tied cost unitary is applied 5 times, so any error in gamma or `clause_w` adds up fivefold. Three results point the same way:
- **ntrain is non-monotonic.** 2k gives 90, 5k gives 334 with ~0.5 leakage, and 10k gives 38. That pattern looks like runs falling into different basins, not like a generalization curve.
- **A beats B.** A (shallow GNN, high learning rate, short run) beats B (deep GNN, low learning rate, long run). That suggests early exploration matters more than capacity.
- **Untied reached ~12.** The GNN's features are good enough; the trouble is finding the tied solution.

## 2. Ranked ideas (expected impact × ease)

**1. Warm-start from the n=6 L=5 model, then train on mixed sizes.**
- *Why:* The same recipe reaches 7.44 at n=6, and the message-passing GNN works at any formula size. Starting n=8 training from a good tied basin sidesteps the basin-selection problem behind the ntrain cliff.
- *First run:* Anchor A hypers, `init_from` set to the n=6 L=5 checkpoint (GNN + `clause_w`/gamma heads), lr=6e-4, steps=1000, ntrain=10k at n=8.
- *Variant:* a training mix of 30% n=6 and 70% n=8, evaluated on n=8 only.

**2. Measure seed variance, and select checkpoints with a validation set plus a weight average.**
- *Why:* The 5k result at 334 might just be an unlucky seed. If seed-to-seed spread is around 2x, then 38 is a single draw. Picking the best seed by validation could be a free gain, and it tells you how big an effect has to be before you trust it.
- *First run:* A at 10k with seeds {1,2,3,4}, and A at 5k with seeds {1,2}. Hold out 500 validation formulas and evaluate every 100 steps. Keep a running average of the weights (EMA, decay 0.999) and report test Med for three choices: final weights, best-validation checkpoint, and EMA.

**3. Depth curriculum within the tied circuit: L=3, then L=5.**
- *Why:* Tied parameters have the same shape at every L. L=3 trains more easily (B@L3 reached 62.5, and A should do better). Starting L=5 from an L=3 solution avoids the ill-conditioned start. The final training and test are still at L=5, so this is only an initialization.
- *First run:* A hypers at L=3 for 800 steps (ntrain=10k), then L=5 for 1000 steps at lr=6e-4 with 100 warmup steps. A cheaper variant is a 5-step lr re-warm with no curriculum, for comparison.

**4. Initialize the tied model from the untied ~12 run.**
- *Why:* That run is a known good basin. Averaging its per-layer parameters gives a tied starting point that is probably much closer to good than vacuum plus a random GNN.
- *First run:* Set tied `clause_w`/gamma heads to the layer average of the untied run's outputs (or to its middle layer, as a second arm) and copy the GNN weights. Then train at lr=5e-4 for 1200 steps on 10k formulas.

**5. Ramp the leakage penalty to block the leakage basin.**
- *Why:* The 5k collapse was specifically a leakage failure (~0.5). A strong early penalty keeps the run out of that basin, and lowering it later frees the optimizer for p_succ.
- *First run:* A at 10k with `leakpen` 0.4 for the first 30% of steps, then cosine-decayed to 0.1. Repeat the same schedule at 5k as a stress test; if 5k no longer collapses, the cliff was about the basin.

**6. Learning-rate schedule plus a slower rate for the cost head.**
- *Why:* A's high learning rate wins early but probably adds noise at the end. Errors in the tied cost parameters are amplified five times, so they need a smaller step than the GNN trunk.
- *First run:* Peak lr 1.5e-3, 100 linear warmup steps, cosine decay to 1e-5 over 2000 steps, gradient clipping at norm 1.0, and `lrbase` (cost-head multiplier) at 0.3.

**7. Fresh formulas for every batch, or ntrain=20k.**
- *Why:* 10k is the only data level that works, and it sits right at the edge of the cliff. Sampling new formulas online removes ntrain as a variable entirely.
- *First run:* A hypers with formulas resampled online (or ntrain=20k), steps=2000 with the cosine schedule from idea 6.

**8. Small capacity bump and median-focused loss trimming (lowest priority).**
- *Why:* B shows that more message-passing rounds hurt, so widen rather than deepen. Separately, hard instances can dominate early gradients and pull the median up.
- *First run:* H=192, rounds=4, everything else A. Separately, drop the hardest 10% of each batch from the loss for the first 50% of steps, then switch to full loss with `tailw` 0.3 to protect B90.

## 3. Try first this week

1. **Seed baseline:** A at 10k, seeds 1–4, with validation checkpointing and EMA (idea 2). This sets the noise bar for every comparison below.
2. **n=6 to n=8 transfer:** warm start, lr=6e-4, 1000 steps (idea 1).
3. **L=3 to L=5 curriculum:** 800 steps then 1000 steps (idea 3).
4. **Leakage ramp:** 0.4 to 0.1, at both 10k and 5k (idea 5).
5. **Untied to tied initialization:** layer-average arm (idea 4).
6. **Best of runs 2–5** combined with the cosine schedule, clipping, cost-head `lrbase`=0.3 and online data (ideas 6 and 7), 2000 steps, 2 seeds.

If runs 2–4 each beat the seed spread from run 1 but none gets close to ~20, that would suggest tied parameters can't express the solution at L=5. In that case the next lever is capacity in the per-formula `clause_w` head (idea 8), not more optimization tricks.
