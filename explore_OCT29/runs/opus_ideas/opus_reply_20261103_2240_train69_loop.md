**Bottom line:** run Idea 1 first, at L=16. I expect L=16 to clear-beat all four sizes under train{6,9}. I don't expect any train{6,9} protocol to reach R ≤ 0.98 at n15 for L=2 under this gate head. Even when n15 was in the training set, the best L=2 results at n15 were 0.993 (`core_ratio`), 0.9947 (`trust_residual`) and 0.9995 (`rule_scaled`), and the dense-search ceiling is 0.9803. Removing n15 from training can only make that worse. Below I still give a best-effort shallow plan, but the overnight compute should go to L=16.

**What the existing results show**

- **Unscaled shared c2,c3 gets better with n at L=16.** The champion's R falls from 0.942 at n6 to 0.898 at n9, 0.885 at n12 and 0.882 at n15. So at depth, "shared c2,c3 overshoots at larger n" isn't happening. n9's R is already close to the n12 and n15 values, which makes n9 a usable stand-in for the larger sizes.
- **Scaling c2,c3 by 2/(n-1) hurt at n15.** At L=2, `rule_scaled` got 0.9995 at n15 versus 0.993 unscaled. If anything, the evidence points to no shrinkage, or even mild growth, with n.
- **`kappa` in `rule_scaled` is redundant, and it defeats the regularizer.** The loss only sees the product κ·c2 (and κ·c3), but λ penalizes the raw c2 and c3. The optimizer can grow κ and shrink c2, c3, which removes the penalty without changing the circuit. Drop κ or put it inside the penalty.
- **The training sizes are hard-coded.** `train_beat_idea.py` sets `train_ns` inside each `--idea` branch, so every idea below needs a `--train_ns 6,9` flag. Training also gets much cheaper, because the n12 and n15 simulations dominated the cost.

---

## Idea 1 (run first): n9-weighted, unscaled ratio loss with early stopping on held-out n9 instances

**Claim:** at L=16, the unscaled shared head transfers better as n grows. If training leans on n9 and stops on n9 data, n12 and n15 should follow.

**Knobs**
- Free: γ, c1, c2, c3, one shared value per layer (64 parameters at L=16, 8 at L=2).
- No n-scaling of c2,c3 at train or at eval: a_n = 1 for every n.
- Loss: the existing per-instance log(E_θ/E_Q), averaged per n, then combined with weights w6 = 1 and w9 = 2. A softmax-style minimax over the two sizes (temperature 0.01, as in `train_beat_minimax_polish.py`) is an equally good variant. Keep λ on c2²+c3², annealed from 1e-1 to 1e-3.
- Early stop / checkpoint choice: hold out about 20% of the n9 training instances and pick the checkpoint with the best soft-median log-ratio on that split. n12 and n15 are never touched.
- Optimizer: Adam, learning rate 1e-3, 600 steps, then 12 L-BFGS rounds of 8 iterations each (the `core_ratio` defaults), batch 32 per n.
- Init: QAOA dual from the paper-init n6 reference (c1 = −β, c2 = c3 = 0).
- Seeds: 0–5 at L=16, 0–3 at L=2.

**Why it can transfer:** R already improves with n at L=16 for this unscaled head (0.898 → 0.885 → 0.882 from n9 to n15). The failure seen with n6-only polish was the solution sizing itself to n6. Weighting n9 and stopping on n9 removes most of that pull, and 64 parameters are too few to memorize n9 specifically.

**Kill criteria and expected R at n15**
- L=16: kill a seed if its R at n9 on the full B=1000 eval is above 0.93, or if n15 comes out above 0.96. Expected n15 R is about 0.89–0.93, a likely clear beat.
- L=2: kill if n9 R is above 0.985. Expected n15 R is about 0.995–1.00, almost certainly not ≤ 0.98.

**Wall time (rough, on 8 cores):** about 3–5 minutes of training per seed at L=16, plus 5–10 minutes for each B=1000 CORE eval. Six L=16 seeds at 2 threads each, four in parallel, take about 40–60 minutes in total.

---

## Idea 2: learn a power-law exponent for c2,c3 strength, anchored at n9

**Claim:** two training sizes can pin down exactly one size-scaling parameter, so learn that one instead of fixing 2/(n-1).

**Knobs**
- Free: γ, c1, c2, c3 per layer, plus one scalar p. Replace κ with this.
- Scaling at train and at eval: c2,c3 at size n are multiplied by a_n = (8/(n−1))^p, so a9 = 1, a6 = 1.6^p, a12 = 0.727^p, a15 = 0.571^p.
- Init p = 0, which is identical to Idea 1. Add a prior penalty of 1e-2·p² and clip p to [−1, 2]. Penalize the scaled c2,c3 (a_n·c2, a_n·c3) so the gauge problem above doesn't come back.
- Loss, optimizer, steps, seeds and early stopping: same as Idea 1.
- Hedge: run fixed p ∈ {−0.5, 0, 0.5, 1} alongside. Report all four; don't select among them on n12/n15 results.

**Why it can transfer:** n6 versus n9 tells you which way the needed strength moves with n, and a single exponent extrapolates smoothly. The in-cavity pair share argument (2/(n−1)) corresponds to p = 1. The data suggests p ≤ 0, and this lets the training sizes decide.

**Kill criteria and expected R at n15**
- Kill if the learned p sits at either clip bound, or if n9 R is worse than in Idea 1.
- Expected n15 R: about 0.89–0.94 at L=16 (I'd guess roughly 50/50 that it beats Idea 1), and about 0.993–1.00 at L=2.

**Wall time:** about the same per seed as Idea 1. With the fixed-p sweep it's about 1–1.5 hours.

---

## Idea 3 (best-effort shallow, also a low-dimensional L=16 variant): train only c2,c3 in a trust region, against a soft-median loss

**Claim:** at L=2 the gap at n15 is under 1%, so the only realistic gain is from matching the loss to the median metric and changing as little as possible away from QAOA.

**Knobs**
- Frozen: γ and c1 at the paper-init QAOA values, so at ρ = 0 you have exactly QAOA.
- Free: c2,c3 only, projected onto an ℓ2 ball (`project_ball`) with radius ρ ∈ {0.05, 0.1, 0.2}. No n-scaling.
- Loss: a soft quantile at 0.5 of the log-ratio for each n, combined with weights w6 = 1 and w9 = 2.
- Optimizer: Adam, learning rate 5e-4, 600 steps, then 8 L-BFGS rounds.
- Seeds: 0–3 for each ρ.
- At L=16, same idea with c2,c3 restricted to a degree-3 Chebyshev schedule over layer index: 8 parameters instead of 32.

**Why it can transfer:** a small move away from the QAOA dual can't overshoot at large n, and the median loss targets exactly the metric being judged.

**Kill criteria and expected R at n15:** kill if n15 is above 1.00 or n9 is above 0.99. Expected n15 R is about 0.995 at L=2, which is a beat but not ≤ 0.98, and about 0.92–0.95 at L=16.

**Wall time:** about 20–40 minutes for the L=2 ρ sweep.

---

## Order of runs

1. **Idea 1 at L=16, seeds 0–5** (about 1 hour). If n15 R ≤ 0.95 on at least 3 seeds, the L=16 target is met under the lock.
2. **Idea 2 at L=16**, the learned p plus the fixed-p sweep (about 1–1.5 hours), to see if anything improves on Idea 1.
3. **L=2 runs of Ideas 1 and 3** on any spare cores, to document the shallow ceiling.

Ideas 1 and 2 need only small changes to `train_beat_idea.py`: a `--train_ns` flag, per-n loss weights, an n9 held-out split for early stopping, and p in place of κ. Idea 3 also needs the soft-quantile loss. I'm in Ask mode so I haven't made any edits; switch to Agent mode if you want me to implement them.
