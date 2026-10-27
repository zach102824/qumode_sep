# Opus 5.5 round-2 ideas (n=8 L=5 tied cost)

- Model: `claude-opus-5-5-xhigh` via `cursor-agent --mode ask`
- Generated: 2026-10-27 ~14:20 Asia/Shanghai
- Ideas only; flags: leakpen, wd, tailw, init_from/init_th_path, lrbase (noted missing), ema_decay, H, rounds, steps, lr, seed, sched/lr_schedule

---

## 1. Updated diagnosis

38.3 is a real basin, but this recipe rarely gets into it. Most round-1 arms measured seed luck rather than the effect of the arm. I read the run logs and `tiecost_lib.train_one`, and six findings change the round-1 picture.

**The runs fall into three regimes, not a smooth curve.** Every L=5 run ends in one of these:
- **Leak collapse:** Med 300–600, B90 2k–5k, leak about 0.5. This covers seed1, `A_L5_n5000` and the 5k leak ramp. `A_L7` and `cosine_2k` also pass through it early.
- **Near-uniform plateau:** Med 85–150, B90 230–300, which is barely better than uniform (128 / ~256). This covers seeds 2/3/4, leakramp_10k, the curriculum run, `B_L5`, `A_L5_n2000`, and `cosine_2k`, which recovered to val 165 at step 1200.
- **Interference basin:** Med about 38–54, B90 about 117–173. Only seed0 at 10k got here. It jumped between step 280 (val 220) and step 560 (val 54), then descended smoothly to 38–40. The interrupted `L5_s2000` run follows the same trajectory (val 40.3 at step 800).

So the real problem has two parts: the probability of leaving the plateau, then the quality you reach inside the basin.

**Each arm was a single draw.**
- `seed` controls both the initialisation and the minibatch order, because both come from the same random generator.
- At initialisation all the GNN heads are zero, so the starting circuit is the same for every formula. Only the random per-layer displacement init `A0` differs by seed.
- Changing `ntrain` also reshuffles the minibatch order. The 2k/5k/10k grid is therefore effectively three more seeds.
- The "5k cliff" looks exactly like seed1: both were already collapsed by step 280. The non-monotonic `ntrain` result is seed noise, not a data effect.
- Recipe A at L=5, 10k, gives {38, 82, 98, 136, 472} over 5 seeds, a median of about 98.
- Curriculum (120) and leak ramp (110) sit in the middle of that distribution. They weren't shown to be worse, only not lucky. Comparing each one to the best of 5 seeds is a winner's-curse comparison.

**Getting into the basin depends on the training path, not just the init.** `leakramp_10k` used seed0's init *and* seed0's minibatch order, and changed only the early `leakpen` (0.4). It still landed on the plateau. `cosine_2k` (seed0, `lr`=1.5e-3) overshot into a leak transient and then fell back to the plateau. My hypothesis: the way in passes through a moderately leaky intermediate state (seed0 at step 280 was 220/601). A strong early leak penalty blocks that path, and a higher `lr` overshoots it.

**Two round-1 tests weren't valid tests.**
- **EMA was misconfigured.** With `ema_decay`=0.999 over 1400 steps and no bias correction, about 0.999^1400 ≈ 25% of the average is still the random init. The EMA validation Med was 240–900 in every run, including the champion. The EMA result is void, not negative.
- **Every run was under-trained.** In every run that didn't collapse, the best checkpoint was the last evaluation. The L=3 curves were still falling 7–10% per evaluation interval. The champion's val Med had flattened (38.4 → 39.9) while B90 was still falling (128 → 117). That's the noise floor of a constant `lr`, and the champion was never annealed.

**The test set is too small for the next phase.** Test has 100 formulas and val has 200. The median of 1/p over 100 formulas has roughly ±13% noise (one sigma). That resolves 38 vs 100, but not 38 vs 32.

**`leakpen` and `grad_clip` can't stop the collapses.**
- `leakpen`·mean(leak) is at most about 0.14–0.4, against loss terms of order 10.
- `grad_clip` doesn't limit Adam's step size, because Adam normalises the gradient anyway.
- Early collapse looks like a step-size problem. With wide zero-initialised heads, Adam's first steps are about `lr`·sign(g) per parameter. The displacement and SNAP outputs then move together, fast.
- Warmup is the usual fix. It isn't a flag, but you can emulate it with a two-stage `init_from` run.

Three notes on the flags:
- `lrbase` doesn't exist in `tiecost_lib`; there's no per-group learning rate.
- `sched` is `lr_schedule`, either constant or cosine.
- `init_from` is `init_th_path`, and it needs an exact size match. It loads the best-val `th`.

## 2. Answers to (1)–(4)

**(1) Lucky seed or real basin?** Both: the basin is real, and getting into it is lucky.
- **Evidence it's real:**
  - The seed0 EMA rerun reproduced 38.317 exactly, because training is deterministic.
  - Val (39.9) matches test (38.3).
  - B90 is 145 against about 280 on the plateau, so this is a different kind of solution, not noise on the median.
  - The curve shows a sharp drop followed by a smooth descent.
- **Evidence it's lucky:** about 1 in 5 seeds, and about 1 in 10 of all L=5 runs, got in.
- **How to confirm the basin is wide:** fine-tune from the champion with other seeds (experiment 1 below). If they all stay at or below 40, it's a wide basin.

**(2) How to close 38 → 14 with tied cost?** I'd expect optimisation-only levers (anneal, multi-start, best-of-k) to get to roughly 25–32; that's a guess. Getting to 14, and to B90 29 (a 5× drop, which is harder), depends on a number nobody has measured: the ceiling of a tied L=5 circuit.
- Untied reaching ~12 doesn't imply tied can reach 14. With tied cost, the coupling strength between the two cavities is the same in every layer, so there's no QAOA-style gamma ramp.
- Some layer variation is still possible. Per-layer effects that act on one cavity alone (single-variable terms, and terms inside one cavity) can be emulated through the SNAP heads (`ph_lin`, `ph_prod`).
- For scale: `nm_scale` = 8/79 ≈ 0.10, so `g0`=3 gives about 0.3 rad per violated clause per layer. A random assignment violates about 4.9 clauses.
- So measure the ceiling first. Then close the gap with multi-start plus anneal, and use the untied homotopy (gradually forcing the untied model to become tied) only if the diagnostics support it.

**(3) Abandon L=5?** No. It's the only depth that has reached the basin.
- L=3 is the easy landscape: smooth descent to 62–106, and still falling when the runs stopped. One long L=3 run would give a reference value for its ceiling.
- L=7 has a single seed, so it's no evidence either way.
- Leave L=5 only if the ceiling diagnostic says tied L=5 can't get below about 20. In that case go deeper, not shallower.
- Don't retry the L3→L5 curriculum. With tied cost, every added layer necessarily applies the cost once more, even with zero displacement and zero SNAP. Growing depth can never preserve the function. The round-1 curriculum also copied a stage-1 model that was on the plateau (134).

**(4) Are untied and n6 transfer still worth forcing?**
- **Untied transfer: yes, but not as a weight copy, and only after two cheap checks** on the saved Part1 untied checkpoint (`gnn5_n8_pure_cw_theta.npz`):
  - Check how much the untied model's per-layer gamma varies across layers.
  - Replace each per-layer gamma and `w` with the layer average and evaluate Med, with no training.
  - If the projected Med is about 60 or less, or gamma is nearly flat across layers, the homotopy is worth building. Add a penalty in the untied trainer that pulls each layer's (gamma, `w`) toward the layer mean, ramp it up, then project to tied and fine-tune. The fine-tune needs a mapper that matches weights by name.
  - If the projected Med is above 150, or gamma ramps strongly, the untied 12 lives in a basin that tied cost can't express. Drop it.
- **n6 transfer: drop it.**
  - Several weights change shape with K and NU (`TH0`, `Mcav1`, `Wfock`, `Wphi_a`), and the Fock encoding differs (8 vs 16 levels per cavity).
  - The curriculum already showed that copying only the shape-compatible weights from a plateau model gives a plateau model.
  - You would first need a tied n=6 L=5 model that is itself in a basin, which is a whole project.

## 3. Ranked next experiments (impact × ease)

1. **Anneal the champion.**
   - Settings: `init_from`=`A_L5_n10000/weights.npz`, `sched`=cosine, `lr`=3e-4, `lr_min`=1e-5, `steps`=2000, `ema_decay`=0.995. Keep `leakpen`/`tailw`/`H`/`rounds` unchanged. Run `seed`=0 and `seed`=1.
   - Expected: about 38 → 30–34. This also answers (1).
   - Risk: Adam restarts with fresh moments and may kick the model out of the basin. If val at the first evaluation is above 60, rerun with `lr`=1e-4.
   - Keep the already-queued `opus_s2600_lr0p0011` as the control. It replays the champion's path exactly through step 1400 and then just keeps going at constant `lr`.
   - Impact medium (fairly sure), ease trivial.

2. **Basin screen with early stopping.**
   - Run 20 new seeds (5–24) on recipe A with `steps`=700.
   - Promote any run with val Med below 80. The champion was at 54 by step 560, and every run that missed was at 118 or above.
   - Anneal the promoted runs as in experiment 1.
   - Cost: about 200 s each, roughly 70 minutes in total.
   - This gives you the hit rate, which is the metric every future arm should be judged on, plus several basin samples to pick from by val.
   - Impact high, ease high.

3. **Ceiling and gamma diagnostics** (small scripts, no GNN training).
   - **Existing weights:** forward-pass all the saved checkpoints and compare `g_tied`, `w_tied` and per-layer |α| against `amax`=1.8, for the champion versus seeds 2/3/4 and the collapsed runs. A near-uniform plateau would show up as the cost being effectively switched off.
   - **Untied checkpoint:** the two checks from (4).
   - **Per-instance tied oracle:** for about 30 test formulas, optimise the L=5 circuit directly (A, TH, g, `w`; about 200 parameters) with about 16 restarts each.
   - Decision rule: if the oracle Med is about 8 or less, the gap is optimisation, so push experiments 2, 4 and 6. If it's about 20–30, 14 isn't reachable with tied L=5, so move to L=7/9 using the experiment 2 protocol.
   - Impact very high (it decides (2)–(4)), ease medium.

4. **Reverse leak ramp to raise the hit rate.**
   - Settings: `leakpen_start`=0.0, `leakpen_end`=0.14, `leakpen_frac`=0.3. That holds the penalty at 0 for 30% of training, then rises with a cosine.
   - Use 10 seeds at 700 steps with the promotion rule from experiment 2, and compare hit rates.
   - Motivation: raising the early `leakpen` was the only change that knocked seed0 off its path into the basin.
   - Expect more collapses too; count both outcomes.
   - Impact medium-high, ease high.

5. **Emulated warmup.**
   - Stage 1: `lr`=1.5e-4 for 150 steps. Stage 2: `init_from` the stage-1 weights, `lr`=1.1e-3, 550 steps.
   - Use the same seeds as experiment 2 for a paired comparison.
   - This targets the seed1-type and 5k-type collapses, which hit 2 of 7 A@L5 runs.
   - It might also suppress the early exploratory phase that leads into the basin, which is exactly what the paired comparison measures.
   - Impact medium, ease medium; it needs a two-stage wrapper like the curriculum one.

6. **Loss shaping inside the basin** (after experiment 1).
   - From the annealed champion, try `tailw` ∈ {1.0, 2.5} and `bs`=500.
   - The champion's B90/Med ratio is 3.8, against 2.1 for WalkSATlm, so the tail is relatively worse.
   - The exp tail term has zero gradient whenever 1/p is above about 33, because it clips at nl=3.5. Right now only the linear top-16% term pushes on the tail, so `tailw` is the only tail lever.
   - Impact low-medium, ease high.

7. **`g0` init sweep** (only if the weight diagnostics in experiment 3 show it matters).
   - Trigger: the champion's learned gamma sits far from 3.0 while the plateau seeds stay near 3.0 or drift toward 0.
   - Then add a one-line `g0` flag and screen `g0` ∈ {1.5, 4.5, 6} with the experiment 2 protocol.
   - Impact potentially high, ease medium.

8. **Untied → tied homotopy** (only if the untied checks in experiment 3 pass). Highest potential payoff, lowest ease.

**Cancel or deprioritise** `H192_cosine`, `H192_s2600`, and the seed0 grid at `lr` 8e-4/6e-4. Each is a single coin flip with about a 20% hit rate, and neither outcome would tell you anything. Expect `cosine_2k` to finish on the plateau (val 165 at step 1200). Report every future arm as a hit rate plus the best Med inside the basin, never as a single Med.

## 4. What to try this week (ordered)

1. Run experiment 1 (two seeds) next to the queued `s2600_lr1.1e-3` control, and cancel the H=192 runs and the off-seed `lr` grid.
2. Run the forward-pass diagnostic on the existing weights from experiment 3. It takes minutes and decides whether experiment 7 is worth doing.
3. Run the 20-seed screen from experiment 2 and anneal the hits.
4. Run the untied checks and the per-instance tied oracle from experiment 3, in parallel if you can spare CPU.
5. Run the reverse leak ramp screen from experiment 4 on 10 seeds.
6. Build a 1000-formula test set with a WalkSATlm reference before claiming anything below about 30.
7. At the end of the week, decide from the ceiling result: homotopy (experiment 8), deeper L with the screen protocol, or the `g0` sweep (experiment 7).
