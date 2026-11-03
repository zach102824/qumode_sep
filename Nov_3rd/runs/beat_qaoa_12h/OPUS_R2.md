Ranking: (1) fix the n15 shortfall at L=2 by rebalancing the loss, (2) reach L=16 by growing depth step by step from the exact dual, (3) solve L=1 with an exhaustive scan. Keep Idea1 (CORE-RATIO) and kill the other two Round-1 ideas.

These wall times are estimates. I couldn't find timing logs, so I've written them as multiples of the wall time of Idea1 at L=2 (called T₂ below). Every claimed BEAT should also hold on seeds 1 and 2, and the paired-bootstrap 95% upper bound on R should be below 1.0. At B=1000, an R of 0.993 may not differ from 1.0 at all.

Two quick checks first:
- **Train or eval problem?** Compute Idea1's R on the n15 *training* set. If train R is already ≤0.98, the shortfall is overfitting, so add more samples. If train R is around 0.99, the problem is optimization or the penalty, and Idea A below targets that.
- **Protocol conflict:** `Nov/PLAN_Nov3rd_final.md` still locks training to n6 only, 1000 samples and L-BFGS only. Your Round-1 setup (multi-n, Adam) breaks that lock. Update the plan or note the change in `RESULTS.md` so the final table isn't mislabelled.

## Next ideas (ranked)

### A. Minimax CORE-RATIO polish at L=2, warm-started from Idea1 (target: n15 at L=2)
- **Train ns:** {6,9,12,15}. Use 250/n for n6–12 and raise n15 to 1000. n6 has about 4% of margin to give up.
- **Freeze/scale:** nothing frozen. Warm-start all of γ, c1, c2, c3 from the Idea1 best.
- **Loss:** replace the mean over n with a soft-max, τ·logsumexp over n of (mean log(Eθ/EQ))/τ, with τ=0.01. Cut the penalty λ from λ₀ to λ₀/10 and then to 0. The (c2²+c3²) penalty pulls every n back toward the exact dual (R=1), and n15, which has the least room to improve, feels that pull most.
- **Opt:** Adam with 200 steps at lr 3e-4, then L-BFGS for 20 iterations with strong-Wolfe line search.
- **L order:** L=2 only.
- **Success:** R ≤0.98 on all four CORE n, at minimum n12 and n15, on 3 seeds.
- **Wall:** about 0.5·T₂ per seed, so 3 seeds in parallel on 8 cores take about 0.6·T₂.
- **Kill:** stop if n15 *train* R is still above 0.985 after the L-BFGS stage, or if n12 drops back to TIE.

### B. Progressive-depth trust region around the dual (target: n12 and n15 at L=16)
- **Train ns:** {9,12,15} at 250/n, plus n6 at 100 samples as a guard.
- **Freeze/scale:** stage 1 freezes γ and c1 at the QAOA-L16 dual, which is the R=1 point you verified in Round 0. Only per-layer residuals δc2 and δc3 train (32 parameters), held within ‖δ‖ ≤ ρ_L with ρ grown 0.05 → 0.1 → 0.2. Stage 2 unfreezes c1 and γ with ρ=0.2.
- **Loss:** the same soft-max ratio as Idea A, with λ=0 because the trust region does the regularising.
- **Opt:** L-BFGS only, 15 iterations per stage. Adam's noise at depth 16 tends to undo the warm start.
- **L order:** 4 → 8 → 16. Initialise each depth from the shallower one by linear interpolation over layers (the standard depth-to-depth interpolation trick for QAOA), using the QAOA ladder at that L as the base.
- **Success:** R ≤0.98 at L=16 on n12 and n15.
- **Wall:** about 8·T₂ per seed, or 10–12·T₂ for 2 seeds in parallel. Run 2 seeds, not 3.
- **Kill:** stop early if n12 or n15 isn't at or below 0.99 by the end of L=8. If n12 and n15 at L=16 still exceed 0.99 after stage 2, every training path has failed, and only then does the Arm C projector come back into play.
- **Idea1 at L=16:** compare it to B at the L=8 checkpoint and keep whichever is ahead.

### C. Exhaustive scan at L=1 (target: L=1 coverage)
- **Train ns:** {6,9,12,15} at 250/n.
- **Freeze/scale:** with only 4 parameters, use a 15×15 grid of (c2,c3) values out to ±3× the Idea1-L2 layer-1 magnitudes. At each grid point, L-BFGS-optimise γ and c1 only.
- **Loss:** the same soft-max ratio, λ=0.
- **Opt:** for the top 5 grid cells, run a full joint L-BFGS over all 4 parameters for 30 iterations.
- **L order:** L=1. Run it alongside A, since it's cheap.
- **Success:** R ≤0.98 on n12 and n15.
- **Wall:** about 0.3·T₂, since each step is roughly half the cost of L=2 and the scan parallelises trivially.
- **Kill:** if the best grid cell's n15 R is above 0.99, L=1 is capacity-limited. Report it as an honest TIE and spend no more effort on it. A full scan makes that a defensible negative result.

## Round-1 verdicts
- **Idea1 (CORE-RATIO): promote.** It's the base for A, B and C. The one knob to change is the penalty λ, cut to λ₀/10 or 0 in the L-BFGS stage.
- **Idea2 (RULE-SCALED): kill.** It had the worst n15 (0.999) and only one BEAT. Keep κ≈1.107 only as a check on how c2 and c3 should scale with n when you reach n18.
- **Idea3 (TRUST-RESIDUAL): kill as a standalone run.** ρ=0.1 is too tight, and n6 BEAT without being in the training set, which suggests the trust-region shape generalises. Fold it into Idea B with the one knob change of a growing ρ schedule (0.05 → 0.2) in place of the fixed 0.1.

## Suggested schedule on 8 cores
1. Run A (3 seeds) and C together on all 8 cores.
2. Let Idea1-L16 finish. Then start B from whichever is better at L=8, either Idea1 or the progressive warm start.
