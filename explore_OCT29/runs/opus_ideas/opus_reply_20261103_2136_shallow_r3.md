## Recommendation: declare shallow n15 capacity-limited under Hamming-shell

I'd make this #1. The remaining expected value of more shallow-n15 training is low, around 10–15% for a clean eval of 0.98 or better. Spend the 12 hours writing up the L=16 clear beat and the shallow n12 beat.

## Why this is close to proven

- **Every method lands in the same place.** CORE-RATIO, minimax, free-φ weighted, n15-only free-φ+γ, and L=1 all give eval n15 between 0.983 and 0.997. Different losses and freezes converging on the same floor points to a ceiling, not an optimizer problem.
- **A second layer adds almost nothing.** L=1 reaches 0.984 and the best L=2 reaches 0.983. If capacity were the slack, the extra layer would buy more than 0.001.
- **The global search answered the question.** It reached train R = 0.9807 and eval 0.997. The train optimum is no better than the reopen bar, and it overfits the train pack. The train pack is the problem, not the search.
- **The remaining gap is inside eval noise.** You need 0.983 → 0.980. That is probably about one standard error at scale1000. Any "beat" you find by repeatedly checking against scale1000 is likely selection on noise, which a reviewer would rightly flag.

### Closing check before you declare (about 1–1.5 h on 8 cores)

Re-run the same global search (801 starts plus LBFGS on the top 20), but on a large n15 pack that is disjoint from both the train set and scale1000. Use 500–1000 instances drawn from the same distribution as eval.
- If the best R on that pack is above 0.980, capacity is proven at L≤2. Stop and declare. This is the most likely outcome.
- If it is 0.978 or below, the ceiling exists and the old train pack was hiding it. Go to #2.

## #2 (only if the closing check reopens): retrain on the dense pack with held-out early stopping

- **Setup:** Warm-start free-φ+γ at L=2 from the 0.983 eval point and from the dense-pack global optimum. Train on the dense pack. Early-stop on a separate held-out validation split of about 300 instances, never on scale1000.
- **Kill criteria:** Stop if validation R is above 0.981 after about 1 h. Allow one scale1000 evaluation only, and only if validation is 0.979 or better.
- **Wall time:** 2–3 h.

## #3: make the loss match the eval metric, with mild tail control

- **Check the metric definition first.** Make the training objective exactly the eval statistic. Mean of per-instance ratios and ratio of means can differ by a few thousandths, which is your whole gap.
- **Add a tail term.** Use a CVaR term at a tail fraction of 0.2–0.3 on the worst n15 instances, weighted around 0.3 against the mean.
- **Keep the multi-n mix.** Train on n12 plus n15 with an n12 guard. The 0.983 best eval came from multi-n free-φ weighted training, which suggests the mix helps the model generalize.
- **Kill criteria:** Stop if held-out validation R is above 0.981 or n12 eval drifts above 0.978. One scale1000 shot only.
- **Wall time:** 1.5–2 h.

## Bottom line

Run the roughly 1-hour closing check, then most likely declare. If you do pursue #2 or #3, report any eventual shallow-n15 result with its held-out validation number and a single eval shot. A 0.979 found after many scale1000 looks isn't a defensible beat.
