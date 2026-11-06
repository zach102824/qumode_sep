# q≡1 freeze vs learned q

Clean paired A/B on `qtimes_cost`, recipe matched to box BEST **hot0003**.

## Recipe

- L = 10, leakpen = 0.1, lr = 5e-05
- learned α0 (both arms; `head_alpha0` stays in Adam)
- early-stop on S_comb = 0.35·S(n=6) + 0.65·S(n=9), S = (log Med + log B90) / 2
- patience = 60 evals, max steps = 800, min_delta = 1e-06
- CPU threads requested = 8 (intra-op). Host CPUs = 4, interop threads = 1
- arch = baseline factor GNN {"H": 128, "rounds": 4, "agg": "mean", "dropout": 0.0, "layernorm": true, "residual": true, "degree_features": true, "lr": 5e-05, "wd": 0.0001, "ema": 0.0}
- seed = 0, max_train = 32, batch = 4 per n
- select val = first 20 val instances; official reval = first 40
- learned q: `1 + 9·sigmoid(u)`
- freeze q: `q_from_logit(..., freeze_q1=True)` returns exact 1 (logit × 0), logits zeroed, `head_q` removed from Adam
- ratio = Med(1/p) / Med(WalkSAT flips) on the same instances (R=20, p=0.5)
- init checksum of every parameter except `head_q` matched across arms

## Decision rule (fixed before the run)

reval40 S_comb lower by >= 1% relative, mean leak not worse by > 0.05 on either n, and disjoint val[20:40] S_comb agrees.
Select-val S_comb is not used to name a winner.

## Result

**No win claimed.** Reval40 S_comb is lower for learned q (5.4435 vs 5.6759, relative gap 4.09%), and the disjoint val[20:40] slice points the same way (5.5986 vs 5.6581). The pre-registered leak rule blocks a win: learned mean leak on n=6 is 0.0903 versus 0.0395 for q≡1 (difference 0.0508, over the 0.05 margin). n=9 leak is 0.0790 versus 0.0628 (difference 0.0162, inside the margin).

The leak check fires before the holdout flag is set, so `holdout_agrees` is null in the json. The slice numbers themselves are in the table below; they do not reverse the S_comb ordering.

On reval40, lower Med, B90, leak, and ratio are better. The arms trade off:

- n=6: learned is better on Med (36.02 vs 40.98) and ratio (7.20 vs 8.20). q≡1 is better on B90 (145.6 vs 186.1), leak (0.0395 vs 0.0903), and on S(n=6) (4.347 vs 4.405).
- n=9: learned is better on Med (93.5 vs 387.3) and ratio (7.20 vs 29.8). q≡1 is better on B90 (919 vs 1749) and leak (0.0628 vs 0.0790).

WalkSAT median flips on these same 40 val instances are 5 (n=6) and 13 (n=9). Both arms sit above those bars. Uniform-sampling median 1/p is 32 (n=6) and 256 (n=9): learned n=9 Med is below that uniform figure; q≡1 n=9 Med is above it; both n=6 Meds are above it.

Learned q did not collapse to 1. Reval40 per-layer means run from about 3.3 to 8.1 on n=6 (mean 5.08) and from about 2.0 to 9.1 on n=9 (mean 4.65). The freeze arm is exactly 1 on every layer, and its `head_q` weights are zero.

### Select val (early-stop objective, not the claim)

| arm | n | Med | B90 | leak | ratio |
|---|---:|---:|---:|---:|---:|
| learned | 6 | 32.022 | 99.146 | 0.0810 | 6.404 |
| learned | 9 | 73.257 | 661.275 | 0.0785 | 5.635 |
| learned | S_comb | 4.9172 |  |  |  |
| freeze1 | 6 | 43.207 | 141.888 | 0.0414 | 8.641 |
| freeze1 | 9 | 328.505 | 1156.987 | 0.0629 | 25.270 |
| freeze1 | S_comb | 5.7018 |  |  |  |

### Reval max_val=40 (official)

| arm | n | Med | B90 | leak | ratio |
|---|---:|---:|---:|---:|---:|
| learned | 6 | 36.018 | 186.081 | 0.0903 | 7.204 |
| learned | 9 | 93.545 | 1749.032 | 0.0790 | 7.196 |
| learned | S_comb | 5.4435 |  |  |  |
| freeze1 | 6 | 40.982 | 145.620 | 0.0395 | 8.196 |
| freeze1 | 9 | 387.294 | 919.276 | 0.0628 | 29.792 |
| freeze1 | S_comb | 5.6759 |  |  |  |

### Disjoint slice val[20:40] (robustness, not a second claim)

| arm | n | Med | B90 | leak | ratio |
|---|---:|---:|---:|---:|---:|
| learned | 6 | 44.313 | 250.925 | 0.0996 | 9.329 |
| learned | 9 | 114.787 | 1749.032 | 0.0794 | 8.830 |
| learned | S_comb | 5.5986 |  |  |  |
| freeze1 | 6 | 38.308 | 183.526 | 0.0376 | 8.065 |
| freeze1 | 9 | 415.238 | 743.198 | 0.0626 | 31.941 |
| freeze1 | S_comb | 5.6581 |  |  |  |

## Training

| arm | best step | last step | early-stopped | hit step cap | select S_comb | mean q (reval n=6 / n=9) |
|---|---:|---:|---|---|---:|---|
| learned | 287 | 347 | True | False | 4.9172 | 5.083 / 4.649 |
| freeze1 | 80 | 140 | True | False | 5.7018 | 1.000 / 1.000 |

Learned-q per-layer means on reval40 are in `results.json` (`q_per_layer`).
Freeze arm mean q is 1.0 on every layer.

## Notes

- Single seed. This is a paired point estimate, not a significance test.
- Select val[:20] is a prefix of reval val[:40], so reval is not a pure holdout. The disjoint slice is val[20:40].
- Train loss is the unweighted mean of the n=6 and n=9 mini-batch losses. The 0.35/0.65 weights apply only to S_comb.
- WalkSAT is pure-Python SKC (p=0.5) on the val clauses, median flips per instance, then the median across instances.
- `ratio_uniform` in the json is Med(1/p) / median(2^n / n_solutions) and is not the headline ratio.
- No `runs/gnn_best` weights were loaded. If hot0003 used a different evolved arch, this run matches the listed train knobs on the baseline arch only.

