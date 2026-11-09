# N=9 L=1 parameter concentration and transfer (same protocol as LE PR116 / PR118)

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Entropy-peak \(B^*\) only. **Not** an architecture change and **not** a GR retune.

Repeats LE PR116 / this-stream PR118 concentration/transfer protocol at **N=9 L=1**. Completes the lower end of the unsigned-mean scaling ladder: N=9 (this), N=12 (PR118 PASS), N=15 (PR116 PASS), N=18 (PR119 FAIL). Motivated by typical-instance / parameter-concentration structure in the SK-QAOA literature (arXiv:2204.02923 context). Not paper \(\mathcal{V}(M)\).

PR111 compile miss stays closed. PR117 `L3_2XBUDGET_FAIL` stands; no `q=4096`. L=2 holdout FAIL stands. PR118 stands. Seeds `10664–10679` not reused. `10624–10631` unused.

## Split (frozen before any SPSA)

Train **10696–10703**. Test **10704–10711**. Stage A fits **train only**. Then freeze coordinate-wise **mean and median** of the 8 Nov3 vectors and of the 8 QAOA vectors (plus var/std/range). Do not peek test before freeze. Stage B only after that freeze: evaluate those four frozen vectors on test (1 query each) and run independent Nov3/QAOA on the same test seeds as the per-instance ceiling.

**Primary aggregator is mean.** Do **not** switch the primary gate to median because of PR116’s N=15 test-median result, PR118’s N=12 test-median result, or any N=9 test peek. Median is frozen alongside mean and is not selected on test.

## Seeds

Exclusive unread `10696–10711` (n=16). Collision-checked vs LE `10648–63` / LE next `10680–95`, local R33 `10632–47`, unused-hold `10624–31`, unused hole `10560–67`, PR118 `10664–79`, PR116 `10600–15`, PR117 `10616–23`, LE `10432–39` / `10552–59`, R31 `10536–51`, R32 `10584–99`, R30 `10504–19`, this stream `10520–35` / `10568–83` / `10480–87` / `10456–63` / `10376–83` / `10368–75`, holdout `10144–51`, validation `11000–11031`, GR/final `20000–20999`.

## Grid

`results/n9xfer/`. N=9, L=1, `Q=1024`, `T=122`. Interleaved packing m=3. \(B^*\) cut `0..3`. Arms `nov3_center4` (6 params) and `qaoa_center4` (2 params). Method ids 1 and 0.

## Contrasts (same as PR116 / PR118)

1. Train independent Nov3−QAOA (10696–10703)
2a. Test **shared-mean** Nov3− shared-mean QAOA (**primary**)
2b. Test shared-median Nov3− shared-median QAOA (descriptive)
3. Test ceiling independent Nov3−QAOA (10704–10711)
4. Transfer gap: test shared-mean Nov3 vs independently optimized Nov3 (descriptive; not a gate that Nov3 must beat itself)

Bootstrap 4000, `default_rng(3334)`, n=8.

**TRANSFER_MEAN_NEGATIVE_CI** iff contrast 2a n=8 and the shared-mean Nov3−QAOA mean \(\Delta E\) CI is entirely `<0`. Else **UNSTABLE** (same label rule as LE PR116 / PR118). Exploratory; not adoption; does not revive L=2/L=3. Do not switch the primary aggregator to median.

## Train concentration

Report train mean/std/range of packed coordinates, including **s1 / θ_K / θ_M and QAOA β**, with the n=8 caveat, plus s1 sign split. Descriptive comparison to PR118 N=12 and PR116 N=15 train stds (different seeds; not paired):

| coord | PR118 N=12 train std | PR116 N=15 train std |
|---|---:|---:|
| θ_K | 0.0573 | 0.0511 |
| θ_M | 0.0152 | 0.0233 |
| s1 | 0.8265 | 0.8058 |
| QAOA β | 0.5523 | 0.4015 |

Unsigned-mean ladder (primary 2a, already frozen, not this trial’s selector): N=12 PASS ΔE −1.093; N=15 PASS ΔE −0.764; N=18 FAIL ΔE +1.826 (LE PR119, train s1 4/4 sign split).

## Queries

Independent: \(16\times 2\times 1024=32768\). Transfer: \(8\times 2\times 2\times 1=32\). New **32800** if complete. Historical SK **3047456**. Cumulative **3080256**. Classical \(B^*\) separate. Workers 4, BLAS 1.

## Off limits

Compiler, L=3 budget climb, V(M), L=2 copy, pair-shell, GR, Nov9/SYK, switching primary to median on test.
