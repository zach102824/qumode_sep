# N=12 L=1 parameter concentration and transfer (same protocol as LE PR116)

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Entropy-peak \(B^*\) only. **Not** an architecture change and **not** a GR retune.

Repeats LE PR116’s concentration/transfer protocol at **N=12 L=1** so three sizes can be compared: N=12 (this trial), N=15 (PR116), N=18 (LE’s next trial). Motivated by typical-instance / parameter-concentration structure in the SK-QAOA literature (arXiv:2204.02923 context). Not paper \(\mathcal{V}(M)\).

PR111 compile miss stays closed. PR117 `L3_2XBUDGET_FAIL` stands; no `q=4096`. L=2 holdout FAIL stands. Seeds `10552–10559` not reused. `10624–10631` unused.

## Split (frozen before any SPSA)

Train **10664–10671**. Test **10672–10679**. Stage A fits **train only**. Then freeze coordinate-wise **mean and median** of the 8 Nov3 vectors and of the 8 QAOA vectors (plus var/std/range). Do not peek test before freeze. Stage B only after that freeze: evaluate those four frozen vectors on test (1 query each) and run independent Nov3/QAOA on the same test seeds as the per-instance ceiling.

**Primary aggregator is mean.** Do **not** switch the primary gate to median because of PR116’s N=15 test-median result. Median is frozen alongside mean and is not selected on test.

## Seeds

Exclusive unread `10664–10679` (n=16). Collision-checked vs LE `10432–39` / `10552–59` / PR116 `10600–15` / next `10648–63`, local R31 `10536–51`, R32 `10584–99`, R30 `10504–19`, local-next reserved `10632–47`, this stream `10520–35` / `10552–59` / `10568–83` / `10480–87` / `10456–63` / `10376–83` / `10368–75` / `10616–23` (PR117), unused-hold `10624–31`, unused hole `10560–67`, holdout `10144–51`, validation `11000–11031`, GR/final `20000–20999`.

## Grid

`results/n12xfer/`. N=12, L=1, `Q=1024`, `T=122`. Interleaved packing m=4. \(B^*\) cut `0..5`. Arms `nov3_center4` (6 params) and `qaoa_center4` (2 params). Method ids 1 and 0.

## Contrasts (same as PR116)

1. Train independent Nov3−QAOA (10664–71)
2a. Test **shared-mean** Nov3− shared-mean QAOA (**primary**)
2b. Test shared-median Nov3− shared-median QAOA (descriptive)
3. Test ceiling independent Nov3−QAOA (10672–79)
4. Transfer gap: test shared-mean Nov3 vs independently optimized Nov3 (descriptive; not a gate that Nov3 must beat itself)

Bootstrap 4000, `default_rng(3334)`, n=8.

**TRANSFER_MEAN_NEGATIVE_CI** iff contrast 2a n=8 and the shared-mean Nov3−QAOA mean \(\Delta E\) CI is entirely `<0`. Else **UNSTABLE** (same label rule as LE PR116). Exploratory; not adoption; does not revive L=2/L=3. Do not switch the primary aggregator to median.

## Train concentration

Report train mean/std/range of packed coordinates, including **s1 / θ_K / θ_M and QAOA β**, with the n=8 caveat. Descriptive comparison to PR116 N=15 train stds (different seeds; not paired):

| coord | PR116 N=15 train std |
|---|---:|
| θ_K | 0.0511 |
| θ_M | 0.0233 |
| s1 | 0.8058 |
| QAOA β | 0.4015 |

## Queries

Independent: \(16\times 2\times 1024=32768\). Transfer: \(8\times 2\times 2\times 1=32\). New **32800** if complete. Historical SK **3014656**. Cumulative **3047456**. Classical \(B^*\) separate. Workers 4, BLAS 1.

## Off limits

Compiler, L=3 budget climb, V(M), L=2 copy, pair-shell, GR, Nov9/SYK, switching primary to median on test.
