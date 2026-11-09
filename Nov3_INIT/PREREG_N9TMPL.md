# N=9 L=1 two-template (bimodal cluster-mean) transfer

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Entropy-peak \(B^*\) only. **Not** an architecture change, **not** a GR retune, **not** an unsigned-mean copy, and **not** a sign-aligned-only copy.

Same protocol as the LE N=18 two-template follow-up queued this hour, at the other unsigned-failure size (N=9), which also showed a clean 4/8 s1 split on PR120. Completes a two-template test where unsigned mean failed. Does not reopen the unsigned-mean ladder.

PR120 `UNSTABLE` stands. PR118 PASS stands. PR117 `L3_2XBUDGET_FAIL` stands; no `q=4096`. PR111 compile miss stays closed. L=2 holdout FAIL stands. Do not reuse `10696–10711`, `10664–79`, unused-hold `10624–31`, or hole `10560–67`. LE PR121 is development-only and is not copied as an unsigned-mean rerun.

## Split (frozen before any SPSA)

Train **10744–10751**. Test **10752–10759**. Stage A fits **train only**. Then freeze, from train only:

- Nov3 cluster+ \(\mu_+\) = coordinate-wise mean of vectors with \(s_1>0\)
- Nov3 cluster− \(\mu_-\) = coordinate-wise mean of vectors with \(s_1<0\)
- QAOA cluster+ \(\nu_+\) = coordinate-wise mean of vectors with \(\beta>0\)
- QAOA cluster− \(\nu_-\) = coordinate-wise mean of vectors with \(\beta<0\)
- Secondary only: Nov3 unsigned mean, Nov3 s1-sign-aligned mean (flip **only** \(s_1\) if \(s_1<0\); no extra gauge), QAOA unsigned mean

If a cluster is empty, record that and **do not invent a template**. Do not peek test before freeze. Stage B only after that freeze: evaluate every nonempty frozen template on test (1 query each) and run independent Nov3/QAOA on the same test seeds as the per-instance ceiling.

**Primary gate is two-template min**, locked before test. Do **not** switch the primary after seeing test. Unsigned-mean and sign-aligned-mean are secondary descriptives only.

## Seeds

Exclusive unread `10744–10759` (n=16). Collision-checked vs local R34 `10712–27`, LE `10728–43`, unused-hold `10624–31`, unused hole `10560–67`, PR120 `10696–10711`, PR118 `10664–79`, PR121 `10680–95`, PR119 `10648–63`, PR116 `10600–15`, PR117 `10616–23`, LE `10432–39` / `10552–59`, R31 `10536–51`, R32 `10584–99`, R30 `10504–19`, this stream `10520–35` / `10568–83` / `10480–87` / `10456–63` / `10376–83` / `10368–75`, holdout `10144–51`, validation `11000–11031`, GR/final `20000–20999`.

## Grid

`results/n9tmpl/`. N=9, L=1, `Q=1024`, `T=122`. Interleaved packing m=3. \(B^*\) cut `0..3`. Arms `nov3_center4` (6 params) and `qaoa_center4` (2 params). Method ids 1 and 0. Same FT v4 center-4 recipe as PR118/PR120 (`Nov3_INIT/alloc.py`; LE copy sha `de595de59`).

## Two-template rule

On each test instance, evaluate both nonempty Nov3 cluster templates and both nonempty QAOA cluster templates. Two-template energy for a method = **min** of its frozen cluster-template energies (energy-only choice among frozen train templates; not GS-fidelity postselection; not test-time SPSA). If a method has only one nonempty cluster, that method reduces to its single cluster mean; report this.

## Contrasts

Primary: test better-of-two Nov3 − better-of-two QAOA.

Required secondaries (do not replace the primary):

1. Train independent Nov3−QAOA (10744–51)
2. Test ceiling independent Nov3−QAOA (10752–59)
3. Two-template Nov3 vs independently-fitted test QAOA
4. Two-template Nov3 vs independently-fitted test Nov3 (transfer gap)
5. Each single Nov3 cluster template vs unsigned-mean QAOA and vs independent QAOA (no min-selection)
6. Unsigned-mean and sign-aligned-mean Nov3 vs unsigned-mean QAOA as replication descriptives only — do not reopen the unsigned-mean ladder

Bootstrap 4000, `default_rng(3334)`, n=8.

**TWO_TEMPLATE_TRANSFER_NEGATIVE_CI** iff primary n=8 and the better-of-two Nov3−QAOA mean \(\Delta E\) CI is entirely `<0`. Else **UNSTABLE**. Exploratory; not adoption; does not revive L=2/L=3.

## Queries

Independent: \(16\times 2\times 1024=32768\). Transfer: every nonempty frozen template × 8 test seeds × 1. If both Nov3 s1 clusters and both QAOA \(\beta\) clusters are nonempty: \(8\times 7=56\) (2 Nov3 cluster + 2 QAOA cluster + Nov3 unsigned + Nov3 sign-aligned + QAOA unsigned). New **32824** in that case. Empty clusters omit their template. Historical SK **3080256**. Cumulative **3113080** if 56 transfer evals. Classical \(B^*\) separate. Workers 4, BLAS 1.

## Off limits

Compiler, L=3 budget climb, V(M), L=2 copy, pair-shell, GR, Nov9/SYK, unsigned-mean copies at other N, switching primary after test.
