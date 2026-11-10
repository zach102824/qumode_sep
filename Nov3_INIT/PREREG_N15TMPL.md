# N=15 L=1 two-template (bimodal cluster-mean) confirmatory transfer

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Entropy-peak \(B^*\) only. **Not** an architecture change, **not** a GR retune, **not** an unsigned-mean copy, and **not** a sign-aligned-only copy.

Independent-stream confirmation of LE PR125 at N=15. Same protocol as PR122 (N=9), PR124 (N=12), PR127 (N=18), and LE PR123/PR125/PR126. Asks especially whether the transfer-gap CI stays including 0 (as in LE PR125’s \(+0.016\) \([−0.016,+0.046]\)) or becomes entirely \(>0\). Does **not** copy LE PR125, does **not** reuse its seeds `10760–75`, and does **not** claim those results as this stream’s. Two-template ladder is already complete (N9/N12/N15/N18 PASS); this is a confirmatory transfer at N=15, not a new untested size. Do not open N=6 or N=21. After this confirmatory, do not open another two-template size in this stream without a new parent steer.

PR127 `TWO_TEMPLATE_TRANSFER_NEGATIVE_CI` stands. PR124 stands. PR122 stands. PR120 `UNSTABLE` stands. PR118 PASS stands. PR117 `L3_2XBUDGET_FAIL` stands; no `q=4096`. PR111 compile miss stays closed. L=2 holdout FAIL stands.

## Split (frozen before any SPSA)

Train **10856–10863**. Test **10864–10871**. Stage A fits **train only**. Then freeze, from train only:

- Nov3 cluster+ \(\mu_+\) = coordinate-wise mean of vectors with \(s_1>0\)
- Nov3 cluster− \(\mu_-\) = coordinate-wise mean of vectors with \(s_1<0\)
- QAOA cluster+ \(\nu_+\) = coordinate-wise mean of vectors with \(\beta>0\)
- QAOA cluster− \(\nu_-\) = coordinate-wise mean of vectors with \(\beta<0\)
- Secondary only: Nov3 unsigned mean, Nov3 s1-sign-aligned mean (flip **only** \(s_1\) if \(s_1<0\); no extra gauge), QAOA unsigned mean

If a cluster is empty, record that and **do not invent a template**. Do not peek test before freeze. Stage B only after that freeze: evaluate every nonempty frozen template on test (1 query each) and run independent Nov3/QAOA on the same test seeds as the per-instance ceiling.

**Primary gate is two-template min**, locked before test. Do **not** switch the primary after seeing test. Unsigned-mean and sign-aligned-mean are secondary descriptives only.

## Seeds

Exclusive unread `10856–10871` (n=16). Collision-checked vs LE this hour `10840–55`, local R35 `10824–39`, PR127 `10808–23`, LE PR126 `10792–10807`, PR124 `10776–91`, LE PR125 `10760–75`, PR122 `10744–59`, LE PR123 `10728–43`, local R34 `10712–27`, PR120 `10696–10711`, PR121 `10680–95`, PR118 `10664–79`, PR119 `10648–63`, PR116 `10600–15`, PR117 `10616–23`, unused-hold `10624–31`, unused hole `10560–67`, LE `10432–39` / `10552–59`, R31 `10536–51`, R32 `10584–99`, R30 `10504–19`, this stream `10520–35` / `10568–83` / `10480–87` / `10456–63` / `10376–83` / `10368–75`, holdout `10144–51`, validation `11000–11031`, GR/final `20000–20999`. Do not post-hoc-gate any prior test set.

## Grid

`results/n15tmpl/`. N=15, L=1, `Q=1024`, `T=122`. Interleaved packing m=5. \(B^*\) cut `0..6`. Arms `nov3_center4` (6 params) and `qaoa_center4` (2 params). Method ids 1 and 0. Same FT v4 center-4 recipe as PR122/PR124/PR127 (`Nov3_INIT/alloc.py`; LE copy sha `de595de59`).

## Two-template rule

On each test instance, evaluate both nonempty Nov3 cluster templates and both nonempty QAOA cluster templates. Two-template energy for a method = **min** of its frozen cluster-template energies (energy-only choice among frozen train templates; not GS-fidelity postselection; not test-time SPSA). If a method has only one nonempty cluster, that method reduces to its single cluster mean; report this. Matching QAOA opportunity.

## Contrasts

Primary: test better-of-two Nov3 − better-of-two QAOA.

Required secondaries (do not replace the primary):

1. Train independent Nov3−QAOA (10856–63)
2. Test ceiling independent Nov3−QAOA (10864–71)
3. Two-template Nov3 vs independently-fitted test QAOA
4. Two-template Nov3 vs independently-fitted test Nov3 (transfer gap)
5. Each single Nov3 cluster template vs unsigned-mean QAOA and vs independent QAOA (no min-selection)
6. Unsigned-mean and sign-aligned-mean Nov3 vs unsigned-mean QAOA as replication descriptives only — do not reopen the unsigned-mean ladder

Bootstrap 4000, `default_rng(3334)`, n=8.

**TWO_TEMPLATE_TRANSFER_NEGATIVE_CI** iff primary n=8 and the better-of-two Nov3−QAOA mean \(\Delta E\) CI is entirely `<0`. Else **UNSTABLE**. Exploratory; not adoption; does not replace per-instance SPSA; does not revive L=2/L=3.

## Queries

Independent: \(16\times 2\times 1024=32768\). Transfer: every nonempty frozen template × 8 test seeds × 1. If both Nov3 s1 clusters and both QAOA \(\beta\) clusters are nonempty: \(8\times 7=56\) (2 Nov3 cluster + 2 QAOA cluster + Nov3 unsigned + Nov3 sign-aligned + QAOA unsigned). New **32824** in that case. Empty clusters omit their template. Historical SK **3178728**. Cumulative **3211552** if 56 transfer evals. Classical \(B^*\) separate. Workers 4, BLAS 1.

## Off limits

Compiler, L=3 budget climb, V(M), L=2 copy, pair-shell, GR, Nov9/SYK, unsigned-mean copies, two-template copies at new untested sizes (N=6/N=21 not opened), switching primary after test, copying LE PR125, reusing `10760–75` / `10840–55` / `10824–39` / `10808–23` / `10792–10807` / `10776–91` / `10744–59` / `10728–43`. After this confirmatory, do not open another two-template size without a new parent steer.
