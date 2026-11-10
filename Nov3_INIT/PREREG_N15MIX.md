# N=15 L=1 mixer-angle freeze (concentrating θ_K, θ_M)

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Entropy-peak \(B^*\) only. **Not** an architecture change, **not** a GR retune, **not** an unsigned-mean copy, **not** a sign-aligned copy, and **not** a two-template / better-of-two `sign(s1)`/`sign(β)` copy.

## Ideas considered (this turn)

1. **Picked: freeze train-mean mixer angles.** Prior L=1 fits show \(\theta_K\) and \(\theta_M\) concentrate even when \(s_1\) (and \(\gamma\)) are bimodal. Freeze those two coordinates from train only, then per-instance SPSA on \((\gamma,s_1,s_2,s_3)\) at test. Could still beat equal-layer QAOA because the L=1 edge does not appear to live in the mixer angles. Addresses the two-template transfer gap without copying better-of-two energy min.
2. **Not run: 1-NN \(J\)-similarity parameter transfer.** Typical-instance test of arXiv:2204.02923, but \(n=8\) train in \(\dim J = N(N-1)/2\) is underpowered (nearest neighbor is noise). Not worth a real batch.

Two-template original+confirmatory ladder is complete. Do not open N=6/N=21. Do not copy better-of-two. Do not reopen unsigned-mean as primary. L=2 holdout FAIL stands. PR117 `L3_2XBUDGET_FAIL` stands; no `q=4096`. PR111 closed. No compiler, V(M), pair-shell, tied-SNAP, L3 freeze-order, start-count, or GR patch.

## Split (frozen before any SPSA)

Train **10888–10895**. Test **10896–10903**. Stage A: independent Nov3 and QAOA on **train only**. Then freeze, from train Nov3 only:

- \(\bar\theta_K\) = mean of \(\theta_K\)
- \(\bar\theta_M\) = mean of \(\theta_M\)

Do **not** freeze \(\gamma,s_1,s_2,s_3\). Do **not** invent extra gauges. Do **not** partition by \(\mathrm{sign}(s_1)\). Do not peek test before freeze. Do not switch the frozen coordinates after seeing test (do not add \(s_2\) later).

Stage B only after that freeze:

- Independent Nov3 and QAOA on test (ceiling / matching QAOA)
- Mixer-freeze Nov3: 4-D SPSA on \((\gamma,s_1,s_2,s_3)\) with \(\theta_K,\theta_M\) locked to the frozen train means. Same v4 equal4 schedule, \(Q=1024\), \(T=122\), 4 starts. Start 0 is the center-4 slice \((\gamma,s_1,s_2,s_3)=(0.25,-0.5,0,0)\). Energy-only selection; F post hoc.

## Seeds

Exclusive unread `10888–10903` (n=16). Collision-checked vs other-stream reserved `10872–87`, PR128 `10856–71`, LE `10840–55`, R35 `10824–39`, PR127 `10808–23`, LE PR126 `10792–10807`, PR124 `10776–91`, LE PR125 `10760–75`, PR122 `10744–59`, LE PR123 `10728–43`, R34 `10712–27`, PR120 `10696–10711`, PR118 `10664–79`, unused-hold `10624–31`, unused hole `10560–67`, width `10568–83`, gauge `10552–59`, holdout `10144–51`, validation `11000–11031`, GR/final `20000–20999`. Local-later `10904+` unused. Do not post-hoc-gate any prior test set.

## Grid

`results/n15mix/`. N=15, L=1, `Q=1024`, `T=122`. Interleaved packing m=5. \(B^*\) cut `0..6`. Independent arms `nov3_center4` (6 params, method id 1) and `qaoa_center4` (2 params, method id 0). Mixer-freeze arm method id **10** (RNG only; not a new circuit). Same FT v4 center-4 recipe (`Nov3_INIT/alloc.py` `spsa_block`; LE copy sha `de595de59`). 4-D sigmas are the v4 Nov3 \(\gamma\) and shell sigmas: \((0.10, 0.15, 0.15, 0.15)\).

## Contrasts

Primary: test mixer-freeze Nov3 − independently-fitted test QAOA.

Required secondaries (do not replace the primary):

1. Train independent Nov3−QAOA (10888–95)
2. Test ceiling independent Nov3−QAOA (10896–10903)
3. Mixer-freeze Nov3 vs independently-fitted test Nov3 (constraint gap)

Bootstrap 4000, `default_rng(3334)`, n=8.

**MIXER_FREEZE_NEGATIVE_CI** iff primary n=8 and the mixer-freeze Nov3−QAOA mean \(\Delta E\) CI is entirely `<0`. Else **UNSTABLE**. Exploratory; not adoption; does not replace per-instance SPSA on the full 6-vector; does not revive L=2/L=3.

## Queries

Independent: \(16\times 2\times 1024=32768\). Mixer-freeze: \(8\times 1024=8192\). New **40960**. Historical SK **3211552**. Cumulative **3252512**. Classical \(B^*\) separate. Workers 4, BLAS 1.

## Off limits

Two-template copies, unsigned-mean or sign-aligned as primary, compiler, L=3 budget climb, V(M), L=2 copy, pair-shell, GR, Nov9/SYK, N=6/N=21, switching frozen coordinates after test, retuning v4 \(c,A,\alpha,\gamma\).
