# cloud_day1 CPU explore

Target is the box **NEW BEST**: L=10, leakpen=0.1, lr=5e-5, learned per-cavity α0. Small val n6 Med≈10.2 / n9≈18. **reval40 ≈ 14.5 / 19.5** is the bar. A result is a win only when a val-40 reval beats both of those medians and the n9/n6 ratio stays in [0.75, 2.0]. Search-val numbers are not a win.

The question for this farm is whether **L∈{12,14}** at a similar lr and leakpen can beat that bar. An L=10 run at the same lr, leakpen, and step count is a control, not a new box config. Learned α0 only. Earlier r014 (L=10, lr=1e-4, small-val 18/23) is an easier, superseded reference.

## Protocol

- Workers: 4 processes × 1 torch thread (host CPUs: 4).
- Search: 1000 Adam steps, batch 4 per n, train pool 128, val 16 (early-stop on S_comb, patience 5 evals).
- S_comb = 0.35·S6 + 0.65·S9, S = (log Med + log B90) / 2. Lower is better.
- Promote only when S_comb improves **and** n9 Med does not regress. A healthier n6→n9 ratio with a worse n9 Med is rejected (`ratio_only`).
- Reval: val n=40 and test n=100 (test is reported, not used to pick the winner). nf=24 (library default).
- Grad clip 5.0, same as `short_train`. nf=24. No inter-trial kill: deeper L is slower early and keeps its full step budget.

## Box reference

| | L | leakpen | lr | α0 | n6 Med | n9 Med | what it is |
|---|---:|---:|---:|---|---:|---:|---|
| box NEW BEST, small val | 10 | 0.1 | 5e-5 | learned | 10.2 | 18 | optimistic |
| **box NEW BEST, reval40** | 10 | 0.1 | 5e-5 | learned | **14.5** | **19.5** | win bar |
| old r014, small val | 10 | 0.1 | 1e-4 | learned | 18 | 23 | superseded |

## Search trials

Med/B90 are 1/p on the **search val** (16 formulas). Lower is better. keep=1 means that trial promoted when it finished.

| id | tag | L | lp | lr | H | r | steps | status | S_comb | n6 Med/B90 | n9 Med/B90 | ratio | leak9 | keep |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| 16 | grid | 12 | 0.2 | 0.0001 | 128 | 4 | 1000 | ok | 3.145 | 11.6/25.4 | 15.7/47.5 | 1.36 | 0.060 | 0 |
| 4 | core | 14 | 0.2 | 5e-05 | 128 | 4 | 1000 | ok | 3.246 | 8.5/22.5 | 24.6/52.4 | 2.90 | 0.070 | 0 |
| 0 | core | 12 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.247 | 12.6/23.5 | 11.1/91.6 | 0.88 | 0.032 | 1 |
| 14 | seed | 14 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.411 | 21.2/123.5 | 11.6/45.2 | 0.55 | 0.033 | 0 |
| 15 | l10_control | 10 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.473 | 12.4/28.6 | 27.2/68.3 | 2.20 | 0.061 | 0 |
| 3 | core | 12 | 0.2 | 5e-05 | 128 | 4 | 801 | early_stop | 3.509 | 12.5/25.4 | 21.7/101.5 | 1.73 | 0.050 | 1 |
| 11 | grid | 12 | 0.3 | 5e-05 | 128 | 4 | 1000 | ok | 3.563 | 9.8/16.7 | 25.4/145.7 | 2.58 | 0.052 | 0 |
| 2 | l10_control | 10 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.631 | 14.9/28.4 | 22.0/124.6 | 1.48 | 0.047 | 0 |
| 1 | core | 14 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.676 | 15.9/100.4 | 18.8/81.8 | 1.18 | 0.037 | 0 |
| 5 | core | 12 | 0.05 | 5e-05 | 128 | 4 | 841 | early_stop | 3.685 | 8.5/40.9 | 23.7/151.7 | 2.78 | 0.034 | 0 |
| 10 | grid | 14 | 0.1 | 3e-05 | 128 | 4 | 841 | early_stop | 3.699 | 22.6/190.6 | 14.1/68.4 | 0.62 | 0.040 | 0 |
| 13 | seed | 12 | 0.1 | 5e-05 | 128 | 4 | 761 | early_stop | 3.746 | 7.7/39.6 | 38.4/121.4 | 5.02 | 0.083 | 0 |
| 17 | grid | 14 | 0.2 | 0.0001 | 128 | 4 | 1000 | ok | 3.749 | 43.9/121.6 | 16.0/63.0 | 0.36 | 0.068 | 0 |
| 20 | grid | 12 | 0.2 | 3e-05 | 128 | 4 | 1000 | ok | 3.777 | 11.6/35.5 | 25.9/168.3 | 2.22 | 0.039 | 0 |
| 21 | grid | 14 | 0.2 | 3e-05 | 128 | 4 | 921 | early_stop | 3.854 | 26.7/47.9 | 34.7/86.4 | 1.30 | 0.042 | 0 |
| 6 | core | 14 | 0.05 | 5e-05 | 128 | 4 | 1000 | ok | 3.937 | 12.2/55.1 | 36.8/148.6 | 3.00 | 0.098 | 0 |
| 18 | gnn_H192 | 12 | 0.1 | 5e-05 | 192 | 4 | 761 | early_stop | 4.241 | 48.4/167.9 | 30.5/119.6 | 0.63 | 0.049 | 0 |
| 8 | grid | 14 | 0.1 | 0.0001 | 128 | 4 | 681 | early_stop | 4.298 | 34.0/314.0 | 35.9/104.4 | 1.05 | 0.063 | 0 |
| 7 | grid | 12 | 0.1 | 0.0001 | 128 | 4 | 921 | early_stop | 4.523 | 42.4/107.5 | 46.6/254.3 | 1.10 | 0.077 | 0 |
| 19 | gnn_rounds6 | 14 | 0.1 | 5e-05 | 128 | 6 | 801 | early_stop | 4.599 | 38.7/94.8 | 86.9/193.5 | 2.25 | 0.082 | 0 |
| 9 | grid | 12 | 0.1 | 3e-05 | 128 | 4 | 761 | early_stop | 5.415 | 12.3/55.5 | 320.4/1600.1 | 26.07 | 0.215 | 0 |
| 12 | grid | 14 | 0.3 | 5e-05 | 128 | 4 | 481 | early_stop | 5.679 | 54.3/270.3 | 207.9/1064.7 | 3.83 | 0.090 | 0 |

Trials logged: **22** (finite scores: 22).

## Winner

Search champion was trial **0** (S_comb=3.247 on val 16). Final BEST is trial **0** (`core`, L=12, leakpen=0.1, lr=5e-05, H=128, rounds=4, seed=0), chosen on the val-40 reval (same trial).

| split | n | B | Med | B90 | S | mean leak |
|---|---:|---:|---:|---:|---:|---:|
| val | 6 | 40.0 | 12.01 | 29.71 | 2.938 | 0.0901 |
| val | 9 | 40.0 | 16.98 | 85.26 | 3.639 | 0.0353 |
| test | 6 | 100.0 | 15.12 | 36.26 | 3.153 | 0.0868 |
| test | 9 | 100.0 | 16.79 | 120.84 | 3.807 | 0.0383 |

### Versus box reval40 (14.5 / 19.5)

**Val-40 full win: trial 0**, L=12, leakpen=0.1, lr=5e-5, learned α0. Medians **12.01 / 16.98** versus the box bar **14.5 / 19.5**. Ratio n9/n6 = 1.41, inside [0.75, 2.0]. This is the saved BEST. It was chosen on val-40 S_comb among candidates that did not regress n9; trial 16 has a slightly lower val-40 S (3.366 vs 3.394) but n9=21.3, which misses 19.5, so it was not promoted.

The same weights still clear the bar on larger val slices (not used to pick the trial; checked after the farm):

| val formulas | n6 Med | n9 Med | ratio | under 14.5 / 19.5 |
|---:|---:|---:|---:|---|
| 40 | 12.01 | 16.98 | 1.41 | yes |
| 80 | 11.50 | 18.64 | 1.62 | yes |
| 120 | 12.59 | 18.66 | 1.48 | yes |
| 200 (all val) | 13.66 | 18.07 | 1.32 | yes |

Held-out test (100 formulas, not used to select): n6 Med 15.12, n9 Med 16.79, ratio 1.11. n9 stays under 19.5. n6 is 15.12, which is above 14.5. The n6 edge is real on val and does not repeat on this test split.

Same-budget L=10 control (trial 15, val-40): n6=10.56, n9=27.75, ratio=2.63. That control beats the box on n6 and loses on n9. Trial 0's extra depth is what brings n9 from 27.8 down to 17.0, with n6 still under 14.5 on val. Our L=10 control is not a reproduction of the box checkpoint; the win claim is against the published 14.5 / 19.5 numbers.

Trial 16 (L=12, leakpen=0.2, lr=1e-4) is the best deep run by val-40 S alone (n6=12.52, n9=21.32) and is not a full win, because n9 misses 19.5.

### Other revals (val 40 / test)

| id | tag | L | lp | lr | H | r | val n6 | val n9 | val ratio | val S | test n6 | test n9 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | core | 12 | 0.1 | 5e-05 | 128 | 4 | 12.0 | 17.0 | 1.41 | 3.394 | 15.1 | 16.8 |
| 16 | grid | 12 | 0.2 | 0.0001 | 128 | 4 | 12.5 | 21.3 | 1.70 | 3.366 | 15.5 | 19.4 |
| 4 | core | 14 | 0.2 | 5e-05 | 128 | 4 | 10.0 | 28.9 | 2.89 | 3.538 | 13.6 | 22.6 |
| 15 | l10_control | 10 | 0.1 | 5e-05 | 128 | 4 | 10.6 | 27.8 | 2.63 | 3.563 | 15.0 | 22.4 |

## Forks

- `gnn_H192` trial 18: L=12 lp=0.1 lr=5e-05 H=192 rounds=4 → search S=4.241 n6=48.4 n9=30.5 ratio=0.63 keep=0 (ratio_only).
- `gnn_rounds6` trial 19: L=14 lp=0.1 lr=5e-05 H=128 rounds=6 → search S=4.599 n6=38.7 n9=86.9 ratio=2.25 keep=0 (s_comb).

A stronger H or extra rounds is a thin fork on the same ansatz (learned α0, exact cost phase). It is not an OCT20 calibrated SNAP–D–SNAP mixer; that stack is a different layer primitive and was not ported.

## Files

- `trials.csv` — every trial, including early-stops.
- `BEST/` — final weights (`weights_n6.pt`, `weights_n9.pt`), `best.json`, `arch.json`. Mid-saved on each search promote, then refreshed if the val-40 reval picks a different trial.
- `BEST_reval.json` — val-40 and test metrics.
- `BEST_search/` — present only if the reval winner differs from the search champion.

## n=12 scale of the L=12 BEST

Checkpoint is trial 0 (L=12, leakpen=0.1, lr=5e-5, learned α0), frozen in `BEST_round1/` before later fork promotes. n=6 and n=9 load `weights_n6.pt` / `weights_n9.pt` directly. There is no `weights_n12.pt`: the n=12 circuit uses the n=9 network, with the message-passing body, q, and gamma copied exactly and the fourth cavity's α0 / displacement / SNAP head set equal to the third. Pack is `data/data_n12.npz` from `gen_teb_packs` (val 200, test 100, dense E). Uniform 1/p on that pack is about 2048 / 4096.

| n | split | B | Med | B90 | mean leak |
|---:|---|---:|---:|---:|---:|
| 6 | val | 200 | 13.66 | 64.68 | 0.089 |
| 6 | test | 100 | 15.12 | 36.26 | 0.087 |
| 9 | val | 200 | 18.07 | 124.95 | 0.036 |
| 9 | test | 100 | 16.79 | 120.84 | 0.038 |
| 12 | val | 200 | 3.45e8 | ~1e12 | ~0 |
| 12 | test | 100 | 1.92e8 | ~1e12 | ~0 |

n=6 and n=9 match the earlier full-val / test numbers for this checkpoint. n=12 does not transfer. Median 1/p is orders of magnitude above the uniform baseline, while leakage stays about 0, so the state remains in the computational subspace and almost none of it sits on solutions. Zero-shot cavity repeat is not a scale result. Full table: `n12_scale.json`.

## Fork from L=12 BEST

Warm-started from trial 0 (L=12, leakpen=0.1, lr=5e-5, learned α0). Grid is L∈{12,14,16} × lr∈{3e-5,5e-5,1e-4} × leakpen∈{0.05,0.1,0.2}. No L≈10. The parent cell is `best_continue` (more Adam steps, patience 10 evals). Other cells are shorter forks. Promote only if val-40 beats **12.01 / 16.98** on both medians and on S_comb, and test medians stay ≤ 1.20× the cloud test (15.12 / 16.79).

| id | tag | L | lp | lr | steps | status | search n6/n9 | val40 n6/n9 | val40 S | test n6/n9 | decision |
|---:|---|---:|---:|---:|---:|---|---|---|---:|---|---|
| 22 | best_continue | 12 | 0.1 | 5e-05 | 1521 | early_stop | 6.3/12.0 | 7.5/15.2 | 3.172 | 8.9/14.3 | ratio |
| 23 | fork | 16 | 0.1 | 5e-05 | 640 | ok | 9.6/7.7 | 8.9/9.7 | 3.047 | 11.1/8.8 | promote |
| 24 | fork | 16 | 0.05 | 5e-05 | 640 | ok | 6.4/9.3 | 7.1/12.6 | 3.166 | 7.9/9.9 | val_med9 |
| 25 | fork | 16 | 0.2 | 5e-05 | 561 | early_stop | 19.1/8.8 | 20.9/9.2 | 3.231 | 18.9/9.1 | ratio |
| 26 | fork | 16 | 0.1 | 0.0001 | 321 | early_stop | 37.7/9.1 | 37.3/13.9 | 3.917 | 52.5/12.9 | ratio |
| 27 | fork | 16 | 0.05 | 0.0001 | 401 | early_stop | 43.2/9.9 | 38.0/11.7 | 3.676 | 64.4/10.0 | ratio |
| 28 | fork | 16 | 0.2 | 0.0001 | 441 | early_stop | 79.5/8.0 | 86.6/10.7 | 3.874 | 59.6/9.0 | ratio |
| 29 | fork | 16 | 0.1 | 3e-05 | 640 | ok | 12.7/8.5 | 15.2/9.6 | 3.392 | 18.1/10.0 | ratio |
| 30 | fork | 16 | 0.05 | 3e-05 | 640 | ok | 8.5/9.2 | 12.4/9.2 | 3.078 | 13.1/9.6 | ratio |
| 31 | fork | 16 | 0.2 | 3e-05 | 640 | ok | 10.3/9.0 | 10.2/10.6 | 2.971 | 10.7/9.5 | val_med6 |
| 32 | fork | 14 | 0.1 | 5e-05 | 640 | ok | 9.2/8.4 | 9.4/11.7 | 3.074 | 9.0/10.9 | val_med6 |
| 33 | fork | 14 | 0.05 | 5e-05 | 521 | early_stop | 17.7/9.6 | 27.3/12.6 | 3.443 | 20.6/13.3 | ratio |
| 34 | fork | 14 | 0.2 | 5e-05 | 640 | ok | 14.1/9.4 | 22.1/13.6 | 3.471 | 24.8/11.2 | ratio |
| 35 | fork | 14 | 0.1 | 0.0001 | 640 | ok | 73.3/8.6 | 64.7/11.3 | 4.006 | 89.3/10.3 | ratio |
| 36 | fork | 14 | 0.05 | 0.0001 | 361 | early_stop | 161.3/13.8 | 121.2/17.7 | 4.349 | 223.1/16.3 | ratio |
| 37 | fork | 14 | 0.2 | 0.0001 | 441 | early_stop | 30.3/12.6 | 30.8/16.3 | 3.586 | 39.1/14.9 | ratio |
| 38 | fork | 14 | 0.1 | 3e-05 | 640 | ok | 33.2/12.4 | 27.7/13.8 | 3.517 | 29.3/12.3 | ratio |
| 39 | fork | 14 | 0.05 | 3e-05 | 640 | ok | 41.7/9.5 | 46.1/13.1 | 3.654 | 53.5/10.9 | ratio |
| 40 | fork | 14 | 0.2 | 3e-05 | 601 | early_stop | 17.0/10.4 | 18.0/12.5 | 3.308 | 16.3/10.7 | ratio |
| 41 | fork | 12 | 0.05 | 5e-05 | 281 | early_stop | 12.6/11.1 | 12.0/17.0 | 3.394 | 15.1/16.8 | val_med6 |
| 42 | fork | 12 | 0.2 | 5e-05 | 561 | early_stop | 9.5/14.8 | 10.7/18.1 | 3.271 | 12.0/15.2 | val_med6 |
| 43 | fork | 12 | 0.1 | 0.0001 | 281 | early_stop | 12.6/11.1 | 12.0/17.0 | 3.394 | 15.1/16.8 | val_med6 |
| 44 | fork | 12 | 0.05 | 0.0001 | 281 | early_stop | 12.6/11.1 | 12.0/17.0 | 3.394 | 15.1/16.8 | val_med6 |
| 45 | fork | 12 | 0.2 | 0.0001 | 401 | early_stop | 9.3/18.0 | 11.6/20.8 | 3.324 | 13.0/19.4 | val_med6 |
| 46 | fork | 12 | 0.1 | 3e-05 | 281 | early_stop | 12.6/11.1 | 12.0/17.0 | 3.394 | 15.1/16.8 | val_med6 |
| 47 | fork | 12 | 0.05 | 3e-05 | 281 | early_stop | 12.6/11.1 | 12.0/17.0 | 3.394 | 15.1/16.8 | val_med6 |
| 48 | fork | 12 | 0.2 | 3e-05 | 281 | early_stop | 12.6/11.1 | 12.0/17.0 | 3.394 | 15.1/16.8 | val_med6 |

Promoted trial **23** (L=16, leakpen=0.1, lr=5e-5, learned α0): val-40 **8.89 / 9.67**, test **11.07 / 8.79**. Both val medians beat the cloud L=12 bar of 12.01 / 16.98, and both test medians beat that checkpoint's test of 15.12 / 16.79. Val ratio 1.09 is inside [0.75, 2]. This is the saved `BEST/`. The L=12 parent remains in `BEST_round1/`.

27 trials, one promote. The longer L=12 continuation (trial 22, 1521 steps) reached val-40 7.53 / 15.23 and test 8.90 / 14.34, which also beats 12.01 / 16.98 and does not hurt test, but the val ratio is 2.02, just outside the gate, so it was not promoted. Several later L=12 warm-starts never left the parent checkpoint (val-40 still 12.0 / 17.0).

## n=12 joint fine-tune (no cavity repeat)

FactorGNN sets C = n/3 when the module is constructed. The readout widths for α0, displacement, and SNAP are fixed Linear layers, not a per-cavity map applied to however many cavities the graph has. A variable-C forward is not already there, so this is path (b): a C=4 network, body plus q and gamma copied from the n=9 L=12 BEST, cavity heads left at their own init, then Adam on n=6, n=9, and 32 n=12 training instances together (2 n=12 + 1 n=6 + 1 n=9 per step). Learned α0. No cavity repeat.

Uniform 1/p on the n=12 pack is about 2048 / 4096.

| stage | steps | n | split | Med | B90 | mean leak |
|---|---:|---:|---|---:|---:|---:|
| 120 | 120 | 12 | val 40 | 384 | 8836 | 0.111 |
| 120 | 120 | 12 | test 100 | 560 | 12383 | 0.110 |
| 120 | 120 | 6 | val 40 | 14.8 | 30.2 | 0.054 |
| 120 | 120 | 6 | test 100 | 11.5 | 46.8 | 0.052 |
| 120 | 120 | 9 | val 40 | 17.9 | 93.7 | 0.050 |
| 120 | 120 | 9 | test 100 | 16.5 | 88.3 | 0.050 |
| 300 (saved) | 300 | 12 | val 40 | 201 | 3799 | 0.165 |
| 300 (saved) | 300 | 12 | test 100 | 256 | 3984 | 0.161 |
| 300 (saved) | 300 | 6 | val 40 | 20.6 | 77.4 | 0.103 |
| 300 (saved) | 300 | 6 | test 100 | 19.7 | 90.5 | 0.102 |
| 300 (saved) | 300 | 9 | val 40 | 19.8 | 76.8 | 0.057 |
| 300 (saved) | 300 | 9 | test 100 | 18.4 | 107.0 | 0.058 |

n=12 median 1/p comes down from the zero-shot ~1e8 into the low hundreds, past the uniform median, after a few hundred joint steps. The tail is still about the uniform B90, and leakage rises to ~0.16. The saved 300-step point is the better n=12 checkpoint and the worse n=6/n=9 one: n=6 test moves from 15.1 on the frozen L=12 BEST to 19.7. Weights: `n12_finetune/`. The n=6/n=9 BEST used above is still trial 23, not this fine-tune.

## Stronger leakpen on n=12 only

Another 200 Adam steps from that checkpoint, leakpen 0.5 on n=12, n=6 and n=9 frozen. `BEST/` (L=16 trial 23) was not written. Its test medians stay **11.07 / 8.79**.

| n | split | Med | B90 | mean leak |
|---:|---|---:|---:|---:|
| 12 | val 40 | 143 | 1783 | 0.103 |
| 12 | test 100 | 172 | 2268 | 0.101 |
| 6 | val 40 | 20.6 | 77.4 | 0.103 |
| 6 | test 100 | 19.7 | 90.5 | 0.102 |
| 9 | val 40 | 19.8 | 76.8 | 0.057 |
| 9 | test 100 | 18.4 | 107.0 | 0.058 |

Against the 300-step point, n=12 val B90 falls from 3799 to 1783 and test B90 from 3984 to 2268. Medians fall too (201 → 143 val, 256 → 172 test), and n=12 leak falls from 0.16 to 0.10. Frozen n=6/n=9 match the 300-step numbers exactly. They are not the L=16 test.

## Size-general FactorGNN

`SizeGeneralGNN` keeps the factor-graph body and replaces `Linear(H, C*2)` (and the matching displacement and SNAP linears) with one MLP applied to each cavity. The cavity vector is the mean of that cavity's three variable states. q and gamma stay a global pool. One module runs n=6, n=9, and n=12. Parameter shapes do not depend on C.

Trained on n=6 and n=9 only (L=12, leakpen 0.3, body and q initialized from the L=12 n=9 checkpoint, cavity MLP fresh), about 720 Adam steps. n=12 was not in the loss.

| n | split | Med | B90 | mean leak |
|---:|---|---:|---:|---:|
| 6 | val 40 | 12.2 | 26.1 | 0.170 |
| 6 | test 100 | 14.5 | 31.7 | 0.189 |
| 9 | val 40 | 28.5 | 150 | 0.012 |
| 9 | test 100 | 32.4 | 133 | 0.013 |
| 12 | val 40 | 2.1e9 | ~1e12 | ~0 |
| 12 | test 100 | 6.1e7 | ~1e12 | ~0 |

The shared module fits n=6. It does not put mass on n=12 solutions: median 1/p stays orders of magnitude above the uniform baseline, with leakage about 0. Weights are in `size_general/`, not in `BEST/`.
