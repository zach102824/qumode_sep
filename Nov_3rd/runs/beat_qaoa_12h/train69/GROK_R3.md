# Grok Round-3 — L=2 train{6,9} shallow ideas (post objective-gap)

Date: 2026-11-03 Asia/Shanghai. Context: ratio-of-medians gate ≠ median-of-ratios loss.
Cloud dense: more starts useless (same basin). Best pre-R3 box: pareto α=0.80 s1 n15=0.9853.
Cloud best n15=0.9878. Capacity NOT declared (knee still moves with α).

## Idea A — med_align / soft-quantile Med(1/p)  [PRIMARY, running]
Loss = Σ_n w_n · log( soft_q(inv_p) / QAOA_Med_n ). Directly optimizes the gate statistic.
Flags: `--loss med_align --soft_q 0.5` on trust69 / n9med / ratio69_w9.

## Idea B — soft_q≠0.5 (left-tail / B40 push)  [if A fails]
Same med_align but soft_q∈{0.4, 0.35} to pull the lower half of 1/p down; may move B90 and help n15.
Risk: can inflate Med if q too low — always report Med R.

## Idea C — pareto-warm + tiny med_align polish
Start from champ params_L2_train69_pareto_a080_s1.pt; freeze γ,c1; polish c2,c3 with med_align
inside ρ≤0.02–0.04 for 100–200 Adam steps. Keeps n6/n9 BEAT while nudging Med ratio on train.

## Idea D — asymmetric hinge on Med gaps
loss = w6·relu(log R6_med + ε) + w9·relu(log R9_med + ε) with ε=log(0.98) so once BEAT, stop
pushing that n and dump capacity into the other / holdout. Shallow; no new gates.

## Idea E — lucky α grid denser around 0.76–0.82 with med-seed polish
Lucky a0.76 already got n12 BEAT (0.979) but n15 slipped. Micro-grid α∈[0.74,0.84] step 0.01
+ 50-step Adam from blend, pick by max(R12,R15) then ALL_CORE. Pure search, no new loss.

If A–C all stall with n15≥0.985 and n12≥0.982 across ≥20 independent starts → consider capacity
near-miss declare for L=2 under locked gates (still NOT declaring yet).
