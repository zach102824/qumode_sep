# L=2 CAPACITY NEAR-MISS DECLARE (train{6,9}, gates locked)

**Declared:** 2026-11-04 ~00:25 Asia/Shanghai (box).
**ALL_CORE_BEAT = False** after extensive shallow hunt. This is an **empirical basin claim**, not a proof.

## Locked setting
- Gates: Hamming-shell SNAP; W=H⊗3; train ONLY {6,9}; eval CORE {6,9,12,15} scale1000 B=1000
- BEAT = R≤0.98 = Med(inv_p)/QAOA_Med

## Best known point (champ)
`pareto_blend/eval_L2_a0.80_s1.json` + `champion/params_L2_train69_pareto_a080_s1.pt`

| n | Med | QAOA Med | R | verdict |
|---|-----|----------|---|---------|
| 6 | — | 5.3257 | **0.9573** | BEAT |
| 9 | — | 12.6416 | **0.9782** | BEAT |
| 12 | — | 29.6063 | 0.9858 | TIE |
| 15 | — | 76.0878 | 0.9853 | TIE |

Knee = max(R12,R15) = **0.9858**. Simultaneous gap to BEAT ≈ **0.005–0.006** on n12 and n15.

## Evidence summary
- **≥163** CORE evals with n6&n9 BEAT; **0** ALL_CORE_BEAT.
- Radial ε scale of champ (c2,c3): **local minimum of knee exactly at ε=0**.
- Champ↔warm Pareto blend: knee minimized at a_champ=1.0.
- Dense trust multi-start (box+cloud): all starts → same basin; more starts useless (cloud bc-ca109616).
- Objective-gap fix (`--loss med_align` = soft_quantile(1/p)/QAOA_Med): QAOA-warm still R9≈1.00; delta-ball around champ best knee 0.9866; soft_q=0.4 slides to n12 BEAT / n15≈0.996.
- Ideas A–D (med_align, soft_q, delta-warm, hinge) + pareto_fine/lucky/dense_micro: none beat knee 0.9858.
- Cloud best pre-med_align n15 = 0.9878 (n9med ρ0.03 s3).

## Claim
Under these locks, L=2 SNAP/QAOA shallow training on train{6,9} appears **capacity-limited** near R≈0.985–0.986 on the n12/n15 knee: improving one of {n12,n15} into BEAT systematically worsens the other past 0.99. Unlock paths would need different gates, train set, L, or ansatz — not more starts with the same loss family.

## Still running (will update if BEAT appears)
soft_q∈{0.35,0.4} delta; cloud bc-0fa5a09f / bc-5198772e; pareto_fine/lucky; soft_med.
