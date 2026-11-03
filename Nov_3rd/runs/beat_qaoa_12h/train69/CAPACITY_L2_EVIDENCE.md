# L=2 CAPACITY NEAR-MISS — FINAL (train{6,9}, gates locked)

**Declared:** 2026-11-04 ~00:25 Asia/Shanghai (box).
**Finalized:** 2026-11-04 ~00:40 Asia/Shanghai — soft_q clouds + box soft_q finished; **no ALL_CORE_BEAT**.
**ALL_CORE_BEAT = False.** Empirical basin claim, not a proof.

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
- **≥230** CORE evals with n6&n9 BEAT on box; **0** ALL_CORE_BEAT.
- Radial ε scale of champ (c2,c3): **local minimum of knee exactly at ε=0**.
- Champ↔warm Pareto blend: knee minimized at a_champ=1.0.
- Dense trust multi-start (box+cloud bc-ca109616): all starts → same basin.
- Objective-gap fix (`--loss med_align`): QAOA-warm still R9≈1.00; delta-ball around champ best knee 0.9866; soft_q∈{0.35,0.4} slides to n12 BEAT / n15≈0.99+.
- Ideas A–E (med_align, soft_q, delta-warm, hinge, α grid) + pareto_fine/lucky/dense_micro: none beat knee 0.9858 under n6&n9 BEAT.
- Cloud Round3 denser (bc-5198772e → `cursor/train69-l2-round3-denser-76da`): 280 pareto_fine evals; best n15=0.9850 but n9 TIE; among n6&n9 BEAT, best knee ≥0.9867; when n6+n9+n12 BEAT, n15≥0.9891. **no ALL_CORE**.
- Cloud med_align (bc-0fa5a09f → `cursor/l2-med-align-sweeps-b563`): 40 soft_q=0.5 starts + Idea C/B/E; n15≥0.985 on every primary start; **no ALL_CORE**. Published knee unchanged.
- Box soft_q_delta (q∈{0.4,0.35} × ρ∈{0.015,0.025,0.04} × seeds 0–3): **24/24** ALL_CORE_BEAT=False. Best softq n6&n9-BEAT knee ≈0.9894.

## Claim (FINAL)
Under these locks, L=2 SNAP/QAOA shallow training on train{6,9} is **capacity-limited** near R≈0.985–0.986 on the n12/n15 knee: improving one of {n12,n15} into BEAT systematically worsens the other past 0.99. Unlock paths would need different gates, train set, L, or ansatz — not more starts with the same loss family.

## Clouds closed
| Agent | Branch | Outcome |
|-------|--------|---------|
| bc-5198772e Round3 denser | cursor/train69-l2-round3-denser-76da | FINISHED no-BEAT |
| bc-0fa5a09f med_align | cursor/l2-med-align-sweeps-b563 | FINISHED no-BEAT |
| bc-ca109616 dense+n9med | cursor/beat-qaoa-l2-train69-28cb | FINISHED no-BEAT (earlier) |

L=2 hunt wrapped. No new shallow arms.
