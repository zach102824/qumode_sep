# Six-start pressure test

The 48 stored procedures were not retrained. Tau is 1e-9. Equal depth and parameter-matched depths are separate ledgers. The L counts are the same chains, not independent samples.

N=12 new QAOA evaluations 50489 of 50489. N=6 extra 12000.

## Equal depth

| N | L | BOTH | Wilson | median error | worst error | worst energy margin | worst F margin |
|---|---|---|---|---|---|---|---|
| 6 | 1 | 48/48 | [0.926, 1.000] | 0.2546 | 0.2729 | 0.4386 | 0.2743 |
| 6 | 2 | 48/48 | [0.926, 1.000] | 0.0726 | 0.0726 | 0.3863 | 0.3720 |
| 6 | 3 | 48/48 | [0.926, 1.000] | 0.0633 | 0.0634 | 0.2630 | 0.3295 |
| 6 | 4 | 48/48 | [0.926, 1.000] | 0.0307 | 0.0310 | 0.2299 | 0.3024 |
| 6 | 5 | 48/48 | [0.926, 1.000] | 0.0189 | 0.0200 | 0.1756 | 0.2458 |
| 6 | 6 | 48/48 | [0.926, 1.000] | 0.0130 | 0.0597 | 0.0779 | 0.1674 |
| 9 | 1 | 48/48 | [0.926, 1.000] | 0.7551 | 0.7551 | 0.4663 | 0.0958 |
| 9 | 2 | 48/48 | [0.926, 1.000] | 0.6940 | 0.6940 | 0.0975 | 0.0498 |
| 9 | 3 | 48/48 | [0.926, 1.000] | 0.4547 | 0.4547 | 0.1439 | 0.1511 |
| 9 | 4 | 48/48 | [0.926, 1.000] | 0.4033 | 0.4033 | 0.0770 | 0.1299 |
| 9 | 5 | 48/48 | [0.926, 1.000] | 0.3648 | 0.3648 | 0.0217 | 0.1122 |
| 9 | 6 | 48/48 | [0.926, 1.000] | 0.3223 | 0.3223 | 0.0144 | 0.1100 |
| 12 | 1 | 48/48 | [0.926, 1.000] | 1.1230 | 1.1230 | 0.6714 | 0.0944 |
| 12 | 2 | 48/48 | [0.926, 1.000] | 1.0449 | 1.0449 | 0.1366 | 0.0580 |
| 12 | 3 | 48/48 | [0.926, 1.000] | 0.6092 | 0.6092 | 0.3189 | 0.2575 |
| 12 | 4 | 48/48 | [0.926, 1.000] | 0.5077 | 0.5077 | 0.2523 | 0.4270 |
| 12 | 5 | 48/48 | [0.926, 1.000] | 0.2280 | 0.2280 | 0.4082 | 0.6314 |
| 12 | 6 | 48/48 | [0.926, 1.000] | 0.1675 | 0.1675 | 0.3888 | 0.6076 |

## Parameter matched, next depth, and p=12

| N | L | matched BOTH | next BOTH | p12 BOTH | matched bar error |
|---|---|---|---|---|---|
| 6 | 1 | p3: 48/48 | p4: 43/48 | p12: 0/48 | 0.3264 |
| 6 | 2 | p5: 48/48 | p6: 48/48 | p12: 0/48 | 0.1956 |
| 6 | 3 | p6: 48/48 | p7: 48/48 | p12: 0/48 | 0.1376 |
| 6 | 4 | p8: 48/48 | p9: 48/48 | p12: 0/48 | 0.0821 |
| 6 | 5 | p9: 48/48 | p10: 48/48 | p12: 0/48 | 0.0626 |
| 6 | 6 | p11: 47/48 | p12: 5/48 | p12: 5/48 | 0.0188 |
| 9 | 1 | p3: 0/48 | p4: 0/48 | p12: 0/48 | 0.5986 |
| 9 | 2 | p5: 0/48 | p6: 0/48 | p12: 0/48 | 0.3865 |
| 9 | 3 | p6: 0/48 | p7: 0/48 | p12: 0/48 | 0.3367 |
| 9 | 4 | p8: 0/48 | p9: 0/48 | p12: 0/48 | 0.2553 |
| 9 | 5 | p9: 0/48 | p10: 0/48 | p12: 0/48 | 0.2233 |
| 9 | 6 | p11: 0/48 | p12: 0/48 | p12: 0/48 | 0.1703 |
| 12 | 1 | p3: 0/48 | p4: 0/48 | p12: 0/48 | 0.9281 |
| 12 | 2 | p5: 0/48 | p6: 0/48 | p12: 0/48 | 0.6363 |
| 12 | 3 | p6: 0/48 | p7: 0/48 | p12: 0/48 | 0.5563 |
| 12 | 4 | p8: 0/48 | p9: 0/48 | p12: 0/48 | 0.4357 |
| 12 | 5 | p9: 48/48 | p10: 48/48 | p12: 48/48 | 0.3903 |
| 12 | 6 | p11: 48/48 | p12: 48/48 | p12: 48/48 | 0.3150 |

Shared-gamma rescue, not a new 48-procedure sample: 10328 evaluations. Resume of stored k=40 ended at error 0.232689, gradient 1.226e-05. Local restarts improved 7 of 48; best error 0.2333892342969941.
Replays: {'N12_p6': {'logged_E': -8.056899807721328, 'replay_E': -8.056899807721328, 'delta': 0.0, 'hash_match': True, 'grad_norm': None, 'F': 0.29467665942546084}, 'N9_procedure_worst': {'logged_E': -6.0148284946152115, 'replay_E': -6.0148284946152115, 'delta': 0.0, 'hash_match': True, 'grad_norm': 5.600711468738038e-05, 'F': 0.6239285797231288}}.

The new N=12 search spent the full 50489 evaluations on p=4, 5, and 6 and did not lower those energies. The new p=6 energy is -8.056899807721328, higher than the stored -8.056899807721482 by about 1.5e-13. N=6’s extra 12000 evaluations reached error 0.14743, which is worse than the stored p=6 error 0.137624. The equal-depth scoreboard is therefore the one above. All 48 stored procedures still win equal depth on both metrics. The worst N=9 L=6 margin is 0.01442253 in energy and 0.11003977 in fidelity.

Parameter-matched depth is a different ledger. At N=9 every stored procedure loses both metrics at every L, including L=6 against p=11 (error 0.170333, fidelity 0.706696) and p=12 (error 0.152058, fidelity 0.730618). Resuming the best stored shared-gamma state, error 0.232740, for 140 further evaluations moved it to 0.232689 and the gradient fell to 1.2e-5. Six fresh second-branch chains reached 0.2333 and 0.2327 twice, and did not go below that. Local restarts of the 48 six-start states improved 7 of them; the best of those is error 0.233389. None of these reaches the p=11 energy. This is one unfinished question about the ansatz family, not a proof that no shared-gamma angles can do so.

At N=6, 47 of 48 beat p=11 on both metrics and 5 of 48 beat p=12. At N=12, L=5 and L=6 beat the matched, next, and p=12 states on both metrics; L=1 through L=4 do not. The six-start recipe remains the equal-depth recommendation. N=15 and N=18 were not run.
