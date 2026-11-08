# Round 4 report: development set only

Protocol frozen at `baa35c55` before any fit. This set (k=4..15, two repeats, all 12 instances at each of N=6, 9, 12) was used to decide what to confirm. It is not the confirmation.

## What advanced

Both variants beat the retained KM reference under the frozen rule, on joint wins as well as fidelity. The rule picks **KM_gl** for confirmation, because the sum of its median-fidelity gains over KM is larger. KM_pack also advanced and is not discarded; it is the fallback if KM_gl does not confirm.

At L=6, best of two repeats, against the same QAOA runs:

| N | KM median dF (wins) | KM_pack median dF (wins, joint) | KM_gl median dF (wins, joint) |
|---|---|---|---|
| 6 | +0.116 (9, all joint) | +0.219 (10, all joint) | +0.230 (11, 11 joint) |
| 9 | +0.029 (5, 3 joint) | +0.056 (6, 5 joint) | +0.102 (9, 9 joint) |
| 12 | -0.020 (4, all joint) | +0.072 (6, all joint) | +0.111 (8, all joint) |

KM_gl's confidence interval for the median fidelity difference is above 0 at all three sizes on this set, and every one of its wins is a joint win on energy too. Mean F_sub at L=6 is 0.867, 0.515 and 0.390 against QAOA's 0.621, 0.417 and 0.279.

## What the two changes actually did

- **KM_gl** gives each layer its own coefficient on the problem diagonal and keeps the two feature coefficients shared. The selected schedules use that freedom: the six layer coefficients differ by a median 0.21, 0.16 and 0.10 at N=6, 9 and 12. It has 2+4L parameters (26 at L=6) against KM's 21 and QAOA's 12. One start per depth is the KM solution with every layer coefficient set equal, so its energy is at most KM's. The fidelity gain is on top of that.
- **KM_pack** keeps the original shared-gamma circuit and only changes which triples the shells and the M feature see, to the partition that captures the most absolute coupling. It is the change that repairs the round-3 failure N9 k7: KM and KM_gl both have ground-space weight 0 there, KM_pack has 0.471 and a joint win. It does not repair N6 k12.

## Losses kept

N6 k12 is a fidelity loss for KM_gl (0.030 vs QAOA 0.146) even though its energy is lower. N9 k7 is the same kind of loss for KM_gl (weight 0, energy 1.07 below QAOA). N9 k15 is a tie with both families below 0.11. At N=12, k5, k7 and k8 stay below 0.2 for everyone.

## Checks

Independent replay of all 3456 selected states agrees within 2.2e-12 in relative energy and 1.1e-12 in F_sub. No selected cell sits on a bound; at N=9 and N=12 the largest diagonal coefficient is under 16% of the box. At N=9 and N=12 the two repeats find the same energy in every cell. 65 of 50688 starts hit the 2000 cap, 53 of them KM_pack at N=6. QAOA hit none. The run took 2803 s. k=28..43 were not fit.
