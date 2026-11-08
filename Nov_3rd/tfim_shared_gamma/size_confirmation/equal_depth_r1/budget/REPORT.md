# Same-budget second branch

Ceiling 946 evaluations per procedure, the mean of the earlier six-start run. The six L1 starts are executed once and counted in both procedures. Tau is 1e-9. The QAOA bars are the current equal-depth states and were not improved by the last rescue at N=9.

Executed optimizer evaluations, with L1 not double-counted: 77022.

## control

| L | n | BOTH | Wilson | median | worst | best | worst F |
|---|---|---|---|---|---|---|---|
| 1 | 48 | 48 | [0.926, 1.000] | 0.7551 | 0.7551 | 0.7551 | 0.3074 |
| 2 | 48 | 48 | [0.926, 1.000] | 0.6940 | 0.6940 | 0.6940 | 0.3498 |
| 3 | 48 | 48 | [0.926, 1.000] | 0.4547 | 0.4547 | 0.4547 | 0.5077 |
| 4 | 48 | 48 | [0.926, 1.000] | 0.4033 | 0.4033 | 0.4033 | 0.5469 |
| 5 | 48 | 48 | [0.926, 1.000] | 0.3648 | 0.3648 | 0.3648 | 0.5889 |
| 6 | 37 | 37 | [0.906, 1.000] | 0.3223 | 0.3223 | 0.3223 | 0.6239 |

## treatment

| L | n | BOTH | Wilson | median | worst | best | worst F |
|---|---|---|---|---|---|---|---|
| 1 | 48 | 48 | [0.926, 1.000] | 0.7551 | 0.7551 | 0.7551 | 0.3074 |
| 2 | 48 | 48 | [0.926, 1.000] | 0.6940 | 0.6940 | 0.6940 | 0.3499 |
| 3 | 48 | 48 | [0.926, 1.000] | 0.4547 | 0.4547 | 0.4547 | 0.5077 |
| 4 | 48 | 48 | [0.926, 1.000] | 0.4034 | 0.4035 | 0.4033 | 0.5459 |
| 5 | 48 | 48 | [0.926, 1.000] | 0.3654 | 0.3670 | 0.2736 | 0.5806 |
| 6 | 6 | 6 | [0.610, 1.000] | 0.3223 | 0.3223 | 0.3223 | 0.6239 |

Raw L1 starts BOTH 185/288, Wilson [0.5854638883206927, 0.695510448442812]. These 288 starts are not 288 procedures.
Paired L6 error, treatment minus control, median 2.616573624436569e-12 over 5 pairs. Treatment lower on 0 pairs.
Treatment selected the second branch on 5 procedures.
L6 error below 0.28: control 0, treatment 0, Wilson [0.0, 0.0741026511527422].
Replay of the treatment best and worst: [{'logged_E': -6.014828494816717, 'replay_E': -6.014828494816717, 'delta': 0.0, 'hash_match': True, 'F': 0.6239299937616728, 'grad_norm': 4.666240908920045e-05, 'energy_error': 0.32225993866947533}, {'logged_E': -6.0148284947336155, 'replay_E': -6.0148284947336155, 'delta': 0.0, 'hash_match': True, 'F': 0.6239307258182561, 'grad_norm': 2.54056621436249e-05, 'energy_error': 0.32225993875257686}].

The treatment does not replace the six-start recipe. It finished L6 on 6 of 48 procedures, the control on 37 of 48. Both completed sets sit at error 0.32225994. The paired difference on the five procedures where both finished is 2.6e-12. No completed treatment state has error below 0.28.

The second branch was present in 44 procedures and was lower at L5 in 5. Two of those, chains 7 and 11, reached L5 errors 0.27355398 and 0.2758 from an L1 error of 1.54014. A replay of chain 7 matches the logged energy and hash. Its gradient norm is 0.325, so that L5 state is not stationary, and its fidelity is 0.604, below the reliable basin’s 0.624. The ceiling then stopped L6. The other 39 second branches were still worse at L5; their median error is 0.467. The branches are not separable by energy before L5, and paying for that comparison spends the L6 budget.

Eight N=12 chains used the same caps, 12842 executed evaluations. The second branch was worse at L5 in every chain that had one, with errors near 1.09 against the greedy branch near 0.228. None switched. L6 was censored. This does not show the N=9 crossover.

Discovery outside this held-out, the direct L1-to-L5 probe, was 648 evaluations. Keep the six-start recipe. A same-budget rule cannot both detect the later basin and finish it. The earlier two-basin run, at about twice the cost, remains the only setting in which 11 of 24 chains reached error near 0.233, and its median did not move. N=15 and N=18 were not run.
