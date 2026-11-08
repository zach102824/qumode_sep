# Equal-depth QAOA rescue

The 48 stored procedures were not retrained. Tau is 1e-9. The bar at each depth is the lower energy of the historical state and this slice, and the fidelity of that same state.

## N=6

New optimizer evaluations 65103 of ceiling 65103. Shortfall 0.

| L | stored BOTH | Wilson | worst error | worst F | bar error | bar F | source | fresh best error |
|---|---|---|---|---|---|---|---|---|
| 1 | 48/48 | [0.926, 1.000] | 0.2729 | 0.7270 | 0.7115 | 0.4527 | historical | 0.7115 |
| 2 | 48/48 | [0.926, 1.000] | 0.0726 | 0.9307 | 0.4589 | 0.5587 | historical | 0.4589 |
| 3 | 48/48 | [0.926, 1.000] | 0.0634 | 0.9593 | 0.3264 | 0.6298 | this_slice | 0.3264 |
| 4 | 48/48 | [0.926, 1.000] | 0.0310 | 0.9827 | 0.2609 | 0.6803 | historical | 0.2609 |
| 5 | 48/48 | [0.926, 1.000] | 0.0200 | 0.9850 | 0.1956 | 0.7393 | this_slice | 0.1956 |
| 6 | 48/48 | [0.926, 1.000] | 0.0597 | 0.9799 | 0.1376 | 0.8124 | this_slice | 0.1376 |

Replay of the deepest new QAOA state: {'p': 6, 'logged_E': -3.9795012707822215, 'replay_E': -3.9795012707822215, 'delta': 0.0, 'hash_match': True, 'F': 0.8124435609387982}.

## N=9

New optimizer evaluations 45388 of ceiling 45388. Shortfall 0.

| L | stored BOTH | Wilson | worst error | worst F | bar error | bar F | source | fresh best error |
|---|---|---|---|---|---|---|---|---|
| 1 | 48/48 | [0.926, 1.000] | 0.7551 | 0.3074 | 1.2215 | 0.2116 | historical | 1.2215 |
| 2 | 48/48 | [0.926, 1.000] | 0.6940 | 0.3498 | 0.7915 | 0.3000 | historical | 0.7915 |
| 3 | 48/48 | [0.926, 1.000] | 0.4547 | 0.5077 | 0.5986 | 0.3566 | historical | 0.5986 |
| 4 | 48/48 | [0.926, 1.000] | 0.4033 | 0.5469 | 0.4803 | 0.4171 | historical | 0.4803 |
| 5 | 48/48 | [0.926, 1.000] | 0.3648 | 0.5889 | 0.3865 | 0.4767 | historical | 0.3865 |
| 6 | 48/48 | [0.926, 1.000] | 0.3223 | 0.6239 | 0.3367 | 0.5139 | historical | 0.3367 |

Replay of the deepest new QAOA state: {'p': 6, 'logged_E': -6.000405967808423, 'replay_E': -6.000405967808423, 'delta': 0.0, 'hash_match': True, 'F': 0.5138887727141089}.

## N=12

New optimizer evaluations 16000 of ceiling 16000. Shortfall 0.

| L | stored BOTH | Wilson | worst error | worst F | bar error | bar F | source | fresh best error |
|---|---|---|---|---|---|---|---|---|
| 1 | 48/48 | [0.926, 1.000] | 1.1230 | 0.1753 | 1.7943 | 0.0809 | historical | 1.7943 |
| 2 | 48/48 | [0.926, 1.000] | 1.0449 | 0.1914 | 1.1816 | 0.1333 | this_slice | 1.1816 |
| 3 | 48/48 | [0.926, 1.000] | 0.6092 | 0.4280 | 0.9281 | 0.1705 | this_slice | 0.9281 |
| 4 | 48/48 | [0.926, 1.000] | 0.5077 | 0.6423 | 0.7600 | 0.2153 | historical | — |
| 5 | 48/48 | [0.926, 1.000] | 0.2280 | 0.8898 | 0.6363 | 0.2584 | historical | — |
| 6 | 48/48 | [0.926, 1.000] | 0.1675 | 0.9023 | 0.5563 | 0.2947 | historical | — |

Replay of the deepest new QAOA state: {'p': 3, 'logged_E': -7.6851276672996365, 'replay_E': -7.6851276672996365, 'delta': 0.0, 'hash_match': True, 'F': 0.17053640871273035}.

## N=9 two-basin continuation

Fresh chains 24. Evaluations 45686 of ceiling 50000. Shortfall 4314.
Beam BOTH against the rescued p=6 bar: 24/24, Wilson [0.8620194241710247, 1.0].
The single-basin arm of the same chains: 24/24.
Second basin lower at L6 in 11 chains. Energy gains: [0.08893578119384493, 0.08889128107101207, 0.08251182751956421, 0.03694959891195104, 0.08438310199776122, 0.088938737226405, 0.08209045261772552, 0.08893936247232315, 0.08884037841700732, 0.08890935137855926, 0.04337797917700925].

N=9’s stored L=6 figure 0.3223/0.6239 is the energy error and the fidelity. Across the 48 procedures the worst error is 0.322259938871 and the worst fidelity is 0.623928579723. The historical p=6 state is error 0.336682465677 and fidelity 0.513888806460. The worst stored chain is lower in energy by 0.014422526806 and higher in fidelity by 0.110039773263. The new search spent 45388 evaluations and its best p=6 error is 0.336682465678, which does not beat that historical state.

N=6 p=6 did move. The new state has error 0.137623721314 and fidelity 0.812443560939, against the historical 0.151002181146 and 0.781859004829. The worst stored procedure is still lower in energy, at 0.059723004038, a margin of 0.077900717276, and higher in fidelity, at 0.979881218541. N=12 p=6 was not searched again. The new p=1 search did not beat the historical energy. The new p=2 and p=3 energies agree with the historical values through 1e-15.

The two-basin continuation is not an equal-cost replacement. It spent 45686 evaluations on 24 chains, about twice the per-chain cost of the six-start recipe. Eleven of the 24 reach an L=6 error near 0.233, from an L=1 energy near 1.540 rather than 0.755. The other 13 stay at 0.3223. Two of the eleven have fidelity 0.594 and 0.602, below the reliable basin’s 0.624 and still above the p=6 fidelity 0.514. The median winner remains 0.3223. The original 48-chain recipe is unchanged.

No search here is a global QAOA optimum. N=15 and N=18 were not run.
