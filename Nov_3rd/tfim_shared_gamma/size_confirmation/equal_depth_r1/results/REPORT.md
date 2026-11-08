# Equal-depth L=1 screen, held out

Recipe frozen in `PREREG.json` before these chains. Six energy-only L=1 starts, then one continuation. Fidelity was not used to choose a start. No angle was copied from N=6, N=9, or N=12.

## N=9

E0 = -6.337088433486. Candidate optimizer evaluations 45388.

| L | procedures | BOTH | Wilson | worseE/betterF | worse both | median error | bar error | bar |
|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.7551 | 1.2215 | stored |
| 2 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.6940 | 0.7915 | stored |
| 3 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.4547 | 0.5986 | stored |
| 4 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.4033 | 0.4803 | stored |
| 5 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.3648 | 0.3865 | stored |
| 6 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.3223 | 0.3367 | stored |

Raw L=1 starts, before the screen keeps one: 182 / 288 BOTH. Wilson [0.5748480817987099, 0.6855671571528309].
Replay of the stored L=6 best: {'logged_E': -6.014828494829463, 'replay_E': -6.014828494829463, 'delta': 0.0, 'hash_match': True}.

## N=6

E0 = -4.117124992097. Candidate optimizer evaluations 65103.

| L | procedures | BOTH | Wilson | worseE/betterF | worse both | median error | bar error | bar |
|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.2546 | 0.7115 | stored |
| 2 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.0726 | 0.4589 | stored |
| 3 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.0633 | 0.3264 | stored |
| 4 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.0307 | 0.2609 | stored |
| 5 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.0189 | 0.1970 | stored |
| 6 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.0130 | 0.1510 | stored |

Raw L=1 starts, before the screen keeps one: 266 / 288 BOTH. Wilson [0.8870541553101507, 0.9490158223763752].
Replay of the stored L=6 best: {'logged_E': -4.111666630645274, 'replay_E': -4.111666630645274, 'delta': 0.0, 'hash_match': True}.

## N=12

E0 = -8.613202458935. Candidate optimizer evaluations 50489.
Fresh equal-depth QAOA evaluations 50524. Shortfall against the candidate ledger: 0.

| L | procedures | BOTH | Wilson | worseE/betterF | worse both | median error | bar error | bar |
|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 1.1230 | 1.7943 | stored |
| 2 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 1.0449 | 1.1816 | stored |
| 3 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.6092 | 0.9281 | stored |
| 4 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.5077 | 0.7600 | this_slice |
| 5 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.2280 | 0.6363 | stored |
| 6 | 48 | 48 | [0.926, 1.000] | 0 | 0 | 0.1675 | 0.5563 | stored |

Raw L=1 starts, before the screen keeps one: 189 / 288 BOTH. Wilson [0.5996617124950765, 0.7087247546843798].
Replay of the stored L=6 best: {'logged_E': -8.445682360983511, 'replay_E': -8.445682360983511, 'delta': 0.0, 'hash_match': True}.

The N=9 energies have no spread: every procedure sits at error 0.3223 and fidelity 0.6239. That beats the stored equal-depth p=6 state, error 0.3367 and fidelity 0.5139, by more than the 1e-9 tolerance. It is the common basin from the earlier 48 chains, not the rarer L=6 state at error 0.2327. It does not beat the earlier parameter-matched p=11 or p=12 states. Those comparisons stay as previously reported.

N=12 also has one energy at each depth. Six stored L=6 vectors fall into two rays, overlap about 0.858 across the rays and above 0.999 inside a ray, both at error 0.16752. The fresh p=6 search, 50524 evaluations against a floor of 50489, reproduced the historical energy 0.5563 and did not lower it. L=4 is the only depth where the fresh search is the comparator, and only by about 2e-14 in the energy. Raw L=1 rates stay well below the procedure rate: 182/288 at N=9, 266/288 at N=6, and 189/288 at N=12.

No angles were transferred across sizes. The exploratory probes that chose this screen cost 4743 evaluations and are not in these 48. Eta is not fidelity. A miss would not have proved an expressivity ceiling. N=15 was checkpointed from stdout and not resumed. N=18 was not started.
