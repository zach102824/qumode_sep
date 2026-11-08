# Historical B=0.1 ferromagnetic triples are partial

Protocol `5175d7f8` before any fit. Running marker `ff1b57d9`. PREREG SHA-256 `7a1799c20c9b5267854fdd356ea4519c5a05510a6dab04f04345cd9015e68e5e`.

This is the historical author field `B = 0.1`. It is not a critical-field result. `B_cri` stays withheld.

## Rule

One partition at every N and every layer: triples that maximize the sum of `-J` on their three edges, with the lexicographically smallest partition on ties. Block means and popcount shells are rebuilt on those triples. `J@J` is unchanged. The circuit is the original shared-gamma KM: one gamma, one thetaK, one thetaM, three shell angles per layer, 21 parameters at L=6. QAOA has 12 parameters at L=6. Both use the round-4 start list. KM_fe does not get extra starts. Interleaved KM is a reference on the same instances and is not the claim.

## Verdict

Overall **partial**. N=6 and N=9 pass the frozen equal-depth rule. N=12 does not.

| family | N | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] |
|---|---|---|---|---|---|
| KM_fe | 6 | 11 (11 / 0) | 0 | 1 | +0.279 [+0.020, +0.430] |
| KM_fe | 9 | 10 (8 / 2) | 0 | 2 | +0.233 [+0.109, +0.382] |
| KM_fe | 12 | 6 (6 / 0) | 1 | 5 | +0.101 [-0.015, +0.222] |

The N=9 fidelity-only wins are k=179 and k=187. The N=12 loss is k=189, joint: fidelity 0.001 against 0.177, and the ferromagnetic energy is 0.077 above QAOA. That selected state is not evaluation-capped.

The same-window interleaved reference also passes N=6 (10 joint, 0 losses, 2 ties, median +0.273 [+0.012, +0.413]) and N=9 (7 joint, 1 loss, 4 ties, median +0.143 [+0.009, +0.308]). It does not pass N=12 (5 joint, 1 loss, 6 ties, median +0.049 [-0.038, +0.171]). The pass bits are not unique to the new packing. The N=6 pass is not a new gain beyond the interleaved search on this window.

## Energy against interleaved KM

Drop is interleaved energy minus ferromagnetic energy. The frozen descriptive rule is a median above 0.01, or at least four instances above 0.01. It is not a second win rule.

| N | median drop | instances above 0.01 |
|---|---|---|
| 6 | +0.0227 | 6 of 12 |
| 9 | +0.1238 | 7 of 12 |
| 12 | +0.0110 | 6 of 12 |

All three N meet that rule. The N=9 median is the clear movement. The N=12 median sits just above 0.01, and several instances go the other way, including k=183 by 0.700. N=9 k=189 is worse by 0.444. The packing changed the energy. It did not confirm an equal-depth advantage at every N.

## Equal parameter count

KM_fe at L=3 against QAOA at L=6 is 12 parameters and is not the verdict. N=6 has 9 wins (8 joint) and 1 loss, median +0.097 [+0.010, +0.289]. N=9 has 4 wins (2 joint) and 6 losses, median -0.068. N=12 has 0 wins and 6 losses, median -0.071.

## Replay and cost

Independent Pauli replay of 432 states at each N passes. Maximum relative energy error is 6.0e-13, 3.9e-13, and 1.6e-13. Subspace-count mismatches are 0.

Optimizer wall 1900.3 s of 4800. Starts 43488. Logged energy evaluations 4294464. L=1 grid evaluations 32256. Capped starts 189. The selected L=6 states at N=9 and N=12 are not capped.

No universal claim. k=190 and above stay unused. k=100 and k=101 stay held out.
