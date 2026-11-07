# Round 6 CD — fresh stride-4 chains

The generator and stride-4 packing are unchanged. Manifest `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`. Packing `0f4330f4a9d4e84db3582beb8b607a2e1092918f7a28b68bb3cff4a1a514c979`.

The L5 fidelity was recovered from the stored round-5 angles, not filled in. L5 energy error 0.2280466201928775, F 0.8898142677626181, variance 0.9729342105760708, state hash `d31ed7766b29e130be4651460973221ff2b40b4c2c29d934fda38e22cca81974`.

L6 reference, the lowest stored energy: error 0.167519957175136, F 0.9023701795284904, variance 0.6571004848726716, hash `88c32377f3f2aa5dff7b0548825a2232f8a9cc1b37c928c83cb8c9b92781cbee`. The arm-0 L6 vector is the same ray. The arm-1 L6 vector matches the energy to 1e-12 and overlaps this ray at 0.926, so it is counted separately.

| arm | L | finished | censored | best E−E0 | F | var | median | IQR | within τ | within 1e-3 | both vs p12 | Wilson 95% | overlap ref | conjugate only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh | 1 | 48 | 0 | 1.1230 | 0.1753 | 1.717 | 1.1230 | 1.1119 | 0 | 0 | 0 | [0.000, 0.074] | 0 | 0 |
| fresh | 2 | 48 | 0 | 1.0449 | 0.1914 | 1.648 | 1.0449 | 0.8805 | 0 | 0 | 0 | [0.000, 0.074] | 0 | 0 |
| fresh | 3 | 48 | 0 | 0.6092 | 0.4280 | 1.142 | 0.6092 | 0.8549 | 0 | 0 | 0 | [0.000, 0.074] | 0 | 0 |
| fresh | 4 | 48 | 0 | 0.5077 | 0.6423 | 1.533 | 0.5077 | 0.7211 | 0 | 0 | 0 | [0.000, 0.074] | 0 | 0 |
| fresh | 5 | 48 | 0 | 0.2280 | 0.8898 | 0.973 | 0.2280 | 0.8511 | 0 | 0 | 29 | [0.463, 0.730] | 0 | 0 |
| fresh | 6 | 48 | 0 | 0.1675 | 0.9024 | 0.657 | 0.1675 | 0.8105 | 1 | 28 | 32 | [0.525, 0.783] | 12 | 17 |
| descend | 3 | 24 | 0 | 0.6092 | 0.4280 | 1.142 | 0.8576 | 0.0000 | 0 | 0 | 0 | [0.000, 0.138] | 0 | 0 |
| descend | 4 | 24 | 0 | 0.5077 | 0.6423 | 1.533 | 0.5077 | 0.0000 | 0 | 0 | 0 | [0.000, 0.138] | 0 | 0 |
| descend | 5 | 24 | 0 | 0.2280 | 0.8898 | 0.973 | 0.2280 | 0.0000 | 0 | 0 | 23 | [0.798, 0.993] | 0 | 0 |
| descend | 6 | 24 | 0 | 0.1675 | 0.9024 | 0.657 | 0.1675 | 0.0000 | 0 | 20 | 20 | [0.641, 0.933] | 20 | 0 |

New evaluations 70477. Fresh marginal 48488. Descending marginal 22373. Descending source cost, charged once, 53927 (round-5 arm-2 family inclusive, which already contains that L6 cell). Family inclusive 124788. Stored QAOA p=12 cumulative cost 89272.

A start is within τ = 1e-9 of the reference only if the energies match at that tolerance. The 1e-3 column is the wider finite-budget neighborhood. Both-metric counts use the frozen p=12 state, error 0.28779490915225203 and fidelity 0.5147185356364429. Conjugate overlaps above 0.99 are not merged into the reference basin.

Each fresh chain starts from its own random L1 point and then carries that chain's energy-selected schedule upward. The L6 median, 0.16752, sits on the reference energy. The 25th percentile is 0.16752 and the 75th is 0.978, so 16 of 48 chains are above error 0.3. One chain is within τ = 1e-9 of the stored reference energy. Twelve overlap the reference ray and seventeen match only its complex conjugate; those seventeen stay in a separate count. Thirty-two of 48 beat the frozen QAOA p=12 state on both metrics. The Wilson interval on that fraction is [0.525, 0.783].

The descending L6 round trip, which is source-conditioned, has median 0.16752, 75th percentile 0.16753, and 4 of 24 starts above error 0.3. Twenty of 24 overlap the reference ray. None of those are conjugate-only. This is not a source-free discovery, and it is not a seal: no new QAOA rescue was run in this slot, and an external replay is still required. The tail of the fresh chains shows that landing on the schedule is common under this continuation and not uniform.

Partial: False. Wall 188.2 s.
