# L1 stopping rule, fresh paired test

One shared gamma. The rule stops after two L1 starts that do not beat the best by more than 1e-4, and never draws more than six. Fixed six is the control on the same seeds. The denominator for BOTH is 48, including censored chains.

## N=9

Executed union 61968. Same L1 choice on 25 chains. Cuts [3, 3, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3]...

| arm | L6 BOTH | joint all L | L6 completed | cost median | cost worst | cost total |
|---|---|---|---|---|---|---|
| rule | 48/48 | 47/48 | 48/48 | 830 | 1124 | 40385 |
| fixed6 | 48/48 | 48/48 | 48/48 | 954 | 1051 | 45832 |

| arm | L | BOTH | Wilson | median | worst |
|---|---|---|---|---|---|
| rule | 1 | 47/48 | [0.891, 0.996] | 0.7551 | 1.5401 |
| rule | 2 | 47/48 | [0.891, 0.996] | 0.6940 | 1.4182 |
| rule | 3 | 47/48 | [0.891, 0.996] | 0.4547 | 0.9199 |
| rule | 4 | 47/48 | [0.891, 0.996] | 0.4033 | 0.7474 |
| rule | 5 | 47/48 | [0.891, 0.996] | 0.3648 | 0.4638 |
| rule | 6 | 48/48 | [0.926, 1.000] | 0.3223 | 0.3223 |
| fixed6 | 1 | 48/48 | [0.926, 1.000] | 0.7551 | 0.7551 |
| fixed6 | 2 | 48/48 | [0.926, 1.000] | 0.6940 | 0.6940 |
| fixed6 | 3 | 48/48 | [0.926, 1.000] | 0.4547 | 0.4547 |
| fixed6 | 4 | 48/48 | [0.926, 1.000] | 0.4033 | 0.4033 |
| fixed6 | 5 | 48/48 | [0.926, 1.000] | 0.3648 | 0.3648 |
| fixed6 | 6 | 48/48 | [0.926, 1.000] | 0.3223 | 0.3223 |

Secondary, L6 error below 0.28: rule 1, fixed6 0.

## N=6

Executed union 87970. Same L1 choice on 27 chains. Cuts [3, 4, 3, 3, 4, 5, 5, 4, 3, 3, 3, 5]...

| arm | L6 BOTH | joint all L | L6 completed | cost median | cost worst | cost total |
|---|---|---|---|---|---|---|
| rule | 46/48 | 46/48 | 48/48 | 1248 | 1359 | 60166 |
| fixed6 | 47/48 | 47/48 | 48/48 | 1351 | 1433 | 65034 |

| arm | L | BOTH | Wilson | median | worst |
|---|---|---|---|---|---|
| rule | 1 | 48/48 | [0.926, 1.000] | 0.2546 | 0.2729 |
| rule | 2 | 48/48 | [0.926, 1.000] | 0.0726 | 0.0726 |
| rule | 3 | 48/48 | [0.926, 1.000] | 0.0633 | 0.0634 |
| rule | 4 | 48/48 | [0.926, 1.000] | 0.0307 | 0.0310 |
| rule | 5 | 48/48 | [0.926, 1.000] | 0.0187 | 0.0199 |
| rule | 6 | 46/48 | [0.860, 0.988] | 0.0129 | 0.3445 |
| fixed6 | 1 | 48/48 | [0.926, 1.000] | 0.2546 | 0.2729 |
| fixed6 | 2 | 48/48 | [0.926, 1.000] | 0.0726 | 0.0726 |
| fixed6 | 3 | 48/48 | [0.926, 1.000] | 0.0633 | 0.0634 |
| fixed6 | 4 | 48/48 | [0.926, 1.000] | 0.0307 | 0.0309 |
| fixed6 | 5 | 48/48 | [0.926, 1.000] | 0.0189 | 0.0214 |
| fixed6 | 6 | 47/48 | [0.891, 0.996] | 0.0132 | 0.2001 |

## N=12

Executed union 63911. Same L1 choice on 31 chains. Cuts [3, 3, 4, 3, 3, 3, 4, 4, 5, 5, 3, 3]...

| arm | L6 BOTH | joint all L | L6 completed | cost median | cost worst | cost total |
|---|---|---|---|---|---|---|
| rule | 47/48 | 47/48 | 48/48 | 924 | 1168 | 45078 |
| fixed6 | 48/48 | 48/48 | 48/48 | 1049 | 1140 | 50520 |

| arm | L | BOTH | Wilson | median | worst |
|---|---|---|---|---|---|
| rule | 1 | 47/48 | [0.891, 0.996] | 1.1230 | 2.2239 |
| rule | 2 | 47/48 | [0.891, 0.996] | 1.0449 | 1.9254 |
| rule | 3 | 47/48 | [0.891, 0.996] | 0.6092 | 1.4703 |
| rule | 4 | 47/48 | [0.891, 0.996] | 0.5077 | 1.2289 |
| rule | 5 | 47/48 | [0.891, 0.996] | 0.2280 | 1.0812 |
| rule | 6 | 47/48 | [0.891, 0.996] | 0.1675 | 0.9924 |
| fixed6 | 1 | 48/48 | [0.926, 1.000] | 1.1230 | 1.1230 |
| fixed6 | 2 | 48/48 | [0.926, 1.000] | 1.0449 | 1.0449 |
| fixed6 | 3 | 48/48 | [0.926, 1.000] | 0.6092 | 0.6092 |
| fixed6 | 4 | 48/48 | [0.926, 1.000] | 0.5077 | 0.5077 |
| fixed6 | 5 | 48/48 | [0.926, 1.000] | 0.2280 | 0.2280 |
| fixed6 | 6 | 48/48 | [0.926, 1.000] | 0.1675 | 0.1675 |

Replays: {'9': {'rule_best': {'logged_E': -6.102950835191637, 'dense_E': -6.102950835191637, 'delta': 0.0, 'hash_match': True, 'grad': 0.11854049563240397, 'error': 0.23413759829455572, 'F': 0.6312191598174617}, 'rule_worst': {'logged_E': -6.014828494599759, 'dense_E': -6.014828494599759, 'delta': 0.0, 'hash_match': True, 'grad': 2.595013939638194e-05, 'error': 0.32225993888643334, 'F': 0.6239295256542975}}, '6': {'rule_best': {'logged_E': -4.111451570692977, 'dense_E': -4.111451570692976, 'delta': 8.881784197001252e-16, 'hash_match': True, 'grad': 0.03923216973174797, 'error': 0.0056734214035820685, 'F': 0.9977932054493647}, 'rule_worst': {'logged_E': -3.7726161750568536, 'dense_E': -3.772616175056854, 'delta': -4.440892098500626e-16, 'hash_match': True, 'grad': 2.0042833812484165e-05, 'error': 0.3445088170397055, 'F': 0.7914353283740997}}, '12': {'rule_best': {'logged_E': -8.445682442348817, 'dense_E': -8.445682442348819, 'delta': -1.7763568394002505e-15, 'hash_match': True, 'grad': 0.0006677723950892109, 'error': 0.16752001658648297, 'F': 0.9023677720079171}, 'rule_worst': {'logged_E': -7.620785125521072, 'dense_E': -7.620785125521072, 'delta': 0.0, 'hash_match': True, 'grad': 0.20904608572221411, 'error': 0.9924173334142274, 'F': 0.24434172973140414}}}.

Per-layer gamma is not part of this test. N=15 and N=18 were not run.

The frozen adoption test fails. The stopping rule's unconditional L6 count is 48, 46, and 47 at N=9, 6, and 12. Fixed six is 48, 47, and 48. The rule is short one success at N=6 and at N=12. Its median cost is lower at every size, about 830 versus 954 at N=9, 1248 versus 1351 at N=6, and 924 versus 1049 at N=12. Every chain finished L6. The ceiling of 2000 did not censor anyone. Retain fixed six.

The rule is the same at every size. It does not use a size-specific energy cut. On this draw it still let one N=9 chain keep the L1 energy 1.540, which reached the secondary deep state at error 0.234 and passed L6, but missed L1 through L5. Fixed six did not find that state. One N=6 chain under the rule ends at error 0.3445, and one N=12 chain at 0.992. Those are the late failures the shorter screen allowed.

Practical limit: six L1 starts, this continuation, and a per-procedure ceiling of 2000. On these 48 fresh chains the unconditional L6 rates are 48/48, 47/48, and 48/48. That is not a guarantee for the next 48.
