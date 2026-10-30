# Freedom ladder on projector-Uc (BinH)

_Generated 2026-10-30 19:44:29 +0800 (Asia/Shanghai)_

Champ baseline train{6,9,12}: n15 **2.754**/10.244, n18 **4.194**/24.302.

## Two-cavity projector factors

Span-2 clauses factor as \(C=C_a\otimes C_b\) with local \((k_a,k_b)\in\{(1,3),(2,2)\}\). Still in the finite GRAPE library (tensor of local projectors).

| n | two-cav frac | (1,3) | (2,2) | span3 | span4 |
|---|--------------|-------|-------|-------|-------|
| 9 | 0.359 | 0.146 | 0.213 | 0.641 | 0.000 |
| 12 | 0.184 | 0.074 | 0.110 | 0.649 | 0.167 |
| 15 | 0.110 | 0.044 | 0.065 | 0.592 | 0.298 |
| 18 | 0.073 | 0.030 | 0.043 | 0.523 | 0.404 |

Tensor check kron projector: err=0.0 pass=True

## Freedom arms scale1000

| arm | n9 | n12 | n15 | n18 | keep |
|-----|----|-----|-----|-----|------|
| span_w | 1.389/3.290 | 1.855/6.046 | 2.695/10.064 | 4.120/24.166 | False |
| eps_delta | 1.388/3.317 | 1.848/6.117 | 2.697/10.261 | 4.123/23.955 | False |

## vs champ

- **span_w**: keep=False — n15 2.695 < 2.699; n18 4.120 >= 4.110
- **eps_delta**: keep=False — n15 2.697 < 2.699; n18 4.123 >= 4.110

## Verdict

DISCARD freedom arms — keep train{6,9,12} projector champ.

