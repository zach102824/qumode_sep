# N3COND B4 — closed

Preregistered before the fits (`PREREG_N3COND_B4.json`, sha `a1801cedfe7e`). One floor, 0.5. The gain is C0's simultaneous-probe gain on the sigma that is stepped. The axial pass sets λ and does not set `a`. Pid 9384, clock 34.4 s, 64/64 ok.

## What was compared

S2 shrinks steep coordinates by at most 2×. S0 spends the same axial queries and throws λ away. Both are fully inside Q=3072. The reference is the already-run C0 at the same Q, with its ordinary profile (8 calibration queries per start, K=379, no axial pass). S0 and S2 have K=367 on Nov3 and K=373 on QAOA, because the axial pass is charged. Seeds 10010–10017 are read development.

Start-0 gains on Nov3 sit next to C0, not 10–50× above it. N9 medians: C0 and S0 both 7.97, S2 14.3. N12: C0 and S0 both 7.40, S2 12.0. S0−C0 energy is a tie (N9 mean −0.0003, CI crosses 0; N12 mean +0.002, CI crosses 0). The sham is the ordinary optimizer with a matched axial bill, not a weakened control.

## Gate

Negative ΔE means S2 is lower. Fidelity is post hoc.

| cell | S2−C0 energy | S2−S0 energy median | gate |
|---|---|---|---|
| N9 L3 Nov3 | median −0.022, mean −0.023, CI [−0.041, −0.007], 7/8 lower | −0.022 | pass |
| N12 L3 Nov3 | median −0.020, mean −0.009, CI [−0.057, +0.042], 6/8 lower | −0.021 | fail |

The branch survives only if both pass. N12's mean interval includes zero, so the branch does not survive. Absolute medians at N12 go the other way from the paired median: C0 −8.012, S2 −8.008. QAOA had the same budget. Its N12 S2−C0 interval also includes zero (mean −0.0004). Post hoc N9 Nov3 median F is 0.846 for S2 and 0.821 for C0. That is not in the gate.

## Close

Scale and gain conditioning stops here. C1 is not adopted. B3 is not adopted. B4 is not adopted. N15 and N18 were not started. The GR benchmark is unchanged.

The blocker for any further claim on this stream is seeds, not another floor. 10010–10017 have been read by C0, C1, B3, and B4. 10018–10025 belong to the init/restart worker. 10026–10031 belong to the local N15 frame pilot. Validation 11000–11031 and final 20000–20999 stay unread. Perturbation directions are that other worker's R7/R8. There is no unread development band left for this stream, and no remaining in-scope SPSA lever here that is not another scale or gain rescue.
