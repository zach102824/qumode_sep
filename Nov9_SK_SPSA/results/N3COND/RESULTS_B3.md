# N3COND B3 read-dev results

Seeds 10010–10017 are read development, not a fresh holdout. M2−M0. Negative ΔE means M2 is lower. Fidelity is post hoc and is not the gate.

Runs 128, failures 0, queries inside Q 147456, outside-cap queries 0, calibration inside Q 6656, summed worker wall 32.9s.

Energy gate passed: True.

| contrast | ΔE med/mean [CI] n(ΔE>0) | ΔF med/mean [CI] n(ΔF>0) |
|---|---|---|
| N9_L1_q128_Nov3_M2_minus_M0 | +0.0375 / +0.0923 [-0.0889,+0.2882] 4/8 | -0.0017 / -0.0141 [-0.0673,+0.0408] 4/8 |
| N9_L1_q128_QAOA_M2_minus_M0 | -0.0001 / +0.0004 [-0.0002,+0.0016] 3/8 | -0.0006 / -0.0017 [-0.0033,-0.0002] 2/8 |
| N9_L1_q1024_Nov3_M2_minus_M0 | +0.0016 / -0.1674 [-0.4039,+0.0087] 4/8 | +0.0004 / +0.0632 [-0.0028,+0.1510] 4/8 |
| N9_L1_q1024_QAOA_M2_minus_M0 | -0.0002 / -0.0002 [-0.0003,-0.0001] 1/8 | +0.0008 / +0.0007 [+0.0004,+0.0010] 7/8 |
| N9_L3_q128_Nov3_M2_minus_M0 | -0.7998 / -0.8378 [-1.1417,-0.5421] 0/8 | +0.2563 / +0.2080 [+0.1059,+0.3038] 8/8 |
| N9_L3_q128_QAOA_M2_minus_M0 | -0.0776 / -0.0875 [-0.1369,-0.0400] 1/8 | +0.0466 / +0.0497 [+0.0224,+0.0782] 7/8 |
| N9_L3_q1024_Nov3_M2_minus_M0 | -0.3399 / -0.5078 [-0.9809,-0.0827] 2/8 | +0.1592 / +0.1643 [+0.0441,+0.2887] 6/8 |
| N9_L3_q1024_QAOA_M2_minus_M0 | -0.0071 / -0.0283 [-0.0568,-0.0056] 1/8 | +0.0088 / +0.0199 [+0.0059,+0.0370] 8/8 |
