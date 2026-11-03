# Gate-0 blockmix identity — **PASS**

_Asia/Shanghai 2026-11-03 12:20:42 +0800_

Pass=20 Fail=0

| tag | n | b | L | max|Δp| | max|rel| | med_gap | pass |
|-----|---|---|---|---------|----------|---------|------|
| a_c_L1 | 9 | 1 | 1 | 1.94e-07 | 1.24e-06 | 8.87e-07 | PASS |
| a_c_L1 | 9 | 3 | 1 | 1.94e-07 | 1.17e-06 | 8.87e-07 | PASS |
| a_c_L1 | 9 | 9 | 1 | 8.20e-08 | 1.23e-06 | 8.60e-07 | PASS |
| a_c_L1 | 12 | 1 | 1 | 6.33e-08 | 1.51e-06 | 1.24e-06 | PASS |
| a_c_L1 | 12 | 2 | 1 | 7.08e-08 | 1.72e-06 | 1.15e-06 | PASS |
| a_c_L1 | 12 | 3 | 1 | 6.33e-08 | 1.72e-06 | 1.15e-06 | PASS |
| a_c_L1 | 12 | 4 | 1 | 5.59e-08 | 1.53e-06 | 1.34e-06 | PASS |
| a_c_L1 | 12 | 6 | 1 | 6.33e-08 | 1.63e-06 | 1.34e-06 | PASS |
| a_c_L1 | 12 | 12 | 1 | 3.91e-08 | 1.60e-06 | 1.47e-06 | PASS |
| a_c_L2 | 9 | 1 | 2 | 6.26e-07 | 1.94e-06 | 1.23e-06 | PASS |
| a_c_L2 | 9 | 3 | 2 | 5.66e-07 | 1.96e-06 | 1.33e-06 | PASS |
| a_c_L2 | 9 | 9 | 2 | 3.13e-07 | 1.86e-06 | 1.55e-06 | PASS |
| a_c_L2 | 12 | 1 | 2 | 2.98e-07 | 2.39e-06 | 2.10e-06 | PASS |
| a_c_L2 | 12 | 2 | 2 | 3.28e-07 | 2.38e-06 | 2.10e-06 | PASS |
| a_c_L2 | 12 | 3 | 2 | 2.98e-07 | 2.38e-06 | 2.10e-06 | PASS |
| a_c_L2 | 12 | 4 | 2 | 3.28e-07 | 2.43e-06 | 1.96e-06 | PASS |
| a_c_L2 | 12 | 6 | 2 | 3.28e-07 | 2.43e-06 | 2.24e-06 | PASS |
| a_c_L2 | 12 | 12 | 2 | 2.24e-07 | 2.46e-06 | 1.90e-06 | PASS |
| a_L16_smoke | 9 | 3 | 16 | 8.46e-06 | 1.32e-05 | 1.24e-05 | PASS |
| b_b1_is_qaoa | 12 | 1 | 1 | 6.33e-08 | 1.51e-06 | 1.24e-06 | PASS |
| d_fock_shell_on_Fock_Bin_vs_Gray | — | — | — | — | — | — | None |

## Notes

- Checks (a)(b)(c): dual-QAOA θ=-β·pc in bit-frame blockmix vs ideal qubit QAOA.
- Check (d): Fock-indexed shell Bin vs Gray is NOT an identity (expected); logical-bit θ is.
