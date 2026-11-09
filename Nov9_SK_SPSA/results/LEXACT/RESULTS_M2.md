# LEXACT M2 measured throughput

Frozen axis: intra-eval parallel in-place FWHT + cis phase, **T=4**. BLAS=1, one process for latency. Numpy ping-pong SIMD FWHT was slower than M1 at N18 and was not adopted. Pid 10878, wall 4.17 s, compile/cache M1 0.111 s / M2 0.013 s (M2 cache hit after tests). RSS 176 → 273 MB. Threads inside kernel: 4. Processes: 1.

524 L1–6 checks on N6/9/12 passed. Nov3 state vs M1 is 0 on those points. Seeds 10010 / 10032 reused. N3COND 772224 unchanged. New queries 768, all inside Q=128.

## Steady energy eval

| N | L | circuit | t_ref (ms) | t_M1 (ms) | t_M2 (ms) | vs M1 | vs ref |
|---|---|---|---|---|---|---|---|
| 6 | 1 | Nov3 | 0.07 | 0.01 | 0.09 | 0.12 | 0.78 |
| 6 | 1 | QAOA | 0.04 | 0.00 | 0.04 | 0.07 | 1.00 |
| 6 | 6 | Nov3 | 0.24 | 0.03 | 0.31 | 0.11 | 0.77 |
| 9 | 1 | Nov3 | 0.13 | 0.03 | 0.13 | 0.24 | 1.00 |
| 9 | 6 | Nov3 | 0.49 | 0.11 | 0.45 | 0.25 | 1.09 |
| 12 | 1 | Nov3 | 0.81 | 0.36 | 0.30 | 1.22 | 2.70 |
| 12 | 6 | Nov3 | 2.54 | 0.87 | 0.99 | 0.88 | 2.57 |
| 12 | 6 | QAOA | 1.63 | 0.69 | 0.52 | 1.33 | 3.13 |
| 15 | 1 | Nov3 | 5.35 | 2.23 | 0.91 | **2.45** | 5.88 |
| 15 | 1 | QAOA | 2.50 | 1.27 | 0.51 | **2.49** | 4.90 |
| 15 | 3 | Nov3 | 10.35 | 4.35 | 1.93 | **2.25** | 5.36 |
| 15 | 3 | QAOA | 6.40 | 3.51 | 1.16 | **3.04** | 5.52 |
| 18 | 1 | Nov3 | 87.97 | 25.83 | **7.30** | **3.54** | 12.05 |
| 18 | 1 | QAOA | 42.97 | 12.74 | **3.81** | **3.34** | 11.28 |

N18 field prep reused (0.00 s this run). M1 N18 B* prep remains 49 s classical, not Q. Observed |Δstate|∞ ≤ 1e-16, |ΔE| ≤ 1.3e-14 on these evals.

## Short inclusive SPSA Q=128

All six traces: same start, same candidate, max |ΔE| ≤ 1.1e-13.

| cell | vs ref wall | max \|ΔE\| |
|---|---|---|
| N9 L3 Nov3 / QAOA | 1.08× / 1.68× | 1.1e-14 / 3.1e-14 |
| N12 L1 Nov3 / QAOA | 2.23× / 2.51× | 2.8e-14 / 1.1e-13 |
| N15 L1 Nov3 / QAOA | 5.43× / 5.05× | 2.1e-14 / 4.8e-14 |

GR frozen recipe was not patched. The ~2200 h figure is a fixed-field probe, not a B* completion estimate.
