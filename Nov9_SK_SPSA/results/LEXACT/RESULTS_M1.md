# LEXACT M1 measured throughput

Candidate exact kernel vs the N3COND/LE reference. float64/complex128. Original Nov3 and QAOA p=L. Pid 10327, wall 57.4 s, compile/cache 0.113 s. Threads=1. Prior N3COND ledger 772224 is unchanged. New LEXACT queries: 768, all inside Q=128.

Seeds: N6/9/12 use 10010 (stored B* reused where present). N15/18 use exclusive 10032. Collision check on 10032–10047: none. GR frozen recipe was not patched.

Tolerances observed: |Δstate|_∞ ≤ 1e-16, |ΔE| ≤ 6e-14 on random evals, SPSA trace max |ΔE| ≤ 1.0e-13. F post hoc, not a selector.

## Steady energy eval (after warmup)

| N | L | circuit | t_ref (ms) | t_fast (ms) | speedup | field prep (s) |
|---|---|---|---|---|---|---|
| 6 | 1 | Nov3 | 0.063 | 0.011 | 5.97 | 0.028 |
| 6 | 1 | QAOA | 0.039 | 0.003 | 13.71 | 0.028 |
| 6 | 6 | Nov3 | 0.234 | 0.032 | 7.27 | reuse |
| 6 | 6 | QAOA | 0.190 | 0.011 | 16.91 | reuse |
| 9 | 1 | Nov3 | 0.127 | 0.034 | 3.75 | reuse stored |
| 9 | 1 | QAOA | 0.085 | 0.015 | 5.64 | reuse stored |
| 9 | 6 | Nov3 | 0.494 | 0.110 | 4.50 | reuse stored |
| 9 | 6 | QAOA | 0.378 | 0.071 | 5.33 | reuse stored |
| 12 | 1 | Nov3 | 0.629 | 0.236 | 2.66 | reuse stored |
| 12 | 1 | QAOA | 0.337 | 0.125 | 2.71 | reuse stored |
| 12 | 6 | Nov3 | 2.354 | 0.824 | 2.86 | reuse stored |
| 12 | 6 | QAOA | 1.619 | 0.671 | 2.41 | reuse stored |
| 15 | 1 | Nov3 | 4.844 | 2.135 | 2.27 | 4.27 |
| 15 | 1 | QAOA | 2.561 | 1.343 | 1.91 | 4.27 |
| 15 | 3 | Nov3 | 10.312 | 4.265 | 2.42 | reuse |
| 15 | 3 | QAOA | 6.306 | 3.448 | 1.83 | reuse |
| 18 | 1 | Nov3 | 82.589 | 22.930 | 3.60 | 49.12 |
| 18 | 1 | QAOA | 42.178 | 12.077 | 3.49 | 49.12 |

RSS 174 MB → 262 MB. Concurrency: 1 process, BLAS=1. N18 B* is classical Lanczos, not Q.

## Short inclusive SPSA (Q=128, 4 starts, energy-only)

| cell | max \|ΔE\| on trace | E_sel | SPSA speedup | F post hoc |
|---|---|---|---|---|
| N9 L3 Nov3 s10010 | 7.1e-15 | −6.943498 | 3.52 | 0.828 |
| N9 L3 QAOA s10010 | 3.8e-14 | −7.165791 | 4.51 | 0.776 |
| N12 L1 Nov3 s10010 | 2.8e-14 | −11.230479 | 2.53 | 0.656 |
| N12 L1 QAOA s10010 | 3.9e-14 | −11.056715 | 2.38 | 0.559 |
| N15 L1 Nov3 s10032 | 7.0e-14 | −14.067065 | 2.15 | 0.546 |
| N15 L1 QAOA s10032 | 1.0e-13 | −13.749905 | 1.94 | 0.460 |

Same start and candidate on every pair. This is not a 1000-seed panel.

GR full-grid ~2200 h on 3 workers is a rough fixed-field probe. It is not a B* completion estimate and was not reproduced here.
