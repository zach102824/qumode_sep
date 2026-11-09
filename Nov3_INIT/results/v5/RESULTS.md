# N3INIT v5 N=18 L1

Exact v4 four-start recipes. Development candidate only. n=8, seeds 10018–10025, not a population.
QAOA start 0 is the derived center `(γ, β) = (0.25, 0.25)`. Energy selects. Fidelity is post hoc.

New quantum queries 24576. Historical 1679360. Total research queries 1703936. Failures 0.
Classical B* wall 900.1s, CPU 898.8s, peak RSS 178928 KiB. Not in the quantum ledger.

| q | Q | n | mean E Nov3 | mean E QAOA | mean (E−E0)/N Nov3 | mean (E−E0)/N QAOA | mean F Nov3 | mean F QAOA | mean ΔE | CI | median ΔE | min | max | wins | start0 Nov3 | start0 QAOA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | 512 | 8 | -17.4475 | -17.2383 | 0.0335 | 0.0451 | 0.636 | 0.545 | -0.2092 | [-0.262, -0.162] | -0.2007 | -0.3507 | -0.1062 | 8/8 | 3/8 | 0/8 |
| 1024 | 1024 | 8 | -17.5046 | -17.2383 | 0.0303 | 0.0451 | 0.657 | 0.545 | -0.2663 | [-0.304, -0.230] | -0.2602 | -0.3595 | -0.1836 | 8/8 | 2/8 | 0/8 |

## Read flags

- q=512: `persists_ci_excludes_0`.
- q=1024: `persists_ci_excludes_0`.
- both budgets persist: `True`.

Do not attribute an energy edge to the coherent center unless start 0 is actually selected.
Fit cell time 1497.0s. Workers 4, BLAS 1. Each ledger equals its Q.
