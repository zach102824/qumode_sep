# N3INIT v7 N=15 L=2,3

Exact v4 four-start recipes, not retuned from N=18. Development candidate only. n=8, seeds 10018–10025.
QAOA start 0 is the derived center `(γ, β) = (0.25, 0.25)`. Energy selects. Fidelity is post hoc. B* reused from v4.
Not LE S1 and not local R10.

New quantum queries 122880. Historical 1826816. Total research queries 1949696. Failures 0.

| L | q | Q | n | mean (E−E0)/N Nov3 | QAOA | mean F Nov3 | QAOA | mean ΔE | CI | median ΔE | min | max | wins | start0 Nov3 | start0 QAOA | flag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 512 | 1024 | 8 | 0.0396 | 0.0483 | 0.648 | 0.577 | -0.1295 | [-0.205, -0.049] | -0.1669 | -0.2591 | +0.0624 | 6/8 | 6/8 | 0/8 | `nov3_lower_ci_excludes_0` |
| 2 | 1024 | 2048 | 8 | 0.0377 | 0.0440 | 0.662 | 0.603 | -0.0949 | [-0.173, -0.011] | -0.1120 | -0.2577 | +0.0860 | 6/8 | 5/8 | 0/8 | `nov3_lower_ci_excludes_0` |
| 3 | 512 | 1536 | 8 | 0.0357 | 0.0357 | 0.684 | 0.657 | +0.0002 | [-0.084, +0.081] | +0.0072 | -0.1776 | +0.1991 | 4/8 | 5/8 | 0/8 | `positive_point_ci_includes_0` |
| 3 | 1024 | 3072 | 8 | 0.0336 | 0.0323 | 0.699 | 0.686 | +0.0193 | [-0.045, +0.075] | +0.0262 | -0.1719 | +0.1512 | 3/8 | 5/8 | 0/8 | `positive_point_ci_includes_0` |

## Read flags

- L=2 q=512: `nov3_lower_ci_excludes_0`.
- L=2 q=1024: `nov3_lower_ci_excludes_0`.
- L=3 q=512: `positive_point_ci_includes_0`.
- L=3 q=1024: `positive_point_ci_includes_0`.

Do not attribute an energy edge to the coherent center unless start 0 is actually selected.
Fit cell time 942.9s. Workers 4, BLAS 1. Each completed ledger equals its Q.
