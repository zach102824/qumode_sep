# N3INIT v6 N=18 L=2,3

Exact v4 four-start recipes, not retuned from N=18 L=1. Development candidate only. n=8, seeds 10018–10025.
QAOA start 0 is the derived center `(γ, β) = (0.25, 0.25)`. Energy selects. Fidelity is post hoc. B* reused from v5.

New quantum queries 122880. Historical 1703936. Total research queries 1826816. Failures 0.

| L | q | Q | n | mean (E−E0)/N Nov3 | QAOA | mean F Nov3 | QAOA | mean ΔE | CI | median ΔE | min | max | wins | start0 Nov3 | start0 QAOA | flag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 512 | 1024 | 8 | 0.0303 | 0.0350 | 0.656 | 0.613 | -0.0856 | [-0.164, -0.003] | -0.1156 | -0.2856 | +0.1210 | 6/8 | 5/8 | 2/8 | `nov3_lower_ci_excludes_0` |
| 2 | 1024 | 2048 | 8 | 0.0289 | 0.0334 | 0.664 | 0.624 | -0.0803 | [-0.159, -0.008] | -0.0663 | -0.2930 | +0.0800 | 6/8 | 6/8 | 2/8 | `nov3_lower_ci_excludes_0` |
| 3 | 512 | 1536 | 8 | 0.0298 | 0.0265 | 0.696 | 0.680 | +0.0597 | [-0.026, +0.153] | +0.0360 | -0.0857 | +0.3051 | 4/8 | 5/8 | 0/8 | `positive_point_ci_includes_0` |
| 3 | 1024 | 3072 | 8 | 0.0275 | 0.0251 | 0.711 | 0.693 | +0.0433 | [-0.030, +0.119] | +0.0290 | -0.1065 | +0.2181 | 4/8 | 4/8 | 1/8 | `positive_point_ci_includes_0` |

## Read flags

- L=2 q=512: `nov3_lower_ci_excludes_0`.
- L=2 q=1024: `nov3_lower_ci_excludes_0`.
- L=3 q=512: `positive_point_ci_includes_0`.
- L=3 q=1024: `positive_point_ci_includes_0`.

Do not attribute an energy edge to the coherent center unless start 0 is actually selected.
Fit cell time 15788.3s. Workers 4, BLAS 1. Each completed ledger equals its Q.
