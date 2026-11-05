# n=6 L=4 Γ=0.5 — fidelity vs LBFGS iteration

Protocol: B_train=64, B_hold=32, 20 LBFGS rounds (max_iter=8 each), seeds [0, 1].

| method | seed | params | final best hold | iters to 95% of final | plateau≈ |
|---|---:|---:|---:|---:|---:|
| free_qaoa | 0 | 8 | 0.8179 | 4 | 6 |
| free_qaoa | 1 | 8 | 0.8171 | 3 | 6 |
| shared_gb | 0 | 5 | 0.7843 | 5 | 7 |
| shared_gb | 1 | 5 | 0.7839 | 5 | 6 |
| shared_gphi | 0 | 5 | 0.7086 | 2 | 4 |
| shared_gphi | 1 | 5 | 0.7092 | 2 | 3 |

Artifacts: `n6_L4_fidelity_vs_iter.csv`, `n6_L4_fidelity_vs_iter.png`.

Notes: iter=0 is init (before first LBFGS step). Dotted verticals / markers = first iter where best-so-far hold ≥ 95% of that run's final best hold.
