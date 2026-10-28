# n=8 L=5 per-instance BFGS expressivity ceiling

Locked: vacuum | (SNAP+D → cost)^L with ONE tied (γ, clause_w). NO GNN.
NF=36 (chk 56), amax=1.8, restarts=3, maxiter=250.
Instances: hard=[0, 2, 4, 6, 9] (nsol=[1, 1, 1, 1, 1]), rand=[63, 52, 30, 34, 82].

| idx | kind | n_gs | best p_succ | best 1/p | leak | chk p | #restarts | nit | wall_s |
|-----|------|------|-------------|----------|------|-------|-----------|-----|--------|
| 0 | hard | 1 | 0.9997 | 1.00 | 0.0001 | 0.9997 | 3 | 250 | 1.4 |
| 2 | hard | 1 | 0.9996 | 1.00 | 0.0002 | 0.9996 | 3 | 250 | 1.1 |
| 4 | hard | 1 | 0.9995 | 1.00 | 0.0001 | 0.9995 | 3 | 250 | 1.1 |
| 6 | hard | 1 | 0.9986 | 1.00 | 0.0007 | 0.9986 | 3 | 250 | 1.1 |
| 9 | hard | 1 | 0.9888 | 1.01 | 0.0038 | 0.9888 | 3 | 250 | 1.1 |
| 63 | rand | 1 | 0.9283 | 1.08 | 0.0226 | 0.9283 | 3 | 250 | 1.1 |
| 52 | rand | 2 | 0.9991 | 1.00 | 0.0004 | 0.9991 | 3 | 250 | 1.1 |
| 30 | rand | 1 | 0.9975 | 1.00 | 0.0009 | 0.9975 | 3 | 250 | 1.1 |
| 34 | rand | 1 | 0.9905 | 1.01 | 0.0036 | 0.9905 | 3 | 250 | 1.1 |
| 82 | rand | 1 | 0.9968 | 1.00 | 0.0015 | 0.9968 | 3 | 250 | 1.2 |

**Aggregate** (best 1/p over 10): median=1.00, B90=1.02, p≥0.5: 10/10, p≥0.9: 10/10. WalkSAT look Med/B90 ≈ 13.99/29.40.

**Verdict:** expressivity OK — most hard instances reach 1/p ≲ 20–30 under per-instance BFGS; amortization/opt is the gap

Wall time: 21.2s (Asia/Shanghai box clock).
JSON: `/workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1/ceiling_bfgs_n8_L5.json`
