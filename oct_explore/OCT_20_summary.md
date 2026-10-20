# OCT summary: native qumode ansatz with network-predicted parameters

> **STRICT-CONTROL RULE (2026-10-20 14:30 UTC+8, Zach):** a result counts only if the training size is strictly smaller than the validation/test size (16-level: train n=8 -> val n=12 -> test n=16; 32-level: train n=10 -> val n=15 -> test n=20). Every earlier number from training on n=10+15 (or n=8+12) with validation/early stopping on n=15 (or n=12) is **not strictly controlled (train/val overlap)**. That includes F32_*, B32_*, B20_*, G16_*, B16_*, the RR4/RR8 runs, and the OpenEvolve net_smoke/net_long runs (which trained on n=10+15). Treat those as indicative only.


## Key finding
- **Architecture:** all data bits live in cavities (Gray code, ≤5 bits/cavity). Native gates only: SNAP, displacement, neighbor-bus diagonal phases.
- **Transfer:** a graph neural network trains at small n and predicts circuit settings for larger n — no per-formula tuning on hardware.
  - 16 levels: train n=8 (4/4) + n=12 (4/4/4), test n=16 (4/4/4/4)
  - 32 levels: train n=10 (5/5) + n=15 (5/5/5), test n=20 (5/5/5/5)
- **Global cost phase** (ideal phases across all cavities): beats QAOA L4 on median and B90 at every tested size.
- **Chain only** (neighbor buses, no 3+ cavity terms): wins only at 2-cavity sizes; the tail collapses at large n. ~40% of the cost function at n=16/20 can't be represented on the chain.
- Both versions have the *same* learned outputs; the difference is which clause phases the hardware can apply.
- Software fixes on the chain (L8, training at n=20, wider bias head, tail-weighted loss, placement choice) did not close the gap. Any-pair coupling helps ~10× in B90 but still loses.

## How it works
Per layer l (L=4):
1. SNAP per cavity: phase γ_l Σ_j w_j [clause j violated], over clauses inside that cavity.
2. Neighbor-bus diagonal phase for clauses shared by the two cavities. Other clauses are least-squares projected onto single-cavity + neighbor-pair terms (weighted by p_v).
3. Calibrated SNAP–D–SNAP mixer ≈ one-Gray-digit flips, strength β_l.

Prep: fixed SNAP+D pairs to a flat state (fidelity ≥0.99998). Readout: photon number → Gray decode.

### Network
- **Input:** formula as a graph (variable and clause nodes, edges with literal sign) + each variable's cavity.
- **Output (one pass):** γ_l, β_l per layer; clause weight w_j (j = clause index); variable bias p_v (v = variable index).
- A fixed compiler turns these into SNAP and bus phases. Code: [gnn_bias.py](v3/native/gnn_bias.py), [fixed.py](v3/native/fixed.py).

## Results (fair, L4 vs L4, median / B90 of 1/p = expected shots)

| n | Chain L4 | Global cost L4 | QAOA L4 |
|---|---|---|---|
| 8 | **4.86/9.29** | **4.76/8.81** | 5.36/10.07 |
| 12 | 18.2/83 | **12.31/26.31** | 12.97/31.76 |
| 16 | 121/4487 | **31.19/74.14** | 34.82/98.20 |
| 10 | **8.07/17.29** | – | 8.44/18.87 |
| 15 | 39.1/251 | **20.2/57** | 23.12/62.53 |
| 20 | 309/8034 (single placement); 290/4455 (top-2 formula-picked placements, shots split) | **77.3/195** | 84.46/226.7 |

Connectivity ladder (same recipe): n=20 chain 309/8034 → any-pair 115.7/810 → 3+ cavity phases 77.3/195.

Notes: fair = one placement (or formula-only pick), no best-of-N oracle. Global-cost rows use ideal gates. Native calibrated mixer ≈ ideal at 16 levels, ~10% cost at 32 levels.

## Raw data
- Full tables and method: [NATIVE_TEMPLATE.md](NATIVE_TEMPLATE.md)
- Results: [results/v3/native](results/v3/native)
- Logs: `logs/v3/native/`


## 2026-10-20 14:30 UTC+8: strictly controlled restart
> **STRICT-CONTROL RULE (2026-10-20 14:30 UTC+8, Zach):** a result counts only if the training size is strictly smaller than the validation/test size (16-level: train n=8 -> val n=12 -> test n=16; 32-level: train n=10 -> val n=15 -> test n=20). Every earlier number from training on n=10+15 (or n=8+12) with validation/early stopping on n=15 (or n=12) is **not strictly controlled (train/val overlap)**. That includes F32_*, B32_*, B20_*, G16_*, B16_*, the RR4/RR8 runs, and the OpenEvolve net_smoke/net_long runs (which trained on n=10+15). Treat those as indicative only.

Launched: OpenEvolve net16_n8 (train 8, val 12 with rr4 over 100 formulas, check 16 every 10 iterations, cores 5-6) and net32_n10 (train 10, val 15, check 20, cores 3-4). Both are seeded from the initial network; the gamma/beta init comes from the *other* family's fixed schedule so no tested size was ever seen. Hand runs: A16_n8_RR4 (train 8, early stopping on n=12, test 12/16) and B32_n10_RR4 (train 10, early stopping on n=15, test 15/20). B32_RR8 was killed (not strictly controlled).

## 2026-10-20 14:35 UTC+8: SPLIT REVISED (Zach). This supersedes the 14:30 rule
16-level: train n=8+12, select/test ONLY on held-out n=16. 32-level: train n=10+15, select/test ONLY on held-out n=20. No separate validation size.
Model/candidate selection uses a fixed held-out SELECTION set (seed base 33e9). Final numbers come from a DIFFERENT held-out TEST set (base 32e9, the standard 100 formulas).
Earlier runs that early-stopped on n=15 / n=12 (a training size) are **not strictly controlled (train/val overlap)**.
Launched: OpenEvolve net16 (selection 60 n=16 formulas, rr4; test-set check every 10 iterations; cores 5-6) and net32 (selection 30 n=20 formulas, rr4; cores 3-4). Hand runs: A16_RR4 (train 8+12, early stopping on 60 n=16 selection formulas, test n=16) and B32_RR4 (train 10+15, early stopping on 30 n=20 selection formulas, test n=20).
