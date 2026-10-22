# net16x STEP 1 — failure analysis of net16r ckpt_10 (Zach's overall best, fair TEB 83.9/2607)

Program: `runs/net16r/checkpoints/checkpoint_10/best_program.py`, retrained once (180 s budget, n=8+12), scored with the frozen circuit
(pair_circuit.circuit_state == frozen circuit_p with no pair gates) on all 100 TEB and 60 VAB n=16 formulas, R=4 placements.
This retrain: TEB rr median 76.9 / B90 4232; VAB 66.6 / 634 (training is stochastic; the official ckpt_10 fair check was 83.9/2607).
Raw data: `runs/net16x/failure16x.json` (script `net/failure16x.py`). Energies = number of violated clauses (integer), so the gap is always 1.

## 10 worst TEB formulas (p = round-robin mean over 4 placements)
| idx | seed | p_gs | 1/p | GS deg | gap (E=1 count) | clauses crossing cavities / non-neighbour | final mass E=1 / E=2 / E≥3 | eff. #states (1/Σp²) | top wrong-state mass | Hamming(top wrong → GS) | pv_acc | p_gs after L1→L4 | p / uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 32160000045 | 4.36e-06 | 229504 | 1 | 1 (8 states) | 1.00 / 0.87 | 0.028 / 0.113 / 0.859 | 359 | 0.020 | 9.2 | 0.50 | 4.4e-06 → 1.0e-05 → 3.9e-06 → 4.4e-06 | 0.3x |
| 50 | 32160000072 | 1.13e-05 | 88509 | 2 | 1 (50 states) | 1.00 / 0.86 | 0.136 / 0.253 / 0.612 | 219 | 0.031 | 8.4 | 0.62 | 1.5e-05 → 1.6e-05 → 2.4e-05 → 1.1e-05 | 0.4x |
| 55 | 32160000080 | 3.39e-05 | 29467 | 2 | 1 (37 states) | 1.00 / 0.86 | 0.121 / 0.217 / 0.662 | 204 | 0.033 | 7.9 | 0.50 | 2.0e-05 → 2.9e-05 → 4.9e-05 → 3.4e-05 | 1.1x |
| 62 | 32160000090 | 3.71e-05 | 26957 | 1 | 1 (15 states) | 1.00 / 0.88 | 0.029 / 0.155 / 0.815 | 325 | 0.026 | 9.0 | 0.44 | 1.8e-05 → 3.6e-05 → 7.7e-05 → 3.7e-05 | 2.4x |
| 17 | 32160000021 | 6.34e-05 | 15771 | 1 | 1 (12 states) | 1.00 / 0.87 | 0.048 / 0.094 / 0.858 | 315 | 0.024 | 6.2 | 0.50 | 4.2e-05 → 1.1e-04 → 1.2e-04 → 6.3e-05 | 4.2x |
| 91 | 32160000134 | 7.78e-05 | 12849 | 1 | 1 (13 states) | 1.00 / 0.88 | 0.020 / 0.067 / 0.914 | 505 | 0.016 | 6.9 | 0.44 | 2.8e-05 → 7.5e-05 → 1.1e-04 → 7.8e-05 | 5.1x |
| 64 | 32160000093 | 8.60e-05 | 11622 | 1 | 1 (25 states) | 1.00 / 0.85 | 0.134 / 0.172 / 0.694 | 132 | 0.053 | 10.4 | 0.38 | 1.2e-05 → 3.4e-05 → 5.5e-05 → 8.6e-05 | 5.6x |
| 78 | 32160000115 | 9.05e-05 | 11048 | 1 | 1 (24 states) | 1.00 / 0.85 | 0.018 / 0.165 / 0.817 | 231 | 0.037 | 9.1 | 0.44 | 1.2e-05 → 2.8e-05 → 5.5e-05 → 9.1e-05 | 5.9x |
| 72 | 32160000106 | 1.67e-04 | 5990 | 1 | 1 (26 states) | 1.00 / 0.87 | 0.094 / 0.209 / 0.696 | 249 | 0.026 | 7.5 | 0.56 | 5.6e-05 → 1.4e-04 → 1.9e-04 → 1.7e-04 | 10.9x |
| 67 | 32160000098 | 2.25e-04 | 4449 | 1 | 1 (48 states) | 0.99 / 0.85 | 0.225 / 0.274 / 0.500 | 107 | 0.054 | 7.6 | 0.50 | 6.9e-05 → 1.5e-04 → 1.7e-04 → 2.2e-04 | 14.7x |

## Worst-10 vs other 90 (TEB)
| quantity | worst 10 | other 90 |
|---|---|---|
| ground-state degeneracy | 1.2 | 3.57 |
| pv_acc (agreement of pv with nearest solution) | 0.49 | 0.77 |
| Hamming distance top wrong states → solution | 8.2 | 4.1 |
| energy of top wrong states | 2.1 | 2.05 |
| p gain from layer 2 to layer 4 | 1.05x | 3.1x |
| clauses crossing cavities / non-neighbour | 1.00 / 0.86 | 1.00 / 0.87 |
| final mass E≥3 | 0.74 | 0.72 |
| effective #states | 265 | 227 |
Spearman with p over the 100 TEB formulas: degeneracy +0.67, pv_acc +0.79, (#E=1 states / deg) −0.54, non-neighbour fraction +0.02.
Placement sensitivity on the worst 10: max/min p over the 4 placements = 9x … 36000x (median ~200x).

## Findings
1. **The tail is unique-ground-state formulas** (deg 1–2) with many E=1 competitors; the circuit reaches only 0.3–15x the uniform 1/65536 there.
2. **pv points to the wrong basin**: pv_acc ≈ 0.5 on the worst 10 (0.77 elsewhere) — the product measure that shapes the compiled cost is centred on a false basin.
3. **Amplitude goes to distant false basins**: top wrong states are low-energy (E≈2) but ~8 bit flips from the solution (4 elsewhere); the mass is spread (≈260 effective states, 74% on E≥3), not trapped in one wrong state.
4. **Later layers stop helping**: p barely grows after layer 2 on the worst 10 (1.05x vs 3.1x) — single-bit in-cavity mixing cannot move amplitude 8 flips across cavities in 4 layers.
5. **Chain geometry does not discriminate**: at n=16 virtually every clause crosses cavities (~86% non-neighbour) in both groups; the tail is about landscape (unique GS, far false basins) plus large placement sensitivity, which argues for correlated multi-bit / cross-cavity moves (non-diagonal pair unitaries, mixer re-encoding) and reflections, not more phase tuning.
