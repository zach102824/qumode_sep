# Output histogram on the bad formulas: net16r ckpt_110 (n=16 TEB, placement r=0)

Program `runs/net16r/checkpoints/checkpoint_110/best_program.py`, retrained once (EV.BUDGET, single-threaded, nice 19), full final |psi|^2
from pair_circuit.circuit_state (== frozen circuit_p with no pair gates; frozen files untouched). Scripts: `net/hist110.py` (dump), `net/hist110_analyze.py`.
This retrain: r=0 median 1/p 75.2, B90 14583 (round-robin median 91.1). Energy E = # violated clauses. Ham = Hamming distance to nearest ground state.
PR = participation ratio 1/sum p^2 (effective # states, out of 65536). S = Shannon entropy in bits (max 16).

| idx | 1/p (r=0) | #GS | max p (E, Ham) | top10 | PR (of 65536) | S bits (/16) | mass E=0/1/2/3/≥4 | uniform E=1/2 | <Ham> vs uniform | mass Ham 0-2/3-5/6-9/10+ |
|---|---|---|---|---|---|---|---|---|---|---|
| **worst** |||||||||||
| 64 | 1214952 | 1 | 0.0652 (1, 11) | 0.248 | 92 | 9.1 | 0.000/0.178/0.183/0.190/0.448 | 0.0004/0.0018 | 9.0 vs 8.0 | 0.00/0.06/0.47/0.46 |
| 17 | 1044408 | 1 | 0.0235 (3, 9) | 0.151 | 263 | 10.4 | 0.000/0.044/0.101/0.245/0.610 | 0.0002/0.0011 | 7.4 vs 8.0 | 0.01/0.17/0.65/0.16 |
| 45 | 115226 | 1 | 0.088 (1, 6) | 0.221 | 88 | 9.7 | 0.000/0.089/0.115/0.257/0.539 | 0.0001/0.0010 | 7.3 vs 8.0 | 0.00/0.12/0.76/0.11 |
| 30 | 62869 | 1 | 0.0204 (3, 6) | 0.113 | 350 | 10.5 | 0.000/0.050/0.129/0.264/0.557 | 0.0003/0.0017 | 7.2 vs 8.0 | 0.01/0.21/0.61/0.16 |
| 50 | 59776 | 2 | 0.0524 (1, 6) | 0.179 | 157 | 9.8 | 0.000/0.229/0.216/0.182/0.373 | 0.0008/0.0033 | 7.6 vs 7.5 | 0.00/0.12/0.73/0.15 |
| 31 | 58511 | 1 | 0.03 (3, 9) | 0.152 | 239 | 10.2 | 0.000/0.034/0.145/0.260/0.562 | 0.0001/0.0014 | 8.4 vs 8.0 | 0.00/0.05/0.71/0.24 |
| 62 | 47643 | 1 | 0.0322 (2, 8) | 0.164 | 214 | 10.2 | 0.000/0.056/0.214/0.157/0.573 | 0.0002/0.0020 | 8.5 vs 8.0 | 0.00/0.08/0.60/0.32 |
| 24 | 30040 | 1 | 0.0177 (2, 4) | 0.126 | 279 | 10.1 | 0.000/0.036/0.130/0.203/0.632 | 0.0003/0.0012 | 6.9 vs 8.0 | 0.02/0.29/0.53/0.16 |
| 78 | 24429 | 1 | 0.0374 (2, 7) | 0.129 | 252 | 10.3 | 0.000/0.031/0.158/0.268/0.543 | 0.0004/0.0018 | 8.5 vs 8.0 | 0.00/0.11/0.55/0.34 |
| 88 | 17373 | 2 | 0.0314 (1, 5) | 0.126 | 306 | 10.9 | 0.000/0.053/0.110/0.141/0.695 | 0.0002/0.0014 | 5.9 vs 6.8 | 0.04/0.35/0.58/0.02 |
| **median** |||||||||||
| 99 | 110 | 2 | 0.0412 (2, 6) | 0.225 | 132 | 9.7 | 0.009/0.147/0.207/0.190/0.447 | 0.0002/0.0016 | 5.9 vs 7.5 | 0.10/0.28/0.59/0.03 |
| 59 | 76 | 3 | 0.0338 (1, 1) | 0.197 | 161 | 9.7 | 0.013/0.125/0.243/0.199/0.419 | 0.0003/0.0026 | 5.0 vs 7.2 | 0.15/0.43/0.41/0.01 |
| 81 | 74 | 2 | 0.0237 (2, 5) | 0.145 | 245 | 9.9 | 0.013/0.071/0.232/0.259/0.424 | 0.0003/0.0031 | 4.8 vs 7.5 | 0.12/0.53/0.34/0.00 |
| 4 | 72 | 1 | 0.0138 (0, 0) | 0.089 | 524 | 10.9 | 0.014/0.019/0.076/0.140/0.751 | 0.0002/0.0011 | 7.0 vs 8.0 | 0.05/0.19/0.59/0.17 |
| 29 | 58 | 1 | 0.0189 (2, 1) | 0.123 | 317 | 10.4 | 0.017/0.004/0.112/0.190/0.676 | 0.0001/0.0010 | 6.2 vs 8.0 | 0.10/0.29/0.51/0.10 |

## Verdict
- **Concentrated, not smeared.** Worst-10 median: PR ≈ 245 states (0.4% of 65536), entropy ≈ 10.2 bits (vs 16 uniform), top-10 states hold ~15%,
  the single largest state 2-9%. Statistically the same as for the median formulas (PR ≈ 245, top-10 ≈ 15%). The circuit always focuses about the same amount; on bad formulas it focuses on the wrong place.
- **GS is suppressed below uniform** on the very worst (idx 64/17: p ≈ 1e-6, ~20x *below* 1/65536): destructive interference, not just lack of amplification.
- **Mass goes to far, mostly not-low-energy states.** E=1 is enriched ~100x over uniform (3-23% vs 0.01-0.08%), but 55-70% of mass sits at E≥3.
  The top wrong state is typically 6-11 bit flips from the GS; 50-75% of mass is 6-9 flips away, and mean Hamming ≈ uniform's (≈8). Median formulas, by contrast, put their mass closer (mean Ham ≈ 5-6 vs 7.5 uniform).
- So the bad cases are not "almost right" states: the circuit concentrates onto a ~250-state cluster in a different region of the hypercube (matches FAILURE_ANALYSIS_16x.md, ~8 flips away; pv guesses wrong).
- ckpt_10 (same analysis, `runs/hist_10.json`, `FAILURE_HISTOGRAM_10.png`): identical picture, worst-10 PR ≈ 234, top-10 ≈ 15%.

Figure: `FAILURE_HISTOGRAM_110.png` (sorted state probabilities, log-log, worst #1, #2 and a median formula; mass by energy).
Raw: `runs/hist_110.json`.
