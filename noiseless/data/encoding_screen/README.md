# Encoding (layout) brute-force screen: raw per-trial data

One gzipped JSONL file per Hamiltonian, `four_sat_XXX.jsonl.gz`. Only Hamiltonians whose
screen is complete are included; more are added as the screen finishes (H0–H19 total).

## Setup
- Problem: n=8 `four_sat` Hamiltonians, `Hamiltonians/four_sat/four_sat_XXX.npz` (in this repo).
- Hardware: 2 data transmons (d, e) + 2 cavities (A, B), 3 bits per cavity, binary Fock code
  n = 4*b2 + 2*b1 + b0 (Fock levels 0–7).
- Circuit: ECD ansatz with fixed `jp` (joint-parity) gate, layer growth L=1→4 (warm start, kick sigma 0.05).
- Optimizer: SPSA-Adam, lr 0.5/0.2/0.05/0.02 per stage, 200 steps per stage, 1604 cost evals per trial.
- Cost: infinite-shot Gibbs cost C = -log sum_x p_theta(x) exp(-eta E(x)), adaptive eta.
- Layouts: all 20160 classes (8!/2; the only symmetry is the joint swap d<->e with A<->B).
  Each layout is run with the same 3 random inits (init 1–3) -> 60480 trials per Hamiltonian.
- Code: `noiseless/run_encoding_screen.py`, `noiseless/encoding.py`.

## Fields (one JSON object per line = one trial)
| field | meaning |
|---|---|
| ham_file, ham_index | Hamiltonian file and index |
| class_idx | layout class id (0 = identity layout) |
| perm | perm[i] = physical slot of variable Z(i+1); slots ordered d, e, A2, A1, A0, B2, B1, B0 |
| init, seed | random-init index (1–3) and its RNG seed (same seeds for every layout) |
| p_gs | final probability of the ground state |
| success | most likely bitstring == ground state |
| energy_mean | final mean energy <E> |
| fun | final cost value |
| nfev | cost evaluations used |
| most_likely_bitstring, ground_bitstring | logical bitstrings (Z1..Z8 order) |
| ground_energy | ground-state energy |
| stage_p_gs | p(GS) at the end of each growth stage L=1,2,3,4 |
| wall_s, t_done | wall time of the trial (s) and finish time (Unix) |

## Load
```python
import pandas as pd
df = pd.read_json("four_sat_000.jsonl.gz", lines=True, compression="gzip")
per_layout = df.groupby("class_idx").p_gs.mean().sort_values(ascending=False)
```

Analysis of H0–H7: `noiseless/results/LAYOUT_PATTERNS.md`.
