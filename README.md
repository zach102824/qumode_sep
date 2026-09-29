# qumode_sep

Materials derived from [zach102824/qumode](https://github.com/zach102824/qumode).

## 4-SAT Hamiltonians

- `Hamiltonians/four_sat.py` — 8-qubit unique-GS generator
- `Hamiltonians/four_sat/` — 20 NPZ instances + manifest
- `tests/test_four_sat.py`
- `Hamiltonians/four_sat_scaling.py` — family F1 for the scaling study: planted, locally rigid,
  unique-solution 4-SAT at n = 8..28 (20 instances/n in `Hamiltonians/four_sat_scaling/`);
  results in `noiseless/results/SCALING_SUMMARY.md`

## Noiseless local-ECD + fixed-U SPSA

See [`noiseless/`](noiseless/) for the 8-bit encoding `|d⟩⊗|e⟩⊗|A⟩⊗|B⟩`, frozen
bus unitaries, SPSA+Gibbs optimizer, and sweep CLI. Results in
`noiseless/results/`.
