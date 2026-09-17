# qumode_sep

Materials derived from [zach102824/qumode](https://github.com/zach102824/qumode).

## 4-SAT Hamiltonians

- `Hamiltonians/four_sat.py` — 8-qubit unique-GS generator
- `Hamiltonians/four_sat/` — 20 NPZ instances + manifest
- `tests/test_four_sat.py`

## Noiseless local-ECD + fixed-U SPSA

See [`noiseless/`](noiseless/) for the 8-bit encoding `|d⟩⊗|e⟩⊗|A⟩⊗|B⟩`, frozen
bus unitaries, SPSA+Gibbs optimizer, and sweep CLI. Results in
`noiseless/results/`.
