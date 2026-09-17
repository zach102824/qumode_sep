# qumode_sep

New version of materials from [zach102824/qumode](https://github.com/zach102824/qumode).

## Included so far

### 4-SAT Hamiltonians

- `Hamiltonians/four_sat.py` — generator (8-qubit target)
- `Hamiltonians/four_sat/` — NPZ instances + manifest
- `tests/test_four_sat.py`

### Noiseless local-ECD + fixed-U SPSA

See [`noiseless/`](noiseless/) for the 8-bit encoding `|d⟩⊗|e⟩⊗|A⟩⊗|B⟩`, frozen
bus unitaries, SPSA+Gibbs optimizer, and sweep CLI. Results land in
`noiseless/results/`.
