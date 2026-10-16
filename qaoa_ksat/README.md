# Random k-SAT QAOA (Boulebnane & Montanaro)

Independent statevector reproduction of
[PRX Quantum 5, 030348](https://doi.org/10.1103/PRXQuantum.5.030348)
(arXiv:2208.06909). Published angles are vendored from
[PhaseCraft/qaoa_ksat_paper_data](https://github.com/PhaseCraft/qaoa_ksat_paper_data)
(Apache-2.0).

This folder is **not** the 8-qubit four_sat Gibbs bake-off in `noiseless/qaoa.py`.
That module uses a different Hamiltonian and `exp(-i β X)` without the paper’s
factor of `1/2`.

## Protocol

- Ensemble: Definition 1. `m ~ Poisson(r n)`, each of `k` literals drawn with
  replacement from `{x_i, ¬x_i}`. 8-SAT threshold `r = 176.54`.
- Cost: unsatisfied-clause count. Success = Born probability of energy 0.
- Only satisfiable instances (energy table min = 0).
- Circuit: `|+\rangle^{\otimes n}`, then each layer `exp(+i γ H / 2)` and
  `Rx(β) = exp(-i β X / 2)` using the published `(β, γ)`.
  The PDF writes `exp(-i γ H / 2)`; that sign with the published *positive* γ
  is worse than random. The authors’ analytic helper uses `exp(+i γ/2)` on the
  forward branch, and that sign matches their n = 12 checkpoint.
- Instance running time = `1 / p_succ` (expected shots). Metric = median TTS.

## Run

```bash
python -m pytest -q qaoa_ksat/tests
python -m qaoa_ksat.evaluate --n 8 --p 1,2,5,10,14 --n-instances 40 --workers 4
python -m qaoa_ksat.evaluate --n 12 --p 14,60 --n-instances 40 --workers 4
python -m qaoa_ksat.evaluate --n 20 --p 14,60 --n-instances 40 --workers 6
```

## Verdict

See [`results/N20_CHECK.md`](results/N20_CHECK.md). Short version: n = 8 and the
paper’s n = 12, p = 60 mean `p_succ ≈ 0.56` **match**. The Table II n = 20
**fit** is pessimistic (about 2× too large a median TTS at p = 60), not
optimistic. QAOA at n = 20 is real; if anything it is stronger than the
published fit.

Saved 8-SAT formulas and partial classical comparison:
[`results/CLASSICAL_VS_PAPER.md`](results/CLASSICAL_VS_PAPER.md).
The classical job was stopped before n = 20.
