# Collective ferromagnetic p=3, round 1

Isolated Family A search. Nothing outside `pspin_nov6/collective_p3/` is part of this run.

```text
H = -N m_z^3 - Gamma * sum_i X_i,    m_z = (sum_i Z_i) / N
```

QAOA minimizes `<H>` from `|+>^N` with independent `(gamma_l, beta_l)`. The cost generator is the diagonal piece `-N m_z^3`. The mixer is `sum_i X_i`.

```bash
python3 validate.py
python3 run_round1.py --out results
```

Dependencies: numpy, scipy, matplotlib. The round-1 search took about one minute on 4 cores after validation. `results/gate.json` records the pre-set low-L1 / depth-gain gate. This run's verdict is **FAIL**: at N=8, 10, and 12, L=1 squared ground-state overlap stays above 0.64 everywhere in the Γ window, so the target does not reproduce a low-L1, sub-0.4, roughly L/N fidelity curve.

See `REPORT.md` for the numbers, the identity check, and what was not claimed.
