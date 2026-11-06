# Round 2 — Steiner quadruple designs with structured signs

Not a random spin glass and not the round-1 all-to-all pure p=4.

```
H = - sum_{b in SQS(N)} J_b Z_b1 Z_b2 Z_b3 Z_b4 - (r * Gamma) sum_i X_i
r = (N-1)(N-2)/6
```

`Gamma` is the per-degree field. The coefficient of each `X` is `Gamma_X = r * Gamma`.
Round-1 `Gamma` was a `Gamma_X`. Both scales are written on every row.

Designs are built and checked in `sqs.py` (every triple in exactly one block).
Signs and certified frustration (`min` unsatisfied blocks over all `2^N` configs)
are in `signs.py`. Covering-radius optimality is claimed only for N=8, where
every coset was enumerated.

```
python3 validate.py
python3 run_round2.py
```
