# Width-4 staggered Baxter–Wu

SQS / pure p=4 are not part of this round. The Hamiltonian matches the triangular-torus enumeration

```
H = -sum_▲ ZZZ + sum_▼ ZZZ - Gamma * sum X
up   = {(x,y),(x+1,y),(x+1,y+1)}
down = {(x,y),(x,y+1),(x+1,y+1)}
index = x + Lx*y, qubit q = bit q
```

`4×3` is an orientation check of `3×4`, not a second system size. `4×4` is the width-4 stress test. Gamma grid is `{0.6, 0.8, 1.0, 1.2}`.

```
python3 run_round3.py
```
