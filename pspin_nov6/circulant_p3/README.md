# Circulant two-window 3-body ring, round 2

Isolated from `collective_p3/`. Same branch.

```text
H = -sum_i Z_i Z_{i+1} Z_{i+3} + kappa * sum_i Z_i Z_{i+2} Z_{i+3} - Gamma * sum_i X_i
```

```bash
python3 verify_algebra.py
python3 run_round2.py --  # full driver; the published depth ladder used 8 random starts and maxiter 40
```

The published conclusion is in `REPORT.md`. The ring is frustrated for every N=6..14, including multiples of 7, and reflection is not a symmetry. The full-space ground state is nevertheless an orbit of a product state, and L=1 squared overlap is already above 0.83 at N=8, 9, 10, and 11.
