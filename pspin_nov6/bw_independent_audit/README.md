# Independent Baxter–Wu audit

Rebuild of the staggered triangular torus. No imports from `baxter_wu_p3/` or from the circulant code. `reference/` is a read-only snapshot of the other branch's term list, angle file, and QAOA table.

```bash
python3 run_audit.py
```

Conclusion: stored angles reproduce to about 1e-14 in fidelity. See `REPORT.md`. Circulant work is not continued here.
