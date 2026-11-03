# Nov_3rd

Self-contained final recipe folder: plan, scripts, settings, data, QAOA angles, runs.

- b=3 / BinH / shell / QAOA-warm / n6×1000 / L-BFGS
- L ∈ {1,4,8,12,16,20,24,28,32}
- Eval n = 6…18 (sim: 6,9,12,15,18)

```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov_3rd.run_final
```

See `PLAN.md`. Paths: `data/`, `qaoa_angles/`, `runs/`.
