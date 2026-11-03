# Nov_3rd

Self-contained final recipe: plan, scripts, tests, data, QAOA angles, runs.

```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov_3rd.tests.run_tests   # must 0 failed
python -m Nov_3rd.run_final         # L small→large; STOP if UNDER QAOA
```

See `PLAN.md`.
