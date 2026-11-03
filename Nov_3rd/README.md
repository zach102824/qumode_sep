# Nov_3rd

Locked final recipe (2026-11-03). All knobs in `settings.py`.

- \(b=3\) bits/cavity, BinH, shell, QAOA-warm, train n6×1000, L-BFGS only
- **L ∈ {1,4,8,12,16,20,24,28,32}**
- **Eval n = 9…18** (sim: n%3==0)

```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov_3rd.run_final
```
