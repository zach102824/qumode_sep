# n=8 Phase-1 (tracked entrypoint)

Active run dir (gitignored artifacts): `../runs/n8_phase1/`
- `STATUS.md`, `RESULTS.md`, `results.csv`, `results.jsonl`
- `runs/<id>/{result.json,weights.npz,log.txt}`
- `phase1.log`

Start / resume:
```
cd /workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1
python3 -u run_phase1.py >> phase1.log 2>&1
```
Skips already-done configs. After core 24 auto-densifies.
