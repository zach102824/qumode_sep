# Nov — Nov1st-final explore

Repo-root folder (not under `explore_OCT29`).

**Nov1st-final:** \(|0\rangle \to [U_c^{X}\to\mathrm{SNAP}]^L \to W \to\) Fock ≡ sealed BinH-proj.

Also explores **SNAP angle heads**: shell / free / bits / fourier / proj / x_pop × shared|perL at fixed `max_train=200`.

See `PLAN_Nov1st_final.md`.

```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov.scripts.angle_explore --phase all --max_train 200 --steps 400
```

Outputs: `runs/angle_explore/` (`snap_bakeoff/RESULTS.md`, `leaderboard.csv`).
