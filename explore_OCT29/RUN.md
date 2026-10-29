# RUN — explore_OCT29 (box / Mac CPU; cloud later)

Theory lock + rewrite order: [`PLAN.md`](PLAN.md).  
**Do not start the locked rewrite until Zach says `go`.**  
Until then these commands exercise the **pre-lock snapshot** (still has free-q / per-cavity mixers).

Chaodao GPU is **deferred** for this track. Prefer box overnight CPU, Mac smoke, or cloud GPU later with the same flags.

## Setup

```bash
# from repo root
cd /workspace/qumode_sep   # box
# or ~/qumode_sep on Mac

pip install -r explore_OCT29/requirements.txt
export PYTHONPATH=/workspace/qumode_sep   # or ~/qumode_sep
```

Mac sync: folder lives in git — after a push from the box, `git pull` on Mac `~/qumode_sep`.

**Box tip:** if system Python is PEP-668 managed, use the existing CPU venv:
`source /workspace/qtimes_venv/bin/activate` (or `/workspace/.venv-qtimes` for CUDA builds),
then the same `python -m …` / `pytest` lines below.

## Tests

```bash
pytest explore_OCT29/tests -q
```

## M0 — TEB / Hamiltonian packs

`data/data_n6.npz` and `data_n9.npz` are already in this snapshot. Regenerate or add n12+ with:

```bash
python -m explore_OCT29.scripts.gen_teb_packs --ns 6,9,12,15,18 --overwrite
```

## M1 — classical WalkSAT bars (CPU ok)

```bash
python -m explore_OCT29.scripts.run_walksat_bars --ns 6,9 --R 20 --max_inst 20
```

## Short train smoke (CPU)

```bash
python -m explore_OCT29.scripts.train --steps 5 --device cpu --save_best
```

Writes under `explore_OCT29/runs/` (and `runs/gnn_best/` if `--save_best`).

## Eval

```bash
python -m explore_OCT29.scripts.eval_scale --ns 6,9 --max_count 16 --device cpu
```

After n12+ packs exist: `--ns 6,9,12,15,18`.

## Evolve (CPU; slow — overnight box)

```bash
python -m explore_OCT29.scripts.evolve_gnn --max_trials 10 --steps 15 --gpus '' --n_parallel 1
# stop:
touch explore_OCT29/runs/STOP_EVOLVE
```

## After the locked rewrite lands

Same scripts with `--device cpu` or a cloud GPU id. Keep SNAP+D **shared**, Φ = `H_cost + εΔ` (local), only `γ_ℓ` per layer. Chaodao stays deferred unless Zach reopens it.

## Paths cheat-sheet

| Role | Box | Mac |
|------|-----|-----|
| Repo root | `/workspace/qumode_sep` | `~/qumode_sep` |
| Package | `explore_OCT29/` | same |
| Data | `explore_OCT29/data/data_n{n}.npz` | same |
| Best GNN | `explore_OCT29/runs/gnn_best/` | same |
| Stop evolve | `explore_OCT29/runs/STOP_EVOLVE` | same |
