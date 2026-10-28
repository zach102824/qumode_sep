# qtimes_cost

Self-contained campaign: coherent-init `D(α₀)` + `[SNAP → D → U_phase^q]^L` on **fixed 3 bits/cavity** (`C = n/3`).

**Backend: PyTorch** (CUDA on chaodao 2×RTX 5090; CPU fallback for smoke/tests).

- **Train n:** 6, 9  
- **Eval n:** 12, 15, 18, 21  
- **No imports outside this folder** (vendored SAT gen + WalkSAT).  
- Plan: [`PLAN.md`](PLAN.md)

## Layout

```
qtimes_cost/
  lib/           sat_gen, walksat, gray_grid, ansatz (torch), gnn (torch), train_loop, device
  scripts/       gen_teb_packs, run_walksat_bars, evolve_gnn (dual-GPU), train, eval_scale
  tests/         pytest correctness suite
  data/          data_n{n}.npz
  baselines/     walksat_n{n}.json
  runs/gnn_best/ best arch only (+ CSV trial log)
  requirements.txt
```

## Mac → chaodao sync

Prefer run folder `F:\zach\qtimes_cost` (WSL `/mnt/f/zach/qtimes_cost`).

```bash
# on Mac, from repo root (host alias `chaodao` in ~/.ssh/config)
rsync -avz --delete --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  qtimes_cost/ chaodao:/mnt/f/zach/qtimes_cost/
```

```bash
# scp alternative (whole folder)
scp -r qtimes_cost chaodao:/mnt/f/zach/qtimes_cost_upload
```

## Chaodao (2× GPU) — PowerShell

```powershell
cd F:\zach\qtimes_cost
$env:PYTHONPATH = "F:\zach"
pip install -r requirements.txt   # once: torch with CUDA

# M0 packs
python -m qtimes_cost.scripts.gen_teb_packs --ns 6,9 --overwrite
python -m qtimes_cost.scripts.gen_teb_packs --ns 6,9,12,15,18,21 --overwrite

# M1 classical bars
python -m qtimes_cost.scripts.run_walksat_bars --ns 6,9 --R 20 --max_inst 20

# M-GNN dual-GPU evolve: TWO candidates per round on cuda:0 and cuda:1
# best-only overwrite of runs/gnn_best/; losers discarded; CSV append
python -m qtimes_cost.scripts.evolve_gnn --max_trials 0 --steps 15 --gpus 0,1 --n_parallel 2
# stop: New-Item runs\STOP_EVOLVE

# single-GPU train
python -m qtimes_cost.scripts.train --steps 30 --device cuda:0 --save_best
python -m qtimes_cost.scripts.eval_scale --ns 6,9,12 --max_count 32 --device cuda:0
```

## Chaodao (WSL)

```bash
cd /mnt/f/zach/qtimes_cost
export PYTHONPATH=/mnt/f/zach
python -m qtimes_cost.scripts.evolve_gnn --gpus 0,1 --n_parallel 2 --steps 15
touch runs/STOP_EVOLVE   # graceful stop
```

> `PYTHONPATH` must be the **parent** of the `qtimes_cost` package folder.

## Tests (box / laptop CPU)

```bash
cd /path/to/qumode_sep
pip install -r qtimes_cost/requirements.txt
PYTHONPATH=. pytest qtimes_cost/tests -q
```

## Dual-GPU notes

- `evolve_gnn --gpus 0,1 --n_parallel 2` trains **two arch candidates in parallel** (ThreadPool, one per device).
- Each candidate’s `short_train` uses Adam on that device; VAL Med/B90 → `combined_score`.
- Promote only if score beats champion (`runs/gnn_best/`); append every try to `runs/gnn_evolve_trials.csv`.
- `--gpu 0,1` sets `CUDA_VISIBLE_DEVICES` before device resolution if you need to mask cards.
- Without CUDA, both candidates fall back to CPU (still runs; slow).

## Notes

- Weights saved as `runs/gnn_best/weights_n{n}.pt` (torch state_dict).
- `runs/gnn_best/` overwritten only on VAL improvement.
