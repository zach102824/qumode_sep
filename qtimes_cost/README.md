# qtimes_cost

Self-contained campaign: coherent-init `D(α₀)` + `[SNAP → D → U_phase^q]^L` on **fixed 3 bits/cavity** (`C = n/3`).

- **Train n:** 6, 9  
- **Eval n:** 12, 15, 18, 21  
- **No imports outside this folder** (vendored SAT gen + WalkSAT).  
- Plan: [`PLAN.md`](PLAN.md)

## Layout

```
qtimes_cost/
  lib/           sat_gen, walksat, gray_grid, ansatz, gnn, train_loop
  scripts/       gen_teb_packs, run_walksat_bars, evolve_gnn, train, eval_scale
  data/          data_n{n}.npz
  baselines/     walksat_n{n}.json
  runs/gnn_best/ best arch only (+ CSV trial log)
```

## Mac → chaodao sync

Prefer run folder `F:\zach\qtimes_cost` (WSL `/mnt/f/zach/qtimes_cost`).

```bash
# on Mac, from repo root (host alias `chaodao` in ~/.ssh/config)
rsync -avz --delete --exclude '__pycache__' --exclude '*.pyc' \
  qtimes_cost/ chaodao:/mnt/f/zach/qtimes_cost/
```

```bash
# scp alternative (whole folder)
scp -r qtimes_cost chaodao:/mnt/f/zach/qtimes_cost_upload
# then on chaodao WSL:
# rsync -av --delete /mnt/f/zach/qtimes_cost_upload/ /mnt/f/zach/qtimes_cost/
```

## Chaodao first commands (PowerShell)

```powershell
cd F:\zach\qtimes_cost
# If package root is the folder itself, set PYTHONPATH to its parent:
#   parent = F:\zach   and folder = qtimes_cost
$env:PYTHONPATH = "F:\zach"

# M0 smoke then train packs
python -m qtimes_cost.scripts.gen_teb_packs --ns 6,9 --ntrain 200 --nval 20 --ntest 20 --overwrite
python -m qtimes_cost.scripts.gen_teb_packs --ns 6,9 --overwrite
# full ladder (18/21 = clauses-only by default, no dense E)
python -m qtimes_cost.scripts.gen_teb_packs --ns 6,9,12,15,18,21 --overwrite

# M1 classical bars (start small)
python -m qtimes_cost.scripts.run_walksat_bars --ns 6,9 --R 20 --max_inst 20

# M-GNN nonstop (best-only); stop with:  New-Item runs\STOP_EVOLVE
python -m qtimes_cost.scripts.evolve_gnn --max_trials 0 --steps 15 --gpu 0

# M2
python -m qtimes_cost.scripts.train --steps 30 --save_best
python -m qtimes_cost.scripts.eval_scale --ns 6,9,12 --max_count 32
```

## Chaodao (WSL)

```bash
cd /mnt/f/zach/qtimes_cost
export PYTHONPATH=/mnt/f/zach
python -m qtimes_cost.scripts.gen_teb_packs --ns 6,9 --overwrite
python -m qtimes_cost.scripts.evolve_gnn --gpu 0   # or --gpu 0
touch runs/STOP_EVOLVE   # graceful stop
```

> If you synced so that `F:\zach\qtimes_cost` **is** the package root (contains `lib/`, `scripts/`),  
> `PYTHONPATH` must be the **parent** (`F:\zach`) for `python -m qtimes_cost...` to work.  
> Alternatively: `cd F:\zach` and run the same `-m` commands.

## Smoke (box / laptop)

```bash
cd /path/to/qumode_sep
PYTHONPATH=. python -m qtimes_cost.scripts.gen_teb_packs --ns 6 --ntrain 200 --nval 20 --ntest 20 --overwrite
PYTHONPATH=. python -c "from qtimes_cost.lib.ansatz import AnsatzConfig, MultiCavityAnsatz, random_params; from qtimes_cost.lib.sat_gen import gen_instance; import numpy as np; n=6; m,cl,E=gen_instance(n,51_000_000_000+10_000_000*n); cfg=AnsatzConfig(n=n,L=2,nf=16); sim=MultiCavityAnsatz(cfg); p=random_params(cfg); print(sim.forward(E,p['alpha0'],p['A'],p['TH'],p['q_logits']))"
```

## Optional fast WalkSAT binary

```bash
gcc -O2 -o lib/walksat_cls qtimes_cost/lib/walksat_cls.c -lm
# Python runner still default; swap later if desired.
```

## Notes

- Numpy reference sim/GNN — good for correctness + evolve scaffolding. Swap accelerator inside `lib/` later; keep scripts’ CLI stable.
- `runs/gnn_best/` is overwritten only on VAL improvement; losers are discarded. History: `runs/gnn_evolve_trials.csv`.
