# RUN — explore_OCT29 (box CPU; step 0 live)

Zach confirms plan in chat; **I run** these on the box. Chaodao deferred.

## Setup

```bash
cd /workspace/qumode_sep
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
```

## Tests

```bash
pytest explore_OCT29/tests -q
```

## Packs

`data_n6.npz` / `data_n9.npz` present. n12 (promote gate):

```bash
python -m explore_OCT29.scripts.gen_teb_packs --ns 12 --ntrain 0 --nval 50 --ntest 50
# optional later:
python -m explore_OCT29.scripts.gen_teb_packs --ns 15 --ntrain 0 --nval 20 --ntest 30
```

## Seed champion

```bash
python -m explore_OCT29.scripts.train --steps 20 --L 5 --device cpu --save_best --max_n12 8
```

## Eval (incl. zero-shot n12)

```bash
python -m explore_OCT29.scripts.eval_scale --ns 6,9,12 --max_count 16 --device cpu --L 5
```

## Nonstop evolve (promote on n12)

```bash
# forever, 1 parallel on CPU:
python -m explore_OCT29.scripts.evolve_gnn --steps 15 --L 5 --gpus '' --n_parallel 1 --max_n12 8
# stop:
touch explore_OCT29/runs/STOP_EVOLVE
```

CSV: `explore_OCT29/runs/gnn_evolve_trials.csv`  
Best: `explore_OCT29/runs/gnn_best/`
