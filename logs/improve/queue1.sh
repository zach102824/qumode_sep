#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8"
python noiseless/run_relayout_kbit.py --n 12 --explore 5 $B --explore-mask sa --sab 144 --trials 5 > logs/improve/n12_sa144_E5.json 2>logs/improve/n12_sa144_E5.err
python noiseless/run_relayout_kbit.py --n 12 --explore 5 $B --explore-mask sa --sab 144 --samode fixed_low --trials 5 > logs/improve/n12_sa144fl_E5.json 2>logs/improve/n12_sa144fl_E5.err
python noiseless/run_relayout_kbit.py --n 14 --explore 9 $B --explore-mask sa --sab 196 --trials 3 > logs/improve/n14_sa196_E9.json 2>logs/improve/n14_sa196_E9.err
echo DONE > logs/improve/queue1.done
