#!/bin/bash
# SA-alone scaling. Stop: touch logs/3cav/saalone/SAALONE.STOP (only this file).
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
python noiseless/sa_alone_scaling.py
