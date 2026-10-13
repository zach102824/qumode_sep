#!/bin/bash
cd /workspace/qumode_sep
logs/ovn/run_variant.sh satlike10 10 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
logs/ovn/run_variant.sh satlike5 5 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
logs/ovn/run_variant.sh tunedsa10 10 "8 10 12 14 16 18" QM_CLS_ALGO=tunedsa
logs/ovn/run_variant.sh satlike10 10 "20" QM_CLS_ALGO=satlike
logs/ovn/run_variant.sh satlike5 5 "20" QM_CLS_ALGO=satlike
touch logs/ovn/queue1.done
