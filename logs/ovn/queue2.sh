#!/bin/bash
cd /workspace/qumode_sep
while [ ! -f logs/ovn/queue1.done ]; do [ -f logs/ovn/STOP ] && exit 0; sleep 60; done
V=logs/ovn/run_variant2.sh
OVN_EMAP="16:2,18:3,20:4" $V sl10_Elow 10 "16 18" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl10_Elow && OVN_EMAP="16:2,18:3,20:4" $V sl10_Elow 10 "20" QM_CLS_ALGO=satlike
OVN_R=100 $V sl10_r100 10 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl10_r100 && OVN_R=100 $V sl10_r100 10 "20" QM_CLS_ALGO=satlike
touch logs/ovn/queue2.done
