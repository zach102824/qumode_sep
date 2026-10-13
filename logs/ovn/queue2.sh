#!/bin/bash
# waits for queue1 (killed after tunedsa10 n18; queue1.done touched by hand)
cd /workspace/qumode_sep
while [ ! -f logs/ovn/queue1.done ]; do [ -f logs/ovn/STOP ] && exit 0; sleep 30; done
V=logs/ovn/run_variant2.sh
$V satlike5 5 "20" QM_CLS_ALGO=satlike
OVN_EMAP="16:2,18:3,20:4" $V sl5_Elow 5 "16 18" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl5_Elow && OVN_EMAP="16:2,18:3,20:4" $V sl5_Elow 5 "20" QM_CLS_ALGO=satlike
$V sl3 3 "18 16 14 12 10 8" QM_CLS_ALGO=satlike
OVN_R=100 $V sl5_r100 5 "18 16 14 12 10 8" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl5_r100 && OVN_R=100 $V sl5_r100 5 "20" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl3 && $V sl3 3 "20" QM_CLS_ALGO=satlike
touch logs/ovn/queue2.done
