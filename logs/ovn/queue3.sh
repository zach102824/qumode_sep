#!/bin/bash
cd /workspace/qumode_sep
while [ ! -f logs/ovn/queue2.done ]; do [ -f logs/ovn/STOP ] && exit 0; sleep 30; done
V=logs/ovn/run_variant2.sh
# (3) K=n (half readout) + satlike5 classical — fewer lookups
OVN_RAWK_MUL=1 $V sl5_Kn 5 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl5_Kn && OVN_RAWK_MUL=1 $V sl5_Kn 5 "20" QM_CLS_ALGO=satlike
# (2) clause-weight fix-up centers, same budget as satlike5
OVN_CW_FIX=1 $V sl5_cw 5 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl5_cw && OVN_CW_FIX=1 $V sl5_cw 5 "20" QM_CLS_ALGO=satlike
# (2b) CW + polt 2 (two fix-up centers), K=2n, satlike5 — more fixups, same SA
OVN_CW_FIX=1 OVN_POLT=2 $V sl5_cw_p2 5 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
# (4) already in q2 as sl5_r100; if that failed, try r=75 with satlike5
OVN_R=75 $V sl5_r75 5 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl5_r75 && OVN_R=75 $V sl5_r75 5 "20" QM_CLS_ALGO=satlike
touch logs/ovn/queue3.done
