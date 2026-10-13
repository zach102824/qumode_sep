#!/bin/bash
# independent of queue2/3; different outdirs (sl5div, sl5rs)
cd /workspace/qumode_sep
V=logs/ovn/run_variant2.sh
# diverse SA centers (Hamming-separated) + satlike5 — targets n20 valley collapse
OVN_SAMODE=fixed_low_div $V sl5div 5 "8 10 12 14 16 18" QM_CLS_ALGO=satlike
python3 logs/ovn/passed.py sl5div && OVN_SAMODE=fixed_low_div $V sl5div 5 "20" QM_CLS_ALGO=satlike
# satlike with restart every 3n lookups, budget 10n (more coverage than satlike5)
$V sl10rs 10 "8 10 12 14 16 18" QM_CLS_ALGO=satlike QM_SL_RESTART=3
python3 logs/ovn/passed.py sl10rs && $V sl10rs 10 "20" QM_CLS_ALGO=satlike QM_SL_RESTART=3
# satlike5 + div + K=n (aggressive lookup cut) only if sl5div passed all including 20 — skip, queue3 handles Kn
touch logs/ovn/queue4.done
