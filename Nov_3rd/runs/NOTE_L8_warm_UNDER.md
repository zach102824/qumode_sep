# NOTE — L=8 warm Hamming-shell dual UNDER vs QAOA (2026-11-03 18:33 Asia/Shanghai)

Warm should be exact QAOA dual (φ_k=-β·k, γ=γ_QAOA). At L=8:
- n6: R=1.000 TIE (exact)
- n9: R=1.061 UNDER
- n12: R=1.083 UNDER
- n15: R=1.099 UNDER

L=1 and L=4 warm were R≈1.000 everywhere. Dual break appears at L≥8 on transfer n.
Polish was also UNDER (worse on n12/n15); t2527u shipped warm, kept climbing.
Not changing STOP policy. Investigate later if persists at higher L.
