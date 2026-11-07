# FM-WS smoke — CORE Med

Lower Med is better (shots to solution, 1/p_succ). CORE B=1000, train {6,9} only.

Champ = free power_p at that L (fresh250 seed0). There is no free L=48 champ;
L=48 is scored against the free **L=32** champ (Opus kill bar: >1% Med at n=15).
Fourier QAOA uses the same Q, the same DST/DCT modes, and the same INTERP polish.

| Q | L | n | FM-WS Med | champ Med | vs champ | free QAOA | vs free Q | Fourier QAOA | vs F-QAOA | L32 champ |
|---|---|---|-----------|-----------|----------|-----------|-----------|--------------|-----------|-----------|
| 4 | 16 | 6 | 1.1619 | 1.1503 | UNDER | 1.2331 | BEAT | 1.1487 | UNDER | 1.0440 |
| 4 | 16 | 9 | 1.4548 | 1.4183 | UNDER | 1.5879 | BEAT | 1.4316 | UNDER | 1.1161 |
| 4 | 16 | 12 | 1.9094 | 1.8389 | UNDER | 2.0854 | BEAT | 1.8012 | UNDER | 1.2279 |
| 4 | 16 | 15 | 2.6529 | 2.5791 | UNDER | 2.9127 | BEAT | 2.4565 | UNDER | 1.4843 |
| 4 | 32 | 6 | 1.0295 | 1.0440 | BEAT | 1.0617 | BEAT | 1.0374 | BEAT | 1.0440 |
| 4 | 32 | 9 | 1.1074 | 1.1161 | BEAT | 1.1626 | BEAT | 1.1037 | UNDER | 1.1161 |
| 4 | 32 | 12 | 1.2185 | 1.2279 | BEAT | 1.3083 | BEAT | 1.2011 | UNDER | 1.2279 |
| 4 | 32 | 15 | 1.4431 | 1.4843 | BEAT | 1.6064 | BEAT | 1.3720 | UNDER | 1.4843 |
| 4 | 48 | 6 | 1.0118 | — | — | — | — | 1.0213 | BEAT | 1.0440 |
| 4 | 48 | 9 | 1.0395 | — | — | — | — | 1.0486 | BEAT | 1.1161 |
| 4 | 48 | 12 | 1.0908 | — | — | — | — | 1.0891 | UNDER | 1.2279 |
| 4 | 48 | 15 | 1.2094 | — | — | — | — | 1.1671 | UNDER | 1.4843 |
| 6 | 16 | 6 | 1.1619 | 1.1503 | UNDER | 1.2331 | BEAT | 1.1471 | UNDER | 1.0440 |
| 6 | 16 | 9 | 1.4358 | 1.4183 | UNDER | 1.5879 | BEAT | 1.4112 | UNDER | 1.1161 |
| 6 | 16 | 12 | 1.8572 | 1.8389 | UNDER | 2.0854 | BEAT | 1.7653 | UNDER | 1.2279 |
| 6 | 16 | 15 | 2.5593 | 2.5791 | BEAT | 2.9127 | BEAT | 2.3853 | UNDER | 1.4843 |
| 6 | 32 | 6 | 1.0266 | 1.0440 | BEAT | 1.0617 | BEAT | 1.0337 | BEAT | 1.0440 |
| 6 | 32 | 9 | 1.0921 | 1.1161 | BEAT | 1.1626 | BEAT | 1.0969 | BEAT | 1.1161 |
| 6 | 32 | 12 | 1.2059 | 1.2279 | BEAT | 1.3083 | BEAT | 1.1877 | UNDER | 1.2279 |
| 6 | 32 | 15 | 1.4058 | 1.4843 | BEAT | 1.6064 | BEAT | 1.3584 | UNDER | 1.4843 |
| 6 | 48 | 6 | 1.0098 | — | — | — | — | 1.0162 | BEAT | 1.0440 |
| 6 | 48 | 9 | 1.0349 | — | — | — | — | 1.0425 | BEAT | 1.1161 |
| 6 | 48 | 12 | 1.0887 | — | — | — | — | 1.0843 | UNDER | 1.2279 |
| 6 | 48 | 15 | 1.2111 | — | — | — | — | 1.1576 | UNDER | 1.4843 |

## Verdict

**needs_edit**

Rule: kill if Q=6 L16 n15 is worse than the free L16 champ by >0.5%, or if neither L32 nor L48 beats the free L32 champ by >1% Med at n15; ship only if both depths clear that 1% bar, both beat Fourier QAOA at n15, and Q=6 L16 stays within 0.5% of the champ on every CORE n; otherwise needs_edit

- Gate A (Q=6 L16 n15 within 0.5% of champ): True
- Gate A all CORE: False rel={'6': -0.0100497911649945, '9': -0.012343478128763765, '12': -0.009979674251255401, '15': 0.0076563684434337274}
- Gate B L32 n15 drop vs L32 champ: 0.0529 pass=True
- Gate B L48 n15 drop vs L32 champ: 0.1840 pass=True
- Gate C beats Fourier QAOA n15: L32=False L48=False

