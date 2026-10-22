# Paper-protocol WalkSATlm on the fair TEB test formulas (2026-10-22)

**Solver.** `qaoa_ksat/walksatlm_c/wslm.c`, the same C WalkSATlm as in
`qaoa_ksat/results/FIG3_WALKSATLM_REPRO.md`. It uses Cai's min-break / linear-make pick rule with
p=0.15, w1=6, w2=5. Each try starts from a fresh uniform random string and makes n flips. Tries
repeat until E==0, so any satisfying string counts, the same success rule as the quantum evaluator.

**Formulas.** `sim.instances(n, 100, TEB=32e9)` with `sim.gen_instance`: 4-SAT, m=round(9.93n),
satisfiable only. These are the exact sets behind QAOA L4 and net16r ckpt_110. The dump script is
`v3/classical/paper_units/dump.py`.

**Statistics.** 500 independent runs per formula, averaged per formula. Then median / B90 (90th
percentile) over the 100 formulas.

**Units:**
- tries: paper units, one try = n flips from a fresh start.
- lookups: bitstrings visited, i.e. each fresh start plus each flip.
- flips.

| n | tries (paper unit) med/B90 | lookups med/B90 | flips med/B90 | QAOA L4 1/p med/B90 | net16r ckpt_110 rr med/B90 |
|---|---|---|---|---|---|
| 12 | 2.92 / 7.33 | 32.2 / 89.1 | 29.3 / 81.8 | 12.97 / 31.76 | — |
| 16 | **4.37 / 10.62** | **67.2 / 173.0** | 62.8 / 162.4 | **34.82 / 98.20** | **82.6 / 4374.7** |
| 20 | 5.56 / 15.54 | 108.3 / 318.0 | 102.8 / 302.5 | 84.9 / 238.1 (make_json table) | — |

Worst formula (mean tries): 13.1 at n=12, 22.0 at n=16, 25.7 at n=20.

**Reading.** The units are not like for like. One quantum sample is one bitstring measured. In
lookups (bitstrings visited), classical WalkSATlm at n=16 needs 67 against QAOA L4's 34.8 and
ckpt_110's 82.6. In paper units (tries), classical looks about 8× cheaper than QAOA L4, because
each try hides n=16 flips. That is how the PRX Quantum paper counts its classical cost.
