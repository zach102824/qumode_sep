# Status: 8-SAT Hamiltonians + classical vs PRX Quantum 5, 030348

Stopped 2026-10-16 ~15:58 CST. Classical job killed while starting n=20 simple WalkSAT.

There is **no** `noiseless/results/SATLIKE_4SAT.md`. Our SATLike numbers are in
`noiseless/results/STRONG_CLASSICAL_BASELINES.md`, on **weighted MAX-4-SAT**
(`Hamiltonians/wmaxsat/`), not the paper’s random 8-SAT. Those cannot be
compared to Table II.

## Done

### 1. QAOA vs the paper (finished)

Exact statevector, published angles, paper ensemble. Writeup:
`qaoa_ksat/results/N20_CHECK.md`.

- n=12, p=60 mean success **0.558** vs paper “approximately 0.56”.
- n=8 and n=12 median TTS match Table II QAOA lines.
- n=20, p=60: our median TTS **4.3 [2.7, 5.9]** vs Table II **9.2**. The fit
  is pessimistic, not fake. QAOA works.

### 2. Hamiltonians saved (finished)

Same SAT formulas used in the QAOA evals, replayed from eval seeds.

`qaoa_ksat/hamiltonians/` (manifest: 600 files)

| n | files |
|---|---|
| 8 | 200 |
| 12 | 120 |
| 16 | 80 |
| 20 | 200 |

JSON per instance: clauses (`vars`, `signs`), `m`, `n_sat`, `sat_index`,
`eval_p`, `eval_seed`. Not the `Hamiltonians/four_sat` or `wmaxsat` NPZs.

Dump: `python -m qaoa_ksat.dump_instances`.

### 3. Classical on those 8-SAT instances (partial)

Solvers in `qaoa_ksat/classical.py`, run on **p=14 QAOA instances**, 40 per n,
3 random starts each (120 jobs). TTS = **flips** until E=0. Table II WalkSATlm
is `log2(median TTS) = -0.309 + 0.325 n` (n=12…20), quoted as “formula
evaluations”.

| n | solver | ok | median flips | median lookups | Table II WalkSATlm | flips / Table II |
|---|---|---|---|---|---|---|
| 8 | walksat_simple | 120/120 | 69 | 70 | 4.9 (extrapolated) | 14× |
| 8 | walksatlm | 120/120 | 88 | 705 | 4.9 | 18× |
| 12 | walksat_simple | 120/120 | 816 | 817 | 12.1 | 68× |
| 12 | walksatlm | 120/120 | 255 | 2037 | 12.1 | 21× |
| 16 | walksat_simple | 120/120 | 5365 | 5366 | 29.7 | 181× |
| 16 | walksatlm | 120/120 | 515 | 4117 | 29.7 | 17× |
| 20 | both | **not run** (killed) | — | — | 73 | — |

WalkSATlm here is p=0.15, w1=6, w2=5 as in the paper, but **our** Python
make/break, not their C WalkSATlm. Simple WalkSAT is the paper’s “random unsat
clause, random variable”.

**They all solve.** The gap vs Table II is a stable ~17–21× on WalkSATlm
flips, not a failure to find solutions. That is almost certainly **accounting**
(what they call a formula evaluation) and/or a stronger official WalkSATlm
implementation, not “classical does not work on these Hamiltonians”.

SATLike on 8-SAT was started then dropped: pure Python + 100k cap was too slow
(n=8 did not finish in 8+ min). Our existing SATLike results remain 4-SAT
wmaxsat only.

## Not done

1. **n=20 classical** on the saved 8-SAT set (simple WalkSAT + WalkSATlm, same
   40×3 protocol). Table II at n=20 is 73; we do not have our number.
2. **SATLike / SKC WalkSAT from `noiseless/strong_classical.py`** on these
   8-SAT files (needs general-k, not hardcoded 4-SAT). That is the actual
   “our code vs their results” check, once it is fast enough (numba is broken
   on this numpy; need a C/numba pin or a tighter inner loop).
3. **Match Table II units.** Reproduce one n=12 WalkSATlm median in *their*
   formula-evaluation counter (or run the Cai et al. binary). Until that is
   aligned, do not treat 88 vs 4.9 as “they under-reported classical cost”.
4. **No `SATLIKE_4SAT.md` to sync.** Do not mix
   `STRONG_CLASSICAL_BASELINES.md` (weighted 4-SAT, unique GS, lookups) with
   Table II (random 8-SAT, formula evals).

## How to resume

```bash
# already on disk
ls qaoa_ksat/hamiltonians/n20 | wc -l

# remaining classical (slow in current Python WalkSATlm)
python -m qaoa_ksat.run_classical --n 20 --p-filter 14 --limit 40 --runs 3 \
  --solvers walksat_simple,walksatlm --cap 50000 --workers 6
```

QAOA side does not need another run unless you want n=20 p=60 on this exact
saved 40-instance p=14 subset (those QAOA numbers were on a larger, overlapping
seed set).
