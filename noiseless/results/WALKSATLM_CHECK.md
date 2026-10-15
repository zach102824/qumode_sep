# WalkSATlm check vs our WalkSAT / SATLike (weighted MaxSAT)

Run 2026-10-15, 15:07–15:09 CST, 8 cores, `setsid nohup`. Script `noiseless/walksatlm_check.py`, raw data `logs/walksatlm/{tune,eval}.jsonl`, configs `logs/walksatlm/best_cfg.json`.

## 1. Paper claim (PRX Quantum 5, 030348 = arXiv:2208.06909)

Boulebnane & Montanaro compare **fixed-angle QAOA** to classical SLS on **random k-SAT at the satisfiability threshold** (decision SAT: find any satisfying assignment). They report **WalkSATlm** (Cai, Luo, Su; AAAI 2013 / Computer Journal 2015) as the strongest classical solver they tested on random 8-SAT, with median formula-eval scaling ≈ 2^{0.325 n}. Their hyperparameters: `p=0.15, w1=6, w2=5`.

That claim is **not** about weighted MaxSAT / MaxCut, and **not** about our ECD+JP circuit. Our instances are **weighted 4-MaxSAT** (`w ∈ {1..5}`, plant unique GS, `m ≈ 9.93 n`).

## 2. What WalkSATlm does differently from SKC WalkSAT

WalkSAT (SKC): pick a random unsatisfied clause; if any var has `break=0`, flip one (ties random); else with noise `p` flip a random var in the clause, otherwise flip a min-break var (ties random).

**WalkSATlm = WalkSAT + linear-make tie-break only.** Among vars tied for min break (including freebies), prefer max

```
lmake(x) = w1 · make1(x) + w2 · make2(x)
```

- `make1(x)`: # of 0-true clauses that become 1-true if `x` flips
- `make2(x)`: # of 1-true clauses that become 2-true if `x` flips
- remaining ties random

Cai et al. 4-SAT defaults: `(w1,w2)=(3,1)`. No dynamic clause weighting (that is SATLike / CCASat territory).

## 3. Code audit of our baselines

| Piece | File | Finding |
|---|---|---|
| WalkSAT | `noiseless/strong_classical.py` `run_walksat` | **Faithful weighted SKC.** Uses weighted break; random ties among min-break. **No lmake / make1 / make2.** So it is WalkSAT, **not** WalkSATlm. Not a bug relative to SKC; it is simply missing the WalkSATlm ingredient. |
| SATLike-style | `run_satlike` | Dynamic clause weights + BMS + age tie-break. Faithful SATLike-3.0-*style* reimplementation (not the official C binary). Hard-clause path unused (all soft). Incremental full-neighbourhood scores undercounted under “strict” lookups (documented in `STRONG_CLASSICAL_BASELINES.md`). |
| Bugs that would flip the old ranking? | — | **No.** Old WalkSAT was correctly SKC. The gap vs SATLike is real under the same strict lookup count. |

## 4. Implementation used here

Faithful WalkSATlm reimplementation for **weighted** MaxSAT (`noiseless/walksatlm_check.py`):

- Same SKC skeleton as `run_walksat`
- Weighted `break`, `make1`, `make2` (clause weights)
- Tie-break by max `lmake = w1·make1 + w2·make2`
- Lookup count identical to WalkSAT: **4 per step** (score all 4 vars of the picked unsat clause); hit = first time the GS string is scored; fresh random start = 1
- Tuned `(p, maxf, w1, w2)` on held-out unique-GS instances (same generator as strong baselines); WalkSAT / SATLike configs reused from `logs/strong/best_cfg.json`
- Official WalkSATlm binary not used: it targets decision SAT CNF, not weighted MaxSAT with our lookup counter

## 5. Head-to-head (same 70 eval instances, 200 runs/n, strict lookups)

| n | WalkSAT B90 / median | WalkSATlm B90 / median | SATLike B90 / median | tuned WalkSATlm cfg |
|---|----------------------|------------------------|----------------------|---------------------|
| 8 | 270 / 50 | 270 / 62 | **161 / 33** | p=0.3, maxf=8, w1=3, w2=1 |
| 10 | 468 / 117 | 399 / 127 | **161 / 37.5** | p=0.1, maxf=10, w1=1, w2=0 |
| 12 | 811 / 214.5 | 744 / 215.5 | **333 / 94** | p=0.05, maxf=12, w1=1, w2=0 |
| 14 | 1155 / 305.5 | 927 / 265.5 | **310 / 88.5** | p=0.15, maxf=14, w1=5, w2=4 |
| 16 | 1470 / 380.5 | 1590 / 430.5 | **526 / 120.5** | p=0.15, maxf=16, w1=6, w2=5 |
| 18 | 2142 / 576.5 | 2674 / 643 | **994 / 240.5** | p=0.1, maxf=18, w1=1, w2=0 |
| 20 | 3980 / 1067.5 | 4107 / 1204.5 | **795 / 257.5** | p=0.5, maxf=80, w1=5, w2=4 |

Success at cap: ≈1.0 for all arms (one SATLike miss at n=12, succ=0.995).

Ratios (B90):

- WalkSATlm / WalkSAT: 0.80×–1.25× (helps at n=10–14; neutral/worse at n=16–20)
- WalkSATlm / SATLike: **1.68×–5.17× worse** (SATLike always better)

## 6. Verdict

**No — we did not make a mistake picking SATLike for our weighted MaxSAT baselines.**

WalkSATlm is the right stronger classical solver for the paper’s **random k-SAT** family. On **our** weighted MaxSAT instances, a faithful WalkSATlm either matches or only mildly beats SKC WalkSAT, and remains **~2–5× worse than SATLike** under the same strict lookup count. The paper’s ranking does not transfer to this problem family.
