## Methods, code and deviations (8-SAT)

Code: `oct_explore/clean16_8sat/` (`common.py`, `qaoa.py`, `gcost.py`, `wsat.py`, `paper_angles.py`, `report.py`); raw per-formula p in `oct_explore/results/clean16_8sat/*.json`, logs in `oct_explore/logs/clean16_8sat/`.

- **Instance filter (checked against the paper text).** Paper §4.2: "sample up to 10000 instances from CNF(n,k,r) ... and retain only satisfiable instances"; Def. 1: m ~ Poisson(rn), k literals uniformly *with replacement*; Table 1: r(8) = 176.54. We do exactly this for **all** train/val/test sets: keep a formula iff its exact energy table has min 0 (≥1 solution); **no** unique-solution filter (typical nsol 1–3). Tautological clauses (x ∨ ¬x) are never violated; repeated literals collapse.
- **QAOA L4** = clean16 `qaoa.py` (fixed.py `--proj full --mixer gray`), only change: nm = 1/r instead of n/m (m now varies per formula). 1000 train formulas per n, L-BFGS 12 starts, 3 seeds.
- **Global cost L4** = clean16 `gcost.py` (G16_full_w recipe, x0 = fresh QAOA winner, from scratch). Changes, identical math: clauses zero-padded to a fixed M(n) = ceil(rn + 8√(rn)) for batching; the dense (2^n,B,m,k) literal tensor (infeasible at m≈3500, k=8) is replaced by exact sparse (clause, violating-string) lists with segment_sum (checked: bincount of the list == energy table for every formula); E_w = nm·(g_l·E + Σ_j w_jl viol_j).
- **Native ckpt_110: not run.** The OpenEvolve program and the hash-locked frozen circuit hard-code 4-literal clauses (V/SG (F,m,4), 15 literal subsets, (m,4) weight tables). Running it at k=8 would require rewriting the program, i.e. it would no longer be "what OpenEvolve found".
- **WalkSATlm** = `qaoa_ksat/walksatlm_c/wslm.c` k=8, p=0.30, w1=6, w2=5, n-flip tries with restarts, 500 runs/formula; lookups = tries + flips; the paper's unit (tries) is also reported.
- **Reference rows**: the paper's published fixed angles (PhaseCraft qaoa_ksat_paper_data, key (8,176.54)), evaluated untrained on the same test formulas with exp(+iγE/2), RX(β).
- Ideal one-bit-flip mixer (= qubit RX) for every quantum method.
