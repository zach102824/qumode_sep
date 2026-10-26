# Classical algorithm head-to-head on 4-SAT TEB (n=6, 8, 10)

Date: 2026-10-26 (Asia/Shanghai). Exploration STOP left in place (no Part2/WAVE14).

## Question

On **our** random 4-SAT TEB test set, is WalkSATlm still the strongest classical baseline among a PRX Quantum–style pack?

## Protocol

- **Instances:** `sim.instances(n, 100, TEB=52e9)` — same Part 1/2 test formulas.
  - 4-SAT, `m = round(9.93 n)`, satisfiable only.
  - Seeds n6: `52060000000…`; n8: `52080000000…`; n10: `52100000000…`.
  - DIMACS export: `oct_explore/results/classical_4sat_head/cnf/`.
- **SLS cost units (both):**
  1. **Tries** (paper): `max_flips = n`, fresh random start every try; count tries until SAT.
  2. **Lookups:** bitstrings checked = starts + flips.
- **Stats:** per formula, mean over runs; then **median / B90** over 100 formulas.
- **WalkSATlm:** audited `qaoa_ksat/walksatlm_c/wslm`, 500 until-SAT runs/formula, `w1=6, w2=5`.
  - Primary: `p=0.30` (Part-matched). Also a TEB p-grid for best-p reference.
- **UBCSAT:** `dtompkins/ubcsat` (LP64 `%lu`/`%ld` patch), binary in `classical_4sat_head/bin/ubcsat`.
  - Algs: walksat (SKC), novelty+, adaptnovelty+, saps, rsaps.
  - Each formula: 2000 independent tries at `cutoff=n`; mean tries = `1/p_success`;
    mean lookups = `((1-p)/p)*(1+E[steps|fail]) + (1+E[steps|ok])`.
  - Light VAB tune: walksat `-wp` ∈ {0.3,0.4,0.5,0.6} → **0.3**; novelty+ `-novnoise` ∈ {0.3,0.5,0.7} → **0.7**.
- **CDCL:** PySAT Glucose4 and Cadical195 (one solve/formula).
  - Report conflicts / decisions / propagations. **Not** the same unit as SLS tries/lookups.

## SLS results (tries and lookups Med / B90)

| Algorithm | n=6 tries | n=6 lookups | n=8 tries | n=8 lookups | n=10 tries | n=10 lookups |
|---|---|---|---|---|---|---|
| WalkSATlm p=0.30 (Part-matched) | 1.691 / 2.816 | 8.466 / 16.039 | 2.014 / 3.739 | 13.989 / 29.396 | 2.481 / 4.950 | 22.915 / 49.657 |
| UBCSAT walksat (wp=0.3) | 1.954 / 3.006 | 11.361 / 18.361 | 2.229 / 4.265 | 16.859 / 34.805 | 3.021 / 5.335 | 29.071 / 54.749 |
| UBCSAT novelty+ (novnoise=0.7) | 1.750 / 2.756 | 10.004 / 16.555 | 1.827 / 3.562 | 13.248 / 28.412 | 2.223 / 4.116 | 20.144 / 41.871 |
| UBCSAT adaptnovelty+ (defaults) | 1.775 / 2.993 | 10.070 / 18.014 | 2.088 / 4.258 | 15.188 / 34.603 | 2.433 / 4.351 | 22.605 / 43.180 |
| UBCSAT saps (defaults) | 2.136 / 3.561 | 12.303 / 21.552 | 2.423 / 5.105 | 18.302 / 42.383 | 2.997 / 5.621 | 27.769 / 56.365 |
| UBCSAT rsaps (defaults) | 2.127 / 3.545 | 12.169 / 21.222 | 2.523 / 5.038 | 18.695 / 41.986 | 3.014 / 5.400 | 28.688 / 54.448 |
| WalkSATlm best-p on this TEB | 1.654 / 2.800 (p=0.35) | 8.422 / 16.001 | 1.963 / 3.880 (p=0.15) | 13.254 / 30.232 | 2.406 / 4.580 (p=0.15) | 21.238 / 45.646 |

### Part-1 sanity check

WalkSATlm `p=0.30` reproduces Part 1 exactly:

- n=6: tries **1.691 / 2.816**, lookups **8.466 / 16.039** (Part1: 1.691 / 2.816 · 8.466 / 16.039)
- n=8: tries **2.014 / 3.739**, lookups **13.989 / 29.396** (Part1: 2.014 / 3.739 · ~14 / 29)
- n=10 (new): tries **2.481 / 4.950**, lookups **22.915 / 49.657**

## CDCL results (different unit — conflicts / decisions Med / B90)

| Solver | n | conflicts | decisions | propagations | wall ms |
|---|---|---|---|---|---|
| CDCL glucose4 | 6 | 3.000 / 7.000 | 7.000 / 12.000 | 13.000 / 25.100 | 0.033 / 0.039 |
| CDCL glucose4 | 8 | 5.000 / 11.000 | 10.000 / 16.000 | 23.500 / 46.000 | 0.041 / 0.048 |
| CDCL glucose4 | 10 | 6.000 / 16.000 | 11.000 / 21.100 | 33.000 / 64.400 | 0.047 / 0.062 |
| CDCL cadical | 6 | 0.000 / 10.000 | 0.000 / 8.000 | 0.000 / 23.000 | 0.047 / 0.061 |
| CDCL cadical | 8 | 1.000 / 12.000 | 4.000 / 12.000 | 10.500 / 34.500 | 0.059 / 0.072 |
| CDCL cadical | 10 | 4.000 / 16.000 | 7.000 / 17.000 | 20.500 / 57.100 | 0.072 / 0.088 |

All formulas SAT under both CDCL solvers. At n≤10 these instances are tiny for CDCL:
median conflicts are single-digit, so CDCL “looks” cheaper numerically than SLS lookups,
but **conflicts ≠ bitstring evaluations**. For the QAOA-shot / lookup comparison used in Part 1/2,
SLS lookups (or paper tries) remain the relevant classical baseline unit.

## Verdict

### Best SLS on our 4-SAT TEB (tries + lookups)

| n | Best tries (Med/B90) | Best lookups (Med/B90) | WalkSATlm p=0.30 still best? |
|---|---|---|---|
| 6 | WalkSATlm best-p=0.35 **1.654/2.800** (p=0.30: 1.691/2.816) | WalkSATlm p=0.30 **8.466/16.039** | **Yes** (narrowly); Novelty+ close on tries B90 |
| 8 | **UBCSAT Novelty+ (0.7) 1.827/3.562** | **Novelty+ 13.248/28.412** | **No** — Novelty+ beats WalkSATlm tries and lookups |
| 10 | **UBCSAT Novelty+ (0.7) 2.223/4.116** | **Novelty+ 20.144/41.871** | **No** — Novelty+ beats WalkSATlm (even vs best-p=0.15: 2.406/4.580) |

**Overall (n=6,8,10):** WalkSATlm is **not** uniquely best on our fixed-m 4-SAT TEB set.
- At **n=6**, WalkSATlm remains the strongest (lookups clear; tries tied/slightly ahead).
- At **n=8 and n=10**, **UBCSAT Novelty+ (`-novnoise 0.7`)** is the strongest SLS baseline on both tries and lookups.
- SAPS / RSAPS / plain WalkSAT-SKC are clearly behind.
- Adaptive Novelty+ is second tier (between Novelty+ and WalkSATlm at n=10).

**Recommended Part1/2 classical baseline update:** keep reporting WalkSATlm (paper protocol continuity),
and add **Novelty+** as the stronger package SLS reference at n≥8. For quantum-vs-classical claims on
lookups, Novelty+ is the harder classical number to beat at n=8,10.

### Does this contradict PRX Quantum 5, 030348 (2024)?

No direct contradiction. That paper’s WalkSATlm ranking was on **threshold random k-SAT (esp. 8-SAT)**
with Poisson m, among the SLS/CDCL pack they tried. Our ensemble is **4-SAT, fixed m≈9.93n**.
Same problem family, different k / clause law — ranking can (and here does) move.

## Artifacts

- Runner: `oct_explore/results/classical_4sat_head/run_head.py`
- CNFs: `oct_explore/results/classical_4sat_head/cnf/`
- Raw JSON: `oct_explore/results/classical_4sat_head/raw/`
- UBCSAT binary: `oct_explore/results/classical_4sat_head/bin/ubcsat` (LP64-patched build of dtompkins/ubcsat)
- Tuned knobs: `oct_explore/results/classical_4sat_head/tuned_params.json`
- This report: `oct_explore/results/CLASSICAL_4SAT_TEB_HEAD.md`
