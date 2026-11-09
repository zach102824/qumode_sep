# N3INIT L=3 annealed SPSA gains

Bounded development. Original shared-Uc Nov3 only. SPSA only. Energy selection. Fidelity post hoc. No GR patch. No candidacy file. Holdout FAIL, pair-shell REJECT, shell-gain REJECT (shared-with-LE-PR101), and spread8 REJECT still stand. Do not re-fit `10368..10375`.

Not shell×gain. Not LE freeze-order. Not paper \(\mathcal{V}(M)\). Not per-cavity unshare.

Hypothesis: the v4 Spall pair \(\alpha=0.602\), \(\gamma=0.101\) stays too large late at L=3; the asymptotically optimal Spall pair \(\alpha=1\), \(\gamma=1/6\) is a smoother \(1/t\)-style anneal and may recover a Nov3 energy edge.

V4 is **already** the in-repo Spall anneal \(a_k=a_0(1+A)^\alpha/(k+1+A)^\alpha\), \(c_k=c/(k+1)^\gamma\) with \(\alpha=0.602\), \(\gamma=0.101\), \(A=25\). It is not piecewise-constant. ANNEAL only changes the exponents.

## Exact schedules

Same a0 calibration (4 two-sided pairs at \(c=0.1\)), \(A=25\), \(c=0.1\), clip \(2\sigma\), Rademacher, σ, center-4, four equal blocks, \(T=378\), inclusive \(Q=3072\). \(a_{k=0}=a_0\) on both arms.

| arm | circuit | method_id | \(\alpha\) | \(\gamma\) | \(A\) |
|---|---|---:|---:|---:|---:|
| A `V4` | shared-shell Nov3 | 1 | 0.602 | 0.101 | 25 |
| B `ANNEAL` | shared-shell Nov3 | 7 | 1 | \(1/6\) | 25 |
| C `qaoa_center4` | QAOA p=3 | 0 | 0.602 | 0.101 | 25 |

At \(T=378\), \(a_{377}/a_0 = (26/403)^\alpha\): V4 \(\approx 0.185\), ANNEAL \(\approx 0.064\). Late \(a_k\) is about \(2.9\times\) smaller. Query count does not depend on the exponents.

## Seeds

Exclusive unread band `10456..10463`, n=8. Provenance, verified before any instance or SPSA:

- Not LE `10432..10439`.
- Not local R27 `10440..10455`.
- Not local R26 `10416..10431`.
- Not `≤10415` (covers train, holdout `10144..10151`, pair-shell, shell-gain `10368..10375`, spread8 `10376..10383`).
- Not validation `11000..11031`.
- Not GR final `20000..20999`.
- No existing `Nov3_INIT/results/**` file for these seeds.

The runner refuses every other seed. `B*` is new, classical, interior entropy max on `0..floor(N/2)−1`, not a quantum query.

## Grid

New directory `results/anneal/`. N=15, L=3, `q=1024`, `Q=3072`. Three arms. Energy-only selection. Fail any cell → reject; no GR.

## Gates (prespecified, energy only, before any SPSA)

Bootstrap 4000, `default_rng(3334)`, seeds resampled.

**Primary:** ANNEAL vs QAOA mean \(\Delta E\) CI entirely `< 0`.

**Secondary:** ANNEAL mean E `≤` V4 mean E.

Fail primary or secondary, or any infeasible cell → **REJECT_ANNEAL_L3**. Pass both → **ANNEAL_L3_EDGE**. Either label: no GR patch, no candidacy file.

## Queries

Each cell ledger equals `Q=3072`. Report new queries and cumulative total `2498560 + new`. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, paper \(\mathcal{V}(M)\), re-fit of `10368..10375`, extra agents, candidacy notes, frozen-benchmark retune.
