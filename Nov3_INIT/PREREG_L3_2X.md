# L=3 doubled per-layer budget (q=2048, Q=6144)

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Entropy-peak \(B^*\) only. **Not** a closed L=3 architecture/schedule lever.

Scientific question: is L=3 simply **budget-starved** because Nov3 has more SPSA coordinates (12) than QAOA p=3 (6) at the historical `q=1024` (`Q=3072`) cap?

This trial **doubles the per-layer query cap for both methods**: `q=2048`, `Q=qL=6144`. Matching QAOA gets the same increased cap. It is a **total-budget increase**, not:

- redistributing old `Q=3072` across more starts (spread8 / S8, already `REJECT_SPREAD8_L3`)
- annealing exponents (`REJECT_ANNEAL_L3`)
- tying / freeze-order / two-stage / start-count tweaks
- pair-shell, shell-unshare, paper \(\mathcal{V}(M)\)

If this FAILs, **stop** further L=3 budget-increase retries on this stream. Do not climb to `q=4096`.

PR111 compile miss stays closed. PR112/PR113 stand. L=2 holdout FAIL stands. No GR.

## Seeds

Exclusive unread `10616..10623` (n=8). **Hold `10624..10631` unused** on this stream. Collision-checked vs:

- LE `10432..39`, LE PR115 collision band `10552..59`, LE next `10600..15`
- local R31 `10536..51`, R32 `10584..99`, R30 `10504..19`
- this stream `10520..35`, `10552..59`, `10568..83`, `10480..87`, `10456..63`, `10376..83`, `10368..75`
- unused hole `10560..67`
- holdout `10144..51`, validation `11000..11031`, GR/final `20000..20999`

Note: `10552..59` were also used by LE PR115 (L3TIE). Different experiment; both results stand; this panel does **not** reuse that band.

## Grid

`results/l3_2x/`. N=15, L=3, `q=2048`, `Q=6144` **each** method.

| arm | method | params | method_id | Q | T=Q/8−6 |
|---|---|---:|---:|---:|---:|
| `nov3_center4` | shared-Uc Nov3, untied per-layer SNAP | 12 | 1 | 6144 | 762 |
| `qaoa_center4` | QAOA p=3 | 6 | 0 | 6144 | 762 |

Exact v4 `center4`. Interleaved packing m=5. \(B^*\) interior entropy max on `0..6`.

## Label (prespecified, energy only)

Bootstrap 4000, `default_rng(3334)`, n=8 instances resampled. \(\Delta E=E_{\mathrm{Nov3}}-E_{\mathrm{QAOA}}\).

- mean \(\Delta E<0\) and CI entirely `<0\) → **L3_2XBUDGET_ESTIMATE_NEGATIVE_CI**
- complete but that fails → **L3_2XBUDGET_FAIL**
- infeasible/incomplete → **L3_2XBUDGET_UNSTABLE**

Not an adoption gate. Not a GR patch.

**Secondary (descriptive only, different seeds, not paired):** historical L=3 `q=1024` (`Q=3072`) equal4 Nov3−QAOA on this stream:

- v7 `10018..25`: mean \(\Delta E=+0.0193\), CI \([-0.045,+0.075]\)
- spread8 equal4 `10376..83`: mean \(\Delta E=-0.0315\), CI \([-0.106,+0.043]\)
- anneal V4 `10456..63`: mean \(\Delta E=-0.0477\), CI \([-0.121,+0.022]\)

Report this trial’s mean/CI next to those numbers. Do not treat them as a paired test.

## Queries

\(8\times 2\times 6144=98304\) energy queries. Historical SK **2916352**. Cumulative **3014656** if complete. Classical \(B^*\) separate. Workers 4, BLAS 1.

## Off limits

`q=4096`, L=2, extra starts, anneal, pair-shell, V(M), compiler, ε, GR, validation, final, Nov9/SYK, seeds `10624..10631`.
