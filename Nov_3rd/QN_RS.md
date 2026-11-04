# QN-RS — SNAP residual on matched Fourier QAOA

Opus rank 1 after the joint-idea round. Hamming-shell SNAP, ungraded \(U_c^X = WDW\),
\(W = H^{\otimes 3}\). Train only \(n\in\{6,9\}\). CORE is \(\{6,9,12,15\}\),
scale1000 B=1000.

Code: `Nov_3rd/qn_rs.py`, trainer `scripts/train_qn_rs.py`, smoke
`scripts/run_qn_rs_smoke.py`. Math gates: `python -m Nov_3rd.tests.test_qn_rs`
(or `python Nov_3rd/tests/test_qn_rs.py`).

## Formulas

Layer \(\ell=1..L\), mode \(q=1..Q\), \(t_\ell=(\ell-\tfrac12)/L\).

\[
\gamma_\ell=\sum_{q=1}^{Q} u_q\sin\!\big((q-\tfrac12)\pi t_\ell\big),\qquad
\beta_\ell=\sum_{q=1}^{Q} v_q\cos\!\big((q-\tfrac12)\pi t_\ell\big)
\]

\[
r2_\ell=\sum_{q=1}^{Q'} s_q\cos\!\big((q-\tfrac12)\pi t_\ell\big),\qquad
r3_\ell=\sum_{q=1}^{Q'} w_q\cos\!\big((q-\tfrac12)\pi t_\ell\big)
\]

\[
a(n)=\Big(\frac{8}{n-1}\Big)^p,\qquad
\varphi_{\ell,k}=-\beta_\ell\,k+a(n)\,(r2_\ell\,k^2+r3_\ell\,k^3)
\]

\(U_c^X=W\exp(-i\gamma_\ell f)W\) with the same normalized ungraded cost the
qubit Fourier QAOA uses. \(\gamma\) is the DST-IV schedule and \(\beta,r2,r3\)
are DCT-IV schedules, the same modes as the matched baseline.

\((u,v)\) is initialized to that Fourier-QAOA solution (\(Q=6\)) and \(s=w=0\).
Then \(\varphi_{\ell,k}=-\beta_\ell k\), which is the QAOA transverse mixer, so
the circuit is Fourier QAOA. \(p\) does not move the circuit until \(r2\) or
\(r3\) leaves zero. \(Q=Q'=6\) is 25 parameters.

## Smoke

* Gate 0: \(s=w=0\) reproduces Fourier-QAOA Med at n=15 (2.385 at L=16) within \(10^{-3}\).
* Phase A: freeze \(u,v\); train \(s,w,p\) (Adam 600, then L-BFGS 12×8).
* Phase B: unfreeze all.
* Pass (one depth): on ≥2/3 seeds, n=12 and n=15 Med are below Fourier QAOA and
  the paired bootstrap 95% CI of the median gap excludes 0, and
  \(\lVert a(15)\,(r2,r3)\rVert\ge 0.1\,\lVert\beta\rVert\).
* Kill: n=15 is no better than the Fourier QAOA median, or \(r2,r3\) collapse.
* L=32 (against 1.358) runs only after an L=16 pass. Ship requires both depths.

The loss is the mean log ratio against the same Fourier QAOA angles
(\(w_6=1\), \(w_9=2\), 20% n=9 holdout). There is no penalty on \(r2,r3\):
shrinking them is the failure this smoke is built to catch.

## Results

<!-- RESULTS_BEGIN -->

# QN-RS smoke — SNAP residual on matched Fourier QAOA

Lower Med is better (shots to solution). Train only {6, 9}. CORE is scale1000 B=1000, n∈{6, 9, 12, 15}.

Circuit: ungraded Fourier γ_ℓ = Σ u_q sin((q−½)π t_ℓ), t_ℓ=(ℓ−½)/L;
φ_{ℓ,k} = −β_ℓ k + a(n)(r2_ℓ k² + r3_ℓ k³), β and (r2, r3) on the cosine modes,
a(n)=(8/(n−1))^p. Init (u, v) = Fourier QAOA Q=6, s=w=0.

Phase A freezes (u, v) and trains (s, w, p). Phase B unfreezes all 25 parameters.
Adam 600 @ 1e-3 then L-BFGS 12×8, each phase. Loss is mean log(inv/inv_FourierQAOA), w6=1, w9=2.
No residual penalty. Checkpoint is the best n9 holdout (20%).

Published Fourier QAOA n15 Med: L=16 **2.3853**, L=32 **1.3584**.

## Gate 0

**PASS**. s=w=0 n15 Med 2.385327 vs published 2.385319 (|Δ|=8.160e-06, tol 0.001). Paired Fourier Med 2.385319. Median |instance diff| 8.345e-06.

| n | SNAP s=w=0 Med | paired Fourier Med | published |
|---|-----------------|--------------------|-----------|
| 6 | 1.1471 | 1.1471 | 1.1471 |
| 9 | 1.4112 | 1.4112 | 1.4112 |
| 12 | 1.7654 | 1.7653 | 1.7653 |
| 15 | 2.3853 | 2.3853 | 2.3853 |

## CORE

| L | seed | n | QN-RS Med | Fourier Med | Δ Med | 95% CI | CI excludes 0 | residual ratio | p |
|---|------|---|-----------|-------------|-------|--------|---------------|----------------|---|
| 16 | 0 | 6 | 1.1298 | 1.1471 | -0.0173 | [-0.0257, -0.0091] | True | 0.0887 | 0.1012 |
| 16 | 0 | 9 | 1.3669 | 1.4112 | -0.0443 | [-0.0613, -0.0135] | True | 0.0887 | 0.1012 |
| 16 | 0 | 12 | 1.7395 | 1.7653 | -0.0258 | [-0.0529, +0.0093] | False | 0.0887 | 0.1012 |
| 16 | 0 | 15 | 2.3839 | 2.3853 | -0.0014 | [-0.0625, +0.0620] | False | 0.0887 | 0.1012 |
| 16 | 1 | 6 | 1.1271 | 1.1471 | -0.0200 | [-0.0293, -0.0123] | True | 0.0924 | 0.3718 |
| 16 | 1 | 9 | 1.3603 | 1.4112 | -0.0509 | [-0.0666, -0.0230] | True | 0.0924 | 0.3718 |
| 16 | 1 | 12 | 1.7387 | 1.7653 | -0.0266 | [-0.0533, +0.0030] | False | 0.0924 | 0.3718 |
| 16 | 1 | 15 | 2.3736 | 2.3853 | -0.0118 | [-0.0711, +0.0405] | False | 0.0924 | 0.3718 |
| 16 | 2 | 6 | 1.1209 | 1.1471 | -0.0263 | [-0.0348, -0.0195] | True | 0.0922 | 0.2630 |
| 16 | 2 | 9 | 1.3667 | 1.4112 | -0.0444 | [-0.0593, -0.0187] | True | 0.0922 | 0.2630 |
| 16 | 2 | 12 | 1.7609 | 1.7653 | -0.0044 | [-0.0351, +0.0232] | False | 0.0922 | 0.2630 |
| 16 | 2 | 15 | 2.3909 | 2.3853 | +0.0056 | [-0.0406, +0.0835] | False | 0.0922 | 0.2630 |

## Verdict

**needs_edit**

- L=16: **needs_edit** (0/3 seeds). pass if ≥2/3 seeds have n12 and n15 Med below Fourier QAOA with paired bootstrap 95% CI of (Med−Med_FQ) excluding 0 on the better side, and ‖a(15)·(r2,r3)‖/‖β‖ ≥ 0.10; kill if n15 is no better than Fourier QAOA on every seed, or every residual ratio is < 0.02; otherwise needs_edit
  - seed_pass=[False, False, False] n15_better=[True, True, False] ratios=['0.0887', '0.0924', '0.0922']
- L=32: not run (L=16 did not pass)

A seed passes only when n12 and n15 are both better than Fourier QAOA and each paired bootstrap 95% CI lies entirely below 0, and ‖a(15)·(r2, r3)‖ / ‖β‖ ≥ 0.10. Ship requires that bar at L=16 and at L=32. Kill is an L=16 failure (n15 never beats 2.385, or the residual collapses on every seed). Needs-edit is a split: one seed, a residual under 10%, or an L=16 pass that does not repeat at L=32.

## Reading

Gate 0 passed. With s=w=0 the Hamming-shell circuit reproduces Fourier QAOA: n15 Med 2.385327 versus the published 2.385319.
L=16 is **needs_edit** (0/3 seeds). n6 and n9 beat Fourier QAOA on every seed, with the paired CI entirely below 0. n12 and n15 do not: every CI includes 0.
- seed 0: n15 2.3839 (Δ -0.0014, CI [-0.0625, +0.0620]), n12 Δ -0.0258, p=0.101, residual ratio=0.0887, checkpoint B-lbfgs1
- seed 1: n15 2.3736 (Δ -0.0118, CI [-0.0711, +0.0405]), n12 Δ -0.0266, p=0.372, residual ratio=0.0924, checkpoint B-adam525
- seed 2: n15 2.3909 (Δ +0.0056, CI [-0.0406, +0.0835]), n12 Δ -0.0044, p=0.263, residual ratio=0.0922, checkpoint B-adam375
The residual did not collapse and it did not clear the bar. ‖a(15)·(r2, r3)‖/‖β‖ is 0.0887, 0.0924, 0.0922, all under 0.10 and all above the 0.02 collapse line. p stayed in (0.10, 0.38), so a(n) is a mild mixer-side rescaling. Each reported checkpoint is that seed's phase-B n9-holdout best.
Overall **needs_edit**. L=32 was not run. The SNAP-only terms learn a gain on the sizes they were trained on. They leave n15 at 2.385 inside the bootstrap interval. The matched Fourier QAOA angles are the baseline, so that gap is the residual. A follow-up would need n12 and n15 CIs entirely below 0 on at least two seeds, with the residual ratio still at least 0.10.


<!-- RESULTS_END -->
