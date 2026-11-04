# FM-WS — shared Fourier modes + locked n-scaling (Opus Idea 3)

Hamming-shell SNAP + Walsh-graded \(U_c^X\), aimed at deep \(L\). Train only on
\(n\in\{6,9\}\). Eval CORE is \(\{6,9,12,15\}\), B=1000, fresh250 packs
(disjoint from scale1000).

Code: `Nov_3rd/fm_ws.py`, trainers `scripts/train_fm_ws.py` and
`scripts/train_qaoa_fourier.py`, smoke `scripts/run_fm_ws_smoke.py`.
Math gates: `python -m Nov_3rd.tests.test_fm_ws`.

## Formulas

Layer \(\ell\) and mode \(q\) are 1-based. \(Q\) is the number of kept modes.

\[
u_q(\ell)=\sin\!\big[(q-\tfrac12)(\ell-\tfrac12)\pi/L\big],\qquad
v_q(\ell)=\cos\!\big[(q-\tfrac12)(\ell-\tfrac12)\pi/L\big].
\]

\(u_q\) is a DST-IV column and \(v_q\) is a DCT-IV column. The matrix
\(M\) of either family at \(Q=L\) obeys \(M^\top M=(L/2)I\), so the analysis
\(\mathrm{coeff}=(2/L)\,M\,\mathrm{schedule}\) inverts `synthesize` exactly.

\[
s_n=\frac{8}{n-1}\quad(s_9=1),\qquad p_1=0,\qquad p_2,p_3,\lambda\ \text{free}.
\]

Walsh degrees \(d=1,2,3\) of the normalized cost (graded \(H_c\), labeled as such):

\[
\gamma_{\ell,d}(n)=s_n^{\lambda p_d}\sum_{q=1}^{Q} A_{q,d}\,u_q(\ell).
\]

Shell polynomial, then \(\phi_k=c_1 k+c_2 k^2+c_3 k^3\):

\[
c_{\ell,k}(n)=s_n^{p_k}\sum_{q=1}^{Q} B_{q,k}\,v_q(\ell),\qquad k=1,2,3.
\]

The ansatz still wants a length-4 Walsh vector. This cost is random 4-SAT
(\(m=\mathrm{round}(9.93\,n)\)), so the Walsh degree is at most 4 and degrees
≥ 5 are identically zero. Opus writes \(d=1..3\). The degree-4 weight is tied
to degree 1, \(\gamma_{\ell,4}=\gamma_{\ell,1}\), and is not an extra
parameter. With that tie, equal \(A_{q,d}\) and \(\lambda=0\) reproduce scalar
\(\gamma\cdot\mathrm{cost}_{C_n}\) exactly (the degree-0 piece is a global
phase). Dropping degree 4 would not recover the champ: it is a real fraction
of \(H_c\).

## How the two sides are coupled

* Same \(Q\) and the same INTERP rule.
* The exponents \(p_d\) are shared. SNAP uses \(p_k\). The cost side uses
  \(\lambda p_d\), so the cost/mixer ratio drifts with \(n\) on both sides
  together. \(p_1=0\) keeps the linear SNAP term and the degree-1 Walsh weight
  independent of \(n\), which is the power_p champ (that champ scales only
  \(c_2,c_3\)).

Parameter count: \(6Q+p_2+p_3+\lambda\), independent of \(L\).

## Warm start and INTERP

Warm start at the champ depth (\(L=16\) power_p, fresh250 seed0):

* \(A_{q,d}\) = DST of the champ \(\gamma\), the same coefficients for every
  degree, then truncated to \(Q\).
* \(B_{q,1},B_{q,2},B_{q,3}\) = DCT of unscaled \(c_1,c_2,c_3\).
* \(\lambda=0\), \(p_2=p_3=p_{\mathrm{champ}}\).

At \(Q=L\) this is the champ: \(\lambda=0\) kills the \(n\)-dependence of
\(\gamma\), tied \(A_{\cdot,d}\) makes every Walsh degree share that \(\gamma\),
and one \(p\) scales \(c_2\) and \(c_3\) exactly as power_p. Truncating \(Q\) is
the orthogonal projection onto the lowest modes; training then moves those
coefficients.

INTERP (pad): keep \((A,B,p_2,p_3,\lambda)\) and rebuild \(u_q,v_q\) at the new
depth. Modes \(q>Q\) stay zero. This is Zhou-style Fourier interpolation, not a
linear resample of the layer schedule. The smoke **polishes** those
coefficients at the target depth (same loss, same Adam+LBFGS budget) so the
depth change is an initialization, not a zero-shot stretch.

Fourier QAOA uses the same modes and the same pad:

\[
\beta_\ell=\sum_{q=1}^{Q} B_q\,v_q(\ell),\qquad
\gamma_\ell=\sum_{q=1}^{Q} A_q\,u_q(\ell).
\]

It is warmed from the DST/DCT of the free L=16 fresh250 QAOA, then INTERP-polished
at 32 and 48. No Walsh grading and no \(s_n\) — it is qubit QAOA, included so a
Med gap is not just “Fourier was easier to optimize.”

## Train / eval lock

| knob | value |
|---|---|
| packs | fresh250 n=6 and n=9, disjoint from scale1000 |
| batch | 32+32 |
| FM-WS loss | mean \(\log(\mathrm{inv}/\mathrm{inv}_{QAOA})\), \(w_6=1\), \(w_9=2\), 20% n9 holdout |
| FM-WS step budget | Adam 600 @ \(10^{-3}\), then LBFGS 12×8, same as power_p |
| exponent box | \(p\in[-1,2]\), \(\lambda\in[-3,3]\), plus \(10^{-2}(p_2^2+p_3^2+\lambda^2)\) |
| Fourier QAOA loss | \(\tfrac12(\overline{-\log p_6}+\overline{-\log p_9})\), the fresh250 QAOA loss |
| Fourier QAOA steps | same Adam/LBFGS round counts; lr \(0.05\) (QAOA angle scale) |
| eval | scale1000 CORE B=1000 |
| champ reference | `train69_ladder_fresh250/power_p` L=16 and L=32 seed0 |

Free QAOA angles exist at L=16 and L=32 only. L=48 has no free champ and no free
QAOA; the kill reference at L=48 is the free **L=32** champ.

## Kill rule (Opus)

Judged on Q=6, raw Med, lower is better.

* **Kill** if Q=6 at L=16 is worse than the free L=16 champ by more than 0.5%
  Med at n=15, **or** if neither L=32 nor L=48 beats the free L=32 champ by
  more than 1% Med at n=15.
* **Ship** only if both deep depths clear that 1% bar, both beat same-depth
  Fourier QAOA at n=15, and Q=6 at L=16 stays within 0.5% of the champ on
  every CORE \(n\).
* **Needs edit** otherwise (one depth only, Q=4 instead of Q=6, or Fourier
  QAOA picks up the same gain).

Q=4 is reported and does not by itself flip a kill into a ship.

## Results

<!-- RESULTS_BEGIN -->

Smoke results are written here by `scripts/run_fm_ws_smoke.py` when CORE evals finish.
See `runs/beat_qaoa_12h/fm_ws/RESULTS.md`.

<!-- RESULTS_END -->
