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


<!-- RESULTS_END -->

## Reading

**Verdict: needs edit.** Not a kill, and not a ship.

Q=6 at L=16 stays inside the champ at n=15 (Med 2.559 vs 2.579, about 0.8% better), so the “Q=6 is not within 0.5%” kill does not fire. The same schedule is about 1.0–1.2% worse than the free L=16 champ at n=6, 9, and 12, so it is not a lossless copy of that champ on every CORE size. Q=4 at L=16 is coarser: n=15 Med 2.653 is about 2.9% worse than the champ.

INTERP to deeper L does clear the champ bar. Against the free L=32 power_p champ (n=15 Med 1.484), Q=6 lands at 1.406 at L=32 (5.3% lower) and 1.211 at L=48 (18.4% lower). Both are past the >1% Med line, and both beat the unrestricted fresh250 QAOA wherever that baseline exists. Q=4 clears the same 1% line too (1.443 at L=32, 1.209 at L=48) but is behind Q=6 at L=32.

The matched Fourier QAOA takes a larger share of that depth gain. At n=15 its Med is 2.385 (L=16), 1.358 (L=32), and 1.158 (L=48), all better than FM-WS at the same Q=6. So the improvement over the free L=32 champ is the smooth low-mode schedule, and qubit QAOA absorbs it. That is the failure mode Opus named.

The coupling parameters agree with that reading. For Q=6, λ goes 0.033 → 0.009 → −0.007 from L=16 to 32 to 48, so the shared n-exponent on the cost side shuts off as depth grows. The Walsh degrees do split (degree MAD stays about 0.23–0.25), and that split still loses to Fourier QAOA at n=12 and n=15.

Do not ship FM-WS as the deeper-L ansatz. A useful edit is to keep the Fourier/INTERP schedule and drop the claim that Walsh grading plus λ beats a Fourier QAOA trained the same way. A follow-up that would change this verdict is one where FM-WS beats this Fourier QAOA at n=15 with λ staying away from 0.
