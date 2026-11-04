# Idea1 — Walsh-degree-graded γ (graded H_c)

Arm label: **graded H_c**. Never “non-H_c”. No projector gadgets.

**Formula.** \(U_c^X = W\,\exp(-i\sum_{w=1}^{4}\gamma_{\ell,w} E_w)\,W\), with \(E_w\) the degree-\(w\) Walsh component of unsat \(E\) (FWHT → popcount mask → iFWHT), same `cost_Cn` \(n/m\) scale. \(w=0\) is a global phase. Equal \(\gamma_w\) recovers \(\gamma\cdot E\).

**Lock.** Motif Nov1st X-sandwich + Hamming-shell SNAP \(\varphi_k=c_1 k+c_2 k^2+c_3 k^3\). \(W=H^{\otimes 3}\) BinH. Train **only** \(n\in\{6,9\}\). Eval CORE \(\{6,9,12,15\}\).

**Smoke.** B=100 train + B=100 CORE (not full scale1000). \(R=\mathrm{Med}/\mathrm{Med}_{QAOA}\) on the **same B samples**. BEAT iff \(R\le 0.98\) on all four CORE sizes. Published L=2 B=1000 knee ≈0.9858; rank vs same-sample plain-γ champ first. B=1000 confirm only if B=100 clearly beats that.

This agent owns **Idea1 only**. Idea2 (cavity-locality) and Idea3 (clause features) are out of scope.

## Sanity (equal \(\gamma_w\) recovers \(\gamma\cdot E\))

`Nov_3rd/scripts/test_ucx_bit_a_walsh.py`

| check | rel err | bar |
|-------|---------|-----|
| FWHT involution | 4.4e-16 | — |
| 4-SAT Walsh recon \(\sum_w E_w+\mathrm{mean}=E\) | 0 | 1e-5 |
| equal-γ cost-grid vs `cost_Cn` | 1.4e-8 | 1e-5 |
| equal-γ `forward_nov3rd_graded` vs `forward_nov3rd` inv_p | 2.5e-7 | 1e-5 |

**PASS.**

## Same-sample plain-γ baseline (B=100)

Warm: L=2 champ `params_L2_train69_pareto_a080_s1.pt` (single \(\gamma_\ell\), shell \(c_{1,2,3}\)).

| n | QAOA Med (B=100) | Med | R | verdict |
|---|--------------|-----|---|---------|
| 6 | 4.9980 | 4.7700 | 0.9544 | BEAT |
| 9 | 11.1948 | 10.8775 | 0.9717 | BEAT |
| 12 | 33.1726 | 32.4195 | 0.9773 | BEAT |
| 15 | 71.0018 | 71.9662 | 1.0136 | TIE |

Baseline knee \(\max(R_{12},R_{15})\) = **1.0136**. This B=100 n15 draw is noisier than the published B=1000 knee 0.9858 (QAOA Med here 71.00 vs pub 76.09). Rank Idea1 against **this** baseline, not the B=1000 number.

## Idea1 seeds (L=2, champ-warm, all \(\gamma_w\leftarrow\gamma_{\mathrm{champ}}\))

Free \(\gamma_{\ell,w}\) (4/layer) + \(c_1,c_2,c_3\). No `power_p`. Adam 300 + LBFGS 8, \(w_6=1,w_9=2\), n9 holdout 20%.

| seed | n6 R | n9 R | n12 R | n15 R | knee | ALL_CORE | γ rms-dev | holdout logR |
|------|------|------|-------|-------|------|----------|-----------|--------------|
| 0 | 0.9326 | 0.9609 | 0.9517 | 1.0153 | 1.0153 | no | 0.2276 | −0.0771 |
| **1** | 0.9437 | 0.9535 | **0.9432** | **1.0035** | **1.0035** | no | 0.2281 | −0.0521 |
| 2 | 0.9218 | 0.9707 | 0.9500 | 1.0207 | 1.0207 | no | 0.2317 | −0.0438 |

Best seed **1** vs same-sample baseline: n6/9/12 all better (n12 0.977→0.943), n15 1.0136→1.0035 (still TIE, not ≤0.98).

### Seed 1 CORE table

| n | QAOA Med (B) | Med | R | R vs pub1000 | verdict |
|---|--------------|-----|---|---------------|---------|
| 6 | 4.9980 | 4.7166 | 0.9437 | 0.8856 | BEAT |
| 9 | 11.1948 | 10.6739 | 0.9535 | 0.8443 | BEAT |
| 12 | 33.1726 | 31.2892 | 0.9432 | 1.0568 | BEAT |
| 15 | 71.0018 | 71.2488 | 1.0035 | 0.9364 | TIE |

### \(\gamma_w\) (seed 1) — extra freedom **is used** (not collapsed)

| layer | \(\gamma_{w=1}\) | \(\gamma_{w=2}\) | \(\gamma_{w=3}\) | \(\gamma_{w=4}\) |
|-------|------------------|------------------|------------------|------------------|
| 0 | −4.434 | −4.406 | −3.968 | −3.924 |
| 1 | −6.601 | −6.850 | −7.158 | −7.085 |

Champ \(\gamma\approx(-4.213,-6.898)\). Layer 0 puts more weight on \(E_1,E_2\); layer 1 on \(E_3,E_4\). rms-dev from equal ≈0.23 on every seed. Opus “γ_w drift back to equal → kill” does **not** fire.

## Verdict

| question | answer |
|----------|--------|
| ALL_CORE \(R\le 0.98\)? | **no** (0/3 seeds; n15 never BEAT) |
| Pass (Opus L=2 near-miss)? | **no** |
| Kill? | **yes** for L=2 ALL_CORE / keep |
| Keep? | **no** as an L=2 BEAT arm |
| B=1000 confirm? | **no** — n15 never ≤0.98 on B=100 |

**Why kill.** Train holdout and n6/9/12 R improve, but n15 stays ≥1.003 (Opus: kill if n12/15 no better than ~0.986 **on the published CORE**; here even the generous B=100 n15 bar 0.98 is missed). Matches the named failure mode: once \(\gamma_w\) differ, \(E_1\) can act as a cheap local field that helps small \(n\) and does not carry to n15.

**What is real.** Walsh split is correct (equal-γ recovery). The extra 3 params/layer are used. n12 BEAT on this B=100 sample is a real lift vs champ. That is not enough to keep the arm under the ALL_CORE gate.

Artifacts: `Nov_3rd/runs/beat_qaoa_12h/train69/ucx_bit_a/idea1_walsh/` (`params_L2_seed*.pt`, `eval_L2_seed*.json`). Code: `Nov_3rd/graded_cost.py`, `Nov_3rd/ansatz.py::forward_nov3rd_graded`, `Nov_3rd/scripts/train_ucx_bit_a.py`.
