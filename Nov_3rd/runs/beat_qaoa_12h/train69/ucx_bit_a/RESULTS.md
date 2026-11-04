# ucx_bit_a RESULTS — Opus Idea1 Walsh-degree-graded γ (graded H_c)

_Updated 2026-11-04 ~11:45 Asia/Shanghai (CST, UTC+8)_

**Label:** graded H_c (NOT non-Hc).  
**Formula:** \(U_c^X = W\exp(-i\sum_{w=1}^{4}\gamma_{\ell,w}E_w)W\), \(E_w=\) degree-\(w\) Walsh piece of `cost_Cn`.  
**Warm:** \(\gamma_{\ell,w}=\gamma_{\mathrm{QAOA},\ell}\) ∀w (recovers scalar champ).

## Verdict — Idea1 **PASS** (L=2 decisive)

Prior L=2 capacity knee under Hamming-shell + train{6,9}: **R≈0.9858** (pareto_blend a0.80 s1; n12/n15 TIE).  
Idea1 (smoke train B=100, CORE eval B=1000 sealed Q) → **ALL_CORE_BEAT on seeds 0–3**, best knee **0.9567** (seed1).

γ_w did **not** collapse to equal: mean |γ_w − mean_w| ≈ 0.26–0.28 all seeds.

### B=1000 CORE (sealed QAOA scale-ref Med) — smoke-trained params

| seed | n6 R | n9 R | n12 R | n15 R | knee | ALL_CORE |
|------|------|------|-------|-------|------|----------|
| 0 | 0.9343 | 0.9474 | 0.9566 | 0.9595 | 0.9595 | **True** |
| **1** | **0.9432** | **0.9467** | **0.9469** | **0.9567** | **0.9567** | **True** |
| 2 | 0.9383 | 0.9431 | 0.9528 | 0.9618 | 0.9618 | **True** |
| 3 | 0.9446 | 0.9474 | 0.9523 | 0.9599 | 0.9599 | **True** |

Best seed1 Med/R (sealed Q):

| n | QAOA Med | Med | R | verdict |
|---|----------|-----|---|---------|
| 6 | 5.3257 | 5.0231 | **0.9432** | BEAT |
| 9 | 12.6416 | 11.9681 | **0.9467** | BEAT |
| 12 | 29.6063 | 28.0352 | **0.9469** | BEAT |
| 15 | 76.0878 | 72.7966 | **0.9567** | BEAT |

Params: `walsh_degree/params_L2_seed1_B100.pt`  
(also `params_L2_best_seed1_smokeTrain100.pt`)

### Fair B=100 relative ranking (same first-100 scale slice)

| arm | n6 | n9 | n12 | n15 | knee |
|-----|----|----|-----|-----|------|
| prior champ pareto a0.80 s1 | 0.9533 | 0.9697 | 0.9764 | 1.0040 | 1.0040 |
| walsh s0 | 0.9254 | 0.9591 | 0.9464 | 1.0035 | 1.0035 |
| walsh s1 | 0.9435 | 0.9512 | 0.9427 | 0.9981 | 0.9981 |
| walsh s2 | 0.9311 | 0.9592 | 0.9510 | 0.9914 | **0.9914** |
| walsh s3 | 0.9460 | 0.9490 | 0.9407 | 1.0072 | 1.0072 |

B=100 alone understates n15 (champ itself shows n15≈1.004 vs sealed 0.9853). Relative: Idea1 ≥ baseline; B=1000 sealed confirms clear pass.

### Train recipe (smoke)

- L=2, train{6,9}, `max_train=100`, `eval_B=100` then confirm `eval_B=1000`
- Adam 400 × lr=1e-3 + LBFGS 8×8; w6=1,w9=2; holdout 20% n9
- Script: `Nov_3rd/scripts/train69_ucx_walsh_degree.py`
- Ansatz: `Nov_3rd/ansatz.py` accepts `gamma` (L,) or (L,4)

### Caveats / follow-ups

1. **Smoke train** (B=100 instances). Pass is robust across 4 seeds on sealed CORE, but seal harder with full train packs (`--max_train 1000`) + optional `power_p`.
2. Kill criteria checked: γ_w did not drift back to equal; n12/n15 improved vs 0.9858 knee (not “loss↓ but R stuck ~0.986”).
3. Idea2 (cavity-locality γ_in/γ_x) not run yet — scaffold next if wanted.

### ETA note (first seed)

Seed0 train+CORE B=100 ≈ **70s** wall. Seeds 1–3 parallel ≈ +70s. B=1000 CORE confirm ≈ **12s**/seed.
