# Workspace compile audit

PR109 `WCOMP_FAIL_BUDGET` is kept as history. Its mean retained probability 0.402 was repeated \(P_{16} D_{48} P_{16}\) after every displacement, not Euclidean column norm and not a valid full-oscillator leakage budget. That objective could improve by discarding pad. This trial does not reuse that propagator.

## This trial (frozen before SPSA)

- Common workspace \(W=96\) for the whole \(K=3\) sequence. No per-gate NF16 projection or renormalization.
- Preflight: three aligned \(D(1.8)\) on \(\lvert 7\rangle\) has \(p_{\rm lost}=2.16\times 10^{-11}\) at 96; \(W=64\) loses 8.3% and was rejected.
- Finite-generator \(D\) is `restricted` (exp in \(W+32\), restrict once as the gate). Sequence stays in \(\mathbb C^{96}\).
- Selector is coherent \(L=\lVert V_\phi-T\rVert_F^2/8\) on the full 8-column \(W_{\rm BinH}\) isometry after one global phase. Not \(F_{\rm avg}\).
- Identity \(x=0\) is start-0 init (\(L=2\), \(F_{\rm avg}=0.125\), \(F_{\rm HS}=0\)) and is in the min-\(L\) pool.
- Frozen pulse replayed at \(W=128\) without refit.

## Returned pulse

| | identity | selected (start 1, last) | replay 128 |
|---|---|---|---|
| \(L\) | 2.000 | 1.129 | 1.129 |
| \(F_{\rm HS}\) | 0 | 0.190 | 0.190 |
| \(F_{\rm avg}\) (not selected) | 0.125 | 0.225 | 0.225 |
| \(p_{\rm code}\) | 1 | 0.639 | 0.639 |
| \(p_{\rm pad}\) | 0 | 0.361 | 0.361 |
| \(p_{\rm lost}\) | 0 | \(\sim 0\) | \(\sim 0\) |

Partition \(p_{\rm code}+p_{\rm pad}+p_{\rm lost}=1\). Replay full-column overlap 1.000, \(\Delta L\sim 10^{-15}\). Code-block Frobenius \(3\times 10^{-14}\). Convergence is numerical, not inconclusive.

Selected beats identity on \(L\) **with pad retained**, so the drop is not projection. \(F_{\rm avg}\) also beats identity (0.225 vs 0.125) but is not the selector and is far from 0.99. Basis \(F\) alone would not certify a gate.

\(\alpha \approx (-1.447,-0.897,1.534)\), inside \(\lvert\alpha\rvert\le 1.8\). Not tanh-saturated the way PR109's selected pulse was.

## Label

**WCOMP_WS_FAIL_BUDGET.** \(L=1.129\not\le 0.02\), \(F_{\rm HS}=0.190\not\ge 0.99\), mean \(p_{\rm code}=0.639\not\ge 0.99\). Workspace and \(p_{\rm lost}\) bars hold, so this is a bounded **budget/depth** miss, not `INCONCLUSIVE` and not a physical impossibility. Stop. No extra \(K\), complex \(\alpha\), or pad SNAP.

Compiler evals this trial 2048 (init 4 + cal 32 + iter 2000 + cand 12). Historical PR109 compiler 2048 separate. SK queries 0. SK total **2818048**.

Not a hardware claim. No GR / freeze / final change.
