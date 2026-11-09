# Corrected-workspace W_BinH compile (K=3, Q=2048)

Bounded simulator correction. **No SK fits.** Historical SK ledger stays **2818048**. PR109 **WCOMP_FAIL_BUDGET** remains history under a projection-exploitable propagator; it is not a valid full-oscillator trial and not an impossibility claim. No extra controls, no extra depth, no ε sweep, no GR.

Same native primitives as PR109: real \(\alpha\), \(K=3\), \(D\) then SNAP, SNAP on 0..7 pad phase 0, \(\alpha=1.8\tanh u\), SPSA \(Q=2048\), four starts, start 0 = identity.

## Necessary differences (only these, before fits)

| item | PR109 | this trial | why |
|---|---|---|---|
| Inter-gate space | `P_{16} D_{48} P_{16}` each D | one workspace \(W=96\) for the whole sequence | PR110: repeated restriction is not oscillator evolution |
| Objective | \(\lVert V_\phi-T\rVert_F^2/8\) on the truncated register | same formula on \(W=96\) where \(p_{\rm lost}\) is preflight-bounded | pad is retained, so losing pad **raises** \(L\); projection cannot help |
| Replay cutoff | 32 (still truncated) | frozen pulse at \(W=128\), no reopt | larger than fit, PR110 ladder |
| RNG tuple | `SeedSequence([33001, 16, 3, start_id, stream])` | **same tuple** (tag 16 kept) | hold start 1–3 draws |

Not changed: \(K\), real-\(\alpha\) only, \(A_{\max}=1.8\), SNAP support 0..7, \(Q=2048\), Spall \(c=0.1\) \(A=25\) \(0.602/0.101\), \(\sigma_u=0.15\) \(\sigma_\theta=0.25\), identity start 0, starts 1–3 \(u\sim U(-1,1)\) \(\theta\sim U(-\pi,\pi)\). Complex \(\alpha\) and pad SNAP are still not added.

## Preflight (before SPSA)

Worst case: three aligned \(D(1.8)\) on code kets (no SNAP). Allowed \(\lvert\alpha\rvert\le 1.8\).

| W | \(\lvert7\rangle\) \(p_{\rm lost}\) | \(\lvert7\rangle\) last-4 | \(\lvert0\rangle\) \(p_{\rm lost}\) |
|---|---|---|---|
| 64 | \(8.3\times10^{-2}\) | 0.13 | \(1.7\times10^{-8}\) |
| **96** | \(2.2\times10^{-11}\) | \(9\times10^{-10}\) | \(\sim10^{-15}\) |
| 128 | \(\sim0\) | \(\sim0\) | \(\sim0\) |

Fit workspace **96**. Replay **128**. If a returned pulse has mean \(p_{\rm lost}>10^{-8}\) at 96 or 128 → **WCOMP_WS_INCONCLUSIVE** (numerical), not success or failure.

A finite-generator \(D\) on \(W\) (exp in \(W+32\), restrict once as the **gate definition**) is used. It is nearly isometric on occupied low-\(n\) states at \(W=96\); residual non-unitarity lives on high-\(n\) basis states near the cutoff. Sequence stays in \(\mathbb C^{96}\) (no per-gate \(P_{16}\)).

## Target and objective

\(T\) is the \(W\times 8\) isometry: \(H^{\otimes 3}\) on rows 0..7, zeros on the pad.

One global phase \(\phi=\arg\operatorname{tr}(T^\dagger V)\). SPSA **minimizes** the coherent isometry error

\[
L=\frac18\lVert V_\phi-T\rVert_F^2.
\]

This is **not** mean basis \(F_{\rm avg}\). With \(p_{\rm lost}\approx0\), pad mass is inside \(V\) and **increases** \(L\) (T is 0 there). Identity has \(L=2\) because \(\operatorname{tr}(H^{\otimes3})=0\).

Selector: lowest \(L\) among the four SPSA starts and their init/last/polyak/mid candidates, including **measured identity** (start 0 init). Tie: lower start_id, then init. Never select by post-hoc \(F_{\rm avg}\).

Reported, not selected: \(F_{\rm HS}=\lvert\operatorname{tr}(T^\dagger V)\rvert^2/64\), per-column \(F_k=\lvert\langle T_k\vert V_{\phi,k}\rangle\rvert^2\), plus/random superpositions, partition \(p_{\rm code}+p_{\rm pad}+p_{\rm lost}=1\).

## Success (prespecified)

**WCOMP_WS_CANDIDATE** iff all hold at \(W=96\) and replay \(W=128\):

- \(L\le 0.02\) and \(F_{\rm HS}\ge 0.99\)
- mean \(p_{\rm code}\ge 0.99\), mean \(p_{\rm lost}\le 10^{-8}\)
- replay full-column overlap (pad-embed, one \(\phi\)) \(\ge 0.999\), \(\lvert\Delta L\rvert\le 10^{-3}\), replay \(p_{\rm lost}\le 10^{-8}\)

Else if \(p_{\rm lost}\) bound fails → **WCOMP_WS_INCONCLUSIVE**. Else **WCOMP_WS_FAIL_BUDGET**. Bounded; not an impossibility claim. No second setting.

## Accounting

Compiler evaluations this trial: 2048 inclusive. Historical compiler 2048 (PR109) recorded separately. SK queries **0**. SK total **2818048**.

## Off limits

SK, GR, validation, final, Nov9/SYK, BFGS/Adam, extra \(K\), complex \(\alpha\), pad SNAP, hardware claim.
