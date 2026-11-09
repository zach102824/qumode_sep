# Addendum: PR107 is a phenomenological \(G_{7,8}(0.10)\) sensitivity test

Numerical results in `summary.json` and the cell files are unchanged. Ledger **2818048**. No new fits.

**Relabel.** `WL_ROBUST_L1` remains the prespecified gate string. It means: under this one illustrative Givens residual, L=1 Nov3−QAOA mean \(\Delta E\) CI was entirely \(<0\) on seeds 10528–35. It does **not** mean compiled-\(W_L\) robustness, hardware validation, or a physical error model. D5 still unmodelled. No GR. No further \(\varepsilon\) sweeps.

**Secondary (corrected).** B−A \(+0.3068\) CI \([+0.273,+0.331]\) is Nov3 degradation vs ideal Nov3 only. There is no ideal QAOA arm. Equal \(p_{\rm leak}\approx 2.35\%\) is not equal energy bias.

**Selected energy** is unconditional \(\langle H^{\rm ext}\rangle\). Oracle replay of frozen \(x\) (0 ledger queries):

| arm | mean \(p_{\rm code}\) | mean \(E_{\rm uncond}\) | mean \(E_{\rm cond}\) (not selected) | mean leak-block |
|---|---|---|---|---|
| A | \(1\) | −11.3335 | −11.3335 | \(\sim0\) |
| B | 0.9765 | −11.0267 | −11.3246 | +0.0306 |
| C | 0.9765 | −10.7169 | −11.0098 | +0.0342 |

Leakage raises the selected energy. Do not reclassify previous \(E\) as postselected.

See `Nov3_INIT/AUDIT_WL.md`.
