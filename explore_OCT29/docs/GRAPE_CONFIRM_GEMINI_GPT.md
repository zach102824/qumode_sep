# GRAPE confirm: Gemini 4 Argon + GPT-5.6 Sol (LIVE)

_Asia/Shanghai — 2026-10-31 ~15:40 CST — Zach t2203_

## One-line for Zach

| Model | Verdict | Gradients force full \(2^n\) matrix? | Bypass densified \(U_c\)? |
|-------|---------|--------------------------------------|--------------------------|
| **Gemini 4 Argon High** (`gemini-4-argon-high`) | **YES-bypass** | **NO** | **YES** |
| **GPT-5.6 Sol High** (`gpt-5.6-sol-high`) | **YES-bypass** | **NO** | **YES** |
| Opus 5.5 high (prior, t2202) | YES-bypass | NO | YES |

**Answer to Zach’s belief** (“GRAPE uses gradients ⇒ we MUST need the full matrix”):

> Gradients are computed on the **small controlled space** \(N=d_1 d_2\) (or \(d^k\), or \(2^k\) for qubits) — forward \(U_j\) and backward \(\lambda_j\) are \(N\times N\). That is **not** the densified instance \(2^n\times 2^n\). Gradients do **not** enlarge the Hilbert space.

Raw LIVE dumps: `GEMINI4_ARGON_GRAPE_LIVE.txt`, `GPT56_SOL_GRAPE_LIVE.txt`.  
Prior Opus: `OPUS_GRAPE_TWOMODE_LIVE.txt`, summary `GRAPE_KLOCAL_BYPASS.md`.

---

## Short usual GRAPE procedure (both models agree; Khaneja 2005)

1. **Model** on truncated gadget space dim \(N\): \(H(t)=H_0+\sum_a u_a(t)H_a\).
2. **Piecewise-constant** controls \(u_a(j)\) on \(M\) slices of width \(\Delta t\).
3. **Slice prop** \(G_j=\mathrm{e}^{-i\Delta t H_j}\) (\(N\times N\)).
4. **Forward** \(U_0=I\), \(U_j=G_j U_{j-1}\); final \(U_f=U_M\).
5. **Fidelity** \(\Phi=|\mathrm{Tr}(U_{\mathrm{tgt}}^\dagger U_f)|^2/N^2\).
6. **Backward** \(\lambda_M=U_{\mathrm{tgt}}\), \(\lambda_{j-1}=G_j^\dagger\lambda_j\).
7. **Gradient** (short-slice): \(\partial\Phi/\partial u_a(j)\propto\mathrm{Im}\,\mathrm{Tr}(\lambda_j^\dagger H_a U_j)\); update (L-BFGS / CG / ascent).
8. **Freeze** pulse; at instance time only a **classical schedule** of \(m\) gadgets — never form \(2^n\times 2^n\) \(U_c\).

**\(N\)** = gadget truncation (e.g. \(d_1 d_2\), cavity⊗transmon, or \(2^k\) for a \(k\)-qubit clause) — **independent of SAT size \(n\)**.

---

## Matrix-size table (from Gemini; GPT agrees)

| Quantity | GRAPE actually uses | Wrong belief |
|----------|---------------------|--------------|
| Space | \(\mathcal{H}_{\mathrm{gadget}}=\mathbb{C}^{d_1}\otimes\mathbb{C}^{d_2}\) | \((\mathbb{C}^2)^{\otimes n}\) |
| Matrix size | \((d_1 d_2)\times(d_1 d_2)\) e.g. \(64\times64\) | \(2^n\times 2^n\) |
| Scaling in \(n\) | \(O(1)\) offline + \(O(m)\) schedule | \(O(4^n)\) memory |

---

## Caveats (both + Opus agree)

1. **Commuting clause projectors** — product \(=\mathrm{e}^{-i\gamma H_c}\) **exact**, no Trotter.
2. **Fixed \(k\)** — gadget cost \(d^k\) or \(O(2^k)\) entanglers; if \(k\) grows with \(n\), exp returns.
3. **Error budget** — \(\|\prod\widetilde G-\prod G\|\le m\,\varepsilon_{\mathrm{gadget}}\) (op-norm); need \(\varepsilon\ll 1/m\); coherent errors can be worse (\(O(m^2\varepsilon)\) infidelity). Leakage, guard levels, spectators/crosstalk. Pulse for one \(\gamma\) ≠ all \(\gamma\) unless parameterized / grid / robust GRAPE.

---

## LIVE excerpts

### Gemini 4 Argon High — on Zach’s belief

> GRAPE indeed requires an explicit target matrix \(U_{\mathrm{tgt}}\) and dense matrix propagators… **on the Hilbert space of the gate being synthesized**. However, GRAPE is only ever run **offline on a local 1-, 2-, or \(k\)-mode gadget** (dimension \(N\ll 2^n\)), never on the global \(n\)-variable SAT instance… The global \(2^n\times 2^n\) matrix is never formed, differentiated, or stored.

**(A) NO · (B) NO · (D) YES · VERDICT: `YES-bypass`**

### GPT-5.6 Sol High — on Zach’s belief

> Zach’s inference is false: differentiating the controlled evolution does not enlarge its Hilbert space. The gradients use local \(N\times N\) quantities… Whether a derivative is first-order approximate or evaluated exactly changes gradient accuracy and runtime—not \(N\).

**(A) NO · (B) NO · (D) YES under caveats · VERDICT: YES—BYPASS; the full \(2^n\times 2^n\) instance matrix is not needed.**

---

## Cross-links

- `GRAPE_KLOCAL_BYPASS.md` — Opus + lit
- `GRAPE_TARGET_SPEC.md` / `.pdf`
- `PROJECTOR_EQ_EXP_HC.pdf` — projector product \(=\mathrm{e}^{-i\gamma H_c}\)
