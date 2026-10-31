# GRAPE k-local / two-mode bypass of densified \(U_c\)

_Asia/Shanghai — 2026-10-31 ~15:35 CST — Zach t2201 + clarify t2202_

## One-line verdict for Zach

**YES.** GRAPE can compile a **general two-mode** (or few-mode) target \(U_{\mathrm{tgt}}\) offline on a truncated Fock⊗Fock / cavity⊗transmon space; for k-SAT the instance \(U_c=\mathrm{e}^{-i\gamma H_c}\) is then a **classical schedule** of those frozen gadgets. You **do not** densify or matrix-exp the full \(2^n\) (or \(d^n\)) instance unitary.

## What Zach asked (refined)

1. **t2201:** With local gadgets \(\mathrm{e}^{-i\theta ZZ}\) (2-body) or \(k\)-body max for \(k\)-SAT, can we bypass the full matrix?
2. **t2202 clarify:** He means GRAPE for an **arbitrary unitary between TWO MODES** (cavity–cavity or cavity–transmon), not only \(\mathrm{e}^{-i\theta ZZ}\).

Both map to the same YES: compile small \(U_{\mathrm{tgt}}\) once → schedule for the instance.

## Claim (confirmed)

| Step | What | Size |
|------|------|------|
| Offline GRAPE / SNAP | Arbitrary \(U_{\mathrm{tgt}}\) on 2 modes (or \(k\) modes for a clause), truncated \(d_1\times d_2\) | \(O((d_1 d_2)^3)\) per pulse opt — **not** \(2^n\) |
| Library | Frozen pulses for gadget classes (ZZ, projector \(I+(e^{-i\gamma}-1)C_c\), or any two-mode SU) | ~const # classes |
| Instance \(U_c\) | Classical product / schedule of \(m\) clause gadgets | \(O(m\cdot\mathrm{poly})\) |
| Identity | \(\prod_c\big(I+(e^{-i\gamma}-1)C_c\big)=\mathrm{e}^{-i\gamma H_c}\) (commuting diagonal) | exact, no densify |

Cross-links: `GRAPE_TARGET_SPEC.md`, `PROJECTOR_EQ_EXP_HC.pdf`.

---

## LIVE Opus 5.5 high reply (2026-10-31, model `claude-opus-5-5-high`)

> **VERDICT: YES.** Two-mode or few-mode GRAPE, used to build an offline library of frozen pulses that are then run as a classical schedule, implements \(U_c = e^{-i\gamma H_c}\) for fixed-k k-SAT/TEB without ever forming the \(2^n\) or \(d^n\) instance matrix. This holds under three conditions…: the clause terms commute, \(k\) is fixed, and there is a per-gadget error budget.

**A) Can GRAPE compile general two-mode \(U_{\mathrm{tgt}}\) offline without instance \(2^n\)?**  
> **YES.** GRAPE only ever works with the propagator of the system it controls… \(N\) is the dimension of the simulated device model (for example \(d_1 d_2\), or \(d_c\cdot d_q\) for cavity⊗transmon)… Nothing about an instance on \(n\) variables enters the calculation. A completely general \(U_{\mathrm{tgt}}\in SU(d_1 d_2)\) is allowed, provided the controls make the system fully controllable.

**B) Frozen two-mode / k-mode gadgets + schedule for k-SAT without densifying full \(U_c\)?**  
> **YES, for fixed \(k\) and commuting clause terms.**

**Proof sketch (Opus, condensed):**
1. \(e^{-i\gamma w_c C_c}=I+(e^{-i\gamma w_c}-1)C_c\) for projector \(C_c\).
2. Commuting diagonals ⇒ product \(=\mathrm{e}^{-i\gamma H_c}\) exact.
3. Each gadget is \(d^k\times d^k\) (or decomposed into two-mode gates); GRAPE on that support only.
4. Circuit = classical list of \(m\) gadgets — no \(d^n\) multiply.
5. Generic \(n\)-qubit unitaries need exp many locals (Knill); \(U_c\) is **not** generic — it factorizes into \(m\) locals.

Full raw reply: `OPUS_GRAPE_TWOMODE_LIVE.txt`.

---

## Literature verdict (independently checked this turn)

### QAOA / k-local cost — product of locals, no densify

**Farhi, Goldstone, Gutmann (2014)** arXiv:1411.4028:
\[
U(C,\gamma)=e^{-i\gamma C}=\prod_{\alpha=1}^{m}e^{-i\gamma C_{\alpha}}.
\]
> “All of the terms in this product commute because they are diagonal… and each term’s locality is the locality of the clause \(\alpha\).”  
> Circuit depth “at most \(mp+p\)”.

This is exactly the classical-schedule story for diagonal \(H_c\).

### GRAPE targets small analytic / truncated space

- **Khaneja et al. (2005)** — GRAPE; example target \(\mathrm{e}^{-i\alpha I_z I_z I_z/4}\) on controlled spins, not densified many-body \(H_c\). DOI: 10.1016/j.jmr.2004.11.004
- **Rowland & Jones (2012)** arXiv:1203.6260 — fidelity to \(U_t\); “computational complexity scales badly with the number of qubits” → compile **local**, not full instance.
- **Heeres et al. (2017)** arXiv:1608.02430 / Nat. Commun. — GRAPE on **cavity⊗transmon**, truncated Fock; arbitrary drives → universal logical gate set. Supports: GRAPE of general ops on truncated two-subsystem space, offline.
- **Fösel et al. (2020)** arXiv:2004.14256 — SNAP+D compile;  
  > “The action of the target operation **does not need to be defined on the entire Hilbert space** \(\mathcal{H}\), but can be restricted to a logical subspace of dimension \(L\).”  
  Contrasts modular SNAP with full GRAPE (Heeres).

### Two-mode / oscillator universality (arbitrary \(U\), not only ZZ)

- **Krastanov et al. (2015)** PRA 92, 040303 / arXiv:1502.08015 — SNAP + displacement = **universal single-mode** control.
- **Heeres et al. (2015)** PRL — experimental SNAP.
- **Fösel / Krastanov (2020)** — efficient SNAP sequences for target ops (subspace OK).
- **Beamsplitter + single-mode control → two-mode:** Gao et al. PRX/Nature; Chapman et al. arXiv:2212.11929 (programmable beamsplitter between two cavities); Eigen-SNAP / two-cavity SWAP proposals. CV universality: Lloyd & Braunstein PRL 1999.
- **Wang et al. Science 2016** (“cat in two boxes”) — optimal control on cavity–transmon–cavity (two bosonic modes + ancilla).

**Net:** Literature supports (i) GRAPE/SNAP for **arbitrary** targets on truncated 1–2 mode spaces, and (ii) QAOA-style **product of local** cost unitaries. Combining them = library + schedule, **bypass densified instance \(U_c\)**.

---

## Caveats (from Opus + lit — still YES under these)

1. **Fixed \(k\)** — gadget dim \(d^k\); if \(k\) grows with \(n\), cost returns.
2. **Commuting / diagonal cost** — exact product; non-commuting cost ⇒ Trotter error (our dead `nonhc_Uc` arms).
3. **\(\gamma\)-dependence** — either put \(\gamma\) in a virtual-Z / SNAP angle between frozen entanglers, or grid-\(\gamma\) library; still not \(2^n\).
4. **Truncation / leakage / spectators** — Heeres-style guard levels; if all spectators modeled brute-force, dim can grow (engineering, not instance densify).
5. **Error budget** — \(\varepsilon_{\mathrm{gadget}}\ll 1/m\).
6. **Bypass ≠ solve SAT** — compilation is poly; finding \(\min H_c\) remains hard.

---

## Plain answer to Zach

- **Arbitrary two-mode \(U_{\mathrm{tgt}}\):** YES — GRAPE (Heeres) or SNAP+D (+ beamsplitter for cavity–cavity) on truncated two-mode space, offline, once per gadget class.
- **k-SAT instance \(U_c\):** YES — classical schedule of those frozen few-mode pulses; product of clause gadgets \(=\mathrm{e}^{-i\gamma H_c}\) without ever writing the \(2^n\) matrix.
- **Not claimed:** that GRAPE of a generic full-\(n\) unitary is poly; that this solves SAT classically; that ZZ is the only allowed target (any two-mode SU on the truncated space is fine).

## Files

- This note: `explore_OCT29/docs/GRAPE_KLOCAL_BYPASS.md`
- Live Opus raw: `explore_OCT29/docs/OPUS_GRAPE_TWOMODE_LIVE.txt`
- Prior: `GRAPE_TARGET_SPEC.md`, `PROJECTOR_EQ_EXP_HC.pdf`
