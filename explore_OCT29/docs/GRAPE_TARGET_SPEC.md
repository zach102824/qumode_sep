# GRAPE target specification: analytic local \(U_{\mathrm{tgt}}\) vs full \(2^n\) matrix

_Asia/Shanghai — 2026-10-31 ~14:30 CST — Zach t2198: "if using GRAPE one has to know the matrix or simple exp iZZ is ok"_

## Verdict (one line)

**Simple analytic targets are enough.** Standard GRAPE takes a **small** target unitary \(U_{\mathrm{tgt}}\) (often written as \(\mathrm{e}^{-i\theta Z\otimes Z}\), \(\mathrm{e}^{-i\alpha ZZZ/4}\), a SNAP, or a projector gadget) on a few levels / few modes — **not** the densified full-instance \(2^n\times 2^n\) cost unitary. Instance \(U_c=\mathrm{e}^{-i\gamma H_c}\) is then a **classical schedule** of many such frozen pulses.

## What GRAPE actually optimizes

GRAPE (Khaneja et al. 2005) discretizes controls \(u_k(j)\) and maximizes a fidelity between the realized propagator \(U_f=U_N\cdots U_1\) and a **chosen target**:

- **Unitary synthesis:** \(\Phi \propto |\mathrm{Tr}(U_{\mathrm{tgt}}^\dagger U_f)|^2\) (phase-insensitive Hilbert–Schmidt overlap).
- **State-to-state / subspace:** average fidelity of a list of transfers \(\lvert\psi_{\mathrm{in}}^{(i)}\rangle\mapsto\lvert\psi_{\mathrm{out}}^{(i)}\rangle\) (Heeres et al.; Fösel et al.).

So the optimizer needs **some** description of \(U_{\mathrm{tgt}}\) (or of the transfers) that it can compare to \(U_f\) inside a **simulable** Hilbert space. That description can be:

| Spec form | Typical size | Example |
|-----------|--------------|---------|
| Analytic generator / Pauli string | \(2^k\) with \(k\) small | \(\exp(-i\alpha\,I_z I_z I_z/4)\), \(\exp(-i\theta ZZ)\) |
| Explicit small matrix | \(2\)–\(d\) levels, \(1\)–few modes | CNOT \(4\times4\); SNAP diag phases; projector gadget |
| Subspace map \(V=\sum_l\lvert y_l\rangle\langle x_l\rvert\) | logical dim \(L\ll\dim\mathcal{H}\) | cat encode/decode; binomial recovery |
| Full \(n\)-qubit densified \(U\) | \(2^n\times 2^n\) | **possible in principle, exponentially hard** — not the literature default |

**Key:** “Must know the matrix” means “must know \(U_{\mathrm{tgt}}\) on the **controlled** space,” which for local gadgets is \(O(1)\) or \(O(\mathrm{poly})\) — **not** “must densify instance \(H_c\) to length \(2^n\) then matrix-exp.”

## Distinction for our TEB / qumode claim

**(a) Library GRAPE of frozen local gadgets (our claim)**  
Target per compile = one clause pattern, e.g.
\[
U_c(\gamma)=I+(e^{-i\gamma}-1)C_c
\quad\text{or}\quad
\mathrm{e}^{-i\theta Z_i Z_j}
\quad\text{or SNAP phases on a few Fock levels.}
\]
GRAPE (or exact SNAP) runs on that **small** support once offline → frozen pulse library (~26 locals). Instance \(U_c=\prod_c U_c(\gamma)=\mathrm{e}^{-i\gamma H_c}\) is classical product / schedule. **No \(2^n\) densify.** Same physics as Arm C; complexity win is densification only.

**(b) GRAPE of the full instance unitary**  
Target = densified \(\mathrm{e}^{-i\gamma E}\) or the full \(2^n\) matrix of \(U_c\). Simulation / gradient cost scales exponentially in \(n\). Literature warns this “scales badly with the number of qubits” (Rowland & Jones 2012). **Not** what we mean by “GRAPE the projector gadgets.”

## Supporting papers (verified links)

1. **Khaneja, Reiss, Kehlet, Schulte-Herbrüggen, Glaser (2005)** — original GRAPE.  
   Target example is the **analytic** propagator
   \(U_{zzz}(\alpha)=\exp\{-i\alpha\,4^{-1}I_{1z}I_{2z}I_{3z}\}\), optimized with \(U(0)=I\), \(U(T)=U_{zzz}(\alpha)\).  
   - DOI: https://doi.org/10.1016/j.jmr.2004.11.004  
   - PDF: https://www.ch.nat.tum.de/fileadmin/w00bzu/ocnmr/pdf/94_GRAPE_JMR_05_.pdf  
   - Related arXiv: https://arxiv.org/abs/quant-ph/0502173

2. **Rowland & Jones (2012)** — “Implementing quantum logic gates with GRAPE.”  
   Fidelity \(\Phi=|\langle U_t|U_f\rangle|^2\) with target propagator \(U_t\); “any desired unitary transform can be generated,” but “computational complexity scales badly with the number of qubits.” → Practically GRAPE **local / few-qubit** gates.  
   - arXiv: https://arxiv.org/abs/1203.6260  
   - HTML: https://ar5iv.labs.arxiv.org/html/1203.6260

3. **Heeres et al. (2017)** — cavity + transmon; GRAPE for logical cat gates / encode–decode.  
   Targets are **logical** ops and state transfers on truncated oscillator⊗transmon — not a full SAT \(U_c\). Explicitly cites Khaneja GRAPE.  
   - arXiv: https://arxiv.org/abs/1608.02430  
   - Nature Comm.: https://doi.org/10.1038/s41467-017-00045-1

4. **Fösel, Krastanov, Marquardt, Jiang (2020)** — “Efficient cavity control with SNAP gates.”  
   Quote: “The action of the target operation **does not need to be defined on the entire Hilbert space** \(\mathcal{H}\), but can be restricted to a logical subspace of dimension \(L\).” Contrasts full GRAPE pulses (Heeres) with modular SNAP+D.  
   - arXiv: https://arxiv.org/abs/2004.14256  
   - HTML: https://ar5iv.labs.arxiv.org/html/2004.14256

5. **Glaser et al. (2015)** — “Training Schrödinger’s cat: quantum optimal control” (review).  
   GRAPE as mainstream for gate / propagator synthesis across platforms.  
   - arXiv: https://arxiv.org/abs/1508.00442  
   - EPJ D: https://doi.org/10.1140/epjd/e2015-60464-1

6. **Machnes / Wilhelm lecture notes (2020)** — “An introduction into optimal control for quantum technologies.”  
   GRAPE section: state-to-state and unitary control; target is the desired gate on the modeled system.  
   - arXiv: https://arxiv.org/abs/2003.10132

## Short quotes (for paper / chat)

> “Here, we used … the target operator is \(U(T)=U_{zzz}(\alpha)\)” — Khaneja et al. 2005 (analytic generator, not densified many-body \(H_c\)).

> “The action of the target operation does not need to be defined on the entire Hilbert space \(\mathcal{H}\), but can be restricted to a logical subspace of dimension \(L\).” — Fösel et al. 2020.

> “It is … relatively straightforward to apply [GRAPE] to a system of arbitrary size, although the computational complexity scales badly with the number of qubits involved.” — Rowland & Jones 2012 → reinforces: compile **local** gadgets, schedule classically.

## Answer to Zach (plain)

For \(\mathrm{e}^{-i\theta ZZ}\) or a frozen clause projector \(I+(e^{-i\gamma}-1)C_c\), GRAPE (or SNAP) uses that **small analytic \(U_{\mathrm{tgt}}\)** — typically a few qubits / Fock levels. You do **not** hand GRAPE the full \(2^n\) instance \(U_c\). The instance unitary is the product of many such local pulses (classical schedule). That is exactly the non-exponential path vs Arm C densified \(E\). Same \(U=\mathrm{e}^{-i\gamma H_c}\) physics; no densify.

## Cross-links in this repo

- Projector \(=\mathrm{e}^{-i\gamma H_c}\) proof: `PROJECTOR_EQ_EXP_HC.pdf` / `.md` confirm log  
- Robust / non-\(H_c\) taxonomy: `ROBUST_GRAPE.md`  
- Offline 26-local compile: `COMPILE_OFFLINE.md`

## Optional PDF

Same content can be rendered from this note; markdown is the source of truth.

---

## Update (t2201/t2202)

Zach clarified: not only \(\mathrm{e}^{-i\theta ZZ}\), but **arbitrary two-mode** \(U_{\mathrm{tgt}}\) (cavity–cavity / cavity–transmon). Live Opus + lit confirm **YES** — still bypass densified instance \(U_c\). See **`GRAPE_KLOCAL_BYPASS.md`** and `OPUS_GRAPE_TWOMODE_LIVE.txt`.
