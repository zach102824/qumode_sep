# Nov9th: physical-definition decisions for the user, and what is safe to run now

No reviewed contract defines the physical action of SNAP or \(U_{cZ}\) on Fock levels ≥ 8, or how energy is measured when a cavity leaks. Nov3rd never needed it, because it uses NF = 8 and an ideal \(W\). So Nov9th with a true displacement is **not defined** until D2–D5 are chosen. Below, ★ marks the consultation's recommended default. Workers may launch with the ★ choices, labeled "provisional-★". Any headline claim needs the user to confirm.

| id | question | options | ★ recommendation and reason |
|---|---|---|---|
| D1 | logical ↔ Fock encoding | binary j = 0..7 (Nov3rd, now confirmed by the local stream) / Gray (Oct-29 historical) | ★ **binary**. It matches Nov3rd's \(S\) and \(W\), and gives \(s_3\) an exclusive handle on level 7, the leaking level. Gray only as the labeled variant (Idea 2). |
| D2 | SNAP phase on f ≥ 8 | 0 / periodic \(\theta_{f\bmod8}\) / free | ★ **0**. This is the repository pad-0 convention (`_snap_theta`) and adds no control. "Free" would add unapproved controls. |
| D3 | \(U_{cZ}\) on leaked components | **Uz**: zero-extended \(\tilde Z\); terms touching a leaked cavity are identity, the rest act / **U0**: whole component unphased (legacy slice code) | ★ **Uz** primary, **U0** mandatory sensitivity arm. Uz is what pairwise conditional-phase compilation does if each pair gate is only specified on the code. U0 needs a global "all cavities in code" condition, which is not a pairwise gate. |
| D4 | energy observable with leakage | per-term \(H^{\rm ext}\) / global penalty \(\sum\lvert J\rvert+bN\) / H0 (leak scored 0) | ★ **per-term \(H^{\rm ext}\)** (proof below). Global penalty is valid but higher-variance. **H0 is forbidden for selection**: it can reward leaking out of positive-energy regions. |
| D5 | readout model | ideal number-resolved readout with a "≥ 8" bin; X setting via ideal \(W_L\) (= Nov3rd's \(W\)) on the code and identity on ≥ 8 / compiled \(W_L\) / bit-wise mod-16 ancilla readout | ★ **ideal number-resolved + ideal \(W_L\)** for simulation, labeled. This is the same idealization Nov3rd already uses \(2L+1\) times per circuit. Run mod-16 aliasing as a diagnostic. **Blocker for any hardware-feasibility claim:** the compiled \(W_L\) and its leakage are not modelled. |
| D6 | accept the strict-logical model \(D_8=\exp[\alpha(Pa^\dagger P-PaP)]\) (8-level qudit or blockaded cavity) as a reported hardware variant? | yes, as a diagnostic / yes, as a co-primary / no | ★ **diagnostic only**, labeled "Nov9th-B", never called a displacement. It is needed to separate leakage loss from expressivity loss (Idea 1, K1/K2). |
| D7 | complex α (adds 1 real per layer) | not now / round 2 if L = 1 \(O(\gamma^2)\) weakness is confirmed | ★ **not now**. Revisit after Idea 1's K2 verdict. |
| D8 | 4-body Z strings from the \(\sum_ih_i^4\) expansion | not now / restricted subset | ★ **not now**. The compiled cost is \(O(N^4)\) dense terms and is unapproved. |
| D9 | \(\alpha_{\max}\) bound | 0.30 / other | ★ 0.30. First-order leak per cavity per layer is about \(\alpha^2\), so 0.30 allows at most about 9% per cavity, and the energy penalty will push far below that. |

## Proof sketch for D4 (per-term \(H^{\rm ext}\) cannot be exploited)

\(H^{\rm ext}\) commutes with every cavity's code projector \(P^L_c\). It is therefore block diagonal over leak patterns Λ, the set of leaked cavities.

In block Λ:

\[
H_\Lambda=H_{\rm rest}+\sum_{(i,j)\,\text{touching}\,\Lambda}|J_{ij}|+3b|\Lambda|,
\]

where \(H_{\rm rest}\) is \(H\) restricted to the qubits of unleaked cavities. The leaked internal Fock levels appear only as a multiplicity.

Take any state φ of the unleaked qubits, and any Z-basis assignment \(z_\Lambda\) of the leaked qubits. In \(H\):

- the pair terms touching Λ are at most \(|J_{ij}|\);
- the X terms on Λ have zero expectation in \(z_\Lambda\).

So \(E_0\le\langle\phi\otimes z_\Lambda|H|\phi\otimes z_\Lambda\rangle\le\langle\phi|H_\Lambda|\phi\rangle\). Hence \(\min\mathrm{spec}\,H^{\rm ext}=E_0\), attained only at Λ = ∅ and the GS. T8 checks this exhaustively at N = 6.

Estimator per shot:

- **Z setting:** sum \(J_{ij}z_iz_j\) over pairs with both cavities in the code, plus \(|J_{ij}|\) over pairs touching a leaked cavity.
- **X setting:** \(-b\,x_i\) for qubits in coded cavities, plus \(+b\) per qubit in a leaked cavity.

## Safe to run now (no user decision needed)

- Nov3rd vs QAOA at N = 12, 15, 18 (local stream, already running). Add columns: \(F_+\), η, \(\Delta_{\rm odd}\), budget curve, shortlist re-ranking.
- S0 for everything: classical ED diagnostics, \(F_+\), \(\Delta_{\rm odd}\), and T1–T14.
- Nov9th-B (strict-logical, no off-code space) exact-energy SPSA screen. It is fully defined.
- Nov9th-leaky under the ★ defaults with the U0 sensitivity arm, exact energy, dev seeds only, labeled provisional-★.

## Not safe until decided

- Any Nov9th-leaky number presented as "physical" or "hardware-feasible": needs D2–D5.
- A shot-mode Nov9th claim: needs D5 (the readout model), and its \(W_L\) cost is unmodelled.
- FREEZE / validation / final for Nov9th: needs D1–D5 confirmed and NF_sim frozen.
