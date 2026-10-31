# Opus confirm: PROJECTOR_EQ_EXP_HC

_Generated 2026-10-31 14:11 Asia/Shanghai_

## What was asked

PDF: `explore_OCT29/docs/PROJECTOR_EQ_EXP_HC.pdf` (3 pages) + source `.tex`.

Prompt (ask mode): confirm or refute each claim —
1. \(I+(e^{-i\gamma}-1)P=\exp(-i\gamma P)\) for orthogonal projector \(P\)
2. \(\prod_c U_c(\gamma)=\exp(-i\gamma H_c)\) when \([C_c,C_{c'}]=0\)
3. Sparse / Fock-local apply ⇒ no densify \(E\in\mathbb{C}^{2^n}\), no dense \(2^n\times 2^n\) matrix-exp
4. \(W=H^{\otimes 3}\) is frame-only; identity holds in Fock/computational frame
5. Same physics as Arm C densified \(\mathrm{diag}(E)\) — complexity win only

Full note text pasted into `BRIEF.txt` (see `/tmp/ca_opus_proj_confirm/BRIEF.txt` on box).

## Live Opus 5.5 high

| Attempt | Command | Result |
|---------|---------|--------|
| 1 | `cursor-agent --print -f --mode ask --model claude-opus-5-5-high` from `explore_OCT29/docs` | **REGION BLOCK** — `ActionRequiredError: Model not available This model provider is not supported in your region.` |
| 2 (retry) | same | **REGION BLOCK** (identical) |

No live Opus reply. Prior explore_OCT29 Opus calls were likewise CN-region blocked.

## Self-check (NOT live Opus)

**Label: NOT live Opus — agent algebraic self-check only.**

| # | Claim | Verdict | Why |
|---|-------|---------|-----|
| 1 | \(I+(e^{-i\gamma}-1)P=\mathrm{e}^{-i\gamma P}\) | **CONFIRM** | \(P^2=P\Rightarrow P^k=P\) for \(k\ge1\); power series collapses. Equiv. spectrum \(\{0,1\}\). |
| 2 | \(\prod_c U_c=\mathrm{e}^{-i\gamma H_c}\) | **CONFIRM** | Needs mutual commutation (given: all \(C_c\) diagonal in same Fock basis). Then \(\prod\mathrm{e}^{A_i}=\mathrm{e}^{\sum A_i}\). Order free. |
| 3 | Sparse apply, no densify / no dense matrix-exp | **CONFIRM** | Complexity claim about *implementation*, not a new unitary. Each gadget is local SNAP / joint SNAP; product identity supplies correctness. |
| 4 | \(W=H^{\otimes3}\) frame only | **CONFIRM** | \(W^\dagger\mathrm{e}^{-i\gamma H_c}W=\mathrm{e}^{-i\gamma W^\dagger H_c W}\); spectrum unchanged; Fock-frame product identity untouched. |
| 5 | Same physics as Arm C | **CONFIRM** | On basis \(\lvert x\rangle\), both multiply by \(\mathrm{e}^{-i\gamma E(x)}\) with \(E(x)=\sum_c\langle x\rvert C_c\rvert x\rangle\). |

### Gaps noted (self-check, not Opus)

- Hypothesis \([C_c,C_{c'}]=0\) is essential; without it BCH does not truncate and claim 2 fails. TEB Fock projectors satisfy it because they share the computational eigenbasis.
- “Any product order” is free only under that commutation.
- Claim 3 is algorithmic; a naive simulator may still materialize amplitudes of length \(2^n\) for statevec — that does not contradict “no densify of \(E\) / no matrix-exp of \(U\)”.
- Span-\(>1\) clauses need joint (not product-of-local) SNAP; still sparse relative to \(2^n\), still equals \(\mathrm{e}^{-i\gamma C_c}\) for that clause projector.

## Overall one-liner

**Live Opus: REGION BLOCKED (×2).** Self-check (NOT live Opus): all five claims **CONFIRM**; projectors are an exp-avoiding implementation of \(\mathrm{e}^{-i\gamma H_c}\), not different cost physics.

## Artifacts

- `PROJECTOR_EQ_EXP_HC.tex` / `.pdf` / `.txt`
- This file: `PROJECTOR_EQ_EXP_HC_CONFIRM.md`
