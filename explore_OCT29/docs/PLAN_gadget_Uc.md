# PLAN — instance-free GRAPE story via X-gadget U_c

_Asia/Shanghai — explore_OCT29 locked path (Opus + Zach GO t2151)_

## Goal

Keep Arm C skeleton (BinH \(W=H^{\otimes 3}\), shared shell SNAP mixer) but **stop treating
diag\(E_{\mathrm{unsat}}(s)\) as the GRAPE target**. Cost layer is built from a **fixed
operator family** of X-gadgets; only \(\gamma_\ell\) (and shared \(\phi_{pc}\)) train.

## Instance-blind vs instance-aware

| Piece | Blind? | Notes |
|-------|--------|-------|
| Gadget **family** \(e^{-i\gamma X_j}\), \(e^{-i\gamma X_jX_k}\), \(e^{-i\gamma XXXX}\) | **YES** | GRAPE once per family / generator+time |
| Control Hamiltonians SNAP, \(D\), beamsplitter | **YES** | device model, not SAT |
| **Which** gadgets fire (clause → bit support + polarity) | NO | classical schedule from instance |
| \(\gamma_\ell\), shell \(\phi_{pc}\) | shared across instances | trained |
| Fixed stencil \(\prod_j e^{-i\gamma X_j}\) | **YES** fully | no clause list in \(U_c\) |

## Arms

1. **gadget-clause (BinH)**  
   \(U_c(\gamma)=\prod_{\mathrm{clauses}\,c}\exp(-i\gamma\,P_c),\quad P_c=(\prod_i s_i)\prod_i X_{v_i}\).  
   All pure-X strings commute ⇒ \(U_c=W^\dagger e^{-i\gamma\lambda}W\) with
   \(\lambda(z)=\sum_c\prod_i s_i(-1)^{z_{v_i}}\) (parity / XOR cost — **not** unsat count).  
   Sim may use Z-parity diag + BinH sandwich for speed; GRAPE story uses X-ops.

2. **fixed-stencil**  
   \(U_c(\gamma)=\prod_{j=0}^{n-1}\exp(-i\gamma X_j)\) (product of 1-local X). Fully instance-blind.

Mixer both: Arm C `W · SNAP(φ_pc) · W†`, prep `W`, binary Fock packing, L=16.

## Train / eval

- Train `{9,12}`, L=16, warm γ/φ from Arm C if present else champ PROJ.
- Eval: val/test packs + scale1000 when feasible (B=1000 n9/12/15; n18 as time allows).
- **Promote** only if Med/B90 competitive with Arm C B1000:
  n9 1.441/3.314, n12 1.991/6.252, n15 2.921/10.959, n18 4.573/27.810.

## Non-goals

- Do not densify unsat \(E(s)\) as a GRAPE pulse target.
- Do not revive Arm B (X-only spectral θ=β·pc) as cost.

## Result (2026-10-30 ~19:18 CST)

- parity / stencil: **negative** on SAT Med/B90.
- unsat clause-projector (BinH): **matches Arm C** scale1000 n9–15; this is the GRAPE path (per-clause projector gadgets, conjugated to X under BinH).
- Do not promote XOR-string or blind stencil for 4-SAT.
- RESULTS: `runs/gadget_Uc/RESULTS.md`

## SOFT-PROMOTE (2026-10-30 ~19:30 CST)

- Projector-gadget unsat: scale1000 n9–18 competitive with Arm C; **promoted family**.
- n18 Med/B90 **4.329/24.342** vs Arm C 4.573/27.810.
- Drop parity & stencil from further runs.
- Fourier-γ deepen: **discard** (see fourier_deepen/RESULTS.md).
