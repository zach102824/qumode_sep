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

## GRAPE library + train6912 (2026-10-30 ~19:38 CST)

### Library
- 26 local C on 3-bit cavity; U(γ)=I+(e^{-iγ}-1)C; two-cavity C_a⊗C_b.
- Exact SNAP (Z) and W†SNAP W (X/BinH). Doc: `runs/gadget_Uc/grape_library/LIBRARY.md`.

### Performance
- train{6,9,12} **promoted** over train{9,12}:
  - n15 Med 2.754 < prior 2.820
  - n18 Med 4.194 < prior 4.329
- Champ: `runs/gadget_Uc/unsat_params.pt`

### Still banned
- GRAPE targeting densified instance diag(E) as the pulse target.
- parity XOR / stencil_x / Fourier deepen for SAT.

## Freedom ladder + two-cavity census (2026-10-30 ~19:45 CST)

### Two-cavity factors
- Span-2 ⇒ \(C=C_a\otimes C_b\) with local \((1,3)\) or \((2,2)\); in GRAPE library.
- Frac of clauses that are two-cavity: n9 36%, n12 18%, n15 11%, n18 7% (declines with n as span-3/4 dominate).

### Freedom probes (train{6,9,12}) — DISCARD
- **span_w**: learned softplus weights on E^(span=2,3,4); converged ≈(1,1,1); n15 2.695 / n18 4.120 — fails joint 2% gate on n18.
- **eps_delta**: local δ on Fock 0..7 + bounded ε; ε→0; n15 2.697 / n18 4.123 — same.
- Keep champ `unsat_params.pt` (train{6,9,12} plain projector).

Artifacts: `runs/gadget_Uc/freedom/`.

## Depth L=20 PROMOTE (2026-10-30 ~19:53 CST)

- Pad L20/L24: see `depth/RESULTS.md` (interp alone).
- **Retrain L=20** on {6,9,12}: n15 Med **2.168**, n18 Med **3.170** (gate pass).
- Champ → L=20 `unsat_params.pt`. b_Med≈1.1079 (best vs Arm C 1.137 / QAOA 1.151).
- Next: L=24 retrain.

## Depth L=24 CHAMP (2026-10-30 ~19:59 CST)

- Pad L20/L24 and retrain L20/L24 all beat L16 gate.
- **Champ = retrain L24**: n15 **1.820**, n18 **2.492**; b_Med≈1.0868.
- Params: `runs/gadget_Uc/unsat_params.pt`.
- Next: L=28 or span-split γ_ℓ^(s) on L24.

## Depth L=28 CHAMP (2026-10-30 ~20:06 CST)

- Retrain L28 vs L24 gate: n15 **1.655**, n18 **2.183** — PROMOTE.
- b_Med≈**1.0749** (best so far vs Arm C 1.137 / QAOA 1.151).
- Next: span-split γ on L28, or one L=32 check then stop depth ladder.
