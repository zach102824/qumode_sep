# Two-cavity projector compile (simulation)

_Generated 2026-10-30 20:34:09 +0800 (Asia/Shanghai)_

## Setup

Clause spanning two cavities: \(C=C_a\otimes C_b\) with local projectors from the frozen library.
Target \(U(\gamma)=I+(e^{-i\gamma}-1)C\) on \(\mathbb{C}^8\otimes\mathbb{C}^8\) (Fock trunc 0..7).

Representatives:

- **(1,3):** \(C_a=\Pi_{b0}\), \(C_b=\Pi_{b0,b1,b2}\) (rank 4 ⊗ rank 1)
- **(2,2):** \(C_a=\Pi_{b0,b1}\), \(C_b=\Pi_{b0,b1}\) (rank 2 ⊗ rank 2)

## Does \(U\) factor as locals?

**No.** If \(U_a=I+(e^{-i\gamma}-1)C_a\) and \(U_b=I+(e^{-i\gamma}-1)C_b\), then

\[ U_a\otimes U_b = I + (e^{-i\gamma}-1)(C_a\otimes I + I\otimes C_b) + (e^{-i\gamma}-1)^2 C_a\otimes C_b \]

which is **not** \(I+(e^{-i\gamma}-1)C_a\otimes C_b\). The joint unitary phases only the
**AND** of supports \(\mathrm{supp}(C_a)\times\mathrm{supp}(C_b)\); product of local
projector-unitaries also phases when only one cavity is in support.

### (1,3)

- ranks (Ca, Cb, C) = (4, 1, 4)
- supp_a = [0, 2, 4, 6], supp_b = [0]
- Fid(\(U\), \(U_a\otimes U_b\)) = **0.773743** (fro=5.753e+00)
- factors as local projector-unitaries? **False**

### (2,2)

- ranks (Ca, Cb, C) = (2, 2, 4)
- supp_a = [0, 4], supp_b = [0, 4]
- Fid(\(U\), \(U_a\otimes U_b\)) = **0.773743** (fro=5.074e+00)
- factors as local projector-unitaries? **False**

## Compile paths

| path | what | exact? |
|------|------|--------|
| **Joint Fock SNAP** (Z-frame) | phase γ on \|fa,fb⟩ iff fa∈supp_a and fb∈supp_b | **YES** |
| BinH sandwich | \(W^\dagger\otimes W^\dagger\) · jointSNAP · \(W\otimes W\) | **YES** (X-frame) |
| Local SNAP⊗SNAP | separable phases θa(fa)+θb(fb) | **NO** (misses AND) |
| BS · (SNAP⊗SNAP) · BS† | entangling sandwich | approximate (table) |
| Cross-Kerr / EigenSNAP | native #a#b-dependent phase | hardware-native AND |

## Fidelity table

| split | γ | Fid joint SNAP | Fid best local SNAP⊗SNAP | Fid BS sandwich | Fid BinH local⊗ in X | needs entangling |
|-------|---|----------------|--------------------------|-----------------|----------------------|------------------|
| (1,3) | 0.5000 | 1.000000 | 0.9932 | 0.9932 | 0.9397 | True |
| (1,3) | 1.0000 | 1.000000 | 0.9731 | 0.9732 | 0.7737 | True |
| (1,3) | 3.1416 | 1.000000 | 0.7811 | 0.7812 | 0.0156 | True |
| (2,2) | 0.5000 | 1.000000 | 0.9911 | 0.9913 | 0.9397 | True |
| (2,2) | 1.0000 | 1.000000 | 0.9651 | 0.9657 | 0.7737 | True |
| (2,2) | 3.1416 | 1.000000 | 0.7656 | 0.7656 | 0.0156 | True |

## Takeaway for GRAPE / hardware

1. Single-cavity factors stay **exact local SNAP** (library FREEZE).
2. Two-cavity clauses **require a joint / entangling compile** — not SNAP_a⊗SNAP_b.
3. Preferred sim compile: **joint number-dependent diagonal** (or BinH-conjugated).
4. Preferred hardware story: **cross-Kerr / EigenSNAP / BS+SNAP sandwich** GRAPE with
   target \(U(\gamma)=I+(e^{-i\gamma}-1)C_a\otimes C_b\) — still **not** densified diag(E).
5. Instance only picks which \((C_a,C_b)\) pair from the finite library.

Artifacts: `compile_twocav.json`. Champ classical angles unchanged.

