# GRAPE library — clause-projector gadgets (3-bit cavity, BinH)

_Generated 2026-10-30 19:36:14 +0800 (Asia/Shanghai)_

## Finite operator family

4-SAT + K=3 packing ⇒ clause cavity-spans ∈{2,3,4} (never 1). Local support per cavity k∈{1,2,3}.

Each local factor is a **fixed** projector

\[ C_S = \prod_{(j,s)\in S} \frac{I - \mathrm{sgn}'(s)\,Z_j}{2} \]

with BinH image \(C^X_S\) (Z→X). Unitary

\[ U(\gamma)= I + (e^{-i\gamma}-1)\,C \]

— **same C**, only γ changes. GRAPE / pulse library = these finitely many C (and two-cavity products \(C_a\otimes C_b\)).

- Patterns k=1: **6** (choose 1 bit × 2 signs)
- Patterns k=2: **12**
- Patterns k=3: **8**
- **Total local: 26**

Two-cavity clause: \(C=C^{(a)}\otimes C^{(b)}\) with \((k_a,k_b)\in\{(1,3),(3,1),(2,2)\}\). Still fixed op × γ.

### Catalog (labels)

- **k=1** (rank=4): b0_s-, b0_s+, b1_s-, b1_s+, b2_s-, b2_s+
- **k=2** (rank=2): b01_s--, b01_s-+, b01_s+-, b01_s++, b02_s--, b02_s-+, b02_s+-, b02_s++, b12_s--, b12_s-+, b12_s+-, b12_s++
- **k=3** (rank=1): b012_s---, b012_s--+, b012_s-+-, b012_s-++, b012_s+--, b012_s+-+, b012_s++-, b012_s+++

## Exact pulse identities (proof sketch)

1. **Z-frame (lab Fock):** \(C_Z\) is diagonal 0/1 on Fock 0..7 ⇒ \(U_Z(\gamma)=\mathrm{SNAP}(\theta)\) with \(\theta_f=\gamma\) on \(\mathrm{supp}(C_Z)\), else 0. **Exact** — no GRAPE needed for single-cavity factors.
2. **X-frame (BinH):** \(U_X=W^\dagger U_Z W\) with \(W=H^{\otimes 3}\) ⇒ \(U_X=W^\dagger\mathrm{SNAP}(\theta)W\). Exact if W is available (Arm C sandwich).
3. **SNAP+D without W** fitting \(U_X\) in Fock basis: numerical (table below) — optional compile path.

## Fidelity table (d=8)

| pattern | k | γ | Fid SNAP↔UZ | Fid W†SNAP W↔UX | Fid SNAP·D↔UX | Fid D·SNAP·D↔UX | BinH conj fro |
|---------|---|---|-------------|-----------------|---------------|-----------------|---------------|
| b0_s- | 1 | 0.2 | 1.000000 | 1.000000 | 0.9952 | 0.9975 | 1.28e-15 |
| b0_s- | 1 | 0.5 | 1.000000 | 1.000000 | 0.9703 | 0.9852 | 1.39e-15 |
| b0_s- | 1 | 1.0 | 1.000000 | 1.000000 | 0.8869 | 0.9427 | 1.76e-15 |
| b0_s- | 1 | 1.5 | 1.000000 | 1.000000 | 0.7647 | 0.8777 | 1.26e-15 |
| b0_s+ | 1 | 0.2 | 1.000000 | 1.000000 | 0.9952 | 0.9976 | 1.59e-15 |
| b0_s+ | 1 | 0.5 | 1.000000 | 1.000000 | 0.9703 | 0.9852 | 1.39e-15 |
| b0_s+ | 1 | 1.0 | 1.000000 | 1.000000 | 0.8869 | 0.9427 | 1.48e-15 |
| b0_s+ | 1 | 1.5 | 1.000000 | 1.000000 | 0.7647 | 0.8777 | 1.26e-15 |
| b01_s-- | 2 | 0.2 | 1.000000 | 1.000000 | 0.9928 | 0.9945 | 1.27e-15 |
| b01_s-- | 2 | 0.5 | 1.000000 | 1.000000 | 0.9560 | 0.9679 | 1.64e-15 |
| b01_s-- | 2 | 1.0 | 1.000000 | 1.000000 | 0.8341 | 0.8777 | 1.73e-15 |
| b01_s-- | 2 | 1.5 | 1.000000 | 1.000000 | 0.6626 | 0.8125 | 1.52e-15 |
| b01_s-+ | 2 | 0.2 | 1.000000 | 1.000000 | 0.9955 | 0.9957 | 1.58e-15 |
| b01_s-+ | 2 | 0.5 | 1.000000 | 1.000000 | 0.9719 | 0.9765 | 1.62e-15 |
| b01_s-+ | 2 | 1.0 | 1.000000 | 1.000000 | 0.8920 | 0.9101 | 1.72e-15 |
| b01_s-+ | 2 | 1.5 | 1.000000 | 1.000000 | 0.7730 | 0.8169 | 1.53e-15 |
| b012_s--- | 3 | 0.2 | 1.000000 | 1.000000 | 0.9958 | 0.9962 | 1.58e-15 |
| b012_s--- | 3 | 0.5 | 1.000000 | 1.000000 | 0.9743 | 0.9831 | 1.60e-15 |
| b012_s--- | 3 | 1.0 | 1.000000 | 1.000000 | 0.9030 | 0.9172 | 1.96e-15 |
| b012_s--- | 3 | 1.5 | 1.000000 | 1.000000 | 0.8024 | 0.8635 | 1.70e-15 |
| b012_s--+ | 3 | 0.2 | 1.000000 | 1.000000 | 0.9957 | 0.9961 | 1.58e-15 |
| b012_s--+ | 3 | 0.5 | 1.000000 | 1.000000 | 0.9733 | 0.9766 | 1.91e-15 |
| b012_s--+ | 3 | 1.0 | 1.000000 | 1.000000 | 0.8998 | 0.9110 | 1.96e-15 |
| b012_s--+ | 3 | 1.5 | 1.000000 | 1.000000 | 0.7972 | 0.8216 | 2.00e-15 |

### Takeaway

- SNAP↔UZ and W†SNAP W↔UX should be **=1** (exact) within numerics.
- SNAP+D fit to UX is a fallback if W is not native; expect high but not exact fidelity.
- **GRAPE never needs instance `diag(E)`** — only library ops \(U(\gamma;C)\) for C in this finite set, scheduled by the clause list.

