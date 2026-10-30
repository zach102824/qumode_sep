# Cheap Med probe — γ multiplicative jitter (renames Hc)

_Generated 2026-10-31 01:13:13 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. Per-layer `γ_ℓ ← γ_ℓ(1+σ ξ_ℓ)`, ξ~N(0,1). Not blind.

| σ | n9 Med/B90 | n12 Med/B90 | vs champ n9 Med |
|---|------------|-------------|-----------------|
| 0.00 | 1.092/1.776 | 1.197/2.843 | ×1.000 |
| 0.01 | 1.095/1.767 | 1.201/2.845 | ×1.003 |
| 0.02 | 1.095/1.788 | 1.204/2.859 | ×1.003 |
| 0.05 | 1.125/1.796 | 1.247/2.910 | ×1.030 |

Verdict: small σ (≤2%) keeps n9/12 Med within ~few %; σ=5% starts to hurt.

