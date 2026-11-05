# Shared-γ vs free per-layer γ (L=8, n=9, Γ=0.5)

Comparison:
- **FREE**: per-layer γ_ℓ + β_ℓ → **16 params** (current QAOA/fourier)
- **SHARED**: one optimized γ (same all layers) + β_ℓ → **9 params**

Same compare data (B_train=64, B_hold=32), 20 LBFGS rounds, seeds 0–2.

| case | params | best hold MedF | Δ vs free | seed holds | notes |
|---|---:|---:|---:|---|---|
| FREE per-layer γ | 16 | 0.4380 | 0.000 | 0.4373, 0.4380, 0.4380 | existing fourier L8 n9 |
| SHARED one γ | 9 | 0.3922 | -0.0458 | 0.3922, 0.3896, 0.3884 | g*=-0.4378 (best seed0) |

## Takeaway

- Shared γ hold **0.3922** vs free **0.4380** → **Δ = −0.0458** (−10.5% relative).
- Locking to one γ across layers costs ~0.046 hold MedF at L=8 n9 Γ=0.5; per-layer γ still helps.
