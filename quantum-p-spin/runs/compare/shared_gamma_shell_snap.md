# Shared γ + shell SNAP vs QAOA

Protocol: B_train=64, B_hold=32, cost 1−F, 20 LBFGS rounds, seeds 0–2, Γ=0.5.

## Warm re-init (Zach)

Cold shared-γ+shared-φ(5) underperformed shared-γ+β. Nested manifold `φ_k=-β·k` should let free φ recover ≥ shared-γ+β if optim lands on it.

Warm protocol:
1. Train/reuse shared-γ+β champs (save β_ℓ, g*).
2. Init shared φ_k = mean_ℓ(−β_ℓ·k), keep g*.
3. L-BFGS free `{γ, φ_0..3}` (5 params).
4. Control: freeze per-layer `φ_{ℓ,k}=-β_ℓ k` (exact shared-γ+β eval).

### n=9, L=8

| case | params | best hold MedF | Δ vs free | Δ vs shared-γ+β | seed holds | notes |
|---|---:|---:|---:|---:|---|---|
| FREE QAOA (γ_ℓ+β_ℓ) | 16 | 0.4380 | 0.0000 | +0.0458 | — | existing fourier L8 n9 |
| SHARED γ + β_ℓ | 9 | 0.3922 | -0.0458 | 0.0000 | 0.3922, 0.3896, 0.3884 | retrained+saved β |
| SHARED γ + shared φ_k **cold** | 5 | 0.3278 | -0.1102 | -0.0644 | 0.3246, 0.3278, 0.3268 | prior cold init |
| SHARED γ + shared φ_k **warm** | 5 | 0.3301 | -0.1079 | -0.0621 | 0.3301, 0.3280, 0.3286 | from β-proj; recovers≥gb? **False** |
| SHARED γ + φ_{ℓ,k} (per-layer) | 33 | 0.4197 | -0.0183 | +0.0276 | 0.4197, 0.4164, 0.4070 | prior cold; **does** nest β_ℓ |
| control freeze φ=-βk (per-ℓ) | 9 | 0.3922 | -0.0458 | +0.0000 | 0.3922, 0.3896, 0.3884 | should ≈ shared-γ+β |

Warm0 (proj only, no BFGS): 0.2784, 0.2864, 0.2781 → best 0.2864.

### Takeaway (n9 L8)

- Warm shared-φ hold **0.3301** vs cold **0.3278** (Δcold +0.0023) vs shared-γ+β **0.3922** (Δgb -0.0621).
- Recovers ≥ shared-γ+β: **False** (shared-across-layers φ cannot express per-layer β_ℓ).
- Control freeze-linear max hold 0.3922 (Δ vs gb +0.000000) — φ_{ℓ,k}=-β_ℓ k reproduces gb exactly.
- Per-layer shell φ (33) **does** nest shared-γ+β and already beat it cold (0.4197).

**Answer to nesting:** free shared φ_k (same all layers) nests only `shared γ + one β` (2 params), not `shared γ + β_ℓ`. Warm cannot fall back to the β_ℓ champ when L>1. L=1 is the exception (shared ≡ per-layer) — and there warm does recover ≥ gb.

## n=6, L ∈ {1,2,4,8}

| L | free | shared-γ+β | φ cold | φ warm | φ warm0 | ctrl freeze | Δ(warm−gb) | recovers≥gb? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.6869 | 0.6869 | 0.6835 | 0.6870 | 0.6869 | 0.6869 | +0.0001 | True |
| 2 | 0.7411 | 0.7102 | 0.6996 | 0.7007 | 0.6667 | 0.7102 | -0.0095 | False |
| 4 | 0.8179 | 0.7843 | 0.7094 | 0.6896 | 0.6841 | 0.7843 | -0.0947 | False |
| 8 | 0.9203 | 0.8529 | 0.7150 | 0.7790 | 0.7790 | 0.8529 | -0.0739 | False |

### Takeaway (n6)

- L=1: free **0.6869**, gb **0.6869**, cold φ **0.6835**, warm φ **0.6870** (Δgb +0.0001, recovers≥gb=True).
- L=2: free **0.7411**, gb **0.7102**, cold φ **0.6996**, warm φ **0.7007** (Δgb -0.0095, recovers≥gb=False).
- L=4: free **0.8179**, gb **0.7843**, cold φ **0.7094**, warm φ **0.6896** (Δgb -0.0947, recovers≥gb=False; warm < cold — BFGS path, not nesting).
- L=8: free **0.9203**, gb **0.8529**, cold φ **0.7150**, warm φ **0.7790** (Δgb -0.0739, recovers≥gb=False).

Artifacts: `shared_gamma_shell_snap_warm.json`, `shared_gb_ckpts/`, `shared_gamma_shell_snap.json` (cold), `n6_shared_gamma_sweep.json`, `fixed_gamma_ablation.json`.
