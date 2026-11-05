# quantum-p-spin

Quantum p=3 spin glass with Nov_3rd **Hamming-shell SNAP** (4 φ_k per cavity).

\[
H=-\frac{\sqrt3}{n}\sum_{i<j<k} J_{ijk} Z_i Z_j Z_k - \Gamma\sum_i X_i,\quad J\sim\mathcal N(0,1)
\]

## Smoke

```bash
cd /workspace/qumode_sep/quantum-p-spin
bash run_smoke.sh
```

- Train fidelity `1-|⟨ψ|GS⟩|²` on n∈{6,9} (joint + per-n for 1/n interp)
- L∈{1,2,4,8}, Γ∈{0.5,1.0,1.5}, seeds 0–2, B_train=64 / hold=32
- Freeze + linear interp in `1/n` → eval median F at n=12,15

Does **not** modify `Nov_3rd/` (imports ansatz only).

## QAOA/Fourier vs SNAP (Γ=0.5)

```bash
bash run_qaoa_vs_snap.sh
# OUT_DIR=... REUSE_SNAP=runs/smoke/20261105_smoke1 INCLUDE_L16=0
```

Fourier = φ_k=−βk (2L params); SNAP = free φ_0..3 (5L). See `runs/compare/*/RESULTS.md`.
