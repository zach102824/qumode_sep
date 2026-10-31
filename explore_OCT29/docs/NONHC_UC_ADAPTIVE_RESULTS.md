# Adaptive non-Hc Uc probes — verdict

_Generated 2026-10-31 10:22:07 (Asia/Shanghai)_

Ask (Zach): different Uc that does **not** secretly encode Hc.
This track: **measurement / sample-feedback adaptive** cost (not fixed exp(−iγ Hc)).
Prior fixed non-Hc (surrogate_pair / noncomm_xz / wrong_graph) all KILL — left intact in `runs/nonhc_Uc/`.

## Feature / honesty docs

### `fb_marginal`

After each mixer, read computational bit-marginals m_i=P(bit_i=1) from the state (soft 'shots'). Classical detach → exploratory 1-body field h_i=−(2m_i−1) (exploratory); next cost diagonal E[x]=∑_i h_i z_i(x) with z=1−2x. No clauses enter the cost. Sequence is measurement-feedback adaptive → not a fixed unitary exp(−iγ Hc) / projector product / densified E.

### `fb_break`

After each mixer, take soft expected assignment (or mode) from bit marginals; score WalkSAT-style break/make per variable on that assignment using the instance clauses; classical detach → 1-body field h_i ∝ (make_i − break_i). Cost diagonal E[x]=∑ h_i z_i — 1-local, ≠ densified unsat count Hc=∑_c C_c. Adaptive across layers from samples.

### Algebra / channel note

```json
{
  "fixed_unitary": false,
  "equals_densified_Uc": false,
  "equals_exp_Hc": false,
  "note": "Cost diagonal at layer \u2113+1 is a classical function of layer-\u2113 computational marginals/samples (detached). Channel is adaptive measurement-feedback, not a fixed U=exp(-i\u03b3 Hc)."
}
```

## Verdict table (L=16, train{9,12}, W=H^{⊗3} BinH, screen B=200)

| arm | =exp(Hc)? | fixed Uc? | n9 Med/B90 | n12 | n15 | vs ArmC n15 | vs trunc_Hc n15 | verdict |
|-----|-----------|-----------|------------|-----|-----|-------------|-----------------|---------|
| Arm C densify | YES | YES | 1.441/3.314 | 1.991/6.252 | 2.921/10.959 | ×1.0000 | ×65.7912 | PROMOTE (≈ArmC — investigate leakage to Hc) |
| projector (=Hc) | YES | YES | 1.406/3.288 | 1.864/6.106 | 2.754/10.244 | ×1.0606 | ×69.7807 | PROMOTE (≈ArmC — investigate leakage to Hc) |
| trunc_Hc (prior) | NO (partial) | YES | 20.796/64.953 | 63.160/214.865 | 192.176/912.036 | ×0.0152 | ×1.0000 | KILL (Med=192.2 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) |
| fb_marginal | NO | NO (adaptive) | 256.001/512.002 | 2048.005/4096.013 | 16384.057/32768.141 | ×0.0002 | ×0.0117 | KILL (Med=16384.1 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) |
| fb_break | NO | NO (adaptive) | 115.558/929.332 | 927.799/7621.431 | 6078.449/69208.806 | ×0.0005 | ×0.0316 | KILL (Med=6078.4 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) |

## Interpretation

- **fb_marginal**: KILL (Med=16384.1 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) — =exp(Hc)? **NO**; fixed Uc? **NO**. After each mixer, read computational bit-marginals m_i=P(bit_i=1) from the state (soft 'shots'). Classical detach → exploratory 1-body field…
- **fb_break**: KILL (Med=6078.4 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) — =exp(Hc)? **NO**; fixed Uc? **NO**. After each mixer, take soft expected assignment (or mode) from bit marginals; score WalkSAT-style break/make per variable on that assignment…

## Artifacts

- Out: `/workspace/qumode_sep_fresh/explore_OCT29/runs/nonhc_Uc_adaptive`
- Script: `scripts/probe_nonhc_Uc_adaptive.py`
- JSON: `runs/nonhc_Uc_adaptive/results.json`
- Prior fixed non-Hc: `runs/nonhc_Uc/` (untouched)

ArmC L44 fair-bar stopped / not needed.

## One-line verdicts

1. **fb_marginal** — KILL. Soft bit-marginal feedback → exploratory 1-body diagonal (no clauses; adaptive ≠ fixed exp(−iγ Hc); honesty affine R²≈0.02). n15 Med≈16384 ≈ 2^{n-1} uniform. B=200; not promoted.
2. **fb_break** — KILL. WalkSAT make−break 1-body field from soft mode (1-local ≠ densified Hc; adaptive; R²≈0.07). n15 Med≈6078 ≫ trunc_Hc 192 / ArmC 2.92. B=200; not promoted.

**Bottom line:** adaptive/sample-feedback non-Hc Uc also fails the screen. Still no competitive Uc that does not secretly encode Hc.

Screen B=200 (promote threshold: beat trunc_Hc n15). Honesty samples in `results.json`.
