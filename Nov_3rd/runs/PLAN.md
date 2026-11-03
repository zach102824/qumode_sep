# PLAN — Nov_3rd (final data)

_Asia/Shanghai 2026-11-03 — Zach t2442 / t2443_

Fresh package under `Nov_3rd/` — **does not** reuse `Nov/` CLI defaults (no Adam, no max_train=200, no train {6,9}, no Gray, no b>3).

## Motif
\[
|0\rangle \xrightarrow{[U_c^{X}\to\mathrm{bare\,SNAP}]^{L}} W \to \text{Fock},
\quad U_c^{X}=W\,D\,W,\quad D=\mathrm{e}^{-i\gamma E},\quad W=H^{\otimes 3}.
\]

## Locked (`settings.py`)
| knob | value |
|------|-------|
| b | **3** |
| W | **BinH** |
| SNAP | **shell**, shared across cavities, per layer |
| Init | **QAOA-warm** \(\phi_j=-\beta j\), \(\gamma=\gamma_{\mathrm{QAOA}}\) |
| Train | **n=6 only**, **1000** samples |
| Opt | **L-BFGS only** (no Adam) |
| L | **16** |
| Eval | scale1000 **B=1000**, n∈{6,9,12,15,18} |

## Run
```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov_3rd.run_final
```

Outputs → `Nov_3rd/runs/`.
