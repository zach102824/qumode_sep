# OpenEvolve on the native cavity-chain ansatz (Cursor models only)
- `cursor_adapter.py`: OpenAI-compatible `/v1/chat/completions` on 127.0.0.1:8765 that runs `cursor-agent -p --model <m>` in an empty temp dir. The key comes only from env `CURSOR_API_KEY`; it is never stored or logged.
- `initial_program.py`: evolvable `design(n, ks, clauses)` that maps a formula to placement (`perm`), projection measure (`pv`), per-layer gamma/beta (beta can be per cavity), clause weights `w`, and optional per-cavity SNAP tables.
- `evaluator.py`: fair single-assignment median/B90 of 1/p at n=15 (5/5/5), ideal one-bit-flip mixers, chain projection. Validation is 100 formulas from seed base 33e9 (`OE_NVAL`, `OE_BASE`); `OE_BASE=32e9` gives the official test set. It reproduces the F32_gray_chain test result exactly (115.9/1046.8). Guards: import whitelist, design time ≤1.5 s per formula, and a 20-point score penalty if pv matches a true solution above 0.85 (to catch classical solving).
- `run_smoke.sh`: starts the adapter if needed and runs `ITER` iterations into `runs/` (gitignored).

## net16r pre-screen tune (2026-10-22)
Measured clean eval ≈ 260-275 s (s1 ≈ 73 s: 55 s train + ~18 s compile/screen/diag; s2 ≈ 190 s: 163 s train + ~27 s VAB).
LLM code call median ≈ 137 s (p90 ≈ 1170 s, host freezes). Old s1 margin 0.35 cut 0/46 programs.
Change: s1 cut margin 0.35 -> 0.20 (live file runs/net16r/s1_margin; picked up when workers reload at phase relaunch),
persistent JAX cache runs/jaxcache16r for --inner children (live). Calibration: 20% of stage-2 programs would be cut,
0 of top-decile cut, ckpt_10 (s1 -3.656) and ckpt_110 (s1 -3.654) pass vs threshold -3.90. Stage-2 fitness unchanged.
