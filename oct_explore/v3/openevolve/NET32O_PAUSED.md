# net32o paused (2026-10-20 Asia/Shanghai)

Zach decision: stop net32o completely; keep only net16o.

- OpenEvolve process for `runs/net32o` stopped (SIGTERM).
- `watch32_chunked.sh` / `run_paused` n=20 checkers stopped.
- Checkpoints and program DB kept on disk: `runs/net32o/checkpoints/checkpoint_{10,20,30}`.
- net16o continues under the **3.2 GB MemAvailable floor** with `eval_governor.py` (one eval at a time). Floor not lowered.
- Do not auto-restart net32o. Do not touch `*.STOP` or PID 2888494.
