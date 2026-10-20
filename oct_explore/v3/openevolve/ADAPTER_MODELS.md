# Adapter models (2026-10-21 Asia/Shanghai)

- stage1 ideas: `claude-opus-5-5-medium` (Zach: Opus only for ideas)
- stage2 code: `composer-2.5` (exact cursor-agent id)
- net32o remains paused (see NET32O_PAUSED.md)
- floor 3.2 GB MemAvailable + `eval_governor.py` unchanged

Smoke (2026-10-21 ~07:32 CST):
- `composer-2.5` / `composer-2.5-fast` direct: OK (PONG)
- adapter code-only path (`model=composer-2.5`): OK (`ADAPTER_OK`)
- all tested Opus 5.5 / 4.8 variants: **region blocked** (`Model provider is not supported in your region`)
- two-stage smoke therefore fails until Opus is available in-region
