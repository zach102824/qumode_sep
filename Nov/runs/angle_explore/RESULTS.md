# Nov1st-final angle + SNAP explore — RESULTS

_Updated 2026-11-01 23:43 Asia/Shanghai — polish wave-2 live + watchdog wave-3 queued_

Fixed `max_train=200`, train n∈{6,9}. Motif **Nov1st-final**.  
**Ping rule:** dual-beat sealed Med **and** B90 at **every** n∈{9,12,15,18}. **Not yet.**

## Sealed bar (L=32)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

## Best Med (n9 dual-beat): `adam800_lbfgs8` ← shell seed0

| n | Med | B90 | Med× | B90× |
|---|-----|-----|------|------|
| 9 | **1.090** | **1.708** | **0.998×** | **0.962×** |
| 12 | 1.236 | 2.869 | 1.033× | 1.009× |
| 15 | 1.485 | 5.697 | 1.047× | 1.035× |
| 18 | 1.905 | 13.543 | 1.087× | 1.193× |

## Best B90 ladder: `cvar_q09_adam600` ← shell seed0

| n | Med | B90 | Med× | B90× |
|---|-----|-----|------|------|
| 9 | 1.099 | **1.677** | 1.006× | **0.944×** |
| 12 | 1.242 | **2.786** | 1.038× | **0.980×** |
| 15 | 1.482 | **5.496** | 1.045× | **0.998×** |
| 18 | 1.891 | 12.999 | 1.079× | 1.145× |

B90 beats sealed at n9/12/15; Med still short everywhere; n18 both short.

## Status

- Wave-1 progressive+SNAP bakeoff: done (pushed).
- Wave-2 `--phase polish`: running (PID alive); tier-B unlocked.
- Watchdog queued for wave-3 heavy polish when wave-2 exits.
- Stop: `Nov/runs/STOP_ANGLE_EXPLORE`. Beat flag: `BEAT_SEALED.json`.
