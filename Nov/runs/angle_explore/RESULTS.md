# Nov1st-final angle + SNAP explore — RESULTS

_Updated 2026-11-01 23:33 Asia/Shanghai_

Fixed `max_train=200`, train n∈{6,9}. Motif **Nov1st-final**.  
**Ping rule:** dual-beat sealed Med **and** B90 at **every** n∈{9,12,15,18}.

## Sealed bar (L=32)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

## Best so far: polish `adam800_lbfgs8` ← shell L32 seed0

| n | Med | B90 | Med× | B90× | dual? |
|---|-----|-----|------|------|-------|
| 9 | **1.090** | **1.708** | **0.998×** | **0.962×** | **YES** |
| 12 | 1.236 | 2.869 | 1.033× | 1.009× | no |
| 15 | 1.485 | 5.697 | 1.047× | 1.035× | no |
| 18 | 1.905 | 13.543 | 1.087× | 1.193× | no |

**n9 dual-beats sealed**; n12–18 still short (n18 largest gap). **No full-table dual-beat → no Zach ping.**

## Wave-1 baseline (pre-polish)

Progressive shell L32: n9 1.103/1.772 (1.010×/0.998×).  
SNAP bakeoff closest: `snap_free_perL` n9 1.130/1.897 — see `snap_bakeoff/RESULTS.md`.

## Wave-2 polish (running)

Warm-start shell/free L32 → Adam 600–1200 + heavy LBFGS + CVaR/lr.  
Other completed: cvar n9 1.096/1.689; lr1e-3 n9 1.092/1.694. Tier-B unlocking when n9 Med×≤1.05.

Flag if full beat: `BEAT_SEALED.json`. Live: `polish/RESULTS.md` on wave end.
