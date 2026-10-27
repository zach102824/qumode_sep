# n=8 Phase-1 STATUS

Updated: 2026-10-27 11:00:30 CST (Asia/Shanghai)

Done: **7** runs | target core=24 then densify

Bars: WS look Med/B90=13.989/29.396 | uniform floor Med≈128.0 | promote val≲64.0 | hot≲40.0


## Leaderboard (by test Med)

| id | L | ntrain | H/r/steps/lr | val Med/B90 | test Med/B90 | leak | best_step | sec | flag |

|---|---:|---:|---|---:|---:|---:|---:|---:|---|

| `A_L5_n10000` | 5 | 10000 | 128/4/1400/0.0011 | 39.9/116.9 | 38.3/145.2 | 0.0543 | 1400 | 400 | ≪floor |
| `B_L3_n10000` | 3 | 10000 | 80/8/2600/0.0006 | 72.7/227.6 | 62.5/238.4 | 0.0224 | 2600 | 331 | ≪floor |
| `A_L3_n5000` | 3 | 5000 | 128/4/1400/0.0011 | 70.8/301.9 | 72.2/378.3 | 0.0466 | 1400 | 306 | <floor |
| `A_L3_n2000` | 3 | 2000 | 128/4/1400/0.0011 | 80.3/351.4 | 80.8/409.4 | 0.0483 | 840 | 298 | <floor |
| `B_L3_n5000` | 3 | 5000 | 80/8/2600/0.0006 | 95.8/252.6 | 98.5/278.2 | 0.0374 | 2600 | 310 | <floor |
| `A_L3_n10000` | 3 | 10000 | 128/4/1400/0.0011 | 115.8/308.2 | 106.0/272.0 | 0.0564 | 1400 | 325 | <floor |
| `B_L3_n2000` | 3 | 2000 | 80/8/2600/0.0006 | 137.1/273.3 | 138.0/274.5 | 0.0348 | 2600 | 311 |  |

## Core grid pivot (test Med)


### Anchor A

| L \ ntrain | 2000 | 5000 | 10000 |

|---|---:|---:|---:|

| **3** | 80.8 | 72.2 | 106.0 |

| **5** | — | — | 38.3 |

| **7** | — | — | — |

| **9** | — | — | — |


### Anchor B

| L \ ntrain | 2000 | 5000 | 10000 |

|---|---:|---:|---:|

| **3** | 138.0 | 98.5 | 62.5 |

| **5** | — | — | — |

| **7** | — | — | — |

| **9** | — | — | — |


## Notes

- Best so far: `A_L5_n10000` test Med **38.32** (B90 145.2) best_step=1400 weights=`/workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1/runs/A_L5_n10000/weights.npz`

- Lock: SNAP+D→cost, tied cost, vacuum, pure GNN+cw, same L train/test, n=8 only.

- Queue auto-continues into densify after core 24.
