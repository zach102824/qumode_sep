# SNAP easiness — RESULTS

_Updated 2026-11-02 11:03 Asia/Shanghai_

Motif locked: Nov1st-final. Train n∈{6,9} max_train=200. Eval scale1000 B=1000.

Sealed: n9 1.092/1.776, n12 1.197/2.843, n15 1.418/5.505, n18 1.752/11.353.



## Ladder (params/layer ↑)

| tag | L | params | n9 Med/B90 | ×Med/×B90 | n12 | n15 | n18 | verdict |
|-----|---|--------|------------|-----------|-----|-----|-----|---------|
| `easy_snap_gauss_perL_L32_s0` | 32 | 3 | 1.071/1.597 (0.981×/0.899×) | 0.9806949249554029/0.8991747020601151 | 1.186/2.218 (0.991×/0.780×) | — | — | PROMOTE_CAND |
| `easy_snap_gauss_shared_L32_s0` | 32 | 3 | 1.436/2.141 (1.315×/1.205×) | 1.3147507940893206/1.2054837636045503 | 1.855/3.346 (1.550×/1.177×) | — | — | PROMOTE_CAND |

## Most-restricted survivors (n9 Med ≤ 1.10× sealed)

- **easy_snap_gauss_perL_L32_s0** params=3 n9 1.071/1.597 (0.981×/0.899×)

**BEAT_SEALED (restricted SNAP dual-beat all n):** NO

Stop file: `Nov/runs/STOP_SNAP_EASY`.
