# PLAN — Nov1st-final (nonstop angle + SNAP-head explore)

_Asia/Shanghai — Zach t2307 GO + SNAP parameterization add._

## Motif (locked)

**Nov1st-final** ≡ sealed BinH-proj:

\[
|0\rangle \xrightarrow{[U_c^{X}\to\mathrm{SNAP}]^{L}} W \to \text{Fock},\quad U_c^{X}=W D W,\ D=\mathrm{e}^{-i\gamma E},\ W=H^{\otimes 3}.
\]

**Killed:** prep \(W\) + \(U_c^{X}\) + bare SNAP + final \(W\). Do not revive.

## Fixed train size (do not grow pack)

| Knob | Lock |
|------|------|
| Train \(n\) | \{6, 9\} |
| `max_train` | **200** (fixed) |
| `max_val` | 40 |
| Eval | scale1000 **B=1000**, n 9/12/15/18 |
| \(L\) | progressive **1,4,8,…,32**; SNAP bakeoff **8,16,32** |

Sealed target Med/B90: n9 **1.092/1.776**, n12 **1.197/2.843**, n15 **1.418/5.505**, n18 **1.752/11.353**.

## Angle-finding arms (optimizer / schedule)

1. Adam lr/steps · 2. Adam→LBFGS · 3. Multi-seed · 4. Init transfer · 5. Loss variants · 6. Progressive L · 7. Sealed twin

## SNAP angle parameterization (KEY add)

So far baseline = **shell** \(\theta_k=\varphi_{\mathrm{pc}(k)}\) (4 / layer). Also train:

| Head | \(\theta_k\) | params / layer (or shared) |
|------|--------------|----------------------------|
| `shell` | \(\varphi_{\mathrm{pc}(k)}\) | 4 |
| `free` | free \(\theta_{0..7}\) | 8 |
| `bits` | \(\sum_b \alpha_b\,\mathrm{bit}_b(k)\) | 3 |
| `fourier` | DC + Fourier \(M=2\) on Fock | 5 |
| `proj` | \(\theta_0=\beta\), else 0 | 1 |
| `x_pop` | \(\beta\cdot\mathrm{pc}(k)\) | 1 |

Each × **per-layer** vs **shared-across-layers**.

Goal: which SNAP head gets scale1000 Med/B90 closest to sealed (fixed data).

Promote / kill: same as before (n9 Med ≫10 → kill; within ~2× sealed → promote cand). Full B=1000 on top-4 screen heads.

## Code / outputs

```text
Nov/
  PLAN_Nov1st_final.md
  scripts/nov1st_final_train.py
  scripts/snap_heads.py
  scripts/angle_explore.py
  runs/angle_explore/
    progressive/          # shell baseline ladder
    snap_bakeoff/         # per-head RESULTS
    leaderboard.csv
```

```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov.scripts.angle_explore --phase all --max_train 200 --steps 400 \
  --seeds 0,1,2 --snap_L 8,16,32 --scale_B 1000
```

Stop: `Nov/runs/STOP_ANGLE_EXPLORE`.

## Status

**Wave-1 DONE** (progressive shell + SNAP bakeoff). Closest: progressive shell L32 scale1000 n9 **1.103/1.772** (1.010×/0.998×) — no dual-beat.

**Wave-2 RUNNING** — `--phase polish`: warm-start shell/free/bits L32 → Adam 600–1200 + heavy LBFGS + CVaR/lr. Fixed max_train=200. Ping only if dual-beats sealed Med **and** B90 at n9/12/15/18.

```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov.scripts.angle_explore --phase polish --max_train 200 --scale_B 1000
```

## SNAP easiness / restriction ladder (Zach t2320)

Angle polish is ≈ sealed on Med → **restrict bare SNAP** so hardware is easiest (Gaussian / Kerr / few-tone / low-degree poly), motif locked.

Papers: `Nov/runs/snap_easy/PAPERS.md`. Runner: `Nov/scripts/snap_easy_explore.py`.

| Mode | params/L | Meaning |
|------|----------|---------|
| `const` | 1 | global phase (sanity kill) |
| `kerr` | 2 | \(\chi k(k-1)/2+\omega k\) |
| `gauss` | 3 | \(\alpha+\beta\,\mathrm{pc}+\delta\,\mathrm{pc}^2\) |
| `poly2_n` | 3 | quadratic on Fock \(n\) |
| `few_m` | M | only first M Fock phases free |
| `gauss_plus1` | 4 | gauss + residual on pc=3 |
| `poly3_*` | 4 | cubic on pc or \(n\) |
| `shell` / `free` | 4 / 8 | reference / upper bound |

Also × **shared-across-L**. Warm-start: LS-project best shell polish → restricted head, then Adam+LBFGS.

```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov.scripts.snap_easy_explore --L 16,32 --seeds 0,1 --steps 400 --scale_B 1000
```

Stop: `Nov/runs/STOP_SNAP_EASY` (independent of angle polish).
Ping only if a **restricted** SNAP dual-beats sealed Med **and** B90 at every n∈{9,12,15,18}.
