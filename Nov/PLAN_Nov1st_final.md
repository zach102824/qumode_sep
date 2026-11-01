# PLAN — Nov1st-final (sandwich-cost BinH ≡ sealed BinH-proj)

_Asia/Shanghai — locked 2026-11-01 (Zach t2296 / t2297)._  
**Status: PLAN + SKELETON only. Do not launch full scale1000 until Zach says go after reviewing this plan.**

## Name

**Nov1st-final** = sandwich-cost BinH rewrite that is unitary-equivalent to sealed BinH-proj.

## Locked gate sequence (Nov1st-final)

\[
|0\rangle \;\xrightarrow{\;[U_c^{X}(\gamma_\ell)\to\mathrm{SNAP}(\phi_\ell)]^{L}\;}\; W \;\to\; \text{Fock measure}
\]

with \(W=W^\dagger=H^{\otimes 3}\) (BinH), \(D=U_c^{Z}=\exp(-i\gamma E)\) (densified unsat / clause projectors on the NU grid), and

\[
U_c^{X}=W\,D\,W.
\]

Steps:

1. Start Fock \(|0\rangle\) — **no prep \(W\)**
2. Each layer \(\ell\): apply \(U_c^{X}(\gamma_\ell)=W\,D(\gamma_\ell)\,W\)
3. Same layer: **bare** \(\mathrm{SNAP}(\phi_\ell)\) (shared shell \(\phi_{pc}\))
4. Repeat 2–3 for \(L\) layers
5. **Final \(W\)**, then Fock measure

### Sealed-code equivalent (same unitary)

What `GadgetAnsatz.forward_gadget(..., cost_mode="unsat")` does:

1. Prep \(W\)
2. Each layer: densified diag \(D=\mathrm{e}^{-i\gamma C_n}\) on NU grid
3. Mixer \(W\,\mathrm{SNAP}\,W^\dagger\)
4. Fock measure — **no** final \(W\)

Algebra (\(W^2=I\)): one sealed layer \(W S W\,D\) telescopes to

\[
U_{\mathrm{sealed}}=W\Big(\prod_{\ell} S_\ell\,(W D_\ell W)\Big)=U_{\text{Nov1st-final}}.
\]

Numeric smoke (n=6, L=2, random angles): Nov1st-final vs sealed overlap \(\approx 0.99999976\), \(\max|\Delta p|\sim 10^{-8}\).

### NOT this arm (killed / wrong)

| Form | Why wrong |
|------|-----------|
| Prep \(W\) → \([W D W^\dagger \to\) bare SNAP\(]^L\) → final \(W\) | Extra outer \(W\)'s; Med blew up vs sealed (cost-sandwich arm killed 2026-11-01) |
| Prep \(W\) → \([U_c^{X}\to\) bare SNAP\(]^L\) → no final \(W\) | Overlap \(\sim 0.997\), not sealed |
| Mixer-sandwich writeup with final \(W\) | Not sealed forward |

## Train / eval (proposed)

| Knob | Value |
|------|-------|
| Motif | **Nov1st-final** (or `--use-sealed` → call `forward_gadget` unsat; same unitary) |
| \(W\) | BinH \(H^{\otimes 3}\) only (no Gray) |
| Cost | densified unsat \(D=\mathrm{e}^{-i\gamma C_n}\) (sealed BinH-proj) |
| Mixer | bare SNAP shell \(\phi_\ell\in\mathbb{R}^4\) (popcount schedule) |
| Train \(n\) | **\(\{6,9\}\)** only (`data/data_n6.npz`, `data/data_n9.npz`) |
| Pack sizes | n6/n9: train **2000** / val **40** / test **40** |
| `max_train` | **48** default (propose 32–64; sibling gadget default was 32) |
| `max_val` | **12** per train \(n\) |
| Opt | Adam `lr=2e-3`, `steps=150`; LBFGS polish at mid + final (gadget_Uc style, `max_iter=4`) |
| Depth | **default single \(L=32\)** (sealed champ lineage); optional ladder later |
| Eval | **scale1000** \(B=1000\), \(n\in\{9,12,15,18\}\) Med / B90 |
| Success | Med/B90 within ~15% of sealed ref below |

### Sealed BinH-proj reference (success bar)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

## Folder layout

```text
Nov/                            # repo root (NOT under explore_OCT29)
  PLAN_Nov1st_final.md          # this file
  README.md
  scripts/
    nov1st_final_train.py       # skeleton: smoke + gated train/eval
  runs/Nov1st_final/            # outputs (params, results.json, RESULTS.md, run.log)
```

## Proposed CLI (skeleton)

```bash
cd /workspace/qumode_sep
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep

# 1) Smoke only (default) — overlap Nov1st-final vs sealed forward_gadget
python -m Nov.scripts.nov1st_final_train --smoke

# 2) Train L=32 on n∈{6,9}  (NO scale1000 until --scale1000)
python -m Nov.scripts.nov1st_final_train \
  --train --L 32 --train_ns 6,9 --max_train 48 --steps 150

# 3) After train: scale1000 B=1000 (Zach go)
python -m Nov.scripts.nov1st_final_train \
  --scale1000 --L 32 --params Nov/runs/Nov1st_final/params_L32.pt

# Optional: train/eval through sealed forward (should match Nov1st-final)
python -m Nov.scripts.nov1st_final_train --train --use-sealed ...
```

## Implementation notes

- Prefer implementing `forward_nov1st_final` explicitly; keep `--use-sealed` as cross-check / alternate path.
- Before claiming train results, `--smoke` must report overlap ≈ 1 vs `forward_gadget(..., "unsat")`.
- Do **not** reuse `binh_proj_L_ladder_n6.py` forward (that was the killed prep-\(W\) + final-\(W\) arm).
- Copy-paste style from `gadget_Uc_explore.py` / `binh_proj_L_ladder_n6.py` train/eval helpers, retargeted to Nov1st-final + train \(\{6,9\}\).


## Smoke status (skeleton)

Ran 2026-11-01 22:04 Asia/Shanghai (CST):

```
smoke n6 L2: |⟨s|n⟩|=0.99999988 max|Δp|=2.980e-08 PASS
```

Skeleton ready. **No train / scale1000 launched.**

## Go / no-go

| Gate | Condition |
|------|-----------|
| Plan review | Zach sees this PLAN + CLI (now) |
| Smoke | overlap ≈ 1 on n6 smoke instance |
| Train | Zach go → `--train` |
| scale1000 | Zach go → `--scale1000`; compare to sealed table |

