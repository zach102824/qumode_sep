# PLAN — BinH-proj n6 L-ladder (cost-sandwich + bare SNAP)

_Asia/Shanghai — locked 2026-11-01 (Zach t2281). **Do not start training/eval until Zach says go.**_

## Status

| Item | Value |
|------|-------|
| Status | **PLAN ONLY — waiting for go** |
| Script (planned) | `explore_OCT29/scripts/binh_proj_L_ladder_n6.py` (**not written yet**) |
| Output dir (planned) | `explore_OCT29/runs/binh_proj_n6_Lladder/` |
| Related champ lineage | `docs/PLAN_gadget_Uc.md`, `runs/gadget_Uc/` (historical **mixer-sandwich** / Arm-C shell writeup) |

This is a **new explore arm**. It is **not** a re-eval of the sealed BinH-proj champ under a different algebra writeup.

## Locked gate sequence (cost-sandwich + bare SNAP)

Zach lock (2026-11-01): **sandwich the cost, bare SNAP as mixer** — no \(W\) sandwich on the mixer.

1. **Prep:** Fock \(\lvert 0\rangle\) → apply \(W = H^{\otimes 3}\) per cavity  
2. **Each layer \(\ell\):** cost \(W\,U_c^{Z}(\gamma_\ell)\,W^\dagger\)  
3. **Same layer:** bare mixer \(\mathrm{SNAP}(\phi_\ell)\) (**no** \(W^\dagger\mathrm{SNAP}\,W\) / \(W\,\mathrm{SNAP}\,W^\dagger\))  
4. Repeat steps 2–3 for \(L\) layers  
5. **Final** \(W\), then Fock measure  

Here \(U_c^{Z}\) is the Z-frame clause-projector / densified cost unitary (e.g. \(\exp(-i\gamma H_c)\) via unsat projectors).  
Cost after conjugation is the X-frame operator \(U_c^{X} = W U_c^{Z} W^\dagger\), but in this writeup the explicit \(W\ldots W^\dagger\) sits on the **cost**, and the mixer stays bare SNAP.

### What this is / is not

| Form | Layer cost | Layer mixer | This arm? |
|------|------------|-------------|-----------|
| **Cost-sandwich + bare SNAP** (LOCKED) | \(W U_c^{Z} W^\dagger\) | bare \(\mathrm{SNAP}\) | **YES** |
| Mixer-sandwich (historical sealed BinH-proj writeup) | \(U_c^{X}=W U_c^{Z} W^\dagger\) | \(W^\dagger\mathrm{SNAP}\,W\) | no — different arm / prior champ |
| Bare \(U_c^{Z}\) + bare SNAP (dropped \(W\)’s) | bare \(U_c^{Z}\) | bare \(\mathrm{SNAP}\) | **NO** — not equivalent |
| Sandwich both / rearrange prep–cost–mixer–readout \(W\)’s freely | — | — | **NO** |

**Warning:** conjugating **one** local gate (\(U_X = W^\dagger U_Z W\)) is **not** license to move every prep / cost / mixer / final \(W\) freely. Cost-sandwich + bare SNAP is the locked circuit for this ladder; do not “simplify” by dropping or one-sided-moving \(W\)’s.

Unitary note (for humans): a correct mixer-sandwich writeup can be algebraically the same circuit as cost-sandwich when the same operators are used consistently. This explore arm still **implements and reports** the **cost-sandwich + bare SNAP** sequence as written above. Historical champ numbers below are from the prior sealed lineage (mixer-sandwich / Arm-C shell implementation in `gadget_Uc_explore.py`: prep \(W\), Z-diag cost in that frame, mixer \(W\,\mathrm{SNAP}\,W^\dagger\)) — use only as a reference bar, not as proof this arm is already done.

## Train / eval

| Knob | Locked plan |
|------|-------------|
| Train \(n\) | **\(n=6\) only** (`data/data_n6.npz`) |
| Pack | existing TEB-style **2000 / 40 / 40** (train / val / test) |
| `max_train` | default **32** (same convention as other explore scripts; subsample of the 2000 train seeds) |
| `max_val` | **12** (match sibling scripts unless go changes it) |
| Depth ladder \(L\) | \(\{1,4,8,12,16,20,24,28,32\}\) |
| Cost | BinH-proj clause projectors / densified \(U_c^{Z}\) → conjugated as \(W U_c^{Z} W^\dagger\) |
| Mixer | **bare SNAP** shared shell \(\phi_\ell\) (or \(\phi_{pc}\) schedule as in sibling BinH-proj scripts — keep shared-across-cavities) |
| \(W\) | BinH \(H^{\otimes 3}\) (no Gray) |
| Eval | **scale1000** Med / B90 (n9 / n12 / n15 / n18 as packs allow) |
| Optimizer | follow sibling BinH-proj practice (Adam ± LBFGS polish) when implementing — exact hyperparams at go time |

## Planned artifacts

```text
explore_OCT29/
  docs/PLAN_binh_proj_n6_Lladder.md          # this file
  scripts/binh_proj_L_ladder_n6.py           # TO WRITE on go (not present yet)
  runs/binh_proj_n6_Lladder/                 # TO CREATE on go
    RESULTS.md
    ladder.json / ladder.csv
    params_L{L}.pt
    run.log
```

## Script requirement (when go)

- There is **no** draft `binh_proj_L_ladder_n6.py` yet.
- Historical runners (`scripts/gadget_Uc_explore.py` and depth / soft-n15 / LBFGS siblings) use the **prior champ sandwich** (prep \(W\); diag cost; mixer `W @ SNAP @ W†`). **Do not** reuse that mixer sandwich unchanged for this arm.
- On go: implement (or fork) so the forward pass matches **cost-sandwich + bare SNAP** exactly as in the locked steps. Comments/docstrings must say cost-sandwich + bare SNAP; do not label it mixer-sandwich.

## Reference bars (historical sealed BinH-proj — not this arm)

Sealed BinH-proj / projector-Uc L=32 LBFGS-style scale1000 (prior lineage; mixer-sandwich / Arm-C shell implementation):

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

(Also see depth ladder tables in `runs/gadget_Uc/depth_ladder/` and `docs/PLAN_gadget_Uc.md`.)  
This n6 L-ladder must produce **its own** scale1000 table under `runs/binh_proj_n6_Lladder/`.

## Do not start until go

- No training, no scale1000 eval, no param writes under `runs/binh_proj_n6_Lladder/` until Zach says **go**.
- Plan/doc updates only are allowed before go (this file).
