# explore_OCT29

OCT 29 **small→large transfer** track for coherent multi-cavity SAT
(`|0⟩ → D(α₀) → [SNAP → D → U_Φ]^L`).

Self-contained snapshot of `qtimes_cost` (2026-10-29). Theory is **locked**;
the PyTorch code in this folder is still the **pre-lock** free-`U_phase^q` /
per-cavity-mixer path — runnable for smoke, then rewrite per `PLAN.md` /
`TODO_REWRITE.md`.

| Doc | What |
|-----|------|
| [`PLAN.md`](PLAN.md) | Theory lock, freedom ladder, train/eval ns, so-far bars |
| [`RUN.md`](RUN.md) | Exact box / Mac CPU commands |
| [`TODO_REWRITE.md`](TODO_REWRITE.md) | File-level step-0 rewrite checklist (do **not** code until Zach says **go**) |

## Locus

- **Box / Mac CPU / cloud** — this track.
- **Chaodao GPU** — deferred (do not park explores there for now).
- Git source of truth: `/workspace/qumode_sep/explore_OCT29` → Mac `~/qumode_sep` via `git pull`.

## Train / eval

- Train: **n = 6 & 9**
- Eval (zero-shot / scale): **12 / 15 / 18** (21 optional later)
- Optional two-model: FactorGNN **L=16** for n≤9; SizeGeneral **L=24** for n≥12

## Ban list

- Free `U_phase^q` / continuous-q diagonal heads
- Per-cavity **private** SNAP+D mixers (breaks fixed-angle / light-cone transfer)
- n-C diagonal / nonlocal memorization heads
- New cost operator per layer (Φ is **one** per instance, shared across layers)

## Quick start

```bash
cd /workspace/qumode_sep   # or ~/qumode_sep on Mac
pip install -r explore_OCT29/requirements.txt
export PYTHONPATH=/workspace/qumode_sep
pytest explore_OCT29/tests -q
```

Full commands: [`RUN.md`](RUN.md).
