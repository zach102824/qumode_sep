# GNN_TIECOST — OpenEvolve adaptive GNN training rule (tied cost)

## Goal
Beat WalkSATlm 4-SAT TEB lookup growth **1.19^n Med / 1.21^n B90** (n=8→20).

## Locked circuit (NOT searched)
- Each layer = **SNAP+D → cost** (order fixed; SNAP+D **params** free).
- Cost = **one unitary per formula**, identical across all L layers (gamma + clause_w tied).
- Vacuum start, pure GNN + clause_w, SNAP+D only (no ECD).
- Same L at train and test; different L = different net.

## OpenEvolve role
Evolve an **adaptive training rule** `rule(L) -> hypers` (H, rounds, arch, lr, steps, bs, wd, leakpen, train_ns, ntrain, …).
Always include {6,8}; rule may add 9 and/or 12 (9/12 support: requested flag; Phase-1 eval uses 6+8 data).

## Models
- Ideas: **claude-opus-5-5-high** (short idea turns; prior CURSOR_API_KEY)
- Code: **cursor-grok-4.6-high-fast**
- Adapter: `cursor_adapter_tiecost.py` on :8767

## Paths
- Run dir: `runs/gnn_tiecost/`
- Config: `config_gnn_tiecost.yaml`
- Evaluator: `net/evaluator_gnn_tiecost.py`
- Trainer lib: `net/tiecost_lib.py`
- Initial program: `net/initial_gnn_tiecost.py`
- Driver: `gnn_tiecost_driver.sh`  Stop: `runs/gnn_tiecost/gnn_tiecost.STOP`
- Status: `runs/gnn_tiecost/STATUS.md`  Monitor: `bash gnn_tiecost_mon.sh`

## OE search L
Default `OE_L=5` (fast iterations). Adaptive rule still conditions on L for Phase 2 ladder.

## Classical bar (lookups, WalkSATlm p=0.30)
n6 8.5/16 | n8 14.0/29.4 | n10 22.9/49.7 | n12 32.5/76.1 | n16 71/147 | n20 110/290
