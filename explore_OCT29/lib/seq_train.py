"""Train / eval helpers for gate-sequence motifs (shared fixed-angle, no GNN)."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .ansatz import AnsatzConfig
from .device import resolve_device
from .seq_ansatz import Motif, MultiCavitySeqAnsatz, init_params
from .train_loop import RUNS, _split_arrays, load_pack


def _S(med: float, b90: float) -> float:
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def _instance_loss(out: dict, leakpen: float = 0.1) -> torch.Tensor:
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


@torch.no_grad()
def eval_motif_pack(
    n: int,
    motif: Motif,
    params: dict,
    split: str = "val",
    max_count: int = 24,
    seed: int = 0,
    device=None,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
) -> dict:
    device = resolve_device(device)
    pack = load_pack(n, data_dir=data_dir)
    seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"Pack n={n} split={split} missing dense E")
    cfg = AnsatzConfig(n=n, L=motif.L, nf=nf)
    sim = MultiCavitySeqAnsatz(cfg, motif, device=device).to(device)
    # move params
    p = {k: v.detach().to(device) for k, v in params.items()}
    invs = []
    leaks = []
    t0 = time.time()
    for b in range(len(seeds)):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_motif(Eb, p)
        invs.append(float(out["inv_p"].detach().cpu()))
        leaks.append(float(out["p_leak"].detach().cpu()))
    invs_a = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n,
        "split": split,
        "B": int(len(invs)),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0,
        "device": str(device),
        "motif": motif.name,
        "freedom_step": 0,
        "shared_mixer": True,
        "param_mode": "fixed_angle_shared",
    }


def short_train_motif(
    motif: Motif,
    steps: int = 80,
    lr: float = 3e-3,
    max_train: int = 32,
    max_val: int = 16,
    max_n12: int = 12,
    seed: int = 0,
    device=None,
    leakpen: float = 0.1,
    nf: Optional[int] = None,
    data_dir: Path | None = None,
    log_fn=None,
) -> Tuple[dict, dict, dict]:
    """Adam on shared fixed-angle params; train n=6&9; keep best by combined S; score n12."""
    device = resolve_device(device)
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), "[seq]", *a, flush=True))
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    params = init_params(motif, seed=seed, device=device)
    opt = torch.optim.Adam(list(params.values()), lr=lr)

    sims: Dict[int, MultiCavitySeqAnsatz] = {}
    packs = {}
    for n in (6, 9):
        pack = load_pack(n, data_dir=data_dir)
        cfg = AnsatzConfig(n=n, L=motif.L, nf=nf)
        sims[n] = MultiCavitySeqAnsatz(cfg, motif, device=device).to(device)
        packs[n] = pack

    best_score = float("inf")
    best_params = None
    best_val = {}
    best_step = -1

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for n, sim in sims.items():
            pack = packs[n]
            seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, "train", max_count=max_train)
            if E is None:
                continue
            B = len(seeds)
            bs = min(4, B)
            idx = np_rng.choice(B, size=bs, replace=False)
            for b in idx:
                Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
                out = sim.forward_motif(Eb, params)
                losses.append(_instance_loss(out, leakpen=leakpen))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 5.0)
        opt.step()

        # val every step is expensive; do every step for short screens, every 5 for longer
        do_val = (steps <= 120) or (step % 5 == 0) or (step + 1 == steps)
        if not do_val:
            continue
        metrics = {}
        for n in sims:
            metrics[n] = eval_motif_pack(
                n, motif, params, split="val", max_count=max_val, seed=seed,
                device=device, nf=nf, data_dir=data_dir,
            )
        # combined like train_loop: 0.35 n6 + 0.65 n9
        sc = 0.35 * metrics[6]["S"] + 0.65 * metrics[9]["S"]
        if sc < best_score:
            best_score = sc
            best_step = step
            best_params = {k: v.detach().cpu().clone() for k, v in params.items()}
            best_val = metrics
            log(
                f"{motif.name} step {step}: NEW best S_comb={sc:.4f} loss={float(loss.detach()):.4f} "
                + " ".join(f"n{n}:Med={metrics[n]['median_inv_p']:.2f}/B90={metrics[n]['B90_inv_p']:.2f}" for n in metrics)
            )
        elif step % 10 == 0:
            log(f"{motif.name} step {step}: S_comb={sc:.4f} stagnant={step - best_step}")

    if best_params is None:
        best_params = {k: v.detach().cpu().clone() for k, v in params.items()}
        best_val = {
            n: eval_motif_pack(n, motif, params, split="val", max_count=max_val, seed=seed, device=device, nf=nf, data_dir=data_dir)
            for n in sims
        }

    # zero-shot n12 with best shared params
    m12 = None
    try:
        load_pack(12, data_dir=data_dir)
        m12 = eval_motif_pack(
            12, motif, best_params, split="val", max_count=max_n12, seed=seed,
            device=device, nf=nf, data_dir=data_dir,
        )
        best_val[12] = m12
        log(f"{motif.name} n12 zero-shot: Med={m12['median_inv_p']:.3f} B90={m12['B90_inv_p']:.3f} S={m12['S']:.4f}")
    except FileNotFoundError:
        log("n12 pack missing")

    info = {
        "best_score": best_score,
        "best_step": best_step,
        "steps": steps,
        "lr": lr,
        "motif": motif.to_dict(),
        "n_params": motif.n_params(),
        "seed": seed,
        "device": str(device),
        "param_mode": "fixed_angle_shared",
        "S_n12": None if m12 is None else m12["S"],
    }
    return best_val, best_params, info


def save_seq_best(
    motif: Motif,
    params: dict,
    metrics: dict,
    info: dict,
    runs_dir: Path | None = None,
    tag: str = "seq_best",
):
    runs_dir = runs_dir or RUNS
    out = runs_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    import json
    payload = {
        "motif": motif.to_dict(),
        "metrics": {str(k): v for k, v in metrics.items()},
        "info": {k: v for k, v in info.items() if k != "motif"},
        "param_mode": "fixed_angle_shared",
        "freedom_step": 0,
    }
    (out / "motif.json").write_text(json.dumps(payload, indent=2) + "\n")
    torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "params.pt")
    return out
