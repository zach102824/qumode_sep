"""Short train / eval loop on n=6&9 packs (PyTorch + Adam). Step 0 shared mixer.

Promotion / transfer scoring uses zero-shot n12 when a pack is available.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .ansatz import AnsatzConfig, MultiCavityAnsatz
from .device import resolve_device
from .gnn import FactorGNN, GNNArch
from .sat_gen import clause_count

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / "data"
RUNS = PKG / "runs"

FREEDOM_STEP = 0
SHARED_MIXER = True


def load_pack(n: int, data_dir: Path | None = None) -> dict:
    path = (data_dir or DATA) / f"data_n{n}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing pack {path}; run scripts/gen_teb_packs.py first")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _split_arrays(pack: dict, split: str, max_count: Optional[int] = None):
    seeds = pack[f"{split}_seeds"]
    cls_v = pack[f"{split}_cls_v"]
    cls_s = pack[f"{split}_cls_s"]
    nsol = pack[f"{split}_nsol"]
    has_E = f"{split}_E" in pack
    E = pack[f"{split}_E"] if has_E else None
    if max_count is not None:
        seeds = seeds[:max_count]
        cls_v, cls_s, nsol = cls_v[:max_count], cls_s[:max_count], nsol[:max_count]
        if E is not None:
            E = E[:max_count]
    return seeds, cls_v, cls_s, nsol, E


def fixed_flat_plus_alpha0(n: int, device, amp: float = 1.6) -> torch.Tensor:
    """Shared flat/plus coherent prep: real α, size-normalized. Shape (2,)."""
    scale_d = (64.0 / float(2 ** n)) ** 0.5
    a = float(amp) * scale_d
    return torch.tensor([a, 0.0], dtype=torch.float32, device=device)


def apply_alpha0_mode(heads: dict, n: int, mode: str, amp: float, device) -> dict:
    """Optionally replace GNN α₀ with fixed flat/plus prep (detach — no grad)."""
    if mode in (None, "", "learned"):
        return heads
    if mode in ("fixed_flat_plus", "flat_plus", "fixed"):
        heads = dict(heads)
        heads["alpha0"] = fixed_flat_plus_alpha0(n, device, amp=amp)
        return heads
    raise ValueError(f"unknown alpha0_mode={mode!r}")


def _forward_ansatz(sim: MultiCavityAnsatz, Eb: torch.Tensor, heads: dict) -> dict:
    return sim.forward(
        Eb,
        heads["alpha0"],
        heads["A"],
        heads["TH"],
        heads["gamma"],
    )


@torch.no_grad()
def eval_pack_metrics(
    n: int,
    arch: GNNArch,
    state: Optional[dict],
    split: str = "val",
    L: int = 5,
    max_count: int = 32,
    seed: int = 0,
    data_dir: Path | None = None,
    device: Union[str, torch.device, None] = None,
    nf: Optional[int] = None,
    alpha0_mode: str = "learned",
    alpha0_amp: float = 1.6,
    gnn_n: Optional[int] = None,
    gnn_m: Optional[int] = None,
) -> dict:
    """Forward GNN+ansatz on a split; return Med/B90 of 1/p_succ.

    ``state`` may come from a different train-n (zero-shot): shared-mixer head
    shapes do not depend on C, so n=9 weights load into an n=12 FactorGNN.
    Pass ``gnn_n``/``gnn_m`` only if you need to force the GNN graph size
    (defaults to pack n / pack m).
    """
    device = resolve_device(device)
    pack = load_pack(n, data_dir=data_dir)
    m = int(pack["m"])
    seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"Pack n={n} split={split} has no dense E; needed for ansatz eval")
    B = len(seeds)
    use_n = int(gnn_n) if gnn_n is not None else n
    use_m = int(gnn_m) if gnn_m is not None else m
    # For zero-shot the GNN must match the *eval* formula size (n,m of pack),
    # while weights come from a train-n checkpoint (same shapes).
    use_n, use_m = n, m
    gnn = FactorGNN(arch, n=use_n, m=use_m, L=L, seed=seed).to(device)
    if state is not None:
        missing, unexpected = gnn.load_state_dict(state, strict=False)
        # shared heads: ignore legacy head_q if present in old ckpt
        unexpected = [k for k in unexpected if not k.startswith("head_q")]
        if unexpected:
            raise RuntimeError(f"unexpected keys loading state into n={n}: {unexpected}")
    gnn.eval()
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = MultiCavityAnsatz(cfg, device=device).to(device)
    inv = []
    leaks = []
    t0 = time.time()
    for b in range(B):
        cv = torch.as_tensor(cls_v[b], device=device)
        cs = torch.as_tensor(cls_s[b], dtype=torch.float32, device=device)
        heads = gnn.forward_graph(cv, cs)
        heads = apply_alpha0_mode(heads, n, alpha0_mode, alpha0_amp, device)
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = _forward_ansatz(sim, Eb, heads)
        inv.append(float(out["inv_p"].detach().cpu()))
        leaks.append(float(out["p_leak"].detach().cpu()))
    inv_a = np.asarray(inv, float)
    med = float(np.median(inv_a))
    b90 = float(np.quantile(inv_a, 0.9))
    return {
        "n": n,
        "split": split,
        "B": B,
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": float((np.log(med) + np.log(b90)) / 2.0),
        "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0,
        "device": str(device),
        "freedom_step": FREEDOM_STEP,
        "shared_mixer": SHARED_MIXER,
    }


def combined_score(metrics_by_n: Dict[int, dict], w6: float = 0.35, w9: float = 0.65) -> float:
    """Lower is better. Primary n9, secondary n6 (train-n monitor)."""
    s = 0.0
    if 9 in metrics_by_n:
        s += w9 * metrics_by_n[9]["S"]
    if 6 in metrics_by_n:
        s += w6 * metrics_by_n[6]["S"]
    return float(s)


def transfer_score(metrics_n12: dict) -> float:
    """Primary promote metric: n12 zero-shot S = (log Med + log B90)/2. Lower better."""
    return float(metrics_n12["S"])


def pick_transfer_state(states_by_n: dict) -> Optional[dict]:
    """Prefer n=9 weights for zero-shot; fallback n=6."""
    if not states_by_n:
        return None
    if 9 in states_by_n:
        return states_by_n[9]
    if 6 in states_by_n:
        return states_by_n[6]
    return next(iter(states_by_n.values()))


def _instance_loss(out: dict, leakpen: float = 0.1) -> torch.Tensor:
    """Differentiable surrogate: log(1/p_succ) + leakpen * p_leak."""
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


def short_train(
    arch: GNNArch,
    steps: int = 20,
    L: int = 5,
    max_val: int = 16,
    max_train: int = 32,
    seed: int = 0,
    data_dir: Path | None = None,
    device: Union[str, torch.device, None] = None,
    leakpen: float = 0.1,
    nf: Optional[int] = None,
    log_fn=None,
    patience: Optional[int] = None,
    min_steps: int = 0,
    alpha0_mode: str = "learned",
    alpha0_amp: float = 1.6,
) -> Tuple[dict, dict, dict]:
    """Adam short-train of FactorGNN on n=6&9; keep best val combined score.

    Returns (val_metrics_by_n, best_state_by_n, info).
    """
    device = resolve_device(device)
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), f"[train:{device}]", *a, flush=True))
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    nets: Dict[int, FactorGNN] = {}
    sims: Dict[int, MultiCavityAnsatz] = {}
    packs: Dict[int, dict] = {}
    for n in (6, 9):
        try:
            pack = load_pack(n, data_dir=data_dir)
        except FileNotFoundError as e:
            log("skip", n, e)
            continue
        m = int(pack["m"])
        gnn = FactorGNN(arch, n=n, m=m, L=L, seed=seed + n).to(device)
        cfg = AnsatzConfig(n=n, L=L, nf=nf)
        sim = MultiCavityAnsatz(cfg, device=device).to(device)
        nets[n] = gnn
        sims[n] = sim
        packs[n] = pack

    if not nets:
        raise FileNotFoundError("No train packs found under data/")

    params = []
    freeze_a0 = alpha0_mode not in (None, "", "learned")
    for gnn in nets.values():
        if freeze_a0:
            for name, p in gnn.named_parameters():
                if name.startswith("head_alpha0"):
                    p.requires_grad_(False)
                else:
                    params.append(p)
        else:
            params.extend(list(gnn.parameters()))
    if not params:
        raise RuntimeError("No trainable parameters after alpha0 freeze")
    opt = torch.optim.Adam(params, lr=float(arch.lr), weight_decay=float(arch.wd))

    best_score = float("inf")
    best_states = None
    best_val = {}
    best_step = -1
    stopped_early = False
    steps_run = 0

    for step in range(steps):
        steps_run = step + 1
        opt.zero_grad(set_to_none=True)
        losses = []
        for n, gnn in nets.items():
            gnn.train()
            pack = packs[n]
            seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, "train", max_count=max_train)
            if E is None:
                continue
            B = len(seeds)
            bs = min(4, B)
            idx = np_rng.choice(B, size=bs, replace=False)
            sim = sims[n]
            for b in idx:
                cv = torch.as_tensor(cls_v[b], device=device)
                cs = torch.as_tensor(cls_s[b], dtype=torch.float32, device=device)
                heads = gnn.forward_graph(cv, cs)
                heads = apply_alpha0_mode(heads, n, alpha0_mode, alpha0_amp, device)
                Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
                out = _forward_ansatz(sim, Eb, heads)
                losses.append(_instance_loss(out, leakpen=leakpen))
        if not losses:
            raise RuntimeError("No train instances with dense E")
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        metrics = {}
        for n, gnn in nets.items():
            gnn.eval()
            metrics[n] = eval_pack_metrics(
                n, arch, gnn.state_dict(), split="val", L=L,
                max_count=max_val, seed=seed, data_dir=data_dir,
                device=device, nf=nf,
                alpha0_mode=alpha0_mode, alpha0_amp=alpha0_amp,
            )
        sc = combined_score(metrics)
        if sc < best_score:
            best_score = sc
            best_step = step
            best_states = {n: {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}
                           for n, gnn in nets.items()}
            best_val = metrics
            log(
                f"step {step}: NEW best S_comb={sc:.4f} loss={float(loss.detach()):.4f} "
                + " ".join(
                    f"n{n}:Med={metrics[n]['median_inv_p']:.2f}/B90={metrics[n]['B90_inv_p']:.2f}"
                    + f"/leak={metrics[n]['mean_leak']:.3f}"
                    for n in metrics
                )
            )
        elif step % 5 == 0:
            stagnant = step - best_step if best_step >= 0 else step
            log(f"step {step}: S_comb={sc:.4f} loss={float(loss.detach()):.4f} stagnant={stagnant}")

        if (
            patience is not None
            and best_step >= 0
            and step + 1 >= int(min_steps)
            and (step - best_step) >= int(patience)
        ):
            stopped_early = True
            log(
                f"early-stop at step {step}: no NEW best for {patience} steps "
                f"(best_step={best_step} S_comb={best_score:.4f})"
            )
            break

    info = {
        "best_score": best_score,
        "best_step": best_step,
        "steps": steps,
        "steps_run": steps_run,
        "stopped_early": stopped_early,
        "patience": patience,
        "min_steps": min_steps,
        "leakpen": float(leakpen),
        "L": int(L),
        "alpha0_mode": alpha0_mode,
        "alpha0_amp": float(alpha0_amp),
        "arch": arch.to_dict(),
        "seed": seed,
        "device": str(device),
        "backend": "torch",
        "freedom_step": FREEDOM_STEP,
        "shared_mixer": SHARED_MIXER,
    }
    return best_val, best_states or {}, info


def save_gnn_best(
    arch: GNNArch,
    states_by_n: dict,
    metrics: dict,
    runs_dir: Path | None = None,
    extra_meta: Optional[dict] = None,
):
    runs_dir = runs_dir or RUNS
    out = runs_dir / "gnn_best"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch.to_dict(),
        "metrics": {str(k): v for k, v in metrics.items()},
        "backend": "torch",
        "freedom_step": FREEDOM_STEP,
        "shared_mixer": SHARED_MIXER,
    }
    if extra_meta:
        payload.update(extra_meta)
    with open(out / "arch.json", "w") as f:
        json.dump(payload, f, indent=2)
    for n, state in (states_by_n or {}).items():
        cpu_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                cpu_state[k] = v.detach().cpu()
            else:
                cpu_state[k] = torch.as_tensor(v)
        torch.save(cpu_state, out / f"weights_n{n}.pt")
        legacy = out / f"weights_n{n}.npz"
        if legacy.exists():
            legacy.unlink()
    with open(out / "meta.json", "w") as f:
        json.dump(
            {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "ns": list((states_by_n or {}).keys()),
                "backend": "torch",
                "freedom_step": FREEDOM_STEP,
                "shared_mixer": SHARED_MIXER,
            },
            f,
        )


def load_gnn_best(runs_dir: Path | None = None) -> Tuple[Optional[GNNArch], dict]:
    runs_dir = runs_dir or RUNS
    arch_path = runs_dir / "gnn_best" / "arch.json"
    if not arch_path.exists():
        return None, {}
    meta = json.load(open(arch_path))
    arch = GNNArch.from_dict(meta["arch"])
    params = {}
    best_dir = runs_dir / "gnn_best"
    for p in best_dir.glob("weights_n*.pt"):
        n = int(p.stem.replace("weights_n", ""))
        params[n] = torch.load(p, map_location="cpu", weights_only=True)
    for p in best_dir.glob("weights_n*.npz"):
        n = int(p.stem.replace("weights_n", ""))
        if n in params:
            continue
        z = np.load(p)
        params[n] = {k: torch.as_tensor(z[k]) for k in z.files}
    return arch, params


def should_promote(cand_score: float, best_score: float, min_improve: float = 0.0) -> bool:
    """Best-only: promote iff cand is strictly better by min_improve (lower is better)."""
    return cand_score + min_improve < best_score


def stage_up_ok(cand_n12_med: float, step0_n12_med: float, min_rel: float = 0.10) -> bool:
    """Clause-w / εΔ stage-up only if n12 Med improves ≥ min_rel vs same-backbone step-0."""
    if step0_n12_med <= 0:
        return False
    return (step0_n12_med - cand_n12_med) / step0_n12_med >= min_rel
