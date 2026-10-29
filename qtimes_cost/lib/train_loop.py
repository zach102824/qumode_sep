"""Short train / eval loop on n=6&9 packs (PyTorch + Adam).

Uses CUDA when available. Dual-GPU evolve launches two short_train calls
on cuda:0 and cuda:1 in parallel (see scripts/evolve_gnn.py).
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .ansatz import AnsatzConfig, MultiCavityAnsatz, check_q_mode
from .device import resolve_device
from .gnn import FactorGNN, GNNArch

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / "data"
RUNS = PKG / "runs"


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


def _to_torch_heads_device(gnn: FactorGNN, device: torch.device):
    gnn.to(device)
    return gnn


def _param_checksum(gnn: FactorGNN, skip_prefix: str = "head_q") -> str:
    """SHA256 of named parameters, excluding skip_prefix (the q head)."""
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in gnn.named_parameters():
            if name.startswith(skip_prefix):
                continue
            h.update(name.encode())
            arr = p.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def _as_float_list(xs) -> list:
    out = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            out.append(float(x.detach().cpu()))
        else:
            out.append(float(x))
    return out


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
    q_mode: str = "learned",
    return_instances: bool = False,
) -> dict:
    """Forward GNN+ansatz on a split; return Med/B90 of 1/p_succ."""
    q_mode = check_q_mode(q_mode)
    device = resolve_device(device)
    pack = load_pack(n, data_dir=data_dir)
    m = int(pack["m"])
    seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"Pack n={n} split={split} has no dense E; needed for ansatz eval")
    B = len(seeds)
    gnn = FactorGNN(arch, n=n, m=m, L=L, seed=seed).to(device)
    if state is not None:
        gnn.load_state_dict(state)
    gnn.eval()
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = MultiCavityAnsatz(cfg, device=device).to(device)
    inv = []
    leaks = []
    q_rows = []
    t0 = time.time()
    for b in range(B):
        cv = torch.as_tensor(cls_v[b], device=device)
        cs = torch.as_tensor(cls_s[b], dtype=torch.float32, device=device)
        heads = gnn.forward_graph(cv, cs)
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward(
            Eb,
            heads["alpha0"],
            heads["A"],
            heads["TH"],
            heads["q_logits"],
            gamma=heads["gamma"],
            q_mode=q_mode,
        )
        inv.append(float(out["inv_p"].detach().cpu()))
        leaks.append(float(out["p_leak"].detach().cpu()))
        q_rows.append(_as_float_list(out["q_cont"]))
    inv_a = np.asarray(inv, float)
    leak_a = np.asarray(leaks, float)
    q_arr = np.asarray(q_rows, float) if q_rows else np.zeros((0, L), float)
    med = float(np.median(inv_a)) if B else float("nan")
    b90 = float(np.quantile(inv_a, 0.9)) if B else float("nan")
    if B and med > 0 and b90 > 0:
        s_val = float((np.log(med) + np.log(b90)) / 2.0)
    else:
        s_val = float("nan")
    result = {
        "n": n,
        "split": split,
        "B": B,
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": s_val,
        "mean_leak": float(np.mean(leak_a)) if B else float("nan"),
        "max_leak": float(np.max(leak_a)) if B else float("nan"),
        "mean_q": float(q_arr.mean()) if q_arr.size else float("nan"),
        "q_per_layer": q_arr.mean(axis=0).tolist() if q_arr.size else [],
        "wall_s": time.time() - t0,
        "device": str(device),
        "q_mode": q_mode,
    }
    if return_instances:
        result["inv_p"] = inv_a.tolist()
        result["p_leak"] = leak_a.tolist()
    return result


def combined_score(metrics_by_n: Dict[int, dict], w6: float = 0.35, w9: float = 0.65) -> float:
    """Lower is better. Primary n9, secondary n6."""
    s = 0.0
    if 9 in metrics_by_n:
        s += w9 * metrics_by_n[9]["S"]
    if 6 in metrics_by_n:
        s += w6 * metrics_by_n[6]["S"]
    return float(s)


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
    q_mode: str = "learned",
    patience: Optional[int] = None,
    eval_every: int = 1,
    num_threads: Optional[int] = None,
    min_delta: float = 1e-6,
) -> Tuple[dict, dict, dict]:
    """Adam short-train of FactorGNN on n=6&9; keep best val combined score.

    patience: if set, stop after this many evals with no S_comb improvement
    (early-stop on S_comb). None keeps the fixed-step loop.
    q_mode: 'learned' (1+9*sigmoid) or 'freeze1' (q≡1, q-head frozen at 0).
    alpha0 is always learned.

    Returns (val_metrics_by_n, best_state_by_n, info).
    """
    q_mode = check_q_mode(q_mode)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    if eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if patience is not None and patience < 1:
        raise ValueError("patience must be >= 1 or None")
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

    init_checksum = {}
    for n, gnn in nets.items():
        if q_mode == "freeze1":
            # Zero logits and keep them out of Adam so weight decay cannot drift q.
            gnn.head_q.requires_grad_(False)
            with torch.no_grad():
                gnn.head_q.weight.zero_()
                gnn.head_q.bias.zero_()
        if not any(p.requires_grad for p in gnn.head_alpha0.parameters()):
            raise RuntimeError("alpha0 head is frozen; recipe requires learned alpha0")
        init_checksum[str(n)] = _param_checksum(gnn, skip_prefix="head_q")

    params = [p for gnn in nets.values() for p in gnn.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.Adam(params, lr=float(arch.lr), weight_decay=float(arch.wd))

    best_score = float("inf")
    best_states = None
    best_val = {}
    best_step = -1
    stall = 0
    history = []
    early_stopped = False
    last_step = -1

    for step in range(steps):
        last_step = step
        opt.zero_grad(set_to_none=True)
        losses = []
        for n, gnn in nets.items():
            gnn.train()
            pack = packs[n]
            seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, "train", max_count=max_train)
            if E is None:
                continue
            B = len(seeds)
            # mini-batch of up to 4 random instances
            bs = min(4, B)
            idx = np_rng.choice(B, size=bs, replace=False)
            sim = sims[n]
            for b in idx:
                cv = torch.as_tensor(cls_v[b], device=device)
                cs = torch.as_tensor(cls_s[b], dtype=torch.float32, device=device)
                heads = gnn.forward_graph(cv, cs)
                Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
                out = sim.forward(
                    Eb,
                    heads["alpha0"],
                    heads["A"],
                    heads["TH"],
                    heads["q_logits"],
                    gamma=heads["gamma"],
                    q_mode=q_mode,
                )
                losses.append(_instance_loss(out, leakpen=leakpen))
        if not losses:
            raise RuntimeError("No train instances with dense E")
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            log(f"step {step}: non-finite loss, skip update")
            stall += 1
            if patience is not None and best_states is not None and stall >= patience:
                early_stopped = True
                log(f"early stop at step {step} (non-finite; best step {best_step})")
                break
            continue
        loss.backward()
        # grad finite check / clip
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        do_eval = (step % eval_every == 0) or (step == steps - 1)
        if not do_eval:
            continue
        # val for best tracking / early-stop on S_comb
        metrics = {}
        for n, gnn in nets.items():
            gnn.eval()
            metrics[n] = eval_pack_metrics(
                n, arch, gnn.state_dict(), split="val", L=L,
                max_count=max_val, seed=seed, data_dir=data_dir,
                device=device, nf=nf, q_mode=q_mode,
            )
        sc = combined_score(metrics)
        history.append({
            "step": step,
            "S_comb": sc,
            "loss": float(loss.detach().cpu()),
            "med": {str(n): metrics[n]["median_inv_p"] for n in metrics},
            "b90": {str(n): metrics[n]["B90_inv_p"] for n in metrics},
            "leak": {str(n): metrics[n]["mean_leak"] for n in metrics},
        })
        improved = math.isfinite(sc) and sc < (best_score - min_delta)
        if improved:
            best_score = sc
            best_step = step
            stall = 0
            best_states = {n: {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}
                           for n, gnn in nets.items()}
            best_val = metrics
            log(
                f"step {step}: NEW best S_comb={sc:.4f} loss={float(loss.detach()):.4f} "
                + " ".join(
                    f"n{n}:Med={metrics[n]['median_inv_p']:.2f}/B90={metrics[n]['B90_inv_p']:.2f}"
                    f"/leak={metrics[n]['mean_leak']:.4f}"
                    for n in sorted(metrics)
                )
            )
        else:
            stall += 1
            if step % 5 == 0 or (patience is not None and stall >= patience):
                sc_s = f"{sc:.4f}" if math.isfinite(sc) else "nan"
                log(f"step {step}: S_comb={sc_s} loss={float(loss.detach()):.4f} stall={stall}")
        if patience is not None and best_states is not None and stall >= patience:
            early_stopped = True
            log(
                f"early stop at step {step} (patience {patience}, "
                f"best step {best_step}, best S_comb={best_score:.4f})"
            )
            break

    info = {
        "best_score": best_score if math.isfinite(best_score) else None,
        "best_step": best_step,
        "last_step": last_step,
        "steps": steps,
        "early_stopped": early_stopped,
        "hit_step_cap": (not early_stopped) and last_step >= steps - 1,
        "patience": patience,
        "eval_every": eval_every,
        "min_delta": min_delta,
        "q_mode": q_mode,
        "leakpen": leakpen,
        "L": L,
        "max_val": max_val,
        "max_train": max_train,
        "alpha0_learned": True,
        "init_checksum": init_checksum,
        "threads": int(torch.get_num_threads()),
        "arch": arch.to_dict(),
        "seed": seed,
        "device": str(device),
        "backend": "torch",
        "history": history,
    }
    return best_val, best_states or {}, info


def save_gnn_best(arch: GNNArch, states_by_n: dict, metrics: dict, runs_dir: Path | None = None):
    runs_dir = runs_dir or RUNS
    out = runs_dir / "gnn_best"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "arch.json", "w") as f:
        json.dump(
            {"arch": arch.to_dict(), "metrics": {str(k): v for k, v in metrics.items()}, "backend": "torch"},
            f,
            indent=2,
        )
    for n, state in (states_by_n or {}).items():
        # state values may be tensors
        cpu_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                cpu_state[k] = v.detach().cpu()
            else:
                cpu_state[k] = torch.as_tensor(v)
        torch.save(cpu_state, out / f"weights_n{n}.pt")
        # also drop legacy npz if present
        legacy = out / f"weights_n{n}.npz"
        if legacy.exists():
            legacy.unlink()
    with open(out / "meta.json", "w") as f:
        json.dump(
            {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "ns": list((states_by_n or {}).keys()),
                "backend": "torch",
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
    # legacy numpy
    for p in best_dir.glob("weights_n*.npz"):
        n = int(p.stem.replace("weights_n", ""))
        if n in params:
            continue
        z = np.load(p)
        params[n] = {k: torch.as_tensor(z[k]) for k in z.files}
    return arch, params


def should_promote(cand_score: float, best_score: float, min_improve: float = 0.0) -> bool:
    """Best-only: promote iff cand is strictly better by min_improve."""
    return cand_score + min_improve < best_score
