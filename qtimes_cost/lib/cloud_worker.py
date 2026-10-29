"""One cloud_day1 trial. Importable so ProcessPool spawn can pickle it."""
from __future__ import annotations

import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from qtimes_cost.lib.ansatz import AnsatzConfig, MultiCavityAnsatz
from qtimes_cost.lib.gnn import FactorGNN, GNNArch
from qtimes_cost.lib.train_loop import (
    _instance_loss,
    _split_arrays,
    combined_score,
    load_pack,
)


def summarize_inv(inv, leak, qmeans) -> dict:
    inv_a = np.asarray(inv, dtype=float)
    med = float(np.median(inv_a))
    b90 = float(np.quantile(inv_a, 0.9))
    return {
        "B": int(len(inv_a)),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": float((math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12))) / 2.0),
        "mean_leak": float(np.mean(leak)) if leak else float("nan"),
        "q_med": float(np.median(qmeans)) if len(qmeans) else float("nan"),
    }


def eval_nets(nets, sims, arrays, max_count_note: int) -> dict:
    metrics = {}
    for n, gnn in nets.items():
        gnn.eval()
        sim = sims[n]
        seeds, cls_v, cls_s, _nsol, E = arrays[n]
        inv, leak, qmeans = [], [], []
        with torch.inference_mode():
            for b in range(len(seeds)):
                heads = gnn.forward_graph(
                    torch.as_tensor(cls_v[b]),
                    torch.as_tensor(cls_s[b], dtype=torch.float32),
                )
                out = sim.forward(
                    torch.as_tensor(E[b], dtype=torch.float32),
                    heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"],
                    gamma=heads["gamma"],
                )
                inv.append(float(out["inv_p"]))
                leak.append(float(out["p_leak"]))
                qmeans.append(float(np.mean(out["q_int"])))
        metrics[n] = summarize_inv(inv, leak, qmeans)
        metrics[n]["n"] = n
        metrics[n]["max_count"] = max_count_note
    return metrics


def _clone_states(nets) -> dict:
    return {
        n: {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}
        for n, gnn in nets.items()
    }


def arch_seed(trial: dict) -> int:
    return 10007 * (int(trial["seed"]) + 1) + int(trial["trial_id"])


def _grow_out(src: torch.Tensor, C: int, src_L: int, dst_L: int, group: int) -> torch.Tensor:
    """Grow a head laid out as (C, L, group) along L. New layers copy the last one."""
    vec = src.ndim == 1
    if vec:
        src_v = src.reshape(C, src_L, group)
        dst = src.new_zeros(C, dst_L, group)
        ncopy = min(src_L, dst_L)
        dst[:, :ncopy] = src_v[:, :ncopy]
        if dst_L > src_L:
            dst[:, src_L:] = src_v[:, src_L - 1:src_L]
        return dst.reshape(C * dst_L * group)
    H = src.shape[1]
    src_v = src.reshape(C, src_L, group, H)
    dst = src.new_zeros(C, dst_L, group, H)
    ncopy = min(src_L, dst_L)
    dst[:, :ncopy] = src_v[:, :ncopy]
    if dst_L > src_L:
        dst[:, src_L:] = src_v[:, src_L - 1:src_L]
    return dst.reshape(C * dst_L * group, H)


def _grow_cavity(src: torch.Tensor, src_C: int, dst_C: int) -> torch.Tensor:
    """Grow a head whose leading axis is cavity. New cavities copy the last one."""
    if src.ndim == 1:
        rest = src.numel() // src_C
        src_v = src.reshape(src_C, rest)
        dst = src.new_zeros(dst_C, rest)
        ncopy = min(src_C, dst_C)
        dst[:ncopy] = src_v[:ncopy]
        if dst_C > src_C:
            dst[src_C:] = src_v[src_C - 1:src_C]
        return dst.reshape(dst_C * rest)
    rest = src.shape[0] // src_C
    H = src.shape[1]
    src_v = src.reshape(src_C, rest, H)
    dst = src.new_zeros(dst_C, rest, H)
    ncopy = min(src_C, dst_C)
    dst[:ncopy] = src_v[:ncopy]
    if dst_C > src_C:
        dst[src_C:] = src_v[src_C - 1:src_C]
    return dst.reshape(dst_C * rest, H)


# Cavity readout. These widths depend on C and must not be copied across n.
CAVITY_HEAD_PREFIXES = ("head_alpha0.", "head_A.", "head_TH.")


def copy_shared_body(gnn: FactorGNN, src: dict) -> list[str]:
    """Copy parameters whose shapes match, skipping cavity heads.

    Message-passing body, q, and gamma transfer across n. α0 / SNAP / displacement
    stay at the destination init (they are not cavity-repeated).
    """
    merged = {k: v.detach().clone() for k, v in gnn.state_dict().items()}
    copied = []
    for key, dst in list(merged.items()):
        if key not in src or key.startswith(CAVITY_HEAD_PREFIXES):
            continue
        parent = src[key].detach().to(dtype=dst.dtype, device=dst.device)
        if tuple(parent.shape) != tuple(dst.shape):
            continue
        merged[key] = parent
        copied.append(key)
    gnn.load_state_dict(merged)
    return copied


def transfer_gnn_n(gnn: FactorGNN, src: dict) -> int:
    """Load a checkpoint trained at another n.

    Message-passing body, q, and gamma copy when shapes match. Cavity heads
    (α0, displacement, SNAP) grow by repeating the last cavity.
    Returns the source cavity count.
    """
    src_C = int(src["head_alpha0.bias"].numel() // 2)
    dst_C = int(gnn.C)
    growers = {
        "head_alpha0.weight", "head_alpha0.bias",
        "head_A.weight", "head_A.bias",
        "head_TH.weight", "head_TH.bias",
    }
    merged = {k: v.detach().clone() for k, v in gnn.state_dict().items()}
    for key, dst in list(merged.items()):
        if key not in src:
            continue
        parent = src[key].detach().to(dtype=dst.dtype, device=dst.device)
        if parent.shape == dst.shape:
            merged[key] = parent
            continue
        if key in growers and src_C != dst_C:
            merged[key] = _grow_cavity(parent, src_C, dst_C)
    gnn.load_state_dict(merged)
    return src_C


def warm_start_gnn(gnn: FactorGNN, src: dict, src_L: int) -> None:
    """Copy a checkpoint, growing SNAP/D/q heads when L increases.

    Body, α0, and gamma copy when shapes match. Extra layers repeat the last
    learned layer so a deeper circuit starts as the parent plus repeated gates.
    """
    from qtimes_cost.lib.packing import NU

    src_L = int(src_L)
    dst_L = int(gnn.L)
    C = int(gnn.C)
    merged = {k: v.detach().clone() for k, v in gnn.state_dict().items()}
    growers = {
        "head_A.weight": (C, 2),
        "head_A.bias": (C, 2),
        "head_TH.weight": (C, NU),
        "head_TH.bias": (C, NU),
        "head_q.weight": (1, 1),
        "head_q.bias": (1, 1),
    }
    for key, dst in list(merged.items()):
        if key not in src:
            continue
        parent = src[key].detach().to(dtype=dst.dtype, device=dst.device)
        if parent.shape == dst.shape:
            merged[key] = parent
            continue
        if key not in growers or src_L <= 0:
            continue
        cavities, group = growers[key]
        merged[key] = _grow_out(parent, cavities, src_L, dst_L, group)
    gnn.load_state_dict(merged)


def _empty_result(trial, status, steps_ran, wall, arch) -> dict:
    return {
        "trial_id": int(trial["trial_id"]),
        "cfg": trial,
        "arch": arch.to_dict() if arch is not None else {},
        "status": status,
        "steps_ran": steps_ran,
        "best_step": -1,
        "wall_s": wall,
        "metrics": {},
        "s_comb": float("nan"),
        "states": {},
        "error": "",
    }


def _run_trial_inner(trial: dict, best_s, best_lock, steps: int, search: dict) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    L = int(trial["L"])
    lr = float(trial["lr"])
    leakpen = float(trial["leakpen"])
    seed = int(trial["seed"])
    arch = GNNArch(
        H=int(trial["H"]),
        rounds=int(trial["rounds"]),
        agg="mean",
        layernorm=True,
        residual=True,
        degree_features=True,
        lr=lr,
        wd=1e-4,
        dropout=0.0,
    )
    torch.manual_seed(10007 * (seed + 1) + int(trial["trial_id"]))
    rng = np.random.default_rng(10_000 + int(trial["trial_id"]) * 17 + seed)

    nets, sims, train_arr, val_arr = {}, {}, {}, {}
    for n in (6, 9):
        pack = load_pack(n)
        gnn = FactorGNN(arch, n=n, m=int(pack["m"]), L=L, seed=arch_seed(trial) + n)
        sim = MultiCavityAnsatz(AnsatzConfig(n=n, L=L))
        nets[n] = gnn
        sims[n] = sim
        train_arr[n] = _split_arrays(pack, "train", max_count=int(search["max_train"]))
        val_arr[n] = _split_arrays(pack, "val", max_count=int(search["max_val"]))

    init_dir = str(trial.get("init_dir") or "")
    if init_dir:
        src_L = int(trial.get("src_L") or L)
        for n, gnn in nets.items():
            parent = torch.load(
                str(Path(init_dir) / f"weights_n{n}.pt"),
                map_location="cpu",
                weights_only=True,
            )
            warm_start_gnn(gnn, parent, src_L)

    params = [p for gnn in nets.values() for p in gnn.parameters()]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    best = None
    stall = 0
    status = "ok"
    steps_ran = 0
    if init_dir:
        # Keep the loaded weights if later steps do not improve search S_comb.
        metrics0 = eval_nets(nets, sims, val_arr, int(search["max_val"]))
        sc0 = combined_score(metrics0)
        if math.isfinite(sc0):
            best = {"s_comb": sc0, "metrics": metrics0, "states": _clone_states(nets), "step": -1}
            print(
                f"t{trial['trial_id']:02d} step -1 WARM S={sc0:.3f} "
                f"n6={metrics0[6]['median_inv_p']:.1f} n9={metrics0[9]['median_inv_p']:.1f} "
                f"L={L} lp={leakpen} lr={lr:g} {trial['tag']}",
                flush=True,
            )
    eval_every = int(trial.get("eval_every") or search["eval_every"])
    patience = int(trial.get("patience_evals") or search["patience_evals"])
    min_steps_cfg = int(trial.get("min_steps") or search["min_steps"])
    min_steps = min_steps_cfg if steps >= min_steps_cfg else steps
    hopeless_after = int(trial.get("hopeless_after") or search["hopeless_after"])
    hopeless_med9 = float(trial.get("hopeless_med9") or search["hopeless_med9"])
    hopeless_S = float(trial.get("hopeless_S") or search["hopeless_S"])

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for n in (6, 9):
            gnn = nets[n]
            gnn.train()
            sim = sims[n]
            seeds, cls_v, cls_s, _nsol, E = train_arr[n]
            bs = min(int(search["batch"]), len(seeds))
            idx = rng.choice(len(seeds), size=bs, replace=False)
            for b in idx:
                b = int(b)
                heads = gnn.forward_graph(
                    torch.as_tensor(cls_v[b]),
                    torch.as_tensor(cls_s[b], dtype=torch.float32),
                )
                out = sim.forward(
                    torch.as_tensor(E[b], dtype=torch.float32),
                    heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"],
                    gamma=heads["gamma"],
                )
                losses.append(_instance_loss(out, leakpen=leakpen))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        steps_ran = step + 1

        if step % eval_every != 0 and step != steps - 1:
            continue
        metrics = eval_nets(nets, sims, val_arr, int(search["max_val"]))
        sc = combined_score(metrics)
        med9 = metrics[9]["median_inv_p"]
        if not math.isfinite(sc):
            status = "nan"
            break
        improved = best is None or sc + 1e-6 < best["s_comb"]
        if improved:
            best = {
                "s_comb": sc,
                "metrics": metrics,
                "states": _clone_states(nets),
                "step": step,
            }
            stall = 0
            print(
                f"t{trial['trial_id']:02d} step {step} NEW S={sc:.3f} "
                f"n6={metrics[6]['median_inv_p']:.1f} n9={med9:.1f} "
                f"L={L} lp={leakpen} lr={lr:g} {trial['tag']}",
                flush=True,
            )
        else:
            stall += 1

        if (
            step >= hopeless_after
            and best is not None
            and best["metrics"][9]["median_inv_p"] > hopeless_med9
            and best["s_comb"] > hopeless_S
        ):
            status = "hopeless"
            break
        if step >= min_steps and stall >= patience:
            status = "early_stop"
            break
        with best_lock:
            global_best = float(best_s["s"])
        if (
            step >= int(search["kill_after"])
            and global_best < 1e8
            and sc > global_best + float(search["kill_margin"])
            and stall >= 2
        ):
            status = "killed_vs_best"
            break

    if best is None:
        return _empty_result(trial, status or "nan", steps_ran, time.time() - t0, arch)

    return {
        "trial_id": int(trial["trial_id"]),
        "cfg": trial,
        "arch": arch.to_dict(),
        "status": status,
        "steps_ran": steps_ran,
        "best_step": int(best["step"]),
        "wall_s": time.time() - t0,
        "metrics": best["metrics"],
        "s_comb": float(best["s_comb"]),
        "states": best["states"],
        "error": "",
    }


def run_trial(payload: dict) -> dict:
    """Process-pool entry. payload: trial, best_s, steps, search."""
    torch.set_num_threads(1)
    trial = payload["trial"]
    try:
        return _run_trial_inner(
            trial, payload["best_s"], payload["best_lock"], int(payload["steps"]), payload["search"],
        )
    except Exception:
        return {
            "trial_id": int(trial["trial_id"]),
            "cfg": trial,
            "arch": {},
            "status": "error",
            "steps_ran": 0,
            "best_step": -1,
            "wall_s": 0.0,
            "metrics": {},
            "s_comb": float("nan"),
            "states": {},
            "error": traceback.format_exc(limit=8),
        }
