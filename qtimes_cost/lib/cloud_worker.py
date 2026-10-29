"""One cloud_day1 trial. Importable so ProcessPool spawn can pickle it."""
from __future__ import annotations

import math
import time
import traceback

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

    params = [p for gnn in nets.values() for p in gnn.parameters()]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    best = None
    stall = 0
    status = "ok"
    steps_ran = 0
    eval_every = int(search["eval_every"])
    min_steps = int(search["min_steps"]) if steps >= int(search["min_steps"]) else steps
    patience = int(search["patience_evals"])

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
            step >= int(search["hopeless_after"])
            and best is not None
            and best["metrics"][9]["median_inv_p"] > float(search["hopeless_med9"])
            and best["s_comb"] > float(search["hopeless_S"])
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
