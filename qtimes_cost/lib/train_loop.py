"""Short train / eval loop on n=6&9 packs (numpy reference).

Not a full production trainer — enough for evolve_gnn scoring and smoke.
Chaodao can swap in a JAX/Torch accelerator later without leaving this folder.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .ansatz import AnsatzConfig, MultiCavityAnsatz
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


def eval_pack_metrics(
    n: int,
    arch: GNNArch,
    params: Optional[Dict[str, np.ndarray]],
    split: str = "val",
    L: int = 5,
    max_count: int = 32,
    seed: int = 0,
    data_dir: Path | None = None,
) -> dict:
    """Forward GNN+ansatz on a split; return Med/B90 of 1/p_succ."""
    pack = load_pack(n, data_dir=data_dir)
    m = int(pack["m"])
    seeds, cls_v, cls_s, nsol, E = _split_arrays(pack, split, max_count=max_count)
    if E is None:
        raise RuntimeError(f"Pack n={n} split={split} has no dense E; needed for ansatz eval")
    B = len(seeds)
    gnn = FactorGNN(arch, n=n, m=m, L=L, seed=seed)
    if params is not None:
        gnn.load_parameters(params)
    cfg = AnsatzConfig(n=n, L=L)
    sim = MultiCavityAnsatz(cfg)
    inv = []
    leaks = []
    t0 = time.time()
    for b in range(B):
        heads = gnn.forward_graph(cls_v[b], cls_s[b])
        out = sim.forward(
            E[b],
            heads["alpha0"],
            heads["A"],
            heads["TH"],
            heads["q_logits"],
            gamma=heads["gamma"],
        )
        inv.append(out["inv_p"])
        leaks.append(out["p_leak"])
    inv = np.asarray(inv, float)
    med = float(np.median(inv))
    b90 = float(np.quantile(inv, 0.9))
    return {
        "n": n,
        "split": split,
        "B": B,
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": float((np.log(med) + np.log(b90)) / 2.0),
        "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0,
    }


def combined_score(metrics_by_n: Dict[int, dict], w6: float = 0.35, w9: float = 0.65) -> float:
    """Lower is better. Primary n9, secondary n6."""
    s = 0.0
    if 9 in metrics_by_n:
        s += w9 * metrics_by_n[9]["S"]
    if 6 in metrics_by_n:
        s += w6 * metrics_by_n[6]["S"]
    return float(s)


def short_train(
    arch: GNNArch,
    steps: int = 20,
    L: int = 5,
    max_val: int = 16,
    seed: int = 0,
    data_dir: Path | None = None,
    log_fn=None,
) -> Tuple[dict, dict, dict]:
    """Tiny train stub: perturb GNN weights, keep best val combined score.

    Returns (val_metrics_by_n, best_params_by_n, info).
    """
    log = log_fn or (lambda *a: print(time.strftime("%H:%M:%S"), "[train]", *a, flush=True))
    rng = np.random.default_rng(seed)
    nets = {}
    for n in (6, 9):
        try:
            pack = load_pack(n, data_dir=data_dir)
        except FileNotFoundError as e:
            log("skip", n, e)
            continue
        m = int(pack["m"])
        nets[n] = FactorGNN(arch, n=n, m=m, L=L, seed=seed + n)

    if not nets:
        raise FileNotFoundError("No train packs found under data/")

    best_score = float("inf")
    best_params = None
    best_val = {}
    lr = float(arch.lr)

    for step in range(steps):
        for n, gnn in nets.items():
            pdata = gnn.parameters()
            for k, W in pdata.items():
                pdata[k] = W + rng.normal(0, lr, size=W.shape)
            gnn.load_parameters(pdata)
        metrics = {}
        for n, gnn in nets.items():
            metrics[n] = eval_pack_metrics(
                n, arch, gnn.parameters(), split="val", L=L,
                max_count=max_val, seed=seed, data_dir=data_dir,
            )
        sc = combined_score(metrics)
        if sc < best_score:
            best_score = sc
            best_params = {n: {k: v.copy() for k, v in gnn.parameters().items()} for n, gnn in nets.items()}
            best_val = metrics
            log(
                f"step {step}: NEW best S_comb={sc:.4f} "
                + " ".join(
                    f"n{n}:Med={metrics[n]['median_inv_p']:.2f}/B90={metrics[n]['B90_inv_p']:.2f}"
                    for n in metrics
                )
            )
        elif step % 5 == 0:
            log(f"step {step}: S_comb={sc:.4f}")

    info = {"best_score": best_score, "steps": steps, "arch": arch.to_dict(), "seed": seed}
    return best_val, best_params, info


def save_gnn_best(arch: GNNArch, params_by_n: dict, metrics: dict, runs_dir: Path | None = None):
    runs_dir = runs_dir or RUNS
    out = runs_dir / "gnn_best"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "arch.json", "w") as f:
        json.dump({"arch": arch.to_dict(), "metrics": {str(k): v for k, v in metrics.items()}}, f, indent=2)
    for n, p in (params_by_n or {}).items():
        np.savez_compressed(out / f"weights_n{n}.npz", **{k: v for k, v in p.items()})
    with open(out / "meta.json", "w") as f:
        json.dump({"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ns": list((params_by_n or {}).keys())}, f)


def load_gnn_best(runs_dir: Path | None = None) -> Tuple[Optional[GNNArch], dict]:
    runs_dir = runs_dir or RUNS
    arch_path = runs_dir / "gnn_best" / "arch.json"
    if not arch_path.exists():
        return None, {}
    meta = json.load(open(arch_path))
    arch = GNNArch.from_dict(meta["arch"])
    params = {}
    for p in (runs_dir / "gnn_best").glob("weights_n*.npz"):
        n = int(p.stem.replace("weights_n", ""))
        z = np.load(p)
        params[n] = {k: z[k] for k in z.files}
    return arch, params
