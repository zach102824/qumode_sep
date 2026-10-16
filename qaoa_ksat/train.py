#!/usr/bin/env python3
"""Train average-instance QAOA angles on a frozen set of SAT instances.

Matches the paper's empirical protocol (Sec. IV A 3), except the instance size
is configurable (they used n=12 with 100 instances; we default to n=8).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qaoa_ksat.angles import paper_angles
from qaoa_ksat.circuit import qaoa_state, success_probability
from qaoa_ksat.ksat import THRESHOLD_R, energy_table, is_satisfiable, random_ksat


def collect_sat_energies(
    n: int,
    k: int,
    r: float,
    n_instances: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    tries = 0
    while len(out) < n_instances:
        tries += 1
        inst = random_ksat(n, k, r, rng=rng)
        energy = energy_table(inst)
        if is_satisfiable(energy):
            out.append(energy)
        if tries > 50 * n_instances:
            raise RuntimeError(f"only {len(out)} SAT instances after {tries} draws")
    return out


def mean_psucc(betas: np.ndarray, gammas: np.ndarray, energies: list[np.ndarray], n: int) -> float:
    acc = 0.0
    for energy in energies:
        psi = qaoa_state(betas, gammas, energy, n)
        acc += success_probability(psi, energy)
    return acc / max(len(energies), 1)


def finite_diff_grad(
    betas: np.ndarray,
    gammas: np.ndarray,
    energies: list[np.ndarray],
    n: int,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    p0 = mean_psucc(betas, gammas, energies, n)
    g_b = np.zeros_like(betas)
    g_g = np.zeros_like(gammas)
    for i in range(betas.size):
        b2 = betas.copy()
        b2[i] += eps
        g_b[i] = (mean_psucc(b2, gammas, energies, n) - p0) / eps
    for i in range(gammas.size):
        g2 = gammas.copy()
        g2[i] += eps
        g_g[i] = (mean_psucc(betas, g2, energies, n) - p0) / eps
    return g_b, g_g, p0


def train(
    n: int,
    p: int,
    *,
    k: int = 8,
    r: float | None = None,
    n_instances: int = 40,
    steps: int = 80,
    lr: float = 0.05,
    seed: int = 0,
    from_paper: bool = True,
) -> dict:
    r = float(THRESHOLD_R[k] if r is None else r)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    energies = collect_sat_energies(n, k, r, n_instances, rng)
    if from_paper:
        try:
            betas, gammas = paper_angles(k, p, r)
        except KeyError:
            betas = np.full(p, 0.01)
            gammas = np.full(p, -0.01)
            from_paper = False
    else:
        betas = np.full(p, 0.01)
        gammas = np.full(p, -0.01)
    history = []
    for step in range(steps):
        g_b, g_g, p0 = finite_diff_grad(betas, gammas, energies, n)
        betas = betas + lr * g_b
        gammas = gammas + lr * g_g
        history.append({"step": step, "mean_psucc": p0, "grad_norm": float(np.linalg.norm(np.concatenate([g_b, g_g])))})
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:3d}  mean p_succ={p0:.6f}  |g|={history[-1]['grad_norm']:.3g}", flush=True)
    p_final = mean_psucc(betas, gammas, energies, n)
    paper_b = paper_g = None
    paper_p = None
    try:
        paper_b, paper_g = paper_angles(k, p, r)
        paper_p = mean_psucc(paper_b, paper_g, energies, n)
    except KeyError:
        pass
    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n": n,
        "k": k,
        "r": r,
        "p": p,
        "n_instances": n_instances,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "init_from_paper": from_paper,
        "mean_psucc_final": p_final,
        "mean_psucc_paper_angles": paper_p,
        "betas": betas.tolist(),
        "gammas": gammas.tolist(),
        "paper_betas": None if paper_b is None else paper_b.tolist(),
        "paper_gammas": None if paper_g is None else paper_g.tolist(),
        "angle_l2_to_paper": None
        if paper_b is None
        else float(np.linalg.norm(np.concatenate([betas - paper_b, gammas - paper_g]))),
        "history": history,
        "wall_s": time.perf_counter() - t0,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-instances", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-paper", action="store_true", default=True)
    ap.add_argument("--from-scratch", action="store_true")
    args = ap.parse_args()
    payload = train(
        args.n,
        args.p,
        k=args.k,
        n_instances=args.n_instances,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        from_paper=not args.from_scratch,
    )
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"train_n{args.n}_p{args.p}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(
        f"final mean p_succ={payload['mean_psucc_final']:.6f}  "
        f"paper-angles mean p_succ={payload['mean_psucc_paper_angles']}  "
        f"L2 to paper={payload['angle_l2_to_paper']}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
