"""scale1000 eval for Nov_3rd."""
from __future__ import annotations

import time

import numpy as np
import torch

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_split_E, log


@torch.no_grad()
def eval_n(n: int, L: int, gamma, phi, device, *, max_count: int = S.SCALE_B) -> dict:
    E, key = load_split_E(
        n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR
    )
    sim = make_sim(n, L, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(E)
    every = 50 if n >= 15 else 100
    for i in range(B):
        out = sim.forward_nov3rd(E[i], gamma, phi)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if (i + 1) % every == 0:
            log(
                f"eval n{n} {i+1}/{B} Med≈{np.median(invs):.3f} "
                f"({(i+1)/(time.time()-t0+1e-9):.2f}/s)"
            )
    invs = np.asarray(invs, dtype=np.float64)
    return {
        "n": n,
        "split": key,
        "B": B,
        "median_inv_p": float(np.median(invs)),
        "B90_inv_p": float(np.quantile(invs, 0.90)),
        "mean_leak": float(np.mean(leaks)),
        "L": L,
        "recipe": S.RECIPE_NAME,
        "wall_s": time.time() - t0,
    }


def eval_ladder(L: int, gamma, phi, device, ns=S.EVAL_NS, max_count=S.SCALE_B) -> dict:
    rows = {}
    for n in ns:
        m = eval_n(n, L, gamma, phi, device, max_count=max_count)
        rows[n] = m
        log(f"scale1000 n{n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} B={m['B']}")
    return rows
