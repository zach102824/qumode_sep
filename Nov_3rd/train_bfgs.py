"""L-BFGS-only trainer for Nov_3rd (Adam forbidden)."""
from __future__ import annotations

import numpy as np
import torch

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import log, shell_phi_from_betas


def _loss(out) -> torch.Tensor:
    return torch.log(out["inv_p"].clamp(min=1e-12))


@torch.no_grad()
def median_inv(sim, gamma, phi, E, max_count=64) -> float:
    invs = []
    B = min(len(E), max_count)
    for i in range(B):
        invs.append(float(sim.forward_nov3rd(E[i], gamma, phi)["inv_p"].cpu()))
    return float(np.median(invs))


@torch.no_grad()
def eval_med_b90(sim, gamma, phi, E) -> dict:
    invs = []
    for i in range(len(E)):
        invs.append(float(sim.forward_nov3rd(E[i], gamma, phi)["inv_p"].cpu()))
    invs = np.asarray(invs, dtype=np.float64)
    return {
        "median_inv_p": float(np.median(invs)),
        "B90_inv_p": float(np.quantile(invs, 0.90)),
        "B": int(len(invs)),
        "S": float(np.median(invs) + 0.1 * np.quantile(invs, 0.90)),
    }


def train_seed(
    *,
    L: int,
    device,
    E_train: torch.Tensor,
    betas0: torch.Tensor,
    gammas0: torch.Tensor,
    seed: int,
    E_val: torch.Tensor | None = None,
):
    S.assert_locked(
        optimizer=S.OPTIMIZER,
        snap_head=S.SNAP_HEAD,
        qaoa_warm=S.QAOA_WARM,
        b=S.B_BITS,
        W_frame=S.W_FRAME,
        train_ns=S.TRAIN_NS,
        max_train=S.MAX_TRAIN,
    )
    if S.OPTIMIZER != "LBFGS_only":
        raise RuntimeError("Adam (or any non-LBFGS) is forbidden in Nov_3rd")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    gamma = torch.nn.Parameter(gammas0.detach().clone().float())
    phi = torch.nn.Parameter(shell_phi_from_betas(betas0).float())
    sim = make_sim(6, L, device)
    N = len(E_train)

    warm_med = median_inv(sim, gamma, phi, E_train, max_count=min(64, N))
    log(f"seed{seed} QAOA-warm Med≈{warm_med:.4f} (n6 subsample)")

    best = {
        "score": float("inf"),
        "round": -1,
        "warm_med": warm_med,
        "train_med": None,
        "val": None,
        "state": None,
    }

    for r in range(S.LBFGS_ROUNDS):
        opt = torch.optim.LBFGS(
            [gamma, phi],
            lr=S.LBFGS_LR,
            max_iter=S.LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        idx = rng.choice(N, size=min(S.LBFGS_BATCH, N), replace=False)

        def closure():
            opt.zero_grad()
            ls = [
                _loss(sim.forward_nov3rd(E_train[int(j)], gamma, phi))
                for j in idx
            ]
            lt = torch.stack(ls).mean()
            lt.backward()
            torch.nn.utils.clip_grad_norm_([gamma, phi], 5.0)
            return lt

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"seed{seed} LBFGS r{r} skip: {e}")
            continue

        tr_med = median_inv(sim, gamma, phi, E_train, max_count=min(128, N))
        val = eval_med_b90(sim, gamma.detach(), phi.detach(), E_val) if E_val is not None else None
        sc = float(val["S"]) if val else tr_med
        if sc < best["score"]:
            best = {
                "score": sc,
                "round": r,
                "warm_med": warm_med,
                "train_med": tr_med,
                "val": val,
                "state": {
                    "gamma": gamma.detach().cpu().clone(),
                    "phi": phi.detach().cpu().clone(),
                    "L": L,
                    "b": S.B_BITS,
                    "W": S.W_FRAME,
                    "snap_mode": S.SNAP_HEAD,
                    "snap_shared_cavities": S.SNAP_SHARED_CAVITIES,
                    "motif": S.MOTIF,
                    "recipe": S.RECIPE_NAME,
                    "train_ns": list(S.TRAIN_NS),
                    "max_train": S.MAX_TRAIN,
                    "optimizer": S.OPTIMIZER,
                    "qaoa_warm": S.QAOA_WARM,
                    "seed": seed,
                },
            }
        if r % 5 == 0 or r + 1 == S.LBFGS_ROUNDS:
            vbit = ""
            if val:
                vbit = f" valMed={val['median_inv_p']:.3f} B90={val['B90_inv_p']:.3f}"
            log(
                f"seed{seed} r{r}/{S.LBFGS_ROUNDS}: loss≈{loss_f:.4f} "
                f"trainMed≈{tr_med:.3f}{vbit} bestS={best['score']:.3f}@r{best['round']}"
            )
    return best
