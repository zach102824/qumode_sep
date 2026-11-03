"""L-BFGS-only trainer for Nov_3rd (Adam forbidden)."""
from __future__ import annotations

import numpy as np
import torch

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import log, shell_phi_from_betas


def _loss(out) -> torch.Tensor:
    return torch.log(out["inv_p"].clamp(min=1e-12))


def reject_polish_for_warm(train_med, warm_med) -> bool:
    """Hard gate: True → discard polish and recover QAOA-warm angles.

    Reject when train Med is missing or strictly worse than warm Med
    (ties keep polish: train_med <= warm_med).
    """
    if train_med is None:
        return True
    return float(train_med) > float(warm_med)


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


def _ckpt_state(
    *,
    gamma: torch.Tensor,
    phi: torch.Tensor,
    L: int,
    seed: int,
) -> dict:
    return {
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

    # Exact QAOA dual warm init — kept for recovery if polish worsens train Med.
    gamma_warm = gammas0.detach().clone().float()
    phi_warm = shell_phi_from_betas(betas0).float()
    gamma = torch.nn.Parameter(gamma_warm.clone())
    phi = torch.nn.Parameter(phi_warm.clone())
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
        "used_warm": False,
        "rejected_polish": False,
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
        # Mid-loop: never crown a polish that already lost to warm on train Med.
        if tr_med <= warm_med and sc < best["score"]:
            best = {
                "score": sc,
                "round": r,
                "warm_med": warm_med,
                "train_med": tr_med,
                "val": val,
                "state": _ckpt_state(gamma=gamma, phi=phi, L=L, seed=seed),
                "used_warm": False,
                "rejected_polish": False,
            }
        if r % 5 == 0 or r + 1 == S.LBFGS_ROUNDS:
            vbit = ""
            if val:
                vbit = f" valMed={val['median_inv_p']:.3f} B90={val['B90_inv_p']:.3f}"
            log(
                f"seed{seed} r{r}/{S.LBFGS_ROUNDS}: loss≈{loss_f:.4f} "
                f"trainMed≈{tr_med:.3f}{vbit} bestS={best['score']:.3f}@r{best['round']}"
            )

    # Mandatory final gate: recover exact QAOA-warm if polish did not help train Med.
    if reject_polish_for_warm(best["train_med"], warm_med):
        why = (
            "no accepted polish round"
            if best["train_med"] is None
            else f"trainMed={best['train_med']:.4f} > warmMed={warm_med:.4f}"
        )
        log(
            f"seed{seed} REJECT polish ({why}) — recovering QAOA-warm "
            f"(φ_j=-β·j, γ=γ_QAOA); warmMed≈{warm_med:.4f}"
        )
        best = {
            "score": warm_med,
            "round": -1,
            "warm_med": warm_med,
            "train_med": warm_med,
            "val": None,
            "state": _ckpt_state(gamma=gamma_warm, phi=phi_warm, L=L, seed=seed),
            "used_warm": True,
            "rejected_polish": True,
        }
        best["state"]["used_warm"] = True
        best["state"]["rejected_polish"] = True
        best["state"]["warm_med"] = warm_med
    else:
        best["used_warm"] = False
        best["rejected_polish"] = False
        if best["state"] is not None:
            best["state"]["used_warm"] = False
            best["state"]["rejected_polish"] = False
            best["state"]["warm_med"] = warm_med
        log(
            f"seed{seed} KEEP polish trainMed≈{best['train_med']:.4f} "
            f"≤ warmMed≈{warm_med:.4f} (round={best['round']})"
        )

    return best
