"""L=3 two-stage SPSA on frozen v4 blocks. Does not modify alloc.py.

Stage 1: 4 equal center-4 joint blocks, Q=1536, all Nov3 params.
Then freeze shared Uc at the single best-so-far x.
Stage 2: 2 equal shell-only blocks from that x, remaining Q=1536.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FTREP"))
import alloc as A  # noqa: E402
import wrap as W  # noqa: E402

Q_TOTAL = 3072
Q_JOINT = 1536
Q_SHELL = 1536
N_JOINT_STARTS = 4
N_SHELL_BLOCKS = 2
T_JOINT = 186
T_SHELL = 378
METHOD_ID_SHELL = 80
UC_SLICE = slice(0, 3)


def iterations_nblocks(Q: int, n_blocks: int) -> int:
    if n_blocks <= 0 or Q % (2 * n_blocks) != 0:
        raise ValueError(f"Q={Q} n_blocks={n_blocks} does not split equally")
    t = Q // (2 * n_blocks) - 6
    if t <= 0:
        raise ValueError(f"Q={Q} n_blocks={n_blocks} leaves no iterations")
    return int(t)


def shell_sigmas(L: int) -> np.ndarray:
    sig = A.sigmas_of("Nov3rd", int(L)).copy()
    sig[UC_SLICE] = 0.0
    return sig


def rng_shell(seed, nq, b, L, start_id):
    return A.make_rng(seed, nq, b, L, METHOD_ID_SHELL, int(start_id), 0)


def merge_ledgers(a: dict, b: dict) -> dict:
    keys = ("screen", "init", "calibration", "iteration", "candidate", "monitoring", "total")
    out = {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in keys}
    out["total"] = (
        out["screen"] + out["init"] + out["calibration"]
        + out["iteration"] + out["candidate"] + out["monitoring"]
    )
    return out


def run_twostage(propagate, L: int, Q: int, rngs_init, rngs_spsa, seed, nq, b) -> dict:
    if int(Q) != Q_TOTAL:
        raise ValueError(f"Q={Q} != {Q_TOTAL}")
    if A.iterations_equal4(Q_JOINT) != T_JOINT:
        raise RuntimeError("joint T drift")
    if iterations_nblocks(Q_SHELL, N_SHELL_BLOCKS) != T_SHELL:
        raise RuntimeError("shell T drift")
    joint = A.run_equal4(
        propagate, "Nov3rd", int(L), Q_JOINT, rngs_init, rngs_spsa,
        x0_fn=W.draw_fn("nov3_center4"),
    )
    x_best = np.asarray(joint["x"], dtype=np.float64).copy()
    uc = x_best[UC_SLICE].copy()
    sig = shell_sigmas(L)
    ledger = A.Ledger()
    best = None
    tails = []
    for start_id in range(N_SHELL_BLOCKS):
        block = A.spsa_block(
            propagate, x_best, sig, T_SHELL, rng_shell(seed, nq, b, L, start_id), ledger
        )
        if not np.allclose(np.asarray(block["x"], dtype=np.float64)[UC_SLICE], uc):
            raise RuntimeError("shared Uc moved during shell stage")
        tails.append({
            "start_id": start_id,
            "e_init": block["e_init"],
            "E": block["E"],
            "which": block["which"],
            "a_gain": block["a_gain"],
        })
        if best is None or block["E"] < best["E"] or (
            block["E"] == best["E"] and start_id < best["start_id"]
        ):
            best = dict(block)
            best["start_id"] = start_id
    q2 = ledger.as_dict()
    if q2["total"] != Q_SHELL or q2["monitoring"] != 0 or q2["screen"] != 0:
        raise RuntimeError(f"shell ledger {q2} != Q_SHELL {Q_SHELL}")
    queries = merge_ledgers(joint["queries"], q2)
    if queries["total"] != Q_TOTAL:
        raise RuntimeError(f"two-stage ledger {queries} != Q {Q_TOTAL}")
    best["queries"] = queries
    best["queries_joint"] = joint["queries"]
    best["queries_shell"] = q2
    best["n_iter_joint"] = int(joint["n_iter"])
    best["n_iter_shell"] = T_SHELL
    best["n_iter"] = T_SHELL
    best["tails_joint"] = joint["tails"]
    best["tails_shell"] = tails
    best["joint_start_id"] = int(joint["start_id"])
    best["joint_which"] = joint["which"]
    best["E_joint"] = float(joint["E"])
    best["x_joint"] = np.asarray(joint["x"], dtype=np.float64).copy()
    best["uc_frozen"] = uc.copy()
    best["kept_start_ids"] = [row["start_id"] for row in tails]
    return best
