"""L=3 shell-first two-stage SPSA. Does not modify alloc.py.

Stage 1: 4 equal center-4 starts, shell-only, Uc frozen at each start's x0.
Then freeze shells at the single best-so-far x.
Stage 2: 2 equal Uc-only blocks from that x, remaining Q=1536.
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
Q_SHELL = 1536
Q_UC = 1536
N_SHELL_STARTS = 4
N_UC_BLOCKS = 2
T_SHELL = 186
T_UC = 378
METHOD_ID_SHELL = 82
METHOD_ID_UC = 83
UC_SLICE = slice(0, 3)
SHELL_SLICE = slice(3, None)


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


def uc_sigmas(L: int) -> np.ndarray:
    sig = A.sigmas_of("Nov3rd", int(L)).copy()
    sig[SHELL_SLICE] = 0.0
    return sig


def rng_lists_shell(seed, nq, b, L):
    rng_init = [None if s == 0 else A.make_rng(seed, nq, b, L, METHOD_ID_SHELL, s, 2)
                for s in range(N_SHELL_STARTS)]
    rng_spsa = [A.make_rng(seed, nq, b, L, METHOD_ID_SHELL, s, 0)
                for s in range(N_SHELL_STARTS)]
    return rng_init, rng_spsa


def rng_uc(seed, nq, b, L, start_id):
    return A.make_rng(seed, nq, b, L, METHOD_ID_UC, int(start_id), 0)


def merge_ledgers(a: dict, b: dict) -> dict:
    keys = ("screen", "init", "calibration", "iteration", "candidate", "monitoring", "total")
    out = {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in keys}
    out["total"] = (
        out["screen"] + out["init"] + out["calibration"]
        + out["iteration"] + out["candidate"] + out["monitoring"]
    )
    return out


def run_shellfirst(propagate, L: int, Q: int, seed, nq, b) -> dict:
    if int(Q) != Q_TOTAL:
        raise ValueError(f"Q={Q} != {Q_TOTAL}")
    if A.iterations_equal4(Q_SHELL) != T_SHELL:
        raise RuntimeError("shell T drift")
    if iterations_nblocks(Q_UC, N_UC_BLOCKS) != T_UC:
        raise RuntimeError("uc T drift")
    rng_init, rng_spsa = rng_lists_shell(seed, nq, b, L)
    draw = W.draw_fn("nov3_center4")
    sig_shell = shell_sigmas(L)
    ledger1 = A.Ledger()
    best1 = None
    tails1 = []
    for start_id, (rng_i, rng_s) in enumerate(zip(rng_init, rng_spsa)):
        x0 = draw("Nov3rd", int(L), start_id, rng_i)
        uc0 = np.asarray(x0, dtype=np.float64)[UC_SLICE].copy()
        block = A.spsa_block(propagate, x0, sig_shell, T_SHELL, rng_s, ledger1)
        x_end = np.asarray(block["x"], dtype=np.float64)
        if not np.allclose(x_end[UC_SLICE], uc0):
            raise RuntimeError("shared Uc moved during shell stage")
        tails1.append({
            "start_id": start_id,
            "e_init": block["e_init"],
            "E": block["E"],
            "which": block["which"],
            "a_gain": block["a_gain"],
        })
        if best1 is None or block["E"] < best1["E"] or (
            block["E"] == best1["E"] and start_id < best1["start_id"]
        ):
            best1 = dict(block)
            best1["start_id"] = start_id
            best1["uc_frozen"] = uc0.copy()
    q1 = ledger1.as_dict()
    if q1["total"] != Q_SHELL or q1["monitoring"] != 0 or q1["screen"] != 0:
        raise RuntimeError(f"shell ledger {q1} != Q_SHELL {Q_SHELL}")
    x_best = np.asarray(best1["x"], dtype=np.float64).copy()
    shells = x_best[SHELL_SLICE].copy()
    sig_uc = uc_sigmas(L)
    ledger2 = A.Ledger()
    best2 = None
    tails2 = []
    for start_id in range(N_UC_BLOCKS):
        block = A.spsa_block(
            propagate, x_best, sig_uc, T_UC, rng_uc(seed, nq, b, L, start_id), ledger2
        )
        x_end = np.asarray(block["x"], dtype=np.float64)
        if not np.allclose(x_end[SHELL_SLICE], shells):
            raise RuntimeError("shells moved during Uc stage")
        tails2.append({
            "start_id": start_id,
            "e_init": block["e_init"],
            "E": block["E"],
            "which": block["which"],
            "a_gain": block["a_gain"],
        })
        if best2 is None or block["E"] < best2["E"] or (
            block["E"] == best2["E"] and start_id < best2["start_id"]
        ):
            best2 = dict(block)
            best2["start_id"] = start_id
    q2 = ledger2.as_dict()
    if q2["total"] != Q_UC or q2["monitoring"] != 0 or q2["screen"] != 0:
        raise RuntimeError(f"uc ledger {q2} != Q_UC {Q_UC}")
    queries = merge_ledgers(q1, q2)
    if queries["total"] != Q_TOTAL:
        raise RuntimeError(f"shell-first ledger {queries} != Q {Q_TOTAL}")
    best2["queries"] = queries
    best2["queries_shell"] = q1
    best2["queries_uc"] = q2
    best2["n_iter_shell"] = T_SHELL
    best2["n_iter_uc"] = T_UC
    best2["n_iter"] = T_UC
    best2["tails_shell"] = tails1
    best2["tails_uc"] = tails2
    best2["shell_start_id"] = int(best1["start_id"])
    best2["shell_which"] = best1["which"]
    best2["E_shell"] = float(best1["E"])
    best2["x_shell"] = x_best.copy()
    best2["uc_frozen"] = np.asarray(best1["uc_frozen"], dtype=np.float64).copy()
    best2["shells_frozen"] = shells.copy()
    best2["kept_start_ids"] = [row["start_id"] for row in tails2]
    return best2
