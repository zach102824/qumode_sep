#!/usr/bin/env python3
"""Unit tests for cavity-locality graded H_c (Opus Idea 2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCR = _ROOT / "Nov_3rd" / "scripts"
for p in (str(_ROOT), str(_SCR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from beat_params import phi_from_c, warm_from_qaoa
from Nov_3rd.ansatz import make_sim
from Nov_3rd.cavity_grade import (
    cavity_pc,
    fwht_np,
    masks_in_x,
    split_E_np,
    walsh_energy_fracs,
)
from Nov_3rd.data_io import load_qaoa


def _load_train_E(n: int, b: int = 4) -> np.ndarray:
    path = _ROOT / "Nov_3rd" / "runs" / "beat_qaoa_12h" / "train_packs" / f"train_n{n}.npz"
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["train_E"][:b], dtype=np.float64)


def test_fwht_involution():
    rng = np.random.default_rng(0)
    for n in (6, 9):
        x = rng.normal(size=1 << n)
        rec = fwht_np(fwht_np(x)) / (1 << n)
        assert np.max(np.abs(rec - x)) < 1e-10, n


def test_split_reconstructs_and_two_classes_only():
    for n in (6, 9, 12):
        E = _load_train_E(n, b=3)
        Ein, Ex = split_E_np(E, n)
        err = np.max(np.abs(Ein + Ex - E.astype(np.float32)))
        assert err < 1e-4, (n, err)
        mask_in, mask_x = masks_in_x(n)
        pc = cavity_pc(n)
        assert not np.any(mask_in & mask_x)
        assert np.all(mask_in | mask_x)
        # two classes: in = pc<=1 (DC+single cavity), x = pc>=2 — no 1/2/3/4 γ split
        assert set(np.unique(pc[mask_in]).tolist()) <= {0, 1}
        assert np.min(pc[mask_x]) >= 2
        # n=12 has 4 cavities so span-4 strings exist, but they still sit in E_x
        if n == 12:
            assert np.any(pc == 4)
            assert np.all(mask_x[pc == 4])


def test_ex_frac_grows_with_n():
    fracs = []
    for n in (6, 9, 12):
        E = _load_train_E(n, b=8)
        f = walsh_energy_fracs(E, n)
        assert f["n_classes"] == 2
        assert abs(f["frac_in"] + f["frac_x"] - 1.0) < 1e-6
        fracs.append(f["frac_x"])
    # share of cross-cavity cost should not shrink with n
    assert fracs[0] < fracs[-1] or abs(fracs[-1] - fracs[0]) < 0.05, fracs


@torch.no_grad()
def test_warm_gamma_matches_locked_forward():
    device = torch.device("cpu")
    n, L = 6, 2
    E = torch.as_tensor(_load_train_E(n, b=2), dtype=torch.float32)
    betas, gammas = load_qaoa(L)
    g0, c10, c20, c30 = warm_from_qaoa(betas, gammas)
    phi = phi_from_c(c10, c20, c30)
    sim = make_sim(n, L, device)
    Ein, Ex = split_E_np(E.numpy(), n)
    for i in range(len(E)):
        a = sim.forward_nov3rd(E[i], g0, phi)
        b = sim.forward_nov3rd_cavity(
            E[i], g0, g0, phi,
            E_in=torch.as_tensor(Ein[i]), E_x=torch.as_tensor(Ex[i]),
        )
        da = float((a["inv_p"] - b["inv_p"]).abs())
        # probabilities should match (global phase drops out of |ψ|²)
        pdiff = float((a["probs"] - b["probs"]).abs().max())
        assert pdiff < 1e-5, (i, pdiff, da)
        assert da / max(float(a["inv_p"]), 1.0) < 1e-5, (i, da)


def main():
    test_fwht_involution()
    print("PASS fwht involution")
    test_split_reconstructs_and_two_classes_only()
    print("PASS split reconstruct + two classes")
    test_ex_frac_grows_with_n()
    print("PASS E_x frac")
    test_warm_gamma_matches_locked_forward()
    print("PASS warm γ_in=γ_x matches locked forward_nov3rd")
    print("ALL cavity-grade tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
