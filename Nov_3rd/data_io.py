"""Data + QAOA-warm loaders for Nov_3rd (locked paths/sizes)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from explore_OCT29.lib.train_loop import _split_arrays, load_pack

from Nov_3rd import settings as S


def log(*a):
    import time

    print(time.strftime("%H:%M:%S"), "[Nov_3rd]", *a, flush=True)


def load_qaoa(L: int = S.L_PRIMARY):
    path = S.qaoa_angles_path(L)
    data = json.loads(path.read_text())
    best = data["best"]
    betas = torch.tensor(best["betas"], dtype=torch.float32).reshape(-1)
    gammas = torch.tensor(best["gammas"], dtype=torch.float32).reshape(-1)
    if betas.numel() != L or gammas.numel() != L:
        raise ValueError(f"QAOA L mismatch: got {betas.numel()} want {L} @ {path}")
    return betas, gammas


def shell_phi_from_betas(betas: torch.Tensor) -> torch.Tensor:
    """QAOA-warm shell: φ_j = -β · j  ⇒  θ_k = -β · pc(k)."""
    L = int(betas.numel())
    pcs = torch.arange(S.B_BITS + 1, dtype=torch.float32)
    return (-betas.reshape(L, 1)) * pcs.reshape(1, -1)


def load_train_E(device):
    S.assert_locked(train_ns=S.TRAIN_NS, max_train=S.MAX_TRAIN)
    if S.TRAIN_NS != (6,):
        raise RuntimeError("Nov_3rd train_ns locked to (6,)")
    pack = load_pack(6, data_dir=S.SEALED_DATA)
    _, _, _, _, E = _split_arrays(pack, "train", max_count=S.MAX_TRAIN)
    if E is None or len(E) == 0:
        raise RuntimeError("empty sealed n6 train")
    if len(E) != S.MAX_TRAIN:
        # sealed has 2000; cap should yield exactly max_train
        log(f"WARN train N={len(E)} (wanted {S.MAX_TRAIN})")
    else:
        log(f"train n6 N={len(E)}")
    return torch.as_tensor(E, dtype=torch.float32, device=device)


def load_split_E(n: int, split: str, max_count: int, device, data_dir: Path | None = None):
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        if split == "scale1000" and "scale1000_seeds" in pack:
            key = "scale1000"
        elif "test_seeds" in pack:
            key = "test"
        else:
            raise KeyError(f"no {split} in pack n={n}")
    _, _, _, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(E) == 0:
        raise RuntimeError(f"empty E n={n} {key}")
    return torch.as_tensor(E, dtype=torch.float32, device=device), key


def load_qaoa_scale_ref(L: int = S.L_PRIMARY) -> dict:
    path = S.qaoa_results_path(L)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for k, v in (data.get("ladder") or {}).items():
        n = int(k)
        out[n] = {
            "Med": float(v["median_inv_p"]),
            "B90": float(v["B90_inv_p"]),
        }
    return out
