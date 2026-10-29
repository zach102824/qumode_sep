"""Ansatz shapes + near-unitarity / probability conservation on tiny n=6."""
from __future__ import annotations
import numpy as np
import torch
import pytest

from explore_OCT29.lib.ansatz import AnsatzConfig, MultiCavityAnsatz, random_params
from explore_OCT29.lib.packing import NU
from explore_OCT29.lib.sat_gen import gen_instance


def test_ansatz_shared_shapes_and_norm_n6():
    n, L, nf = 6, 2, 16
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = MultiCavityAnsatz(cfg, device=torch.device("cpu"))
    m, cl, E = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 3)
    assert E is not None and E.min() == 0
    p = random_params(cfg, np.random.default_rng(0))
    assert p["alpha0"].shape == (2,)
    assert p["A"].shape == (L, 2)
    assert p["TH"].shape == (L, NU)
    assert p["gamma"].shape == (L,)
    assert "q_logits" not in p
    out = sim.forward(
        torch.as_tensor(E, dtype=torch.float32),
        p["alpha0"], p["A"], p["TH"], p["gamma"],
    )
    assert out["p_succ"].ndim == 0
    assert out["p_leak"].ndim == 0
    assert float(out["p_succ"]) >= 0
    assert float(out["p_leak"]) >= -1e-5
    assert abs(float(out["norm"]) - 1.0) < 1e-3
    assert float(out["p_succ"]) + float(out["p_leak"]) <= 1.0 + 1e-3
    assert out.get("freedom_step") == 0


def test_ansatz_accepts_tiled_alpha0():
    n, L, nf = 6, 1, 16
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = MultiCavityAnsatz(cfg, device=torch.device("cpu"))
    _, _, E = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 4)
    p = random_params(cfg, np.random.default_rng(1))
    a0 = p["alpha0"].unsqueeze(0).expand(cfg.C, 2).contiguous()
    out = sim.forward(
        torch.as_tensor(E, dtype=torch.float32),
        a0, p["A"], p["TH"], p["gamma"],
    )
    assert float(out["norm"]) > 0.99
