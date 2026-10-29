"""Ansatz shapes + near-unitarity / probability conservation on tiny n=6."""
from __future__ import annotations
import numpy as np
import torch
import pytest

from explore_OCT29.lib.ansatz import AnsatzConfig, MultiCavityAnsatz, random_params, q_from_logit
from explore_OCT29.lib.sat_gen import gen_instance


def test_q_from_logit_range():
    for u in (-10.0, 0.0, 10.0):
        q = float(q_from_logit(torch.tensor(u)))
        assert 1.0 < q < 10.0 or abs(q - 1.0) < 1e-4 or abs(q - 10.0) < 1e-4


def test_ansatz_shapes_and_norm_n6():
    n, L, nf = 6, 2, 16
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = MultiCavityAnsatz(cfg, device=torch.device("cpu"))
    m, cl, E = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 3)
    assert E is not None and E.min() == 0
    p = random_params(cfg, np.random.default_rng(0))
    out = sim.forward(
        torch.as_tensor(E, dtype=torch.float32),
        p["alpha0"], p["A"], p["TH"], p["q_logits"], gamma=p["gamma"],
    )
    assert out["p_succ"].ndim == 0
    assert out["p_leak"].ndim == 0
    assert float(out["p_succ"]) >= 0
    assert float(out["p_leak"]) >= -1e-5
    # probability conservation (unitary path)
    assert abs(float(out["norm"]) - 1.0) < 1e-3
    assert float(out["p_succ"]) + float(out["p_leak"]) <= 1.0 + 1e-3
