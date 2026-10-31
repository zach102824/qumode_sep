"""q≡1 freeze: q_from_logit is exact 1, logits do not train."""
from __future__ import annotations

import numpy as np
import torch

from qtimes_cost.lib.ansatz import (
    AnsatzConfig,
    MultiCavityAnsatz,
    integer_q,
    q_from_logit,
    random_params,
)
from qtimes_cost.lib.gnn import GNNArch
from qtimes_cost.lib.sat_gen import gen_instance
from qtimes_cost.lib.train_loop import eval_pack_metrics, short_train
from qtimes_cost.scripts.gen_teb_packs import build_dataset


def test_q_from_logit_freeze_exact_one_zero_grad():
    u = torch.tensor(4.0, requires_grad=True)
    q = q_from_logit(u, freeze_q1=True)
    assert float(q.detach()) == 1.0
    q.backward()
    assert float(u.grad) == 0.0
    # learned map at 0 is the midpoint 5.5, not 1
    q_mid = q_from_logit(torch.tensor(0.0))
    assert abs(float(q_mid) - 5.5) < 1e-5
    assert integer_q(0.0, freeze_q1=True) == 1
    assert integer_q(0.0, freeze_q1=False) != 1


def test_learned_q_grad_nonzero():
    u = torch.tensor(0.0, requires_grad=True)
    q = q_from_logit(u, freeze_q1=False)
    q.backward()
    assert u.grad is not None and float(u.grad) != 0.0


def test_ansatz_freeze_q_path():
    n, L = 6, 2
    cfg = AnsatzConfig(n=n, L=L, nf=16)
    sim = MultiCavityAnsatz(cfg)
    _, _, E = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 4)
    p = random_params(cfg, np.random.default_rng(1))
    # nonzero logits must still yield q≡1 on the freeze path
    p["q_logits"] = torch.full((L,), 3.0)
    out = sim.forward(
        torch.as_tensor(E, dtype=torch.float32),
        p["alpha0"], p["A"], p["TH"], p["q_logits"], gamma=p["gamma"],
        q_mode="freeze1",
    )
    assert out["q_int"] == [1, 1]
    assert all(abs(float(q) - 1.0) < 1e-8 for q in out["q_cont"])
    learned = sim.forward(
        torch.as_tensor(E, dtype=torch.float32),
        p["alpha0"], p["A"], p["TH"], p["q_logits"], gamma=p["gamma"],
        q_mode="learned",
    )
    assert any(abs(float(q) - 1.0) > 0.5 for q in learned["q_cont"])


def test_short_train_freeze_matches_init_and_keeps_q1(tmp_path):
    build_dataset(6, tmp_path, ntrain=4, nval=4, ntest=0, store_E=True, overwrite=True)
    arch = GNNArch(H=16, rounds=1, lr=5e-5, wd=1e-4)
    common = dict(
        steps=3, L=2, max_val=2, max_train=4, seed=0,
        data_dir=tmp_path, device="cpu", leakpen=0.1, patience=5, num_threads=1,
    )
    val_l, state_l, info_l = short_train(arch, q_mode="learned", **common)
    val_f, state_f, info_f = short_train(arch, q_mode="freeze1", **common)
    assert info_l["init_checksum"] == info_f["init_checksum"]
    assert info_l["alpha0_learned"] and info_f["alpha0_learned"]
    assert info_f["q_mode"] == "freeze1"
    assert 6 in val_f and 6 in state_f
    for k, v in state_f[6].items():
        if "head_q" in k:
            assert torch.count_nonzero(v) == 0
    # learned q head is not forced to zero
    assert any(
        torch.count_nonzero(v) > 0 for k, v in state_l[6].items() if "head_q" in k
    )
    met_f = eval_pack_metrics(
        6, arch, state_f[6], split="val", L=2, max_count=2, seed=0,
        data_dir=tmp_path, device="cpu", q_mode="freeze1",
    )
    met_l = eval_pack_metrics(
        6, arch, state_l[6], split="val", L=2, max_count=2, seed=0,
        data_dir=tmp_path, device="cpu", q_mode="learned",
    )
    assert abs(met_f["mean_q"] - 1.0) < 1e-6
    assert met_l["mean_q"] > 2.0
    assert info_f["best_step"] >= 0
    assert info_f["patience"] == 5
