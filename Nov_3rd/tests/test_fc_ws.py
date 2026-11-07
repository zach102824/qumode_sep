"""FC-WS identities: O table, Krawtchouk, δ=0 champ recovery, degree-tied Walsh."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCR = Path(__file__).resolve().parents[1] / "scripts"
for p in (str(_ROOT), str(_SCR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.fc_ws import (
    CLASS_DEGREE,
    CLASSES,
    KRAW_M,
    O_TABLE,
    anticommute_fraction,
    effective_delta,
    footprint_class_ids,
    krawtchouk_matrix,
    mu_from_phi,
    o_tilde,
    phi_from_mu,
    rescale_shell_phi,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"PASS {name} {detail}")


def test_o_table():
    for r in (1, 2, 3):
        for t in (1, 2, 3):
            got = anticommute_fraction(r, t)
            exp = O_TABLE[r - 1][t - 1]
            check(f"O_r{r}_t{t}", abs(got - exp) < 1e-12, f"{got} vs {exp}")


def test_krawtchouk_roundtrip():
    M = krawtchouk_matrix(3)
    check("K_hand", torch.equal(M, KRAW_M))
    expect = torch.tensor(
        [
            [1, 1, 1, 1],
            [3, 1, -1, -3],
            [3, -1, -1, 3],
            [1, -1, 1, -1],
        ],
        dtype=torch.float64,
    )
    check("K_values", torch.equal(M, expect), str(M.tolist()))
    check("K_involution", torch.equal(M @ M, 8 * torch.eye(4, dtype=torch.float64)))
    phi = torch.tensor([[0.0, -0.4, 0.2, 0.15], [0.1, -0.2, 0.05, -0.3]], dtype=torch.float64)
    back = phi_from_mu(mu_from_phi(phi))
    check("roundtrip", torch.allclose(back, phi), f"max {(back-phi).abs().max()}")
    # κ=0 leaves φ unchanged in float64
    z = torch.zeros(3, dtype=torch.float64)
    kap = torch.tensor(0.0, dtype=torch.float64)
    out = rescale_shell_phi(phi, z, kap)
    check("kappa0_phi", torch.allclose(out.double(), phi))


def test_footprints():
    # n=6, var0 = MSB = bit 5. Cavity 0 owns bits 5,4,3. Cavity 1 owns bits 2,1,0.
    n = 6
    ids = footprint_class_ids(n)
    def bit_of_var(v):
        return n - 1 - v
    def mask(vars_):
        m = 0
        for v in vars_:
            m |= 1 << bit_of_var(v)
        return m
    cases = {
        mask([0]): 0,             # (1)
        mask([0, 1]): 1,          # (2)
        mask([0, 3]): 2,          # (1,1)
        mask([0, 1, 2]): 3,       # (3)
        mask([0, 1, 3]): 4,       # (2,1)
        0: -1,
    }
    for m, cid in cases.items():
        check(f"fp_n6_{m}", int(ids[m]) == cid, f"got {int(ids[m])} want {cid}")
    check("n6_no_111", int((ids == 5).sum()) == 0)
    ids9 = footprint_class_ids(9)
    m111 = mask_n(9, [0, 3, 6])
    check("n9_111", int(ids9[m111]) == 5, f"got {int(ids9[m111])}")
    # partitions of degree <=3 are exhaustive
    check("classes_cover", len(CLASSES) == 6)
    check("degree_matches", tuple(sum(c) for c in CLASSES) == CLASS_DEGREE)


def mask_n(n, vars_):
    m = 0
    for v in vars_:
        m |= 1 << (n - 1 - v)
    return m


def test_o_tilde_and_pool():
    w = torch.tensor([1.0, 2.0, 0.5, 0.2, 0.3, 0.0], dtype=torch.float64)
    delta = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.4, 9.0], dtype=torch.float64, requires_grad=True)
    de = effective_delta(delta, pool_111=True)
    check("pool_tie", torch.allclose(de[5], de[2]) and float(de[5].detach()) == 0.3)
    ot = o_tilde(w, de)
    check("ot_shape", tuple(ot.shape) == (3,))
    # manual numerator / denom
    from Nov_3rd.fc_ws import N_PIECES, S_O
    numer = (w * de.detach()) @ S_O
    denom = (w * N_PIECES).sum()
    check("ot_formula", torch.allclose(ot, numer / denom))
    ot.sum().backward()
    check("ot_grad_delta", delta.grad is not None and float(delta.grad.abs().sum()) > 0)
    # δ=0 → Õ=0
    z = o_tilde(w, torch.zeros(6, dtype=torch.float64))
    check("ot_zero_delta", torch.allclose(z, torch.zeros(3, dtype=torch.float64)))


def test_circuit_identities():
    device = torch.device("cpu")
    L, n = 2, 6
    sim = make_sim(n, L, device)
    pack = np.load(S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz", allow_pickle=True)
    E = torch.as_tensor(np.asarray(pack["train_E"][0]), dtype=torch.float32, device=device)
    g = torch.tensor([0.3, -0.5], dtype=torch.float32)
    phi = torch.tensor([[0.0, -0.2, 0.05, 0.01], [0.0, -0.4, -0.1, 0.02]], dtype=torch.float32)
    ref = sim.forward_nov3rd(E, g, phi)
    zero = {
        "delta": torch.zeros(6),
        "kappa": torch.tensor(0.0),
        "pool_111": False,
    }
    got = sim.forward_nov3rd(E, g, phi, fcws=zero)
    check(
        "delta0_matches_scalar",
        abs(float(got["inv_p"]) - float(ref["inv_p"])) < 1e-4,
        f"fc {float(got['inv_p'])} scalar {float(ref['inv_p'])}",
    )
    # degree-tied δ reproduces Walsh γ_w = γ*(1+δ_degree), residual degree-4 at γ
    delta = torch.tensor([0.2, -0.1, -0.1, 0.3, 0.3, 0.3])  # tied by degree
    gw = torch.stack([g * (1 + delta[0]), g * (1 + delta[1]), g * (1 + delta[3]), g], dim=1)
    # gw shape (L, 4): columns deg 1,2,3,4
    ref_w = sim.forward_nov3rd(E, gw, phi)
    fc = sim.forward_nov3rd(
        E, g, phi, fcws={"delta": delta, "kappa": torch.tensor(0.0), "pool_111": False}
    )
    check(
        "degree_tied_matches_walsh",
        abs(float(fc["inv_p"]) - float(ref_w["inv_p"])) < 1e-4,
        f"fc {float(fc['inv_p'])} walsh {float(ref_w['inv_p'])}",
    )
    # κ couples: changing κ at fixed nonzero δ changes φ and inv_p, with a gradient
    delta_p = torch.tensor([0.4, -0.3, 0.2, 0.1, -0.2, 0.0], requires_grad=True)
    kap = torch.tensor(0.5, requires_grad=True)
    out = sim.forward_nov3rd(
        E, g, phi, fcws={"delta": delta_p, "kappa": kap, "pool_111": False}
    )
    out["inv_p"].backward()
    check("kappa_grad", kap.grad is not None and abs(float(kap.grad)) > 0, f"grad {None if kap.grad is None else float(kap.grad)}")
    check("delta_grad", delta_p.grad is not None and float(delta_p.grad.abs().sum()) > 0)
    # parts sum + mean reconstructs the centered cost
    parts = sim.footprint_parts(E)
    f = sim.cost_bitstring(E)
    recon_bit = None
    # comps are on the grid; compare grid cost
    Cn = sim.cost_Cn(E)
    summed = parts["comps"].sum(dim=0) + parts["rest"]
    # Cn and summed differ by the degree-0 mean (constant on the full space)
    diff = (Cn - summed).reshape(-1)
    check(
        "components_differ_by_constant",
        float((diff - diff.mean()).abs().max()) < 1e-4,
        f"spread {float((diff-diff.mean()).abs().max())} mean {float(diff.mean())}",
    )
    # (1,1,1) mass is 0 at n=6
    check("w111_n6", float(parts["w"][5]) == 0.0, f"w={parts['w'].tolist()}")
    # Published Walsh seed1 at δ=0, κ=0 matches the graded-(L,4) forward.
    walsh_pt = (
        S.OUT / "beat_qaoa_12h" / "train69" / "ucx_bit_a" / "walsh_degree" / "params_L2_seed1_B100.pt"
    )
    st = torch.load(walsh_pt, map_location="cpu", weights_only=False)
    gw = st["gamma_w"].float()
    from beat_params import phi_from_c
    phi_w = phi_from_c(st["c1"].float(), st["c2"].float(), st["c3"].float())
    ref_w2 = sim.forward_nov3rd(E, gw, phi_w)
    fc_w = sim.forward_nov3rd(
        E, gw, phi_w, fcws={"delta": torch.zeros(6), "kappa": torch.tensor(0.0), "pool_111": False}
    )
    check(
        "walsh_seed1_delta0",
        abs(float(ref_w2["inv_p"]) - float(fc_w["inv_p"])) < 1e-4,
        f"graded {float(ref_w2['inv_p'])} fc {float(fc_w['inv_p'])}",
    )
    _ = recon_bit, f


def main() -> int:
    test_o_table()
    test_krawtchouk_roundtrip()
    test_footprints()
    test_o_tilde_and_pool()
    test_circuit_identities()
    print("ALL FC-WS TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
