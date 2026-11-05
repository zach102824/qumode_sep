"""Quick correctness checks before smoke."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from pspin import make_instance, apply_H
from circuit_snap import make_ansatz, fidelity, init_angles, forward_state

def main():
    inst = make_instance(6, 1.0, seed=0)
    # residual of GS
    Hv = apply_H(inst.gs, inst.Hc_diag, inst.Gamma, inst.n)
    res = np.linalg.norm(Hv - inst.E0 * inst.gs)
    print(f"GS residual n6: {res:.3e} E0={inst.E0:.4f}")
    assert res < 1e-6, res

    device = torch.device("cpu")
    sim = make_ansatz(6, 2, device)
    g, p = init_angles(2, seed=0)
    Hc = torch.as_tensor(inst.Hc_diag, dtype=torch.float32)
    gs = torch.as_tensor(inst.gs, dtype=torch.complex64)
    psi = forward_state(sim, Hc, g, p)
    print("psi norm", float(torch.linalg.vector_norm(psi)))
    F = float(fidelity(sim, Hc, gs, g, p))
    print(f"init F≈{F:.4f}")
    assert 0 <= F <= 1

    # n=9 lanczos not needed; dense OK. Spot-check n=9
    inst9 = make_instance(9, 1.5, seed=1)
    Hv9 = apply_H(inst9.gs, inst9.Hc_diag, inst9.Gamma, inst9.n)
    res9 = np.linalg.norm(Hv9 - inst9.E0 * inst9.gs)
    print(f"GS residual n9: {res9:.3e} E0={inst9.E0:.4f}")
    assert res9 < 1e-5, res9
    print("SELFCHECK OK")

if __name__ == "__main__":
    main()
