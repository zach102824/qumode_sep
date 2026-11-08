"""N=15 checks before any optimizer call. Writes the target manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as h

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref

ROOT = Path(__file__).resolve().parent
B = float("-0.3444")


def check(name, cond, detail=""):
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    print("ok", name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fd(fun, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return g


def main():
    target = h.Target(15, B)
    check("Jnn", abs(target.J[0, 1] - 1.0) < 1e-15, str(target.J[0, 1]))
    check("packing", target.packing == [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]])
    check("features kept", target.feature_rank == 3, str(target.feature_rank))
    check("pair slots", int(np.triu_indices(15, 1)[0].size) == 105)

    # The training matvec and the sparse matrix must agree before the spectrum is trusted.
    rng = np.random.default_rng(15)
    probe = rng.normal(size=target.dim) + 1j * rng.normal(size=target.dim)
    probe /= np.linalg.norm(probe)
    sparse = ref.sparse_h(target)
    a = ref.apply_h(target, probe)
    b = sparse @ probe
    check("two H matvecs", np.linalg.norm(a - b) < 1e-10, str(np.linalg.norm(a - b)))

    spec = ref.ground_state(target)
    check("gs residual", spec["residual"] < 1e-8, str(spec["residual"]))
    sparse_resid = float(np.linalg.norm(sparse @ spec["gs"] - spec["E0"] * spec["gs"]))
    check("sparse residual", sparse_resid < 1e-8, str(sparse_resid))
    gs_parity = spec["parity"]
    print("info E0", spec["E0"], "gap", spec["full_gap"], "parity", gs_parity, "resid", spec["residual"])

    x = rng.normal(scale=0.2, size=6)
    _, _, g, _ = h.logical_grad(target, x)
    gfd = fd(lambda z: h.logical_grad(target, z)[1], x)
    check("candidate grad L1", np.max(np.abs(g - gfd)) < 1e-5, str(np.max(np.abs(g - gfd))))
    y = rng.normal(scale=0.2, size=2)
    _, _, gq, _ = h.qaoa_grad(target, y)
    gqfd = fd(lambda z: h.qaoa_grad(target, z)[1], y)
    check("qaoa grad p1", np.max(np.abs(gq - gqfd)) < 1e-5, str(np.max(np.abs(gq - gqfd))))

    manifest = {
        "n": 15,
        "alpha": 1.1,
        "B": "-0.3444",
        "boundary": "open",
        "kac": False,
        "Jij": [[i, j, format(float(target.J[i, j]), ".17g")] for i in range(15) for j in range(i + 1, 15)],
        "packing": target.packing,
        "rK": format(float(target.rK), ".17g"),
        "rM": format(float(target.rM), ".17g"),
        "feature_rank_of_J_K_M": target.feature_rank,
        "pair_slots": 105,
        "features_not_dropped": True,
        "P_formula": "3L+3",
        "ground_state_method": spec["method"],
    }
    (ROOT / "target_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    packing = {"n": 15, "rule": "[a, a+N/3, a+2N/3]", "packing": target.packing}
    (ROOT / "packing.json").write_text(json.dumps(packing, indent=2) + "\n")
    report = {
        "manifest_sha256": sha256(ROOT / "target_manifest.json"),
        "packing_sha256": sha256(ROOT / "packing.json"),
        "harness_sha256": sha256(ROOT.parents[0] / "harness.py"),
        "reference_sha256": sha256(ROOT / "reference.py"),
        "J01": float(target.J[0, 1]),
        "rK": float(target.rK),
        "rM": float(target.rM),
        "feature_rank_of_J_K_M": target.feature_rank,
        "pair_slots": 105,
        "E0": spec["E0"],
        "ritz_values": spec["evals"],
        "full_gap": spec["full_gap"],
        "gs_residual": spec["residual"],
        "sparse_residual": sparse_resid,
        "gs_parity": gs_parity,
        "excited": spec["excited"],
        "same_parity_gap": spec["same_parity_gap"],
        "degeneracy_assumed": False,
        "dense_matrix": False,
        "features_not_dropped": True,
        "P_formula": "3L+3",
        "method": spec["method"],
    }
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("manifest_sha256", "packing_sha256", "E0", "full_gap", "gs_residual", "gs_parity")}, indent=2))


if __name__ == "__main__":
    main()
