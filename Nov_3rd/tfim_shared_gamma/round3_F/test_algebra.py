"""Verify Hc^2 = c0 I + 2 FK + 2 Q4 and the tied pair coefficients."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

R2 = Path(__file__).resolve().parents[1] / "round2_H"
sys.path.insert(0, str(R2))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import build_features, pair_walsh  # noqa: E402
from target import MANIFEST_SHA256, load_manifest, parse_system, z_table  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name}")


def main():
    raw = (Path(__file__).resolve().parent / "FROZEN_TARGET_R2.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    check("manifest hash", digest == MANIFEST_SHA256, digest)
    manifest, _ = load_manifest()
    for n in (6, 12):
        J, B = parse_system(manifest, n)
        feat = build_features(J)
        check(f"N{n} identity scaled", feat["identity_max_scaled"] < 1e-12, str(feat["identity_max_abs"]))
        check(f"N{n} fourbody count", feat["n_fourbody"] == (15 if n == 6 else 495), str(feat["n_fourbody"]))
        z = z_table(n)
        # Pair coefficients of F and A agree; the difference is pure 4-body.
        i, j = 0, 1
        a = pair_walsh(feat["feature_A"], z, i, j)
        f = pair_walsh(feat["feature_F"], z, i, j)
        check(f"N{n} pair coeff match", abs(a - f) < 1e-10, f"{a} vs {f}")
        diff = feat["feature_F"] - feat["feature_A"]
        check(f"N{n} diff orthogonal to ZZ", abs(pair_walsh(diff, z, i, j)) < 1e-10)
        check(f"N{n} rK positive", feat["rK"] > 0)
        print(
            f"N{n} B={B} rK={feat['rK']:.6f} c0={feat['c0']:.6f} "
            f"pairs={feat['n_pairs']} fourbody={feat['n_fourbody']} "
            f"identity={feat['identity_max_abs']:.3e}"
        )
    # Gradient of the tied generator on N=6.
    from ansatz import Model, legal_energy_grad

    J, B = parse_system(manifest, 6)
    feat = build_features(J)
    model = Model(6, J, B, feat["feature_F"])
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.2, size=2 + 6)

    def fun(z):
        return legal_energy_grad(model, 2, z)[0]

    E, g = legal_energy_grad(model, 2, x)
    fd = np.zeros_like(x)
    eps = 1e-7
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        fd[i] = (fun(x + step) - fun(x - step)) / (2 * eps)
    err = float(np.max(np.abs(g - fd)))
    check("tied grad vs FD", err < 5e-6, str(err))
    print("ALL ALGEBRA CHECKS PASSED")
    return feat


if __name__ == "__main__":
    main()
