"""N=6 smoke: manifest, FK algebra, gradients, full-vs-sector gap. Abort the round on failure."""
from __future__ import annotations

import numpy as np

from ansatz import (
    legal_energy_grad,
    legal_state,
    legal_to_relaxed,
    project_relaxed_to_legal,
    qaoa_energy_grad,
    qaoa_state,
    relaxed_energy_grad,
    relaxed_state,
)
from target import (
    MANIFEST_SHA256,
    apply_H,
    dense_H,
    fk_matrix,
    hc_hash_le,
    load_manifest,
    normalize_feature,
    pair_diagonal,
    parse_system,
    spectrum_diagnostics,
    upper_rms,
    z_table,
)


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name}")


def fd(fun, x, eps=1e-7):
    g = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        g[i] = (fun(x + step) - fun(x - step)) / (2 * eps)
    return g


def build_model(n: int):
    from ansatz import Model

    manifest, raw = load_manifest()
    check("manifest sha256", __import__("hashlib").sha256(raw).hexdigest() == MANIFEST_SHA256)
    J, B = parse_system(manifest, n)
    z = z_table(n)
    hc = pair_diagonal(J, z)
    K = fk_matrix(J)
    rms_J = upper_rms(J)
    Kn, rms_K, scale = normalize_feature(K, rms_J)
    model = Model(n, J, B, pair_diagonal(Kn, z))
    return model, J, B, hc, K, Kn, rms_J, rms_K, scale


def main() -> dict:
    model, J, B, hc, K, Kn, rms_J, rms_K, scale = build_model(6)
    check("NN is 1", abs(J[0, 1] - 1.0) < 1e-15)
    check("string coupling kept", abs(J[0, 2] - float("0.4665164957684037")) < 1e-15)
    check("B string", abs(B - float("-0.1708")) < 1e-15)
    prod = J @ J
    np.fill_diagonal(prod, 0.0)
    check("FK matches J@J offdiag", np.allclose(K, prod, atol=1e-12))
    # Algebra: 0.5 * (sum_i h_i^2 - diag) equals sum_{j<k} K_jk z_j z_k
    rng = np.random.default_rng(1)
    z = model.z
    x = int(rng.integers(0, model.dim))
    zz = z[x]
    h = J @ zz
    left = 0.5 * (float(np.dot(h, h)) - float(np.sum(np.diag(J @ J))))
    right = float(pair_diagonal(K, z[x : x + 1])[0]) if False else float(
        0.5 * zz @ K @ zz
    )
    check("FK quadratic identity", abs(left - right) < 1e-10, f"{left} vs {right}")
    check("normalized RMS matches Hc", abs(upper_rms(Kn) - rms_J) < 1e-12)
    check("Hc diag matches model", np.allclose(hc, model.hc))
    e_plus = model.energy(model.plus)
    check("plus energy B*N", abs(e_plus - B * 6) < 1e-10, str(e_plus))
    H = dense_H(hc, B, 6)
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v /= np.linalg.norm(v)
    check("apply_H matches dense", np.allclose(apply_H(v, hc, B, 6), H @ v, atol=1e-10))
    check("local energy matches dense", abs(model.energy(v) - float(np.real(np.vdot(v, H @ v)))) < 1e-9)
    off = H - np.diag(np.diag(H))
    check("stoquastic", float(off.max()) <= 1e-12)

    # theta=0, c2=c3=0 equals shared-gamma QAOA
    g = 0.31
    betas = np.array([0.2, -0.15, 0.4])
    shell = np.column_stack([betas, np.zeros(3), np.zeros(3)])
    a = legal_state(model, g, 0.0, shell)
    b = qaoa_state(model, np.full(3, g), betas)
    check("theta0 linear shell is shared-gamma QAOA", abs(np.vdot(a, b)) > 1 - 1e-10)

    x_legal = np.array([0.2, -0.4, 0.1, -0.2, 0.05, 0.3, 0.0, -0.1, 0.2, 0.15, -0.05], dtype=np.float64)
    # L=3 legal width = 2+9 = 11
    check("legal width", x_legal.size == 11)
    xr = legal_to_relaxed(3, x_legal)
    check("project roundtrip", np.allclose(project_relaxed_to_legal(3, xr), x_legal))
    check(
        "equal gammas match legal state",
        abs(np.vdot(legal_state(model, 0.2, -0.4, x_legal[2:].reshape(3, 3)), relaxed_state(model, xr[:3], xr[3], xr[4:].reshape(3, 3))))
        > 1 - 1e-10,
    )

    E, g = legal_energy_grad(model, 3, x_legal)
    g_fd = fd(lambda z: legal_energy_grad(model, 3, z)[0], x_legal)
    err = float(np.max(np.abs(g - g_fd)))
    check("legal grad vs FD", err < 5e-6, str(err))

    xr = rng.normal(scale=0.2, size=3 + 1 + 9)
    for lam in (0.0, 0.3, 4.0):
        _E, pen, gr = relaxed_energy_grad(model, 3, xr, lam)

        def fun(z, lam=lam):
            e, p, _g = relaxed_energy_grad(model, 3, z, lam)
            return e + p

        g_fd = fd(fun, xr)
        err = float(np.max(np.abs(gr - g_fd)))
        check(f"relaxed+penalty grad lam={lam}", err < 5e-6, str(err))

    # QAOA grad
    xq = rng.normal(scale=0.2, size=6)

    def qfun(z):
        e, _gg, _gb = qaoa_energy_grad(model, z[:3], z[3:])
        return e

    e, gg, gb = qaoa_energy_grad(model, xq[:3], xq[3:])
    g_fd = fd(qfun, xq)
    err = float(np.max(np.abs(np.concatenate([gg, gb]) - g_fd)))
    check("qaoa grad vs FD", err < 5e-6, str(err))

    spec = spectrum_diagnostics(hc, B, 6)
    check("GS in P+ R+", spec["P_plus_weight"] > 1 - 1e-8 and spec["R_plus_weight"] > 1 - 1e-8)
    check("full gap is the tiny split", spec["full_space_gap"] < 0.01, str(spec["full_space_gap"]))
    check("accessible gap is larger", spec["accessible_PR_plus_gap"] > 0.2, str(spec["accessible_PR_plus_gap"]))
    check("partner is not P+", spec["lowest_states"][1]["P_plus_weight"] < 1e-6)
    check("variational floor", spec["E0"] < e_plus)
    print(
        f"N6 B={B} E0={spec['E0']:.6f} full_gap={spec['full_space_gap']:.6e} "
        f"sector_gap={spec['accessible_PR_plus_gap']:.6e} S={spec['entropy_bits']:.6f} "
        f"Hc_le={hc_hash_le(hc)}"
    )
    print("ALL SMOKE CHECKS PASSED")
    return {
        "full_space_gap": spec["full_space_gap"],
        "accessible_gap": spec["accessible_PR_plus_gap"],
        "E0": spec["E0"],
        "legal_grad_err": err,
        "rms_scale": scale,
        "hc_sha256_le": hc_hash_le(hc),
    }


if __name__ == "__main__":
    main()
