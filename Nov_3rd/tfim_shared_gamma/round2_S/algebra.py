"""Commutation facts for the shell generators. The consultant claim is wrong.

Claim to reject: c2 (sum of intra XX) commutes with every intra ZZ.
Fact: XX and ZZ anticommute when they share one site and commute when they
share two. A sum of intra XX therefore does not commute with a generic intra ZZ.

c3 = XXX on a cavity commutes with every intra ZZ (two overlaps) and
anticommutes with an inter ZZ that has exactly one endpoint in that cavity.
"""
from __future__ import annotations

import numpy as np

from .frozen import FrozenSystem


def _embed_1(n: int, site: int, pauli: np.ndarray) -> np.ndarray:
    """Pauli on one qubit, identity elsewhere. Site 0 is the LSB."""
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        bit = (s >> site) & 1
        # Apply the 2x2 to this qubit. X flips, Z phases.
        if pauli is X:
            t = s ^ (1 << site)
            out[t, s] += 1.0
        elif pauli is Z:
            out[s, s] += 1.0 if bit == 0 else -1.0
        else:
            raise RuntimeError("only X or Z")
    return out


X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _xx(n, i, j):
    return _embed_1(n, i, X) @ _embed_1(n, j, X)


def _zz(n, i, j):
    return _embed_1(n, i, Z) @ _embed_1(n, j, Z)


def _xxx(n, sites):
    op = np.eye(1 << n, dtype=np.complex128)
    for s in sites:
        op = _embed_1(n, s, X) @ op
    return op


def _comm_norm(A, B) -> float:
    return float(np.linalg.norm(A @ B - B @ A))


def _acomm_norm(A, B) -> float:
    return float(np.linalg.norm(A @ B + B @ A))


def shell_zz_algebra(n: int = 6) -> dict:
    """Explicit  N-qubit matrices. n=6 is two cavities and already shows the failure."""
    # Cavity 0 sites 0,1,2. Intra ZZ bonds (0,1), (1,2), (0,2).
    # Inter bond (2,3) has one endpoint in the cavity. Bond (3,4) has none.
    xx01 = _xx(n, 0, 1)
    zz01 = _zz(n, 0, 1)
    zz02 = _zz(n, 0, 2)
    zz12 = _zz(n, 1, 2)
    sum_xx = _xx(n, 0, 1) + _xx(n, 0, 2) + _xx(n, 1, 2)
    xxx = _xxx(n, (0, 1, 2))
    zz_inter_touch = _zz(n, 2, 3)
    zz_inter_far = _zz(n, 3, 4) if n >= 6 else None

    def rel(A, B):
        c = _comm_norm(A, B)
        a = _acomm_norm(A, B)
        if c < 1e-8 and a > 1e-6:
            return "commute"
        if a < 1e-8 and c > 1e-6:
            return "anticommute"
        return f"neither c={c:.3e} a={a:.3e}"

    report = {
        "false_claim": "c2 sum of intra XX commutes with all intra ZZ",
        "xx01_vs_zz01_same_two_sites": rel(xx01, zz01),
        "xx01_vs_zz02_one_shared_site": rel(xx01, zz02),
        "sum_intra_XX_vs_zz01": rel(sum_xx, zz01),
        "sum_intra_XX_vs_zz02": rel(sum_xx, zz02),
        "sum_intra_XX_vs_zz12": rel(sum_xx, zz12),
        "XXX_vs_intra_zz01": rel(xxx, zz01),
        "XXX_vs_intra_zz02": rel(xxx, zz02),
        "XXX_vs_inter_zz_one_endpoint": rel(xxx, zz_inter_touch),
        "correction": (
            "XX and ZZ commute on a shared pair (two overlaps) and anticommute "
            "when they share one site. The intra-cavity sum of XX therefore does "
            "not commute with intra ZZ. XXX commutes with every intra ZZ and "
            "anticommutes with an inter ZZ that touches the cavity once."
        ),
    }
    if zz_inter_far is not None:
        report["XXX_vs_inter_zz_no_endpoint"] = rel(xxx, zz_inter_far)
    # The sum neither commutes nor anticommutes: it contains both a two-site
    # pair, which commutes, and one-site overlaps, which anticommute.
    report["false_claim_rejected"] = report["sum_intra_XX_vs_zz01"] != "commute"
    report["xxx_intra_commutes"] = report["XXX_vs_intra_zz01"] == "commute"
    report["xxx_inter_anticommutes"] = report["XXX_vs_inter_zz_one_endpoint"] == "anticommute"
    return report


def assert_algebra() -> dict:
    report = shell_zz_algebra(6)
    pair_ok = (
        report["xx01_vs_zz01_same_two_sites"] == "commute"
        and report["xx01_vs_zz02_one_shared_site"] == "anticommute"
    )
    if not (report["false_claim_rejected"] and pair_ok and report["xxx_intra_commutes"] and report["xxx_inter_anticommutes"]):
        raise RuntimeError(f"shell/ZZ algebra did not match the correction: {report}")
    return report


def normalization_checks(system: FrozenSystem) -> dict:
    from .frozen import g_diag, upper_rms

    # thetaIn=1, thetaK=0 is exactly Hc.
    g = g_diag(system, "family_S", 1.0, 0.0)
    err_id = float(np.max(np.abs(g - system.hc_full)))
    g_inter = g_diag(system, "inter_K_A", 0.0, 0.4)
    g_s = g_diag(system, "family_S", 1.0, 0.4)
    err_recover = float(np.max(np.abs(g_inter - g_s)))
    # Full-K and inter-K are different operators at the same thetaK.
    g_full = g_diag(system, "full_K_A", 0.0, 0.4)
    differ = float(np.max(np.abs(g_full - g_inter)))
    return {
        "thetaIn1_thetaK0_is_Hc": err_id,
        "thetaIn1_matches_inter_K_A": err_recover,
        "full_K_differs_from_inter_K": differ,
        "rms_J": upper_rms(system.J),
        "rms_K_full_raw": upper_rms(system.K_full),
        "rms_K_inter_raw_including_zeros": upper_rms(system.K_inter),
        "rms_K_full_norm": upper_rms(system.K_full_norm),
        "rms_K_inter_norm": upper_rms(system.K_inter_norm),
        "sK_full": system.sK_full,
        "sK_inter": system.sK_inter,
        "intra_not_shell": system.ledger["intra_level_resolved_not_shell_snap"],
    }
