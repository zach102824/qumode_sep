"""Nov9th, Nov3rd, and QAOA evaluators.

Nov9th: |+> -> [UcZ -> D(alpha_l) -> bare shell SNAP]^L
with one shared gamma and shared theta, real alpha, three shell angles per layer.
Fidelity is the unconditioned code overlap. No renormalization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import ALPHA_MAX, CODE_LEVELS
from encoding import Packing, feature_M
from gates import (
    alpha_from_u,
    displacement_matrix,
    flip_all_table,
    flip_level,
    snap_diagonal,
)
from hamiltonian import (
    LogicalInstance,
    apply_H_site,
    apply_X_sum,
    apply_rx_site,
    logical_energy,
    logical_zz,
    pair_diag,
    plus_state,
)


def _site_index(groups: list) -> dict:
    out = {}
    for c, cav in enumerate(groups):
        for r, site in enumerate(cav):
            out[int(site)] = (c, r)
    return out


def fock_pair_diagonal(coeff: np.ndarray, groups: list, nf: int, encoding: str, mode: str) -> np.ndarray:
    """Diagonal of a pairwise Z operator on the product Fock space.

    mode Uz: pairs that touch a leaked cavity contribute 0.
    mode U0: any leaked cavity zeros the whole component.
    mode energy: pairs that touch a leaked cavity contribute |coeff_ij|.
    """
    C = len(groups)
    grids = np.meshgrid(*[np.arange(nf) for _ in range(C)], indexing="ij")
    leaked = [grids[c] >= CODE_LEVELS for c in range(C)]
    any_leak = np.zeros(grids[0].shape, dtype=bool)
    for c in range(C):
        any_leak |= leaked[c]
    where = _site_index(groups)
    z_site = {}
    for c, cav in enumerate(groups):
        f = grids[c]
        bitsrc = f if encoding == "binary" else (f ^ (f >> 1))
        for r, site in enumerate(cav):
            bit = (bitsrc >> r) & 1
            z_site[int(site)] = 1.0 - 2.0 * bit
    d = np.zeros(grids[0].shape, dtype=np.float64)
    n = coeff.shape[0]
    if mode == "U0":
        zz = np.zeros(grids[0].shape, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                cij = float(coeff[i, j])
                if cij != 0.0:
                    zz = zz + cij * z_site[i] * z_site[j]
        return np.where(any_leak, 0.0, zz)
    for i in range(n):
        for j in range(i + 1, n):
            cij = float(coeff[i, j])
            if cij == 0.0 and mode != "energy":
                continue
            ci, _ri = where[i]
            cj, _rj = where[j]
            both = ~(leaked[ci] | leaked[cj])
            zz = cij * z_site[i] * z_site[j]
            if mode == "Uz":
                d = d + np.where(both, zz, 0.0)
            elif mode == "energy":
                pen = abs(cij)
                d = d + np.where(both, zz, pen)
            else:
                raise ValueError(mode)
    return d


def leak_qubit_count(groups: list, nf: int) -> np.ndarray:
    C = len(groups)
    grids = np.meshgrid(*[np.arange(nf) for _ in range(C)], indexing="ij")
    nleak = np.zeros(grids[0].shape, dtype=np.float64)
    for c in range(C):
        nleak = nleak + 3.0 * (grids[c] >= CODE_LEVELS)
    return nleak


def logical_index_map(groups: list, nf: int, encoding: str) -> np.ndarray:
    C = len(groups)
    grids = np.meshgrid(*[np.arange(nf) for _ in range(C)], indexing="ij")
    logical = np.zeros(grids[0].shape, dtype=np.int64)
    code = np.ones(grids[0].shape, dtype=bool)
    for c, cav in enumerate(groups):
        f = grids[c]
        code &= f < CODE_LEVELS
        bitsrc = f if encoding == "binary" else (f ^ (f >> 1))
        for r, site in enumerate(cav):
            bit = ((bitsrc >> r) & 1).astype(np.int64)
            logical |= bit << int(site)
    return np.where(code, logical, -1)


@dataclass
class FockModel:
    inst: LogicalInstance
    packing: Packing
    nf: int
    ucz: str
    strict8: bool
    dJ_phase: np.ndarray
    dK_phase: np.ndarray
    dM_phase: np.ndarray
    d_energy: np.ndarray
    logical_map: np.ndarray
    flip_all: np.ndarray
    M_scale: float

    @property
    def C(self) -> int:
        return len(self.packing.groups)

    @property
    def gate_name(self) -> str:
        if self.strict8:
            return "D8_strict_logical_not_a_displacement"
        return "D_true_expm_truncated_generator"


def build_fock_model(inst: LogicalInstance, packing: Packing, nf: int, ucz: str, strict8: bool) -> FockModel:
    if strict8:
        nf = CODE_LEVELS
        ucz = "Uz"
    groups = packing.groups
    _raw, _hat, scale, _diag = feature_M(inst.J, groups, inst.rms_J, inst.z)
    mode = ucz
    dJ = fock_pair_diagonal(inst.J, groups, nf, packing.encoding, mode)
    dK = fock_pair_diagonal(inst.K_hat, groups, nf, packing.encoding, mode)
    dM = fock_pair_diagonal(_hat, groups, nf, packing.encoding, mode)
    dE = fock_pair_diagonal(inst.J, groups, nf, packing.encoding, "energy")
    dE = dE + float(inst.b) * leak_qubit_count(groups, nf)
    return FockModel(
        inst=inst,
        packing=packing,
        nf=int(nf),
        ucz=ucz,
        strict8=bool(strict8),
        dJ_phase=dJ,
        dK_phase=dK,
        dM_phase=dM,
        d_energy=dE,
        logical_map=logical_index_map(groups, nf, packing.encoding),
        flip_all=flip_all_table(packing.encoding),
        M_scale=float(scale),
    )


def plus_fock(nf: int, C: int) -> np.ndarray:
    psi = np.zeros((nf,) * C, dtype=np.complex128)
    sl = (slice(0, CODE_LEVELS),) * C
    psi[sl] = 8.0 ** (-0.5 * C)
    return psi


def _apply_unitary(psi: np.ndarray, U: np.ndarray, cav: int) -> np.ndarray:
    p = np.moveaxis(psi, cav, 0)
    sh = p.shape
    p = U @ p.reshape(U.shape[0], -1)
    p = p.reshape(sh)
    return np.moveaxis(p, 0, cav)


def _apply_phase_cav(psi: np.ndarray, phase: np.ndarray, cav: int) -> np.ndarray:
    shape = [1] * psi.ndim
    shape[cav] = phase.shape[0]
    return psi * phase.reshape(shape)


def unpack_nov9(x: np.ndarray, L: int) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    need = 3 + 4 * L
    if x.size != need:
        raise ValueError(f"nov9 parameter length {x.size} != {need}")
    gamma = float(x[0])
    thK = float(x[1])
    thM = float(x[2])
    u = x[3 : 3 + L]
    s = x[3 + L :].reshape(L, 3)
    alpha = np.array([alpha_from_u(float(v)) for v in u], dtype=np.float64)
    return gamma, thK, thM, alpha, s


def x_expectations(psi: np.ndarray, groups: list, encoding: str) -> np.ndarray:
    """<X> for each logical site. Leaked cavities contribute 0."""
    n = sum(len(g) for g in groups)
    out = np.zeros(n, dtype=np.float64)
    for c, cav in enumerate(groups):
        p = np.moveaxis(psi, c, 0).reshape(psi.shape[c], -1)
        for r, site in enumerate(cav):
            acc = 0.0
            for f in range(CODE_LEVELS):
                bits = f if encoding == "binary" else (f ^ (f >> 1))
                if ((bits >> r) & 1) == 0:
                    f2 = flip_level(f, r, encoding)
                    acc += 2.0 * float(np.real(np.dot(np.conj(p[f]), p[f2])))
            out[int(site)] = acc
    return out


def evaluate_nov9(model: FockModel, x: np.ndarray, L: int, keep_state: bool = False) -> dict:
    gamma, thK, thM, alpha, s = unpack_nov9(x, L)
    C = model.C
    psi = plus_fock(model.nf, C)
    ang = gamma * (model.dJ_phase + thK * model.dK_phase + thM * model.dM_phase)
    phase = np.exp(-1j * ang)
    for ell in range(L):
        psi = psi * phase
        U = displacement_matrix(float(alpha[ell]), model.nf, strict8=model.strict8)
        for c in range(C):
            psi = _apply_unitary(psi, U, c)
        sph = snap_diagonal(model.nf, float(s[ell, 0]), float(s[ell, 1]), float(s[ell, 2]), model.packing.encoding)
        for c in range(C):
            psi = _apply_phase_cav(psi, sph, c)
    out = _observe(model, psi, alpha)
    if not keep_state:
        out.pop("psi", None)
        out.pop("amp", None)
    return out


def _observe(model: FockModel, psi: np.ndarray, alpha: np.ndarray) -> dict:
    prob = psi.real * psi.real + psi.imag * psi.imag
    norm = float(prob.sum())
    sl = (slice(0, CODE_LEVELS),) * model.C
    code = psi[sl]
    p_code = float(np.sum(np.abs(code) ** 2))
    p_leak = float(1.0 - p_code)
    amp = np.zeros(1 << model.inst.n, dtype=np.complex128)
    logical = model.logical_map
    mask = logical >= 0
    np.add.at(amp, logical[mask], psi[mask])
    F = float(np.abs(np.vdot(model.inst.gs, amp)) ** 2)
    xs = x_expectations(psi, model.packing.groups, model.packing.encoding)
    E_diag = float(np.sum(model.d_energy * prob))
    E = E_diag - float(model.inst.b) * float(np.sum(xs))
    flipped = code
    for c in range(model.C):
        flipped = np.take(flipped, model.flip_all, axis=c)
    parity = float(np.vdot(code, flipped).real)
    # ZZ-only diagonal uses the energy diagonal without the +b leak qubits.
    # Recomputed as E_diag - b * <n_leak>, and <n_leak> = (E contribution)/b.
    # Store the coded-pair ZZ plus |J| penalties by subtracting the known leak term
    # from a second pass only when needed. Here E_zz_ext is the diagonal minus X.
    E_X = E - E_diag
    return {
        "E": float(E),
        "E_diag": float(E_diag),
        "E_X": float(E_X),
        "F": F,
        "p_leak": p_leak,
        "p_code": p_code,
        "norm": norm,
        "parity": parity,
        "alpha": np.asarray(alpha, dtype=np.float64),
        "psi": psi,
        "amp": amp,
    }


def nov9_x0(L: int, kind: str, rng: np.random.Generator | None) -> np.ndarray:
    x = np.zeros(3 + 4 * L, dtype=np.float64)
    if kind == "E":
        x[0] = 0.25
        mag = 0.06
        # Layer index ell is 1-based in the protocol: alpha = (-1)^{ell+1} * 0.06.
        for ell in range(L):
            alpha = ((-1.0) ** ((ell + 1) + 1)) * mag
            x[3 + ell] = float(np.arctanh(alpha / ALPHA_MAX))
        return x
    if rng is None:
        raise RuntimeError("random init needs an rng")
    x[0] = float(rng.uniform(-0.5, 0.5))
    x[1] = float(rng.uniform(-0.5, 0.5))
    x[2] = float(rng.uniform(-0.5, 0.5))
    for ell in range(L):
        alpha = float(rng.uniform(-0.08, 0.08))
        x[3 + ell] = float(np.arctanh(alpha / ALPHA_MAX))
    for ell in range(L):
        for t in range(3):
            x[3 + L + 3 * ell + t] = float(rng.uniform(-0.3, 0.3))
    return x


def nov9_sigmas(L: int) -> np.ndarray:
    sig = np.empty(3 + 4 * L, dtype=np.float64)
    sig[0] = 0.10
    sig[1] = 0.25
    sig[2] = 0.25
    sig[3 : 3 + L] = 0.02
    sig[3 + L :] = 0.15
    return sig


def nov9_alpha_index(L: int) -> np.ndarray:
    return np.arange(3, 3 + L, dtype=np.int64)


def pop_phase_logical(n: int, sites: list, s1: float, s2: float, s3: float) -> np.ndarray:
    idx = np.arange(1 << n)
    pop = np.zeros(idx.shape, dtype=np.float64)
    for q in sites:
        pop += ((idx >> int(q)) & 1).astype(np.float64)
    return s1 * pop + s2 * (pop * (pop - 1.0) / 2.0) + np.where(pop == 3.0, s3, 0.0)


def unpack_nov3(x: np.ndarray, L: int):
    x = np.asarray(x, dtype=np.float64)
    if x.size != 3 + 3 * L:
        raise ValueError(f"nov3 parameter length {x.size} != {3 + 3 * L}")
    s = x[3:].reshape(L, 3)
    return float(x[0]), float(x[1]), float(x[2]), s


def evaluate_nov3(inst: LogicalInstance, groups: list, dM: np.ndarray, x: np.ndarray, L: int, keep_state: bool = False) -> dict:
    gamma, thK, thM, s = unpack_nov3(x, L)
    dK = pair_diag(inst.K_hat, inst.z)
    gdiag = inst.hc + thK * dK + thM * dM
    psi = plus_state(inst.n)
    for ell in range(L):
        psi = psi * np.exp(-1j * gamma * gdiag)
        for cav in groups:
            for site in cav:
                psi = apply_H_site(psi, int(site))
            psi = psi * np.exp(-1j * pop_phase_logical(inst.n, cav, float(s[ell, 0]), float(s[ell, 1]), float(s[ell, 2])))
            for site in cav:
                psi = apply_H_site(psi, int(site))
    out = _observe_logical(inst, psi)
    if not keep_state:
        out.pop("psi", None)
        out.pop("amp", None)
    return out


def nov3_x0(L: int, kind: str, rng: np.random.Generator | None) -> np.ndarray:
    x = np.zeros(3 + 3 * L, dtype=np.float64)
    if kind == "E":
        x[0] = 0.25
        return x
    if rng is None:
        raise RuntimeError("random init needs an rng")
    x[0] = float(rng.uniform(-0.5, 0.5))
    x[1] = float(rng.uniform(-0.5, 0.5))
    x[2] = float(rng.uniform(-0.5, 0.5))
    for t in range(3 * L):
        x[3 + t] = float(rng.uniform(-0.3, 0.3))
    return x


def nov3_sigmas(L: int) -> np.ndarray:
    sig = np.empty(3 + 3 * L, dtype=np.float64)
    sig[0] = 0.10
    sig[1] = 0.25
    sig[2] = 0.25
    sig[3:] = 0.15
    return sig


def unpack_qaoa(x: np.ndarray, L: int):
    x = np.asarray(x, dtype=np.float64)
    if x.size != 2 * L:
        raise ValueError(f"qaoa parameter length {x.size} != {2 * L}")
    return x[0::2].copy(), x[1::2].copy()


def evaluate_qaoa(inst: LogicalInstance, x: np.ndarray, L: int, keep_state: bool = False) -> dict:
    gammas, betas = unpack_qaoa(x, L)
    psi = plus_state(inst.n)
    for ell in range(L):
        psi = psi * np.exp(-1j * float(gammas[ell]) * inst.hc)
        for q in range(inst.n):
            psi = apply_rx_site(psi, q, float(betas[ell]))
    out = _observe_logical(inst, psi)
    if not keep_state:
        out.pop("psi", None)
        out.pop("amp", None)
    return out


def qaoa_x0(L: int, kind: str, rng: np.random.Generator | None) -> np.ndarray:
    x = np.zeros(2 * L, dtype=np.float64)
    if kind == "E":
        x[0::2] = 0.25
        x[1::2] = 0.25
        return x
    if rng is None:
        raise RuntimeError("random init needs an rng")
    for ell in range(L):
        x[2 * ell] = float(rng.uniform(-0.5, 0.5))
        x[2 * ell + 1] = float(rng.uniform(-0.5, 0.5))
    return x


def qaoa_sigmas(L: int) -> np.ndarray:
    return np.full(2 * L, 0.10, dtype=np.float64)


def _observe_logical(inst: LogicalInstance, psi: np.ndarray) -> dict:
    norm = float(np.vdot(psi, psi).real)
    E = logical_energy(inst, psi)
    Ezz = logical_zz(inst, psi)
    F = float(np.abs(np.vdot(inst.gs, psi)) ** 2)
    dim = 1 << inst.n
    flip = np.arange(dim) ^ (dim - 1)
    parity = float(np.vdot(psi, psi[flip]).real)
    return {
        "E": E,
        "E_diag": Ezz,
        "E_X": float(E - Ezz),
        "F": F,
        "p_leak": 0.0,
        "p_code": norm,
        "norm": norm,
        "parity": parity,
        "alpha": np.zeros(0),
        "psi": psi,
        "amp": psi,
    }


def dense_hext(model: FockModel) -> np.ndarray:
    """Full Hext matrix. Used by T8 at N=6, NF=10. Not an optimizer."""
    nf = model.nf
    C = model.C
    dim = nf ** C
    H = np.diag(model.d_energy.reshape(-1))
    shape = (nf,) * C
    b = float(model.inst.b)
    encoding = model.packing.encoding
    for c, cav in enumerate(model.packing.groups):
        for r, _site in enumerate(cav):
            for flat in range(dim):
                fs = list(np.unravel_index(flat, shape))
                f = int(fs[c])
                if f >= CODE_LEVELS:
                    continue
                bits = f if encoding == "binary" else (f ^ (f >> 1))
                if ((bits >> r) & 1) != 0:
                    continue
                fs[c] = flip_level(f, r, encoding)
                j = int(np.ravel_multi_index(tuple(fs), shape))
                H[flat, j] -= b
                H[j, flat] -= b
    return H


def H0_energy(model: FockModel, psi: np.ndarray) -> float:
    """Leaked shots score 0. Coded amplitude keeps the logical energy, un-normalized."""
    amp = np.zeros(1 << model.inst.n, dtype=np.complex128)
    logical = model.logical_map
    mask = logical >= 0
    np.add.at(amp, logical[mask], psi[mask])
    return logical_energy(model.inst, amp)


def metrics_from_eval(inst: LogicalInstance, out: dict) -> dict:
    F = float(out["F"])
    Fp = float(inst.F_plus)
    denom = 1.0 - Fp
    eta = float((F - Fp) / denom) if denom > 1e-15 else float("nan")
    E = float(out["E"])
    E0 = float(inst.E0)
    resid = float((E - E0) / abs(E0)) if abs(E0) > 1e-15 else float("nan")
    E_plus = -float(inst.b) * inst.n
    span = E_plus - E0
    eta_E = float((E_plus - E) / span) if abs(span) > 1e-15 else float("nan")
    pleak = float(out["p_leak"])
    if pleak < 1.0 - 1e-15:
        f_ps = float(F / (1.0 - pleak))
    else:
        f_ps = float("nan")
    return {
        "E": E,
        "F": F,
        "F_plus": Fp,
        "eta": eta,
        "eta_E": eta_E,
        "E_resid": resid,
        "E0": E0,
        "E1": float(inst.E1),
        "gap": float(inst.gap),
        "delta_odd": float(inst.delta_odd),
        "p_leak": float(out["p_leak"]),
        "parity": float(out["parity"]),
        "norm": float(out["norm"]),
        "E_diag": float(out["E_diag"]),
        "E_X": float(out["E_X"]),
        "F_postselected_not_physical": f_ps,
    }
