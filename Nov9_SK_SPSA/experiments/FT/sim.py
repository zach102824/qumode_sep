"""Provisional Nov9th / Nov3rd / QAOA simulators.

Nov9th primary is the true displacement, built by exponentiating a†-a in
NF+32 Fock levels and restricting. Levels ≥ 8 get SNAP phase 0.
Uz keeps pair phases whose cavities are both inside the code.
U0 drops the whole component's phase if any cavity is outside the code.
Energy is the per-term H^ext. Fidelity is unconditioned and unrenormalized.

Nov9th-B uses D8 = exp[α P(a†-a)P] on levels 0..7 only. That object is a
diagnostic, not a displacement.

Ideal W_L is the logical Hadamard on the code. Its SNAP-displacement
compilation and the leakage of that compilation are not modelled.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from Nov9_SK_SPSA.experiments.FT.instance import Instance, build_instance

# Walsh shell vectors used by Nov 8, rows are popcount 0..3.
_WALSH_V = np.array(
    [[3.0, 3.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-3.0, 3.0, -1.0]],
    dtype=np.float64,
)
_WALSH_D = _WALSH_V[1:] - _WALSH_V[0]


def boson_generator(nf: int) -> np.ndarray:
    a = np.zeros((nf, nf), dtype=np.float64)
    idx = np.arange(1, nf)
    a[idx - 1, idx] = np.sqrt(idx.astype(np.float64))
    return a.T - a


class GenExp:
    def __init__(self, nf: int):
        G = boson_generator(nf)
        w, V = np.linalg.eig(G)
        self.nf = nf
        self.G = G
        self.w = w.astype(np.complex128)
        self.V = V.astype(np.complex128)
        self.Vinv = np.linalg.inv(self.V)

    def exp(self, alpha: float) -> np.ndarray:
        M = (self.V * np.exp(float(alpha) * self.w)) @ self.Vinv
        if np.max(np.abs(np.imag(M))) > 1e-8:
            raise RuntimeError("displacement left the reals")
        return np.real(M)

    def expm_reference(self, alpha: float) -> np.ndarray:
        return np.real(expm(float(alpha) * self.G))


@lru_cache(maxsize=8)
def generator(nf: int) -> GenExp:
    return GenExp(nf)


def displacement_matrix(alpha: float, nf: int, kind: str) -> np.ndarray:
    """kind='restricted' slices a true oscillator displacement from NF+32.

    kind='strict8' is exp[α P(a†-a)P] on 8 levels, not a displacement.
    """
    if kind == "strict8":
        if nf != 8:
            raise ValueError("strict8 displacement diagnostic is 8 levels")
        return generator(8).exp(alpha)
    if kind != "restricted":
        raise ValueError(kind)
    big = generator(nf + 32).exp(alpha)
    return np.array(big[:nf, :nf], dtype=np.float64, copy=True)


def fwht(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.complex128).copy()
    n = int(x.shape[0])
    h = 1
    while h < n:
        x = x.reshape(n // (2 * h), 2, h)
        a = x[:, 0, :].copy()
        b = x[:, 1, :].copy()
        x[:, 0, :] = a + b
        x[:, 1, :] = a - b
        x = np.reshape(x, (n,))
        h *= 2
    x *= n ** -0.5
    return x


def logical_shell_phase(N: int, groups: list[list[int]], s1: float, s2: float, s3: float) -> np.ndarray:
    idx = np.arange(1 << N)
    ph = np.zeros(1 << N, dtype=np.float64)
    for cav in groups:
        local = np.zeros(idx.shape, dtype=np.int64)
        for r, q in enumerate(cav):
            local |= ((idx >> q) & 1) << r
        p = (local & 1) + ((local >> 1) & 1) + ((local >> 2) & 1)
        ph += s1 * p + s2 * (p * (p - 1) / 2.0) + s3 * (p == 3)
    return ph


def snap_level_phase(nf: int, s1: float, s2: float, s3: float) -> np.ndarray:
    ph = np.zeros(nf, dtype=np.float64)
    for f in range(min(8, nf)):
        p = (f & 1) + ((f >> 1) & 1) + ((f >> 2) & 1)
        ph[f] = s1 * p + s2 * (p * (p - 1) / 2.0) + (s3 if p == 3 else 0.0)
    return ph


def t_from_s(s1: float, s2: float, s3: float) -> np.ndarray:
    r = np.array([s1, 2.0 * s1 + s2, 3.0 * s1 + 3.0 * s2 + s3], dtype=np.float64)
    return np.linalg.solve(_WALSH_D, r)


def walsh_phase(t: np.ndarray) -> np.ndarray:
    return _WALSH_V @ np.asarray(t, dtype=np.float64)


def fock_configs(nf: int, C: int) -> np.ndarray:
    grids = np.meshgrid(*([np.arange(nf, dtype=np.int16)] * C), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def logical_table(groups: list[list[int]]) -> np.ndarray:
    C = len(groups)
    table = np.zeros((8,) * C, dtype=np.int64)
    for flat in range(8**C):
        idx = np.unravel_index(flat, (8,) * C)
        bits = 0
        for c, f in enumerate(idx):
            for r, site in enumerate(groups[c]):
                bits |= ((int(f) >> r) & 1) << site
        table[idx] = bits
    return table


class FockTables:
    def __init__(self, inst: Instance, nf: int):
        self.inst = inst
        self.nf = int(nf)
        self.C = len(inst.groups)
        self.N = inst.N
        self.dim = self.nf ** self.C
        fs = fock_configs(self.nf, self.C)
        self.fs = fs
        self.leak_any = np.any(fs >= 8, axis=1)
        self.table = logical_table(inst.groups)
        self.code_index = self.table.ravel().copy()
        # Feature diagonals under Uz: a pair contributes only when both cavities are in code.
        cav_of = np.zeros(self.N, dtype=np.int16)
        for c, sites in enumerate(inst.groups):
            for s in sites:
                cav_of[s] = c
        self.cavity_of = cav_of
        z_site = np.zeros((self.dim, self.N), dtype=np.float64)
        in_code = fs < 8
        for c, sites in enumerate(inst.groups):
            f = fs[:, c]
            coded = in_code[:, c]
            for r, site in enumerate(sites):
                bit = (f >> r) & 1
                z = np.zeros(self.dim, dtype=np.float64)
                z[coded] = 1.0 - 2.0 * bit[coded]
                z_site[:, site] = z
        mats = {
            "J": inst.J,
            "K": inst.K_hat,
            "M": inst.M_hat,
            "R1": inst.R1_hat,
        }
        self.uz = {}
        for name, mat in mats.items():
            diag = np.zeros(self.dim, dtype=np.float64)
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    cij = float(mat[i, j])
                    if cij == 0.0 and name != "J":
                        continue
                    both = in_code[:, cav_of[i]] & in_code[:, cav_of[j]]
                    diag[both] += cij * z_site[both, i] * z_site[both, j]
            self.uz[name] = diag
        # Z-setting of H^ext and the leaked-qubit penalty, for the true J.
        zz = np.zeros(self.dim, dtype=np.float64)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                Jij = float(inst.J[i, j])
                both = in_code[:, cav_of[i]] & in_code[:, cav_of[j]]
                zz[both] += Jij * z_site[both, i] * z_site[both, j]
                zz[~both] += abs(Jij)
        self.diag_zz = zz
        xpen = np.zeros(self.dim, dtype=np.float64)
        for c in range(self.C):
            xpen[fs[:, c] >= 8] += 3.0 * float(inst.b)
        self.diag_xpen = xpen
        self.plus_amp = 2.0 ** (-0.5 * self.N)

    def plus(self) -> np.ndarray:
        psi = np.zeros(self.dim, dtype=np.complex128)
        tens = psi.reshape((self.nf,) * self.C)
        tens[(slice(0, 8),) * self.C] = self.plus_amp
        return psi

    def embed_logical(self, amp: np.ndarray) -> np.ndarray:
        psi = np.zeros(self.dim, dtype=np.complex128)
        block = np.asarray(amp, dtype=np.complex128)[self.code_index]
        psi.reshape((self.nf,) * self.C)[(slice(0, 8),) * self.C] = block.reshape((8,) * self.C)
        return psi


def _apply_matrix(psi: np.ndarray, M: np.ndarray, nf: int, C: int) -> np.ndarray:
    t = psi.reshape((nf,) * C)
    for c in range(C):
        t = np.moveaxis(t, c, 0)
        t = np.tensordot(M, t, axes=(1, 0))
        t = np.moveaxis(t, 0, c)
    return np.ascontiguousarray(t).reshape(-1)


def _x_sum(psi: np.ndarray, tables: FockTables) -> float:
    tens = psi.reshape((tables.nf,) * tables.C)
    total = 0.0
    for c in range(tables.C):
        sl = [slice(None)] * tables.C
        sl[c] = slice(0, 8)
        block = np.moveaxis(tens[tuple(sl)], c, -1)
        for r in range(3):
            perm = np.arange(8) ^ (1 << r)
            total += float(np.vdot(block, block[..., perm]).real)
    return total


def _code_amps(psi: np.ndarray, tables: FockTables) -> np.ndarray:
    tens = psi.reshape((tables.nf,) * tables.C)
    block = tens[(slice(0, 8),) * tables.C].ravel()
    amp = np.zeros(1 << tables.N, dtype=np.complex128)
    amp[tables.code_index] = block
    return amp


def metrics_from_fock(psi: np.ndarray, tables: FockTables) -> dict:
    inst = tables.inst
    p_code = float(np.sum(np.abs(psi.reshape((tables.nf,) * tables.C)[(slice(0, 8),) * tables.C]) ** 2))
    # Population that left the simulated cutoff is also outside the code.
    norm = float(np.vdot(psi, psi).real)
    probs = np.abs(psi) ** 2
    E_zz = float(np.dot(probs, tables.diag_zz))
    E_xpen = float(np.dot(probs, tables.diag_xpen))
    Ex = _x_sum(psi, tables)
    E = E_zz + E_xpen - float(inst.b) * Ex
    amp = _code_amps(psi, tables)
    F = float(np.abs(np.vdot(inst.gs, amp)) ** 2)
    mask = (1 << inst.N) - 1
    parity_u = float(np.vdot(amp, amp[np.arange(amp.size) ^ mask]).real)
    p_leak = float(1.0 - p_code)
    cond = float(parity_u / p_code) if p_code > 1e-18 else float("nan")
    ok = bool(np.isfinite(E) and np.isfinite(F) and abs(norm - 1.0) <= 1e-9)
    return {
        "E": E,
        "F": F,
        "p_leak": p_leak,
        "E_zz": E_zz,
        "E_xpen": E_xpen,
        "parity_code_unnormalized": parity_u,
        "parity_code_conditional": cond,
        "norm": norm,
        "p_code": p_code,
        "ok": ok,
    }


def metrics_from_logical(psi: np.ndarray, inst: Instance) -> dict:
    norm = float(np.vdot(psi, psi).real)
    E_zz = float(np.vdot(psi, inst.zz_J * psi).real)
    Ex = logical_X_sum(psi, inst.N)
    E = E_zz - float(inst.b) * Ex
    F = float(np.abs(np.vdot(inst.gs, psi)) ** 2)
    mask = (1 << inst.N) - 1
    parity_u = float(np.vdot(psi, psi[np.arange(psi.size) ^ mask]).real)
    ok = bool(np.isfinite(E) and np.isfinite(norm) and abs(norm - 1.0) <= 1e-9)
    return {
        "E": E,
        "F": F,
        "p_leak": 0.0,
        "E_zz": E_zz,
        "E_xpen": 0.0,
        "parity_code_unnormalized": parity_u,
        "parity_code_conditional": parity_u / norm if norm > 0 else float("nan"),
        "norm": norm,
        "p_code": norm,
        "ok": ok,
    }


def logical_X_sum(psi: np.ndarray, N: int) -> float:
    total = 0.0
    for q in range(N):
        half = 1 << q
        rest = psi.size // (2 * half)
        a = psi.reshape(rest, 2, half)
        total += float(np.vdot(a[:, 0, :], a[:, 1, :]).real)
        total += float(np.vdot(a[:, 1, :], a[:, 0, :]).real)
    # vdot(a0,a1)+vdot(a1,a0) = 2 Re ⟨a0|a1⟩, and ⟨X⟩ = 2 Re ⟨a0|a1⟩. Correct.
    return total


def apply_rx_all(psi: np.ndarray, beta: float, N: int) -> np.ndarray:
    c = np.cos(beta)
    s = np.sin(beta)
    x = np.array(psi, dtype=np.complex128, copy=True)
    for q in range(N):
        half = 1 << q
        rest = x.size // (2 * half)
        a = x.reshape(rest, 2, half)
        a0 = a[:, 0, :].copy()
        a1 = a[:, 1, :].copy()
        a[:, 0, :] = c * a0 - 1j * s * a1
        a[:, 1, :] = -1j * s * a0 + c * a1
    return x


def _ucz_phase(tables: FockTables, gamma: float, theta: dict, extension: str) -> np.ndarray:
    g = tables.uz["J"].copy()
    for name, value in theta.items():
        g = g + float(value) * tables.uz[name]
    phase = float(gamma) * g
    if extension == "u0":
        phase = phase.copy()
        phase[tables.leak_any] = 0.0
    elif extension not in ("uz", "strict"):
        raise ValueError(extension)
    return phase


def run_nov9(tables: FockTables, unpacked: dict, extension: str, kind: str, return_state: bool = False) -> dict:
    L = len(unpacked["alphas"])
    phase = _ucz_phase(tables, unpacked["gamma"], unpacked["theta"], "uz" if extension == "strict" else extension)
    e_ucz = np.exp(-1j * phase)
    snaps = []
    for ell in range(L):
        s1, s2, s3 = unpacked["shells"][ell]
        snaps.append(np.exp(-1j * snap_level_phase(tables.nf, s1, s2, s3)))
    Ds = [displacement_matrix(float(a), tables.nf, kind) for a in unpacked["alphas"]]
    psi = tables.plus()
    nf, C = tables.nf, tables.C
    tens_shape = (nf,) * C
    for ell in range(L):
        psi *= e_ucz
        psi = _apply_matrix(psi, Ds[ell], nf, C)
        t = psi.reshape(tens_shape)
        for c in range(C):
            shape = [1] * C
            shape[c] = nf
            t *= snaps[ell].reshape(shape)
        psi = t.reshape(-1)
    out = metrics_from_fock(psi, tables)
    if return_state:
        out["_psi"] = psi
    return out


def run_nov3(inst: Instance, unpacked: dict) -> dict:
    L = unpacked["shells"].shape[0]
    g = inst.zz_cost(unpacked["theta"])
    eph = np.exp(-1j * float(unpacked["gamma"]) * g)
    shells = [
        np.exp(-1j * logical_shell_phase(inst.N, inst.groups, *unpacked["shells"][ell]))
        for ell in range(L)
    ]
    psi = np.full(1 << inst.N, 2.0 ** (-0.5 * inst.N), dtype=np.complex128)
    for ell in range(L):
        psi *= eph
        psi = fwht(psi)
        psi *= shells[ell]
        psi = fwht(psi)
    return metrics_from_logical(psi, inst)


def run_nov3_vacuum(inst: Instance, unpacked: dict) -> np.ndarray:
    """Original |0> [W exp(-iγE) W → S]^L → W on NF=8, returned in logical order."""
    nf = 8
    C = len(inst.groups)
    tables = FockTables(inst, nf)
    W = _hadamard3()
    L = unpacked["shells"].shape[0]
    g_logical = inst.zz_cost(unpacked["theta"])
    phase_grid = (float(unpacked["gamma"]) * g_logical)[tables.table]
    psi = np.zeros((nf,) * C, dtype=np.complex128)
    psi[(0,) * C] = 1.0
    psi = psi.reshape(-1)
    for ell in range(L):
        psi = _apply_matrix(psi, W, nf, C)
        psi *= np.exp(-1j * phase_grid.reshape(-1))
        psi = _apply_matrix(psi, W, nf, C)
        s1, s2, s3 = unpacked["shells"][ell]
        snap = np.exp(-1j * snap_level_phase(nf, s1, s2, s3))
        t = psi.reshape((nf,) * C)
        for c in range(C):
            shape = [1] * C
            shape[c] = nf
            t *= snap.reshape(shape)
        psi = t.reshape(-1)
    psi = _apply_matrix(psi, W, nf, C)
    return _code_amps(psi, tables)


def _hadamard3() -> np.ndarray:
    H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)
    W = H
    for _ in range(2):
        W = np.kron(W, H)
    return W


def run_qaoa(inst: Instance, unpacked: dict) -> dict:
    gammas = unpacked["gammas"]
    betas = unpacked["betas"]
    L = gammas.size
    ezz = inst.zz_J
    psi = np.full(1 << inst.N, 2.0 ** (-0.5 * inst.N), dtype=np.complex128)
    for ell in range(L):
        psi *= np.exp(-1j * float(gammas[ell]) * ezz)
        psi = apply_rx_all(psi, float(betas[ell]), inst.N)
    out = metrics_from_logical(psi, inst)
    out["_psi"] = psi
    return out


def qaoa_dense_state(inst: Instance, unpacked: dict) -> np.ndarray:
    """Independent dense-matrix QAOA, N small."""
    N = inst.N
    dim = 1 << N
    zz = inst.zz_J
    X = np.zeros((dim, dim), dtype=np.float64)
    idx = np.arange(dim)
    for q in range(N):
        X[idx, idx ^ (1 << q)] = 1.0
    psi = np.full(dim, 2.0 ** (-0.5 * N), dtype=np.complex128)
    for g, b in zip(unpacked["gammas"], unpacked["betas"]):
        psi *= np.exp(-1j * float(g) * zz)
        psi = expm(-1j * float(b) * X) @ psi
    return psi


_TABLES: dict[tuple, FockTables] = {}
_INST: dict[tuple, Instance] = {}


def get_instance(N: int, seed: int, b: float, role: str) -> Instance:
    key = (N, int(seed), float(b), role)
    if key not in _INST:
        _INST[key] = build_instance(N, seed, b, role=role)
    return _INST[key]


def get_tables(N: int, seed: int, b: float, nf: int, role: str) -> FockTables:
    key = (N, int(seed), float(b), int(nf), role)
    if key not in _TABLES:
        _TABLES[key] = FockTables(get_instance(N, seed, b, role), nf)
    return _TABLES[key]


def clear_caches() -> None:
    _TABLES.clear()
    _INST.clear()
    generator.cache_clear()


def h0_energy(psi: np.ndarray, tables: FockTables) -> float:
    """Leaked outcomes score 0. Diagnostic only; never a selection objective."""
    tens = np.abs(psi.reshape((tables.nf,) * tables.C)) ** 2
    code = tens[(slice(0, 8),) * tables.C]
    # Map code probabilities through the logical H using the embedded amplitudes' ZZ and X.
    amp = _code_amps(psi, tables)
    inst = tables.inst
    E_zz = float(np.vdot(amp, inst.zz_J * amp).real)
    Ex = logical_X_sum(amp, inst.N)
    return float(E_zz - inst.b * Ex)


def apply_Hext(psi: np.ndarray, tables: FockTables) -> np.ndarray:
    shape = (tables.nf,) * tables.C
    out = ((tables.diag_zz + tables.diag_xpen) * psi).reshape(shape)
    tens = psi.reshape(shape)
    b = float(tables.inst.b)
    for c in range(tables.C):
        src = np.moveaxis(tens, c, 0)
        dst = np.moveaxis(out, c, 0)
        code = src[0:8]
        acc = np.zeros_like(code)
        for r in range(3):
            perm = np.arange(8) ^ (1 << r)
            acc += -b * code[perm]
        dst[0:8] += acc
    return np.ascontiguousarray(out).reshape(-1)
