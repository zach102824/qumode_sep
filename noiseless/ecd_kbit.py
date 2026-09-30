"""Local-ECD ansatz with k logical bits per cavity (n = 2 + 2k) and a Fock truncation nf >= 2^k.

Generalizes ``encoding.py`` / ``circuit_local_ecd.py`` / ``spsa_gibbs.NoiselessSimulator`` (n = 8:
k = 3, nf = 8) to the n = 16 chip layout (k = 7): 2 data transmons (d, e) + 2 cavities (A, B);
the middle transmon is a coupler and is not simulated -- the inter-cavity gate U (jp) is applied
as an ideal gate on A (x) B, exactly as in the n = 8 code.

Bit map (MSB-first, identical to encoding.bits_from_denm for k = 3):
    bits = (q_d, q_e | n_A[k-1:0] | n_B[k-1:0]),   logical index = int(bits, 2)
i.e. cavity bits are the binary digits of the Fock number n in 0 .. 2^k - 1 (``encoding="binary"``;
``"gray"`` also supported). Physical flat index = ((d*2 + e)*nf + n_A)*nf + n_B.

Truncation / leakage. The n = 8 code truncates each cavity at exactly nf = 8 = 2^k levels, so it has
NO levels outside the encoding (displacements are the unitary expm of the truncated generator and
are distorted near the cutoff). For n = 16 we simulate nf > 2^k levels (default 160 > 128) so the
ECD displacements are accurate over the encoded range. Fock levels n >= 2^k are *leaked* (invalid)
states: they get the diagonal energy ``E_leak = E_max_valid + 1`` (strictly worse than every valid
bitstring), so the Gibbs cost treats leaked mass as (nearly) lost probability and the η controller's
quantiles see it as the worst level. p(GS) is the Born probability of the GS basis state in the full
truncated space (leaked mass counts against it); success = global argmax (over ALL nf^2 * 4 states)
is the GS, so an argmax in a leaked state is a failure. ``leakage`` = total population with
n_A >= 2^k or n_B >= 2^k.

Circuit methods
  "legacy": exactly the operations of circuit_local_ecd.apply_layer_factored (dense ECD via
            scipy expm of the truncated generator, dense U_AB einsum); bit-for-bit identical to the
            n = 8 code for k = 3, nf = 8. Only usable for small nf.
  "fast":   same unitary (ECD = σ⁻ ⊗ D(β/2) + σ⁺ ⊗ D(-β/2), D(-β/2) = D(β/2)^†), with the truncated
            displacement from a cached eigendecomposition: D(r e^{iφ}) = R_φ exp(r(a†-a)) R_φ^†,
            R_φ = diag(e^{inφ}), exp(rK) = V diag(e^{-irλ}) V^† with H = i(a†-a) = V Λ V^†; applied
            blockwise (two nf×nf matmuls per ECD), jp as a diagonal phase. Agrees with "legacy" to
            ~1e-13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from .circuit_local_ecd import n_parameters, random_parameters, rotation_np, unpack_params  # noqa: F401
from .spsa_gibbs import SampledTailEta, beta_abs_stats, betas_from_x, beta_regularizer, gibbs_objective

_SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def gray_encode(n):
    return n ^ (n >> 1)


def gray_decode_arr(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.int64).copy()
    n = g.copy()
    shift = g >> 1
    while np.any(shift):
        n ^= shift
        shift >>= 1
    return n


def logical_from_denm(d, e, n_a, n_b, k: int, encoding: str = "binary"):
    """Logical MSB-first index for physical (d, e, n_A, n_B) (valid n < 2^k). Vectorized."""
    if encoding == "gray":
        n_a, n_b = gray_encode(n_a), gray_encode(n_b)
    return (np.asarray(d) << (2 * k + 1)) | (np.asarray(e) << (2 * k)) | (np.asarray(n_a) << k) | np.asarray(n_b)


def denm_from_logical(idx, k: int, encoding: str = "binary"):
    idx = np.asarray(idx, dtype=np.int64)
    m = (1 << k) - 1
    d = (idx >> (2 * k + 1)) & 1
    e = (idx >> (2 * k)) & 1
    n_a = (idx >> k) & m
    n_b = idx & m
    if encoding == "gray":
        n_a, n_b = gray_decode_arr(n_a), gray_decode_arr(n_b)
    return d, e, n_a, n_b


def bitstring_from_logical(idx: int, n: int) -> str:
    return format(int(idx), f"0{n}b")


def logical_energies_from_npz(path, n_expected: int | None = None) -> tuple[np.ndarray, dict]:
    """Energies of all 2^n logical bitstrings (MSB-first) from the npz Z-term expansion."""
    d = np.load(path)
    n = int(np.asarray(d["num_spins"]).reshape(-1)[0])
    if n_expected is not None and n != n_expected:
        raise ValueError(f"{path} has num_spins={n}, need {n_expected}")
    idx = np.arange(1 << n, dtype=np.int64)
    z = 1.0 - 2.0 * ((idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)  # (2^n, n)
    sites = np.asarray(d["sites"])
    orders = np.asarray(d["orders"], dtype=np.int64).reshape(-1)
    coeffs = np.asarray(d["coefficients"], dtype=float).reshape(-1)
    ident = float(np.asarray(d["identity"]).reshape(-1)[0]) if "identity" in d.files else 0.0
    E = np.full(1 << n, ident, dtype=float)
    for row, c in enumerate(coeffs):
        prod = np.ones(1 << n)
        for s in sites[row, : int(orders[row])]:
            prod = prod * z[:, int(s)]
        E += float(c) * prod
    meta = {"n": n, "file": str(path), "identity": ident}
    if "ground_bitstring" in d.files:
        meta["ground_bitstring"] = str(d["ground_bitstring"])
    return E, meta


@dataclass
class KbitLayout:
    k: int  # bits per cavity
    nf: int  # simulated Fock levels per cavity (>= 2^k)
    encoding: str = "binary"

    def __post_init__(self):
        if self.nf < (1 << self.k):
            raise ValueError("nf must be >= 2^k")
        if self.encoding not in ("binary", "gray"):
            raise ValueError(self.encoding)

    @property
    def n_qubits(self) -> int:
        return 2 + 2 * self.k

    @property
    def dims(self) -> tuple[int, int, int, int]:
        return (2, 2, self.nf, self.nf)

    def flat_of_logical(self) -> np.ndarray:
        """Physical flat index for every logical index 0..2^n-1."""
        d, e, na, nb = denm_from_logical(np.arange(1 << self.n_qubits), self.k, self.encoding)
        return ((d * 2 + e) * self.nf + na) * self.nf + nb

    def valid_mask(self) -> np.ndarray:
        n = np.arange(self.nf)
        v = n < (1 << self.k)
        return np.broadcast_to(v[None, None, :, None] & v[None, None, None, :], self.dims).reshape(-1)

    def energy_flat(self, E_logical: np.ndarray, leak_energy: float | None = None) -> np.ndarray:
        E_logical = np.asarray(E_logical, dtype=float)
        if leak_energy is None:
            leak_energy = float(E_logical.max()) + 1.0
        out = np.full(int(np.prod(self.dims)), float(leak_energy))
        out[self.flat_of_logical()] = E_logical
        return out

    def logical_of_flat(self, flat: int) -> int | None:
        nb = flat % self.nf
        r = flat // self.nf
        na = r % self.nf
        r //= self.nf
        e, d = r % 2, r // 2
        if na >= (1 << self.k) or nb >= (1 << self.k):
            return None
        return int(logical_from_denm(d, e, na, nb, self.k, self.encoding))


# ---------------------------------------------------------------------------
# circuit
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _destroy(nf: int) -> np.ndarray:
    a = np.zeros((nf, nf), dtype=complex)
    for n in range(1, nf):
        a[n - 1, n] = np.sqrt(n)
    return a


@lru_cache(maxsize=8)
def _disp_eig(nf: int):
    a = _destroy(nf).real
    K = a.T - a  # real antisymmetric
    H = 1j * K  # Hermitian
    lam, V = np.linalg.eigh(H)
    return lam, V


def displace_expm(alpha: complex, nf: int) -> np.ndarray:
    a = _destroy(nf)
    return expm(complex(alpha) * a.conj().T - np.conj(complex(alpha)) * a)


def displace_fast(alpha: complex, nf: int) -> np.ndarray:
    lam, V = _disp_eig(nf)
    r, phi = abs(alpha), np.angle(alpha)
    W = np.exp(1j * phi * np.arange(nf))[:, None] * V
    return (W * np.exp(-1j * r * lam)[None, :]) @ W.conj().T


def jp_diag(nf: int) -> np.ndarray:
    n = np.arange(nf)
    par = (n[:, None] + n[None, :]) % 2
    return np.where(par == 1, -1j, 1.0 + 0j)


def u_diag(u_name: str, nf: int) -> np.ndarray:
    if u_name == "jp":
        return jp_diag(nf)
    if u_name == "identity":
        return np.ones((nf, nf), dtype=complex)
    raise ValueError(f"u {u_name!r} not supported in ecd_kbit (diagonal U only: jp, identity)")


def _layer_legacy(psi, layer, u_ab_dense, nf):
    """Verbatim generalization of circuit_local_ecd.apply_layer_factored (NFOCK -> nf)."""
    psi = psi.reshape(2, 2, nf, nf)
    psi = np.einsum("ij,jklm->iklm", rotation_np(layer["theta_d"], layer["phi_d"]), psi, optimize=True)
    psi = np.einsum("ij,kjlm->kilm", rotation_np(layer["theta_e"], layer["phi_e"]), psi, optimize=True)
    b = layer["beta_d"]
    ecd_d = np.kron(_SM, displace_expm(b / 2.0, nf)) + np.kron(_SP, displace_expm(-b / 2.0, nf))
    tmp = np.transpose(psi, (1, 3, 0, 2)).reshape(2, nf, 2 * nf)
    tmp = np.einsum("ij,abj->abi", ecd_d, tmp, optimize=True)
    psi = np.transpose(tmp.reshape(2, nf, 2, nf), (2, 0, 3, 1))
    b = layer["beta_e"]
    ecd_e = np.kron(_SM, displace_expm(b / 2.0, nf)) + np.kron(_SP, displace_expm(-b / 2.0, nf))
    tmp = np.transpose(psi, (0, 2, 1, 3)).reshape(2, nf, 2 * nf)
    tmp = np.einsum("ij,abj->abi", ecd_e, tmp, optimize=True)
    psi = np.transpose(tmp.reshape(2, nf, 2, nf), (0, 2, 1, 3))
    flat = psi.reshape(2, 2, nf * nf)
    flat = np.einsum("ij,dej->dei", u_ab_dense, flat, optimize=True)
    return flat.reshape(-1)


def _layer_fast(psi, layer, udiag, nf):
    psi = psi.reshape(2, 2, nf, nf)
    rd = rotation_np(layer["theta_d"], layer["phi_d"])
    re = rotation_np(layer["theta_e"], layer["phi_e"])
    # R_d on axis 0, R_e on axis 1
    p0, p1 = psi[0], psi[1]
    psi = np.stack([rd[0, 0] * p0 + rd[0, 1] * p1, rd[1, 0] * p0 + rd[1, 1] * p1])
    q0, q1 = psi[:, 0], psi[:, 1]
    psi = np.stack([re[0, 0] * q0 + re[0, 1] * q1, re[1, 0] * q0 + re[1, 1] * q1], axis=1)
    # ECD_dA: |1_d> <- D(β/2)|0_d>,  |0_d> <- D(-β/2)|1_d>   (acts on A = axis 2)
    D = displace_fast(layer["beta_d"] / 2.0, nf)
    new1 = np.matmul(D, psi[0])  # (2e, nfA, nfB)
    new0 = np.matmul(D.conj().T, psi[1])
    psi = np.stack([new0, new1])
    # ECD_eB on B = axis 3: psi[:, e] (2d, nfA, nfB) @ D^T
    D = displace_fast(layer["beta_e"] / 2.0, nf)
    new1 = np.matmul(psi[:, 0], D.T)
    new0 = np.matmul(psi[:, 1], D.conj())
    psi = np.stack([new0, new1], axis=1)
    psi = psi * udiag[None, None, :, :]
    return psi.reshape(-1)


def apply_circuit(x, n_layers: int, nf: int, *, method: str = "fast", u_name: str = "jp") -> np.ndarray:
    ket = np.zeros(4 * nf * nf, dtype=complex)
    ket[0] = 1.0
    if method == "legacy":
        u = np.diag(u_diag(u_name, nf).reshape(-1))
        for layer in unpack_params(x, n_layers):
            ket = _layer_legacy(ket, layer, u, nf)
    elif method == "fast":
        ud = u_diag(u_name, nf)
        for layer in unpack_params(x, n_layers):
            ket = _layer_fast(ket, layer, ud, nf)
    else:
        raise ValueError(method)
    return ket


def born(ket: np.ndarray) -> np.ndarray:
    p = np.abs(ket) ** 2
    s = float(p.sum())
    return p if s <= 0 else p / s


# ---------------------------------------------------------------------------
# simulator (duck-type compatible with spsa_gibbs.optimize_trial / grow_trial)
# ---------------------------------------------------------------------------
@dataclass
class KbitEcdSimulator:
    layout: KbitLayout
    energies_logical: np.ndarray
    n_layers: int
    ground_bitstring: str
    u_name: str = "jp"
    method: str = "fast"
    leak_energy: float | None = None
    lambda1: float = 0.0
    lam: float = 0.0
    beta_max: float | None = None
    lambda3: float = 0.0
    eta_scale: float = 1.0
    eta_ctrl: SampledTailEta = field(default_factory=SampledTailEta)

    def __post_init__(self):
        lay = self.layout
        self.energies_flat = lay.energy_flat(self.energies_logical, self.leak_energy)
        self.valid = lay.valid_mask()
        self.encoding = lay.encoding
        gl = int(self.ground_bitstring, 2)
        self.ground_flat_index = int(lay.flat_of_logical()[gl])
        self._current_eta = 1.0

    def set_lam(self, value: float) -> None:
        self.lam = float(value)
        self.lambda3 = float(value)

    def probs_from_x(self, x) -> np.ndarray:
        return born(apply_circuit(x, self.n_layers, self.layout.nf, method=self.method, u_name=self.u_name))

    def cost(self, x, eta: float | None = None) -> float:
        probs = self.probs_from_x(x)
        use_eta = self._current_eta if eta is None else float(eta)
        g = gibbs_objective(probs, self.energies_flat, use_eta)
        if float(self.lambda1) == 0.0 and float(self.lam) == 0.0:
            return g
        return float(g + beta_regularizer(betas_from_x(x, self.n_layers), lambda1=self.lambda1,
                                          lam=self.lam, beta_max=self.beta_max))

    def refresh_eta(self, x, step: int) -> float:
        probs = self.probs_from_x(x)
        eta = self.eta_ctrl.update(self.energies_flat, probs, step)
        self._current_eta = eta * float(self.eta_scale)
        return self._current_eta

    def evaluate(self, x) -> dict:
        probs = self.probs_from_x(x)
        idx = int(np.argmax(probs))
        lg = self.layout.logical_of_flat(idx)
        n = self.layout.n_qubits
        ml = "LEAKED" if lg is None else bitstring_from_logical(lg, n)
        mean_abs, max_abs, frac = beta_abs_stats(betas_from_x(x, self.n_layers), self.beta_max)
        leak = float(probs[~self.valid].sum())
        p4 = probs.reshape(self.layout.dims)
        pa = p4.sum(axis=(0, 1, 3))
        pb = p4.sum(axis=(0, 1, 2))
        nn = np.arange(self.layout.nf)
        return {
            "most_likely_bitstring": ml,
            "p_gs": float(probs[self.ground_flat_index]),
            "success": ml == self.ground_bitstring,
            "energy_mean": float(np.dot(probs, self.energies_flat)),
            "energy_mean_valid": float(np.dot(probs[self.valid], self.energies_flat[self.valid]) / max(1e-300, 1 - leak)),
            "probs": probs,
            "mean_abs_beta": mean_abs,
            "max_abs_beta": max_abs,
            "frac_over_beta_max": frac,
            "leakage": leak,
            "mean_n_a": float(pa @ nn),
            "mean_n_b": float(pb @ nn),
            "p_top_levels": float(p4[:, :, -8:, :].sum() + p4[:, :, :, -8:].sum()),  # edge-of-truncation mass
        }
