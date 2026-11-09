"""Logical target, packing-dependent shells, and the four frozen arms.

The Hamiltonian is never permuted for the reported energy. A packing only
chooses which three logical qubits share a shell. A second, physical,
formulation permutes J, runs a contiguous shell, and maps the state back.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST_SHA = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"
PACKING_SHA = "4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717"

ARMS = {
    "R_M": {"packing": "sublattice", "thetas": ("M",), "features": ("M",)},
    "R_full": {"packing": "sublattice", "thetas": ("K", "M"), "features": ("K", "M")},
    "R_blockK": {"packing": "sublattice", "thetas": ("M", "B"), "features": ("M", "B")},
    "N_red": {"packing": "contiguous", "thetas": ("K", "N"), "features": ("K", "N")},
    "A_cont": {"packing": "contiguous", "thetas": ("K",), "features": ("K",)},
    "A_sub": {"packing": "sublattice", "thetas": ("K",), "features": ("K",)},
    "qaoa": {"packing": "none", "thetas": (), "features": ()},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw():
    man = ROOT / "FROZEN_TARGET_R2.json"
    pak = ROOT / "packings.json"
    if _sha(man) != MANIFEST_SHA or _sha(pak) != PACKING_SHA:
        raise RuntimeError("manifest or packing hash mismatch")
    return json.loads(man.read_text()), json.loads(pak.read_text())


def upper(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(v, dtype=np.float64)))))


def z_table(n: int) -> np.ndarray:
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    z = np.empty((n, dim), dtype=np.float64)
    for q in range(n):
        z[q] = 1.0 - 2.0 * ((x >> np.uint32(q)) & np.uint32(1))
    return z


def pair_diag(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = coeff.shape[0]
    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            cij = float(coeff[i, j])
            if cij != 0.0:
                out += cij * z[i] * z[j]
    return out


def pair_K(J: np.ndarray) -> np.ndarray:
    K = np.array(J @ J, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def block_coeff(source: np.ndarray, packing: list[list[int]]) -> np.ndarray:
    """Replace every inter-cavity block by its mean. Intra blocks stay zero."""
    n = source.shape[0]
    C = np.zeros_like(source)
    m = len(packing)
    for a in range(m):
        for b in range(a + 1, m):
            acc = 0.0
            pairs = [(i, j) for i in packing[a] for j in packing[b]]
            for i, j in pairs:
                acc += float(source[i, j])
            w = acc / float(len(pairs))
            for i, j in pairs:
                C[i, j] = C[j, i] = w
    return C


def selector(n: int) -> np.ndarray:
    C = np.zeros((n, n), dtype=np.float64)
    for a in range(n // 3):
        s = 3 * a
        for i, j, w in ((s, s + 1, 1.0), (s + 1, s + 2, 1.0), (s, s + 2, -2.0)):
            C[i, j] += w
            C[j, i] += w
    return C


def hat(raw: np.ndarray, rms_j: float) -> tuple[np.ndarray, float]:
    r = rms(upper(raw))
    if r <= 0:
        raise RuntimeError("zero feature")
    scale = rms_j / r
    return raw * scale, float(scale)


def shell_weights(n: int, packing: list[list[int]]):
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    w1 = np.zeros(dim)
    w2 = np.zeros(dim)
    w3 = np.zeros(dim)
    t1 = np.array([3.0, 1.0, -1.0, -3.0])
    t2 = np.array([3.0, -1.0, -1.0, 3.0])
    t3 = np.array([1.0, -1.0, 1.0, -1.0])
    for cav in packing:
        local = np.zeros(dim, dtype=np.int64)
        for r, q in enumerate(cav):
            local |= ((idx >> q) & 1) << r
        pop = (local & 1) + ((local >> 1) & 1) + ((local >> 2) & 1)
        w1 += t1[pop]
        w2 += t2[pop]
        w3 += t3[pop]
    return w1, w2, w3


def logical_perm(pi: list[int], n: int) -> np.ndarray:
    """perm[logical_index] = physical_index. Physical bit p is logical site pi[p]."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    y = np.zeros(dim, dtype=np.int64)
    for p, site in enumerate(pi):
        y |= ((x >> site) & 1) << p
    return y


class System:
    def __init__(self, n: int):
        raw, packs = load_raw()
        block = raw["systems"][str(n)]
        J = np.zeros((n, n), dtype=np.float64)
        for i, j, s in block["Jij"]:
            J[int(i), int(j)] = J[int(j), int(i)] = float(s)
        self.n = n
        self.B = float(block["B"])
        self.B_string = str(block["B"])
        self.J = J
        self.z = z_table(n)
        self.hc = pair_diag(J, self.z)
        self.K_raw = pair_K(J)
        self.rms_J = rms(upper(J))
        self.K_hat, self.rK = hat(self.K_raw, self.rms_J)
        self.packings = {
            "contiguous": [list(map(int, t)) for t in packs["contiguous"][str(n)]],
            "sublattice": [list(map(int, t)) for t in packs["sublattice"][str(n)]],
        }
        self.diags = {}
        self.scales = {"K": self.rK}
        self.r = {"K": 1.0 / self.rK}
        self.coeffs = {"K": self.K_hat}
        for name, trips in self.packings.items():
            M = block_coeff(J, trips)
            Mh, rM = hat(M, self.rms_J)
            B = block_coeff(self.K_raw, trips)
            Bh, rB = hat(B, self.rms_J)
            self.diags[(name, "M")] = pair_diag(Mh, self.z)
            self.diags[(name, "B")] = pair_diag(Bh, self.z)
            self.diags[(name, "M_raw")] = pair_diag(M, self.z)
            self.diags[(name, "B_raw")] = pair_diag(B, self.z)
            # r* is RMS(raw)/RMS(J). The stored source scale is the reciprocal,
            # and hat = raw / r = raw * RMS(J) / RMS(raw).
            self.scales[(name, "M")] = rM
            self.scales[(name, "B")] = rB
            self.r[(name, "M")] = 1.0 / rM
            self.r[(name, "B")] = 1.0 / rB
            self.coeffs[(name, "M")] = Mh
            self.coeffs[(name, "M_raw")] = M
            self.coeffs[(name, "B")] = Bh
            self.coeffs[(name, "B_raw")] = B
            self.diags[(name, "weights")] = shell_weights(n, trips)
        N = selector(n)
        Nh, rN = hat(N, self.rms_J)
        self.diags[("contiguous", "N")] = pair_diag(Nh, self.z)
        self.scales["N"] = rN
        self.r["N"] = 1.0 / rN
        self.coeffs["N"] = Nh
        self.coeffs["N_raw"] = N
        self.diags[("any", "K")] = pair_diag(self.K_hat, self.z)
        self.hc_sha = hashlib.sha256(np.ascontiguousarray(self.hc, dtype="<f8").tobytes()).hexdigest()
        self.checks = self._checks()

    def g_diag(self, arm: str, theta: dict[str, float]) -> np.ndarray:
        spec = ARMS[arm]
        g = np.array(self.hc, copy=True)
        pack = spec["packing"]
        for name in spec["features"]:
            key = ("any", "K") if name == "K" else (pack, name)
            g = g + float(theta.get(name, 0.0)) * self.diags[key]
        return g

    def partial(self, arm: str, name: str) -> np.ndarray:
        pack = ARMS[arm]["packing"]
        key = ("any", "K") if name == "K" else (pack, name)
        return self.diags[key]

    def weights(self, arm: str):
        pack = ARMS[arm]["packing"]
        if pack == "none":
            pack = "contiguous"
        return self.diags[(pack, "weights")]

    def _checks(self) -> dict:
        n = self.n
        pack = self.packings["sublattice"]
        M = self.coeffs[("sublattice", "M_raw")]
        # Hinter is J with intra pairs zeroed.
        intra = set()
        for cav in pack:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = sorted((cav[a], cav[b]))
                    intra.add((i, j))
        Hinter = np.array(self.J, copy=True)
        for i, j in intra:
            Hinter[i, j] = Hinter[j, i] = 0.0
        proj = block_coeff(Hinter, pack)
        err_proj = float(np.max(np.abs(upper(proj) - upper(M))))
        # rM = RMS(M)/RMS(J). hat = M/rM, so
        # Hc + theta Mhat = Hintra + (Hinter - M) + (1 + theta/rM) M.
        theta = 0.7
        rM = self.r[("sublattice", "M")]
        Hintra = pair_diag(self.J - Hinter, self.z)
        left = self.hc + theta * self.diags[("sublattice", "M")]
        right = (
            Hintra
            + pair_diag(Hinter - M, self.z)
            + (1.0 + theta / rM) * self.diags[("sublattice", "M_raw")]
        )
        err_id = float(np.max(np.abs(left - right)))
        # K is not a linear combination of J and M.
        A = np.column_stack([upper(self.J), upper(M), np.ones(len(upper(self.J)))])
        coef, *_ = np.linalg.lstsq(A, upper(self.K_raw), rcond=None)
        resid = float(np.linalg.norm(A @ coef - upper(self.K_raw)) / np.linalg.norm(upper(self.K_raw)))
        Braw = self.coeffs[("sublattice", "B_raw")]
        k_vs_b = float(np.linalg.norm(upper(self.K_raw) - upper(Braw)) / np.linalg.norm(upper(self.K_raw)))
        # Two-cavity M phase depends only on the two Hamming weights.
        cav = pack
        md = self.diags[("sublattice", "M_raw")]
        buckets = {}
        for bits in range(1 << n):
            pops = tuple(sum((bits >> q) & 1 for q in group) for group in cav)
            buckets.setdefault(pops, set()).add(round(float(md[bits]), 10))
        phase_unique = all(len(v) == 1 for v in buckets.values())
        # Neel selector on one cavity: 0 on {0,7}, -4 on {2,5}, +2 on {1,3,4,6}.
        nd = pair_diag(self.coeffs["N_raw"][:3, :3], z_table(3))
        expect = {0: 0.0, 7: 0.0, 2: -4.0, 5: -4.0, 1: 2.0, 3: 2.0, 4: 2.0, 6: 2.0}
        neel_err = max(abs(float(nd[k]) - expect[k]) for k in expect)
        # Source multiply-scales, verified against the raw J of this manifest.
        src12 = {
            "K": 0.3687169636309484,
            "M_contiguous": 1.8380385770596068,
            "M_sublattice": 1.244371288873176,
        }
        src6 = {
            "K": 0.5288373275351175,
            "M_contiguous": 2.185771867627059,
            "M_sublattice": 1.2375254093858243,
        }
        src = src12 if n == 12 else src6
        scale_err = max(
            abs(self.scales["K"] - src["K"]),
            abs(self.scales[("contiguous", "M")] - src["M_contiguous"]),
            abs(self.scales[("sublattice", "M")] - src["M_sublattice"]),
        )
        return {
            "M_equals_block_projection_of_Hinter": err_proj,
            "identity_max_abs": err_id,
            "rM_sublattice": rM,
            "scale_M_sublattice": self.scales[("sublattice", "M")],
            "scale_M_contiguous": self.scales[("contiguous", "M")],
            "scale_K": self.scales["K"],
            "scale_B_sublattice": self.scales[("sublattice", "B")],
            "scale_N": self.scales["N"],
            "source_scale_abs_err": scale_err,
            "K_residual_vs_span_J_M": resid,
            "rawK_vs_blockK_relative": k_vs_b,
            "two_cavity_M_depends_only_on_weights": phase_unique,
            "neel_eigenvalue_err": neel_err,
            "passed": (
                err_proj < 1e-12
                and err_id < 1e-10
                and resid > 1e-3
                and k_vs_b > 1e-3
                and phase_unique
                and neel_err < 1e-12
                and scale_err < 1e-12
            ),
        }


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


def param_count(arm: str, L: int) -> int:
    if arm == "qaoa":
        return 2 * L
    return 1 + len(ARMS[arm]["thetas"]) + 3 * L


def unpack(arm: str, x: np.ndarray, L: int):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != param_count(arm, L):
        raise ValueError(f"{arm} L={L} wants {param_count(arm, L)} got {x.size}")
    if arm == "qaoa":
        return x[:L].copy(), {}, np.column_stack([x[L:], np.zeros(L), np.zeros(L)])
    names = ARMS[arm]["thetas"]
    theta = {name: float(x[1 + i]) for i, name in enumerate(names)}
    cs = x[1 + len(names) :].reshape(L, 3).copy()
    return np.full(L, float(x[0])), theta, cs


def pack(arm: str, gamma: float, theta: dict, cs: np.ndarray, gammas=None, betas=None) -> np.ndarray:
    if arm == "qaoa":
        return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])
    names = ARMS[arm]["thetas"]
    head = [float(gamma)] + [float(theta.get(n, 0.0)) for n in names]
    return np.concatenate([head, np.asarray(cs, dtype=np.float64).reshape(-1)])


class Runner:
    def __init__(self, system: System):
        self.s = system
        self.plus = np.full(1 << system.n, 2 ** (-0.5 * system.n), dtype=np.complex128)
        self.gs = None
        self.E0 = None
        self.Emax = None
        self.gap_full = None
        self.gap_acc = None

    def set_spectrum(self, gs, E0, Emax, gap_full, gap_acc):
        self.gs = gs
        self.E0 = float(E0)
        self.Emax = float(Emax)
        self.gap_full = float(gap_full)
        self.gap_acc = float(gap_acc)

    def _mix(self, psi, arm, c):
        w1, w2, w3 = self.s.weights(arm if arm != "qaoa" else "A_cont")
        phi = c[0] * w1 + c[1] * w2 + c[2] * w3
        return fwht(fwht(psi) * np.exp(-1j * phi))

    def forward(self, arm, x, L, store=False):
        gammas, theta, cs = unpack(arm, x, L)
        if arm == "qaoa":
            g = self.s.hc
            parts = {}
        else:
            g = self.s.g_diag(arm, theta)
            parts = {name: self.s.partial(arm, name) for name in ARMS[arm]["thetas"]}
        psi = self.plus.copy()
        snaps = []
        for ell in range(L):
            phase = float(gammas[ell]) * g
            mid = psi * np.exp(-1j * phase)
            w1, w2, w3 = self.s.weights(arm if arm != "qaoa" else "A_cont")
            phi = cs[ell, 0] * w1 + cs[ell, 1] * w2 + cs[ell, 2] * w3
            x2 = fwht(mid) * np.exp(-1j * phi)
            psi = fwht(x2)
            if store:
                snaps.append((phase, mid, phi, x2, g, parts, w1, w2, w3))
        return psi, snaps, gammas

    def energy(self, psi) -> float:
        from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum
        Hpsi = self.s.hc * psi + self.s.B * apply_X_sum(psi, self.s.n)
        return float(np.vdot(psi, Hpsi).real), Hpsi

    def energy_only(self, arm, x, L):
        psi, _, _ = self.forward(arm, x, L, store=False)
        E, _ = self.energy(psi)
        return E, psi

    def energy_grad(self, arm, x, L):
        from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum
        psi, snaps, gammas = self.forward(arm, x, L, store=True)
        E, Hpsi = self.energy(psi)
        lam = Hpsi
        Lz = len(snaps)
        g_gamma = np.zeros(Lz)
        names = () if arm == "qaoa" else ARMS[arm]["thetas"]
        g_th = {n: 0.0 for n in names}
        g_c = np.zeros((Lz, 3))
        for ell in range(Lz - 1, -1, -1):
            phase, mid, phi, x2, g, parts, w1, w2, w3 = snaps[ell]
            lam_x = fwht(lam)
            ov = np.conj(lam_x) * x2
            g_c[ell, 0] = 2.0 * float(np.imag(np.sum(ov * w1)))
            g_c[ell, 1] = 2.0 * float(np.imag(np.sum(ov * w2)))
            g_c[ell, 2] = 2.0 * float(np.imag(np.sum(ov * w3)))
            lam_mid = fwht(lam_x * np.exp(1j * phi))
            ovD = np.conj(lam_mid) * mid
            g_gamma[ell] = 2.0 * float(np.imag(np.sum(ovD * g)))
            for name, part in parts.items():
                g_th[name] += 2.0 * float(np.imag(np.sum(ovD * (float(gammas[ell]) * part))))
            lam = lam_mid * np.exp(1j * phase)
        if arm == "qaoa":
            grad = np.concatenate([g_gamma, g_c[:, 0]])
        else:
            grad = np.concatenate([[g_gamma.sum()], [g_th[n] for n in names], g_c.reshape(-1)])
        return E, grad, psi

    def fidelity(self, psi) -> float:
        return float(abs(np.vdot(self.gs, psi)) ** 2)

    def variance(self, psi) -> float:
        E, Hpsi = self.energy(psi)
        return float(np.vdot(Hpsi, Hpsi).real - E * E)
