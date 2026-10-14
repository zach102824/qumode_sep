"""Multi-cavity chip (default: 3 cavities A, B, C, each with its own data transmon d, e, f) for the
explore-then-exploit XOR relabel method of ``relayout_kbit.explore_exploit_trial_kbit``.

Device. P transmon-cavity pairs. Pair c = (transmon t_c, cavity c); each transmon holds 1 logical bit,
cavity c holds k_c bits as the (Gray) code of its Fock number, simulated with nf_c >= 2^k_c levels
(levels >= 2^k_c are leaked: energy E_max + 1, as in ecd_kbit). A cavity with k_c = 0 is dropped with
its transmon and its gates, so ks = (k, k, 0) is exactly the 2-cavity chip of ecd_kbit.

Bit map (MSB first): (t_0 .. t_{P-1} | n_0 code (k_0 bits) | n_1 code | ...). For P = 2 this is
ecd_kbit's (d, e | n_A | n_B). Physical flat index = C-order over dims (2,)*P + (nf_0, .., nf_{P-1}).

One layer: rotation R(theta_c, phi_c) on every transmon, then ECD(beta_c) = sigma- (x) D(beta_c/2) +
sigma+ (x) D(-beta_c/2) on every pair (own displacement), then the joint-parity gate jp
(phase -i on odd n_i + n_j) on each cavity edge: chain (A-B, B-C) or ring (adds A-C). Same order and
conventions as ecd_kbit._layer_fast.

Parameters per layer: 4P = [Re beta (P), Im beta (P), theta (P), phi (P)]; for P = 2 this is the
8-parameter layout of circuit_local_ecd.unpack_params. ``param_layout(P)`` temporarily swaps the
P-pair parameter helpers into spsa_gibbs / relayout_kbit (n_parameters, random_parameters,
small_beta_parameters, transparent_layer_params, betas_from_x) so grow_trial / optimize_trial /
explore_exploit_trial_kbit run unchanged; for P = 2 the helpers are draw-for-draw identical.

XOR relabel: one integer mask over all cavity bits (transmon bits unchanged), carried as xa (xb = 0);
the guess's cavity codes sit at Fock (0, .., 0).
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np

from .ecd_kbit import born, bitstring_from_logical, displace_fast, gray_decode_arr, gray_encode
from .circuit_local_ecd import rotation_np
from .spsa_gibbs import SampledTailEta, beta_abs_stats, beta_regularizer, gibbs_objective


# ---------------------------------------------------------------------------
# P-pair parameter helpers
# ---------------------------------------------------------------------------
def make_param_helpers(P: int):
    P = int(P)
    npl = 4 * P

    def n_parameters(n_layers):
        return npl * int(n_layers)

    def random_parameters(n_layers, rng=None):
        rng = rng or np.random.default_rng()
        out = np.empty(n_parameters(n_layers), dtype=float)
        for ell in range(int(n_layers)):
            base = npl * ell
            for pair in range(P):
                mag = rng.uniform(0.0, 3.0)
                arg = rng.uniform(0.0, np.pi)
                beta = mag * np.exp(1j * arg)
                out[base + pair] = beta.real
                out[base + P + pair] = beta.imag
                out[base + 2 * P + pair] = rng.uniform(0.0, np.pi)
                out[base + 3 * P + pair] = rng.uniform(0.0, np.pi)
        return out

    def small_beta_parameters(n_layers, rng, beta_max=0.1):
        x = random_parameters(int(n_layers), rng)
        for ell in range(int(n_layers)):
            x[npl * ell: npl * ell + 2 * P] *= float(beta_max) / 3.0
        return x

    def transparent_layer_params():
        return np.concatenate([np.zeros(2 * P), np.full(P, np.pi), np.zeros(P)])

    def unpack(x, n_layers):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != n_parameters(n_layers):
            raise ValueError(f"Expected {n_parameters(n_layers)} params for L={n_layers}, got {x.size}")
        out = []
        for ell in range(int(n_layers)):
            b = npl * ell
            out.append((x[b:b + P] + 1j * x[b + P:b + 2 * P], x[b + 2 * P:b + 3 * P], x[b + 3 * P:b + 4 * P]))
        return out

    def betas_from_x(x, n_layers):
        return np.concatenate([lay[0] for lay in unpack(x, n_layers)]) if int(n_layers) else np.zeros(0, complex)

    return dict(n_parameters=n_parameters, random_parameters=random_parameters,
                small_beta_parameters=small_beta_parameters, transparent_layer_params=transparent_layer_params,
                betas_from_x=betas_from_x, unpack=unpack)


@contextlib.contextmanager
def param_layout(P: int):
    """Swap the P-pair parameter helpers into spsa_gibbs and relayout_kbit for the duration (one process)."""
    from . import relayout_kbit as rk
    from . import spsa_gibbs as sg
    h = make_param_helpers(P)
    targets = [(sg, ("n_parameters", "random_parameters", "small_beta_parameters", "transparent_layer_params",
                     "betas_from_x")),
               (rk, ("n_parameters", "small_beta_parameters"))]
    saved = []
    for mod, names in targets:
        for nm in names:
            if hasattr(mod, nm):
                saved.append((mod, nm, getattr(mod, nm)))
                setattr(mod, nm, h[nm])
    try:
        yield h
    finally:
        for mod, nm, f in saved:
            setattr(mod, nm, f)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
@dataclass
class MultiCavLayout:
    ks: tuple  # bits per cavity (0 = cavity and its transmon absent)
    nfs: tuple  # Fock levels per cavity
    encoding: str = "gray"
    ring: bool = False

    def __post_init__(self):
        if len(self.ks) != len(self.nfs):
            raise ValueError("ks / nfs length mismatch")
        keep = [i for i, k in enumerate(self.ks) if int(k) > 0]
        self.ks = tuple(int(self.ks[i]) for i in keep)
        self.nfs = tuple(int(self.nfs[i]) for i in keep)
        for k, nf in zip(self.ks, self.nfs):
            if nf < (1 << k):
                raise ValueError("nf must be >= 2^k")
        if self.encoding not in ("binary", "gray"):
            raise ValueError(self.encoding)
        P = len(self.ks)
        self.edges = [(i, i + 1) for i in range(P - 1)]
        if self.ring and P >= 3:
            self.edges.append((0, P - 1))
        self.k = None  # not a 2-cavity layout

    @property
    def P(self) -> int:
        return len(self.ks)

    @property
    def n_cav_bits(self) -> int:
        return int(sum(self.ks))

    @property
    def n_qubits(self) -> int:
        return self.P + self.n_cav_bits

    @property
    def dims(self) -> tuple:
        return (2,) * self.P + tuple(self.nfs)

    def split_logical(self, idx):
        """(transmon bits list, cavity Fock numbers list) for logical indices (vectorized)."""
        idx = np.asarray(idx, dtype=np.int64)
        n, P = self.n_qubits, self.P
        ts = [(idx >> (n - 1 - t)) & 1 for t in range(P)]
        ns, shift = [], self.n_cav_bits
        for k in self.ks:
            shift -= k
            code = (idx >> shift) & ((1 << k) - 1)
            ns.append(gray_decode_arr(code) if self.encoding == "gray" else code)
        return ts, ns

    def flat_of_logical(self) -> np.ndarray:
        ts, ns = self.split_logical(np.arange(1 << self.n_qubits))
        return np.ravel_multi_index(tuple(ts) + tuple(ns), self.dims)

    def valid_mask(self) -> np.ndarray:
        v = np.ones(self.dims, dtype=bool)
        for c, (k, nf) in enumerate(zip(self.ks, self.nfs)):
            shape = [1] * len(self.dims)
            shape[self.P + c] = nf
            v = v & (np.arange(nf) < (1 << k)).reshape(shape)
        return v.reshape(-1)

    def energy_flat(self, E_logical, leak_energy=None) -> np.ndarray:
        E_logical = np.asarray(E_logical, dtype=float)
        if leak_energy is None:
            leak_energy = float(E_logical.max()) + 1.0
        out = np.full(int(np.prod(self.dims)), float(leak_energy))
        out[self.flat_of_logical()] = E_logical
        return out

    def logical_of_flat(self, flat: int):
        coords = np.unravel_index(int(flat), self.dims)
        P = self.P
        v = 0
        for t in range(P):
            v = (v << 1) | int(coords[t])
        for c, k in enumerate(self.ks):
            nn = int(coords[P + c])
            if nn >= (1 << k):
                return None
            code = gray_encode(nn) if self.encoding == "gray" else nn
            v = (v << k) | code
        return int(v)

    def slots(self) -> list:
        """SA slot order (MSB-first bit positions): transmons, then cavity digits LSB first, cavities alternating."""
        out = list(range(self.P))
        offs = np.concatenate([[0], np.cumsum(self.ks)[:-1]]).astype(int)
        for j in range(max(self.ks)):
            for c, k in enumerate(self.ks):
                if k > j:
                    out.append(self.P + int(offs[c]) + (k - 1 - j))
        return out


# ---------------------------------------------------------------------------
# circuit
# ---------------------------------------------------------------------------
def _apply_axis(M, psi, ax):
    return np.moveaxis(np.tensordot(M, psi, axes=([1], [ax])), 0, ax)


def jp_phase(layout: MultiCavLayout) -> np.ndarray:
    nd = len(layout.dims)
    ph = np.ones(layout.dims[layout.P:], dtype=complex)
    for i, j in layout.edges:
        si = [1] * layout.P
        si = [1] * (nd - layout.P)
        sj = list(si)
        si[i] = layout.nfs[i]
        sj[j] = layout.nfs[j]
        par = (np.arange(layout.nfs[i]).reshape(si) + np.arange(layout.nfs[j]).reshape(sj)) % 2
        ph = ph * np.where(par == 1, -1j, 1.0 + 0j)
    return ph


def apply_circuit_multi(x, n_layers: int, layout: MultiCavLayout, u_name: str = "jp", _cache={}) -> np.ndarray:
    P = layout.P
    h = make_param_helpers(P)
    key = (layout.ks, layout.nfs, tuple(layout.edges), u_name)
    if key not in _cache:
        _cache[key] = jp_phase(layout) if u_name == "jp" else np.ones(layout.dims[P:], dtype=complex)
    ph = _cache[key]
    psi = np.zeros(layout.dims, dtype=complex)
    psi[(0,) * len(layout.dims)] = 1.0
    for betas, thetas, phis in h["unpack"](x, n_layers):
        for t in range(P):
            psi = _apply_axis(rotation_np(float(thetas[t]), float(phis[t])), psi, t)
        for t in range(P):
            D = displace_fast(complex(betas[t]) / 2.0, layout.nfs[t])
            p0 = np.take(psi, 0, axis=t)  # |0_t>
            p1 = np.take(psi, 1, axis=t)
            ax = P + t - 1  # cavity axis after removing transmon axis t
            new1 = _apply_axis(D, p0, ax)
            new0 = _apply_axis(D.conj().T, p1, ax)
            psi = np.stack([new0, new1], axis=t)
        psi = psi * ph
    return psi.reshape(-1)


# ---------------------------------------------------------------------------
# simulators (duck-type compatible with spsa_gibbs.optimize_trial / grow_trial)
# ---------------------------------------------------------------------------
@dataclass
class XorMultiCavSim:
    layout: MultiCavLayout
    energies_logical: np.ndarray
    n_layers: int
    ground_bitstring: str
    u_name: str = "jp"
    method: str = "fast"
    xa: int = 0  # XOR mask over all cavity bits (low n_cav_bits of the logical index)
    xb: int = 0  # unused (kept for the explore_exploit_trial_kbit interface); must be 0
    leak_energy: float | None = None
    lambda1: float = 0.0
    lam: float = 0.0
    beta_max: float | None = None
    lambda3: float = 0.0
    eta_scale: float = 1.0
    eta_ctrl: SampledTailEta = field(default_factory=SampledTailEta)

    def __post_init__(self):
        lay = self.layout
        if int(self.xb) != 0:
            raise ValueError("XorMultiCavSim carries the whole cavity mask in xa; xb must be 0")
        E = np.asarray(self.energies_logical, dtype=float)
        self._mask = int(self.xa)
        leak = float(E.max()) + 1.0 if self.leak_energy is None else float(self.leak_energy)
        flat0 = lay.flat_of_logical()
        self._flat0 = flat0
        ef = np.full(int(np.prod(lay.dims)), leak)
        ef[flat0] = E[np.arange(E.size) ^ self._mask]
        self.energies_flat = ef
        self.valid = lay.valid_mask()
        self.encoding = lay.encoding
        self.ground_flat_index = int(flat0[int(self.ground_bitstring, 2) ^ self._mask])
        self._current_eta = 1.0
        self._h = make_param_helpers(lay.P)

    def set_lam(self, value: float) -> None:
        self.lam = float(value)
        self.lambda3 = float(value)

    def probs_from_x(self, x) -> np.ndarray:
        if os.environ.get("QM_ECD_FAST") == "1":
            from .ecd_bench import apply_circuit_multi_fast2 as apply_circuit_multi_fast
            return born(apply_circuit_multi_fast(x, self.n_layers, self.layout, self.u_name))
        return born(apply_circuit_multi(x, self.n_layers, self.layout, self.u_name))

    def cost(self, x, eta=None) -> float:
        probs = self.probs_from_x(x)
        use_eta = self._current_eta if eta is None else float(eta)
        g = gibbs_objective(probs, self.energies_flat, use_eta)
        if float(self.lambda1) == 0.0 and float(self.lam) == 0.0:
            return g
        return float(g + beta_regularizer(self._h["betas_from_x"](x, self.n_layers), lambda1=self.lambda1,
                                          lam=self.lam, beta_max=self.beta_max))

    def refresh_eta(self, x, step: int) -> float:
        probs = self.probs_from_x(x)
        eta = self.eta_ctrl.update(self.energies_flat, probs, step)
        self._current_eta = eta * float(self.eta_scale)
        return self._current_eta

    def evaluate(self, x) -> dict:
        lay = self.layout
        probs = self.probs_from_x(x)
        idx = int(np.argmax(probs))
        lg = lay.logical_of_flat(idx)
        ml = "LEAKED" if lg is None else bitstring_from_logical(lg ^ self._mask, lay.n_qubits)
        mean_abs, max_abs, frac = beta_abs_stats(self._h["betas_from_x"](x, self.n_layers), self.beta_max)
        leak = float(probs[~self.valid].sum())
        pd = probs.reshape(lay.dims)
        means = []
        for c, nf in enumerate(lay.nfs):
            ax = tuple(a for a in range(len(lay.dims)) if a != lay.P + c)
            means.append(float(pd.sum(axis=ax) @ np.arange(nf)))
        out = {
            "most_likely_bitstring": ml,
            "p_gs": float(probs[self.ground_flat_index]),
            "success": ml == self.ground_bitstring,
            "energy_mean": float(np.dot(probs, self.energies_flat)),
            "energy_mean_valid": float(np.dot(probs[self.valid], self.energies_flat[self.valid]) / max(1e-300, 1 - leak)),
            "probs": probs,
            "mean_abs_beta": mean_abs, "max_abs_beta": max_abs, "frac_over_beta_max": frac,
            "leakage": leak,
            "mean_n": means,
            "mean_n_a": means[0], "mean_n_b": means[1] if len(means) > 1 else 0.0,
        }
        return out


@dataclass
class MultiCavDevice:
    """Device hooks consumed by relayout_kbit.explore_exploit_trial_kbit(device=...)."""
    layout: MultiCavLayout

    @property
    def n(self) -> int:
        return self.layout.n_qubits

    @property
    def slots(self) -> list:
        return self.layout.slots()

    sim_cls = XorMultiCavSim

    def masks(self, v):
        return int(v) & ((1 << self.layout.n_cav_bits) - 1), 0

    def mask_int(self, a, b):
        return int(a)

    def random_masks(self, rng):
        return int(rng.integers(1 << self.layout.n_cav_bits)), 0


def default_nfs(ks, min_margin: int = 16) -> tuple:
    """Fock truncation = code levels + 16 (n = 16, ks 5,4,4 -> 48, 32, 32)."""
    return tuple(int((1 << int(k)) + min_margin) for k in ks)


def explore_exploit_trial_3cav(energies_logical, ground_bitstring, *, ks=(5, 4, 4), nfs=None, code="gray",
                               ring=False, **kw) -> dict:
    """explore_exploit_trial_kbit on the multi-cavity device (same algorithm, same eval / lookup counting)."""
    from .relayout_kbit import explore_exploit_trial_kbit
    nfs = default_nfs(ks) if nfs is None else tuple(nfs)
    lay = MultiCavLayout(tuple(ks), nfs, encoding=str(code), ring=bool(ring))
    if (1 << lay.n_qubits) != np.asarray(energies_logical).size:
        raise ValueError(f"device holds {lay.n_qubits} bits, energy table has {np.asarray(energies_logical).size}")
    dev = MultiCavDevice(lay)
    with param_layout(lay.P):
        rec = explore_exploit_trial_kbit(energies_logical, ground_bitstring, k=0, nf=1, code=code, device=dev, **kw)
    rec.update(device="multicav", ks=list(lay.ks), nfs=list(lay.nfs), ring=bool(ring), edges=[list(e) for e in lay.edges])
    return rec
