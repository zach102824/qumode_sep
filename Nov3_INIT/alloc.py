"""Equal-split baseline and 8-to-2 energy halving. SPSA gains are the frozen ones."""
from __future__ import annotations

import numpy as np

METHOD_ID = {
    "equal4": {"QAOA": 0, "Nov3rd": 1, "Nov3pair": 2, "SHELL2": 3, "SHELL4": 4, "ANNEAL": 7, "VM": 8, "WL": 9},
    "spread8": {"QAOA": 6, "Nov3rd": 5},
    "halve8to2": {"QAOA": 71, "Nov3rd": 70},
}
C_PERT = 0.1
A_SPALL = 25.0
ALPHA_EXP = 0.602
GAMMA_EXP = 0.101
# Asymptotically optimal Spall pair. Same A and a0 calibration as V4; faster late decay.
ANNEAL_ALPHA = 1.0
ANNEAL_GAMMA = 1.0 / 6.0


def iterations_equal4(Q: int) -> int:
    if Q % 8 != 0:
        raise ValueError(f"Q={Q} is not divisible by 8")
    t = Q // 8 - 6
    if t <= 0:
        raise ValueError(f"Q={Q} does not fit four SPSA blocks")
    return int(t)


def iterations_spread8(Q: int) -> int:
    if Q % 16 != 0:
        raise ValueError(f"Q={Q} is not divisible by 16")
    t = Q // 16 - 6
    if t <= 0:
        raise ValueError(f"Q={Q} does not fit eight SPSA blocks")
    return int(t)


def iterations_halve(Q: int) -> int:
    if (Q - 32) % 4 != 0:
        raise ValueError(f"Q={Q} does not fit 8 screens and two SPSA blocks")
    t = (Q - 32) // 4
    if t <= 0:
        raise ValueError(f"Q={Q} leaves no iterations")
    return int(t)


def n_cavities(n: int) -> int:
    if int(n) % 3 != 0:
        raise ValueError(f"N={n} is not divisible by 3")
    return int(n) // 3


def n_params(method: str, L: int, n: int | None = None) -> int:
    L = int(L)
    if method == "QAOA":
        return 2 * L
    if method == "Nov3rd":
        return 3 + 3 * L
    if method == "Nov3pair":
        if n is None:
            raise ValueError("Nov3pair param count needs N")
        return 3 + 3 * n_cavities(n) * L
    if method == "VM":
        if n is None:
            raise ValueError("VM param count needs N")
        return L * (int(n) * (int(n) - 1) // 2 + 1)
    raise ValueError(method)


def sigmas_of(method: str, L: int, C: int | None = None, n: int | None = None) -> np.ndarray:
    if method == "Nov3rd":
        sig = np.empty(3 + 3 * L, dtype=np.float64)
        sig[0] = 0.10
        sig[1] = 0.25
        sig[2] = 0.25
        sig[3:] = 0.15
        return sig
    if method == "Nov3pair":
        if C is None:
            raise ValueError("Nov3pair sigmas need C")
        sig = np.empty(3 + 3 * int(C) * L, dtype=np.float64)
        sig[0] = 0.10
        sig[1] = 0.25
        sig[2] = 0.25
        sig[3:] = 0.15
        return sig
    if method == "QAOA":
        return np.full(2 * L, 0.10, dtype=np.float64)
    if method == "VM":
        if n is None:
            raise ValueError("VM sigmas need N")
        n_pairs = int(n) * (int(n) - 1) // 2
        width = n_pairs + 1
        sig = np.empty(L * width, dtype=np.float64)
        for ell in range(L):
            sl = ell * width
            sig[sl : sl + n_pairs] = 0.15
            sig[sl + n_pairs] = 0.10
        return sig
    raise ValueError(method)


def make_rng(seed, n, b, L, method_id, start_id, stream):
    return np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(n), int(round(10.0 * float(b))), int(L), int(method_id), int(start_id), int(stream)]
        )
    )


def _rademacher(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0


def shell_a_mult(L: int, ratio: float) -> np.ndarray:
    """Per-coordinate multiplier on a_k. Uc (γ,θK,θM) stay 1; shells get `ratio`.

    c_k is not scaled, so shell a_k/c_k is `ratio` times the Uc ratio.
    """
    m = np.ones(3 + 3 * int(L), dtype=np.float64)
    m[3:] = float(ratio)
    return m


def _grad(ep: float, em: float, c_k: float, sig: np.ndarray, delta: np.ndarray) -> np.ndarray:
    g = np.zeros_like(sig)
    scale = ep - em
    for j in range(sig.shape[0]):
        denom = 2.0 * c_k * sig[j] * delta[j]
        if sig[j] == 0.0 or denom == 0.0:
            g[j] = 0.0
        else:
            g[j] = scale / denom
    return g


def x0_equal4(method: str, L: int, start_id: int, rng) -> np.ndarray:
    if method == "Nov3rd":
        x = np.zeros(3 + 3 * L, dtype=np.float64)
        if start_id == 0:
            x[0] = 0.25
            return x
        x[0] = float(rng.uniform(-0.5, 0.5))
        x[1] = float(rng.uniform(-0.5, 0.5))
        x[2] = float(rng.uniform(-0.5, 0.5))
        x[3:] = rng.uniform(-0.3, 0.3, size=3 * L)
        return x
    if method == "QAOA":
        x = np.zeros(2 * L, dtype=np.float64)
        if start_id == 0:
            x[:] = 0.25
            return x
        x[:] = rng.uniform(-0.5, 0.5, size=2 * L)
        return x
    raise ValueError(method)


def x0_correlated(method: str, L: int, start_id: int, rng) -> np.ndarray:
    """Shared draw plus a layer residual. Start 0 is the algebraic coherent center."""
    if method == "Nov3rd":
        x = np.zeros(3 + 3 * L, dtype=np.float64)
        if start_id == 0:
            x[0] = 0.25
            x[3::3] = -0.5
            return x
        x[0] = float(rng.uniform(-0.5, 0.5))
        x[1] = float(rng.uniform(-0.5, 0.5))
        x[2] = float(rng.uniform(-0.5, 0.5))
        base = rng.uniform(-0.5, 0.5, size=3)
        for ell in range(L):
            x[3 + 3 * ell : 6 + 3 * ell] = base + rng.uniform(-0.15, 0.15, size=3)
        return x
    if method == "QAOA":
        x = np.zeros(2 * L, dtype=np.float64)
        if start_id == 0:
            x[:] = 0.25
            return x
        shared = rng.uniform(-0.5, 0.5, size=2)
        for ell in range(L):
            x[2 * ell : 2 * ell + 2] = shared + rng.uniform(-0.15, 0.15, size=2)
        return x
    raise ValueError(method)


class Ledger:
    def __init__(self):
        self.screen = 0
        self.init = 0
        self.calibration = 0
        self.iteration = 0
        self.candidate = 0
        self.monitoring = 0

    def add(self, kind: str) -> None:
        setattr(self, kind, getattr(self, kind) + 1)

    def total(self) -> int:
        return self.screen + self.init + self.calibration + self.iteration + self.candidate + self.monitoring

    def as_dict(self) -> dict:
        return {
            "screen": self.screen,
            "init": self.init,
            "calibration": self.calibration,
            "iteration": self.iteration,
            "candidate": self.candidate,
            "monitoring": self.monitoring,
            "total": self.total(),
        }


def spall_gains(n_iter: int, a0: float, alpha: float, gamma: float, A: float, c: float = C_PERT):
    """a_k = a0 (1+A)^α / (k+1+A)^α, c_k = c / (k+1)^γ. a_{k=0} = a0."""
    a_gain = float(a0) * (1.0 + float(A)) ** float(alpha)
    aks = [a_gain / (k + 1.0 + float(A)) ** float(alpha) for k in range(int(n_iter))]
    cks = [float(c) / (k + 1.0) ** float(gamma) for k in range(int(n_iter))]
    return a_gain, aks, cks


def spsa_block(
    propagate,
    x0: np.ndarray,
    sigmas: np.ndarray,
    n_iter: int,
    rng,
    ledger: Ledger,
    a_mult=None,
    alpha=None,
    gamma=None,
    A=None,
) -> dict:
    """One SPSA block. Default exponents are the frozen v4 0.602/0.101 schedule.

    Optional a_mult multiplies the scalar a_k per coordinate. Calibration uses c=0.1.
    Clip remains 2σ. Query count is independent of the exponents.
    """

    def call(x, kind: str):
        ledger.add(kind)
        energy, aux = propagate(np.asarray(x, dtype=np.float64))
        if not np.isfinite(energy):
            raise RuntimeError("non-finite energy")
        return float(energy), aux

    x = np.array(x0, dtype=np.float64, copy=True)
    if a_mult is None:
        a_mult = np.ones(x.size, dtype=np.float64)
    else:
        a_mult = np.asarray(a_mult, dtype=np.float64)
        if a_mult.shape != (x.size,):
            raise ValueError(f"a_mult shape {a_mult.shape} != {(x.size,)}")
    e_init, aux_init = call(x, "init")
    grads = []
    for _ in range(4):
        delta = _rademacher(rng, x.size)
        step = C_PERT * sigmas * delta
        ep, _aux = call(x + step, "calibration")
        em, _aux = call(x - step, "calibration")
        grads.append(_grad(ep, em, C_PERT, sigmas, delta))
    ratios = [abs(g[j]) / sigmas[j] for g in grads for j in range(x.size) if sigmas[j] > 0.0]
    med = float(np.median(ratios)) if ratios else 0.0
    a0 = 0.0 if med < 1e-15 else 0.5 / med
    alpha = ALPHA_EXP if alpha is None else float(alpha)
    gamma = GAMMA_EXP if gamma is None else float(gamma)
    A = A_SPALL if A is None else float(A)
    a_gain, aks, cks = spall_gains(n_iter, a0, alpha, gamma, A, C_PERT)
    history = []
    for k in range(n_iter):
        ck = cks[k]
        ak = aks[k]
        delta = _rademacher(rng, x.size)
        step = ck * sigmas * delta
        ep, _aux = call(x + step, "iteration")
        em, _aux = call(x - step, "iteration")
        g = _grad(ep, em, ck, sigmas, delta)
        update = np.clip(ak * a_mult * g, -2.0 * sigmas, 2.0 * sigmas)
        x = x - update
        history.append(x.copy())
    last = history[-1]
    poly = np.mean(np.stack(history[-min(64, len(history)) :], axis=0), axis=0)
    mid = history[len(history) // 2]
    e_last, aux_last = call(last, "candidate")
    e_poly, aux_poly = call(poly, "candidate")
    e_mid, aux_mid = call(mid, "candidate")
    options = (
        ("init", e_init, np.array(x0, dtype=np.float64, copy=True), aux_init),
        ("last", e_last, last, aux_last),
        ("polyak64", e_poly, poly, aux_poly),
        ("mid_iterate", e_mid, mid, aux_mid),
    )
    choice = min(options, key=lambda item: (item[1], item[0] != "init"))
    return {
        "E": float(choice[1]),
        "x": np.asarray(choice[2], dtype=np.float64).copy(),
        "aux": choice[3],
        "which": choice[0],
        "e_init": float(e_init),
        "a_gain": float(a_gain),
        "a0": float(a0),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "A": float(A),
        "a_mult": [float(v) for v in a_mult],
    }


def x0_spread8(method: str, L: int, start_id: int, rng) -> np.ndarray:
    """Eight draws. Start 0 is the coherent center; the other seven use the equal4 box."""
    if start_id == 0:
        return x0_center4(method, L, 0, None)
    return x0_equal4(method, L, start_id, rng)


# Nov3 center s1=-0.5, s2=s3=theta=0. H exp(-i s1 n_Z) H = exp(-i s1 n_X)
# with n_X=(I-X)/2, so beta = -s1/2 = +0.25. gamma is shared and unscaled.
QAOA_CENTER_GAMMA = 0.25
QAOA_CENTER_BETA = 0.25


def x0_qaoa_center4(L: int, start_id: int, rng) -> np.ndarray:
    """QAOA counterpart of the Nov3 center. Starts 1-3 stay in the equal4 box."""
    x = x0_equal4("QAOA", L, start_id, rng)
    if start_id == 0:
        derived = np.empty(2 * L, dtype=np.float64)
        derived[0::2] = QAOA_CENTER_GAMMA
        derived[1::2] = QAOA_CENTER_BETA
        if not np.array_equal(x, derived):
            raise RuntimeError("derived QAOA center is not the historical equal4 start 0")
    return x


def x0_center4(method: str, L: int, start_id: int, rng) -> np.ndarray:
    """equal4 boxes. Nov3 start 0 uses the coherent shell center s1=-0.5."""
    x = x0_equal4(method, L, start_id, rng)
    if method == "Nov3rd" and start_id == 0:
        x[3::3] = -0.5
    return x


def x0_pair_center4(C: int, L: int, start_id: int, rng) -> np.ndarray:
    """Per-cavity shells. Start 0 tiles the shared coherent center onto every cavity."""
    C = int(C)
    L = int(L)
    x = np.zeros(3 + 3 * C * L, dtype=np.float64)
    if start_id == 0:
        x[0] = 0.25
        shells = x[3:].reshape(L, C, 3)
        shells[:, :, 0] = -0.5
        return x
    x[0] = float(rng.uniform(-0.5, 0.5))
    x[1] = float(rng.uniform(-0.5, 0.5))
    x[2] = float(rng.uniform(-0.5, 0.5))
    x[3:] = rng.uniform(-0.3, 0.3, size=3 * C * L)
    return x


def x0_vm_center4(n: int, L: int, start_id: int, rng, j_upper: np.ndarray) -> np.ndarray:
    """Paper V(M) p=L. Start 0 sets M=J and β=0.25 (algebraic QAOA)."""
    n = int(n)
    L = int(L)
    n_pairs = n * (n - 1) // 2
    j_upper = np.asarray(j_upper, dtype=np.float64)
    if j_upper.size != n_pairs:
        raise ValueError(f"j_upper len {j_upper.size} != {n_pairs}")
    width = n_pairs + 1
    x = np.zeros(L * width, dtype=np.float64)
    if start_id == 0:
        for ell in range(L):
            sl = ell * width
            x[sl : sl + n_pairs] = j_upper
            x[sl + n_pairs] = QAOA_CENTER_BETA
        return x
    for ell in range(L):
        sl = ell * width
        x[sl : sl + n_pairs] = rng.uniform(-0.5, 0.5, size=n_pairs)
        x[sl + n_pairs] = float(rng.uniform(-0.5, 0.5))
    return x


def run_equal4(
    propagate,
    method: str,
    L: int,
    Q: int,
    rngs_init,
    rngs_spsa,
    x0_fn=None,
    C=None,
    n=None,
    a_mult=None,
    alpha=None,
    gamma=None,
    A=None,
) -> dict:
    t_iter = iterations_equal4(Q)
    sig = sigmas_of(method, L, C=C, n=n)
    ledger = Ledger()
    best = None
    tails = []
    draw = x0_equal4 if x0_fn is None else x0_fn
    for start_id, (rng_i, rng_s) in enumerate(zip(rngs_init, rngs_spsa)):
        x0 = draw(method, L, start_id, rng_i)
        block = spsa_block(propagate, x0, sig, t_iter, rng_s, ledger, a_mult=a_mult, alpha=alpha, gamma=gamma, A=A)
        tails.append({"start_id": start_id, "e_init": block["e_init"], "E": block["E"], "which": block["which"], "a_gain": block["a_gain"]})
        if best is None or block["E"] < best["E"] or (block["E"] == best["E"] and start_id < best["start_id"]):
            best = dict(block)
            best["start_id"] = start_id
    if ledger.total() != Q or ledger.monitoring != 0 or ledger.screen != 0:
        raise RuntimeError(f"equal4 ledger {ledger.as_dict()} != Q {Q}")
    best["queries"] = ledger.as_dict()
    best["n_iter"] = t_iter
    best["tails"] = tails
    best["screen_energies"] = None
    best["kept_start_ids"] = [row["start_id"] for row in tails]
    return best


def run_spread8(propagate, method: str, L: int, Q: int, rngs_init, rngs_spsa, x0_fn=None) -> dict:
    t_iter = iterations_spread8(Q)
    sig = sigmas_of(method, L)
    ledger = Ledger()
    best = None
    tails = []
    if len(rngs_init) != 8 or len(rngs_spsa) != 8:
        raise RuntimeError("spread8 expects eight starts")
    draw = x0_spread8 if x0_fn is None else x0_fn
    for start_id, (rng_i, rng_s) in enumerate(zip(rngs_init, rngs_spsa)):
        x0 = draw(method, L, start_id, rng_i)
        block = spsa_block(propagate, x0, sig, t_iter, rng_s, ledger)
        tails.append({"start_id": start_id, "e_init": block["e_init"], "E": block["E"], "which": block["which"], "a_gain": block["a_gain"]})
        if best is None or block["E"] < best["E"] or (block["E"] == best["E"] and start_id < best["start_id"]):
            best = dict(block)
            best["start_id"] = start_id
    if ledger.total() != Q or ledger.monitoring != 0 or ledger.screen != 0:
        raise RuntimeError(f"spread8 ledger {ledger.as_dict()} != Q {Q}")
    best["queries"] = ledger.as_dict()
    best["n_iter"] = t_iter
    best["tails"] = tails
    best["screen_energies"] = None
    best["kept_start_ids"] = [row["start_id"] for row in tails]
    return best


def run_halve(propagate, method: str, L: int, Q: int, rngs_init, rngs_spsa) -> dict:
    t_iter = iterations_halve(Q)
    sig = sigmas_of(method, L)
    ledger = Ledger()
    screened = []
    for start_id, rng_i in enumerate(rngs_init):
        x0 = x0_correlated(method, L, start_id, rng_i)
        energy, _aux = propagate(x0)
        ledger.add("screen")
        if not np.isfinite(energy):
            raise RuntimeError("non-finite screen energy")
        screened.append((float(energy), start_id, x0))
    screened.sort(key=lambda item: (item[0], item[1]))
    kept = screened[:2]
    best = None
    tails = []
    for energy, start_id, x0 in screened:
        tails.append({"start_id": start_id, "e_screen": energy, "kept": start_id in (kept[0][1], kept[1][1])})
    finalists = []
    for _energy, start_id, x0 in kept:
        block = spsa_block(propagate, x0, sig, t_iter, rngs_spsa[start_id], ledger)
        finalists.append({"start_id": start_id, "e_init": block["e_init"], "E": block["E"], "which": block["which"], "a_gain": block["a_gain"]})
        if best is None or block["E"] < best["E"] or (block["E"] == best["E"] and start_id < best["start_id"]):
            best = dict(block)
            best["start_id"] = start_id
    if ledger.total() != Q or ledger.screen != 8 or ledger.monitoring != 0:
        raise RuntimeError(f"halve ledger {ledger.as_dict()} != Q {Q}")
    best["queries"] = ledger.as_dict()
    best["n_iter"] = t_iter
    best["tails"] = {"screen": tails, "finalists": finalists}
    best["screen_energies"] = [row["e_screen"] for row in sorted(tails, key=lambda r: r["start_id"])]
    best["kept_start_ids"] = [kept[0][1], kept[1][1]]
    return best
