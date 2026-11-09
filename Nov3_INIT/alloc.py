"""Equal-split baseline and 8-to-2 energy halving. SPSA gains are the frozen ones."""
from __future__ import annotations

import numpy as np

METHOD_ID = {"equal4": {"QAOA": 0, "Nov3rd": 1}, "halve8to2": {"QAOA": 71, "Nov3rd": 70}}
C_PERT = 0.1
A_SPALL = 25.0
ALPHA_EXP = 0.602
GAMMA_EXP = 0.101


def iterations_equal4(Q: int) -> int:
    if Q % 8 != 0:
        raise ValueError(f"Q={Q} is not divisible by 8")
    t = Q // 8 - 6
    if t <= 0:
        raise ValueError(f"Q={Q} does not fit four SPSA blocks")
    return int(t)


def iterations_halve(Q: int) -> int:
    if (Q - 32) % 4 != 0:
        raise ValueError(f"Q={Q} does not fit 8 screens and two SPSA blocks")
    t = (Q - 32) // 4
    if t <= 0:
        raise ValueError(f"Q={Q} leaves no iterations")
    return int(t)


def sigmas_of(method: str, L: int) -> np.ndarray:
    if method == "Nov3rd":
        sig = np.empty(3 + 3 * L, dtype=np.float64)
        sig[0] = 0.10
        sig[1] = 0.25
        sig[2] = 0.25
        sig[3:] = 0.15
        return sig
    if method == "QAOA":
        return np.full(2 * L, 0.10, dtype=np.float64)
    raise ValueError(method)


def make_rng(seed, n, b, L, method_id, start_id, stream):
    return np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(n), int(round(10.0 * float(b))), int(L), int(method_id), int(start_id), int(stream)]
        )
    )


def _rademacher(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0


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


def spsa_block(propagate, x0: np.ndarray, sigmas: np.ndarray, n_iter: int, rng, ledger: Ledger) -> dict:
    """One frozen-gain block. propagate(x) -> (energy, aux). Aux of non-selected calls is dropped."""

    def call(x, kind: str):
        ledger.add(kind)
        energy, aux = propagate(np.asarray(x, dtype=np.float64))
        if not np.isfinite(energy):
            raise RuntimeError("non-finite energy")
        return float(energy), aux

    x = np.array(x0, dtype=np.float64, copy=True)
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
    a_gain = a0 * (1.0 + A_SPALL) ** ALPHA_EXP
    history = []
    for k in range(n_iter):
        ck = C_PERT / (k + 1) ** GAMMA_EXP
        ak = a_gain / (k + 1.0 + A_SPALL) ** ALPHA_EXP
        delta = _rademacher(rng, x.size)
        step = ck * sigmas * delta
        ep, _aux = call(x + step, "iteration")
        em, _aux = call(x - step, "iteration")
        g = _grad(ep, em, ck, sigmas, delta)
        update = np.clip(ak * g, -2.0 * sigmas, 2.0 * sigmas)
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
    }


def run_equal4(propagate, method: str, L: int, Q: int, rngs_init, rngs_spsa) -> dict:
    t_iter = iterations_equal4(Q)
    sig = sigmas_of(method, L)
    ledger = Ledger()
    best = None
    tails = []
    for start_id, (rng_i, rng_s) in enumerate(zip(rngs_init, rngs_spsa)):
        x0 = x0_equal4(method, L, start_id, rng_i)
        block = spsa_block(propagate, x0, sig, t_iter, rng_s, ledger)
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
