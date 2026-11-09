"""Parameter layout, seeds, and SPSA scales for N9-FT.

Cold Nov9 inits match IDEAS.md Idea 1 (the shared 4-start protocol):
one deterministic echo start and three independent box draws.
QAOA and Nov3rd have no displacement; their deterministic start is documented
in README.md. The K-prior changes only start 0 of the KM arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA_MAX = 0.30
A_SPALL = 25
ALPHA_EXP = 0.602
GAMMA_EXP = 0.101
C_BASE = 1.0
TARGET_STEP_OVER_SIGMA = 0.5
CLIP_OVER_SIGMA = 2.0
SIGMA = {"gamma": 0.10, "theta": 0.25, "beta": 0.10, "alpha": 0.02, "s": 0.15}

# Nested dev split. Inference pilot and transfer training do not share instances.
# Validation 11000..11031 and final 20000..20999 are not used.
PILOT_SEEDS = list(range(10000, 10008))
TRAIN_SEEDS = list(range(10008, 10016))
N_TRAIN = 9

METHOD_ID = {
    "qaoa": 0,
    "nov3_km": 1,
    "nov9_km_uz": 2,
    "nov9_km_u0": 3,
    "nov9b_km": 4,
    "nov9_kmr1_uz": 20,
    "nov9_km_kprior_uz": 21,
    "train_nov9_km": 30,
    "train_nov9_kmr1": 31,
    "train_qaoa": 32,
    "ft_nov9_km": 40,
    "ft_nov9_kmr1": 41,
    "ft_qaoa": 42,
    "basin_cold_uz": 50,
    "basin_wide_uz": 51,
    "basin_cold_u0": 52,
    "basin_wide_u0": 53,
    "basin_qaoa": 54,
}


def assert_seed_allowed(seed: int, role: str) -> None:
    seed = int(seed)
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"{role} seed {seed} is in the blind validation/final panel")
    if role == "pilot" and seed not in PILOT_SEEDS:
        raise RuntimeError(f"pilot seed {seed} is outside 10000..10007")
    if role == "train" and seed not in TRAIN_SEEDS:
        raise RuntimeError(f"train seed {seed} is outside 10008..10015")


def make_rng(seed, N, b, L, method_id, start_id, stream):
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(N),
                int(round(10 * float(b))),
                int(L),
                int(method_id),
                int(start_id),
                int(stream),
            ]
        )
    )


def family_of(method: str) -> str:
    if "qaoa" in method:
        return "qaoa"
    if "nov3" in method:
        return "nov3"
    if "nov9b" in method:
        return "nov9b"
    return "nov9"


def features_of(method: str) -> tuple[str, ...]:
    if "qaoa" in method:
        return ()
    if "kmr1" in method:
        return ("K", "M", "R1")
    return ("K", "M")


def extension_of(method: str) -> str:
    fam = family_of(method)
    if fam == "nov9":
        return "u0" if "u0" in method else "uz"
    if fam == "nov9b":
        return "strict"
    return "ideal"


def kprior_of(method: str) -> bool:
    return "kprior" in method


@dataclass
class Spec:
    method: str
    names: list[str]
    kind: list[str]
    sigma: np.ndarray
    family: str
    features: tuple[str, ...]
    kprior: bool

    @property
    def n(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)


def spec_for(method: str, L: int, alpha_sigma_scale: float = 1.0) -> Spec:
    family = family_of(method)
    features = features_of(method)
    if family == "qaoa":
        names = [f"gamma{ell}" for ell in range(L)] + [f"beta{ell}" for ell in range(L)]
        kind = ["gamma"] * L + ["beta"] * L
        sigma = np.array([SIGMA["gamma"]] * L + [SIGMA["beta"]] * L, dtype=np.float64)
    else:
        names = ["gamma"] + [f"theta_{f}" for f in features]
        kind = ["gamma"] + ["theta"] * len(features)
        sigma = [SIGMA["gamma"]] + [SIGMA["theta"]] * len(features)
        if family in ("nov9", "nov9b"):
            for ell in range(L):
                names.append(f"u{ell}")
                kind.append("u")
                sigma.append(SIGMA["alpha"] * float(alpha_sigma_scale))
        for ell in range(L):
            for sname in ("s1", "s2", "s3"):
                names.append(f"{sname}_{ell}")
                kind.append("s")
                sigma.append(SIGMA["s"])
        sigma = np.asarray(sigma, dtype=np.float64)
    return Spec(method, names, kind, sigma, family, features, kprior_of(method))


def alpha_to_u(alpha: float) -> float:
    z = float(alpha) / ALPHA_MAX
    z = min(max(z, -1.0 + 1e-15), 1.0 - 1e-15)
    return float(np.arctanh(z))


def u_to_alpha(u: float) -> float:
    return float(ALPHA_MAX * np.tanh(u))


def physical(x: np.ndarray, spec: Spec) -> np.ndarray:
    p = np.array(x, dtype=np.float64, copy=True)
    for i, k in enumerate(spec.kind):
        if k == "u":
            p[i] = u_to_alpha(float(x[i]))
    return p


def shift_physical(x: np.ndarray, dphys: np.ndarray, spec: Spec) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    for i, k in enumerate(spec.kind):
        if k == "u":
            a = u_to_alpha(float(x[i])) + float(dphys[i])
            lim = ALPHA_MAX * (1.0 - 1e-12)
            if a > lim:
                a = lim
            elif a < -lim:
                a = -lim
            y[i] = alpha_to_u(a)
        else:
            y[i] = float(x[i]) + float(dphys[i])
    return y


def unpack(spec: Spec, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if spec.family == "qaoa":
        L = spec.n // 2
        return {"gammas": x[:L].copy(), "betas": x[L:].copy(), "theta": {}}
    theta = {f: float(x[spec.index(f"theta_{f}")]) for f in spec.features}
    L = sum(1 for n in spec.names if n.startswith("s1_"))
    alphas = []
    if spec.family in ("nov9", "nov9b"):
        alphas = [u_to_alpha(float(x[spec.index(f"u{ell}")])) for ell in range(L)]
    shells = np.zeros((L, 3), dtype=np.float64)
    for ell in range(L):
        shells[ell, 0] = float(x[spec.index(f"s1_{ell}")])
        shells[ell, 1] = float(x[spec.index(f"s2_{ell}")])
        shells[ell, 2] = float(x[spec.index(f"s3_{ell}")])
    return {
        "gamma": float(x[0]),
        "theta": theta,
        "alphas": alphas,
        "shells": shells,
    }


def initial_x(spec: Spec, L: int, start_id: int, rng: np.random.Generator, theta_K_prior: float) -> np.ndarray:
    """Start 0 is deterministic. Starts 1..3 draw the Idea 1 box on stream 2.

    K-prior replaces only start 0's theta_K. Later starts consume the same
    uniform draws as the non-prior KM spec of equal length.
    """
    x = np.zeros(spec.n, dtype=np.float64)
    if spec.family == "qaoa":
        if start_id == 0:
            x[:] = 0.25
        else:
            x[:] = rng.uniform(-0.5, 0.5, size=spec.n)
        return x
    if start_id == 0:
        x[0] = 0.25
        if spec.kprior:
            x[spec.index("theta_K")] = float(theta_K_prior)
        if spec.family in ("nov9", "nov9b"):
            for ell in range(L):
                sign = -1.0 if (ell % 2 == 1) else 1.0
                x[spec.index(f"u{ell}")] = alpha_to_u(sign * 0.06)
        return x
    x[0] = float(rng.uniform(-0.5, 0.5))
    for f in spec.features:
        x[spec.index(f"theta_{f}")] = float(rng.uniform(-0.5, 0.5))
    if spec.family in ("nov9", "nov9b"):
        for ell in range(L):
            a = float(rng.uniform(-0.08, 0.08))
            x[spec.index(f"u{ell}")] = alpha_to_u(a)
    for ell in range(L):
        for sname in ("s1", "s2", "s3"):
            x[spec.index(f"{sname}_{ell}")] = float(rng.uniform(-0.3, 0.3))
    return x


def queries_per_start(n_iter: int) -> int:
    """8 calibration + 2 per iteration + 2 candidate evaluations."""
    return 8 + 2 * int(n_iter) + 2


def n_iter_for_start_budget(budget: int) -> int:
    if budget < 10 or (budget - 10) % 2 != 0:
        raise ValueError(f"start budget {budget} is not 8+2*n_iter+2")
    return (int(budget) - 10) // 2
