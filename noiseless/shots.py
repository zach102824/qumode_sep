"""Finite-shot (S shots per circuit eval) model, enabled by QM_SHOTS=S (run_relayout_kbit --shots S).

Every probs read during training (cost, eta refresh, evaluate) is replaced by the empirical distribution
of S shots drawn from the exact Born probs. Leaked (out-of-code) shots stay in the empirical
distribution at the simulator's leak-penalty energy (that energy is a fixed constant, not a table lookup),
but are never counted as strings. The Gibbs cost is computed only over sampled flats; its emin shift
(-log sum p e^{-eta(E-emin)} + eta*emin) cancels exactly, so we shift by the min over the sampled
support (numerical stability only). evaluate() keeps the exact p_gs/success (reported) and returns the
shot histogram as ev["probs"] (used for readout top-2n and round selection) plus ev["probs_exact"].
Per-trial union of sampled logical strings (original problem frame) is kept in STATE['union'].
"""
from __future__ import annotations
import os
import numpy as np

S = int(os.environ.get("QM_SHOTS", "0") or 0)
# Fix variants (FINITE_SHOT_FIXES.md); defaults reproduce the plain shot model.
READOUT = os.environ.get("QM_SHOT_READOUT", "freq")  # "lowE": readout = 2n lowest-E strings sampled in that run
CRN = os.environ.get("QM_SHOT_CRN", "") == "1"       # common random numbers for the +/- SPSA evals of a step
COST = os.environ.get("QM_SHOT_COST", "gibbs")       # "cvar": mean E of the best alpha fraction of shots
ALPHA = float(os.environ.get("QM_SHOT_CVAR_ALPHA", "0.1") or 0.1)
STATE = {"rng": None, "union": set(), "n_shot_evals": 0, "run": set(), "crn_seed": None}


def reset(seed: int) -> None:
    STATE["rng"] = np.random.default_rng(int(seed))
    STATE["union"] = set()
    STATE["n_shot_evals"] = 0
    STATE["run"] = set()
    STATE["crn_seed"] = None


def _orig_map(sim) -> np.ndarray:
    m = getattr(sim, "_shots_orig", None)
    if m is None:
        flat0 = np.asarray(sim._flat0)
        lg = np.arange(flat0.size) ^ int(sim._mask)
        Pm = getattr(sim, "_Pm", None)
        if Pm is not None:
            lg = np.asarray(Pm)[lg]
        m = np.full(int(sim.energies_flat.size), -1, dtype=np.int64)
        m[flat0] = lg
        sim._shots_orig = m
    return m


def sample(sim, probs) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float).reshape(-1), 0, None)
    p = p / p.sum()
    r = STATE["rng"] if STATE["crn_seed"] is None else np.random.default_rng(STATE["crn_seed"])
    counts = r.multinomial(S, p)
    idx = np.flatnonzero(counts)
    orig = _orig_map(sim)[idx]
    vs = [int(v) for v in orig if v >= 0]
    STATE["union"].update(vs)
    STATE["run"].update(vs)
    STATE["n_shot_evals"] += 1
    emp = np.zeros_like(p)
    emp[idx] = counts[idx] / S
    return emp


def gibbs_support(emp, energies, eta) -> float:
    idx = np.flatnonzero(emp)
    p = emp[idx]
    e = np.asarray(energies, dtype=float).reshape(-1)[idx]
    emin = float(e.min())  # min over sampled support; cancels exactly
    avg = float(np.dot(p, np.exp(-float(eta) * (e - emin))))
    return float(-np.log(max(avg, 1e-300)) + float(eta) * emin)


def cvar_support(emp, energies, alpha=None) -> float:
    """Mean energy of the best alpha fraction of the shots (fractional weight at the boundary)."""
    a = ALPHA if alpha is None else float(alpha)
    idx = np.flatnonzero(emp)
    e = np.asarray(energies, dtype=float).reshape(-1)[idx]
    w = emp[idx]
    o = np.argsort(e, kind="stable")
    e, w = e[o], w[o]
    cum = np.cumsum(w)
    take = np.clip(a - (cum - w), 0.0, w)
    return float(np.dot(take, e) / take.sum())


def run_spsa_adam_crn(fun, x0, *, maxiter, rng, lr=None, beta1=None, beta2=None, eps=None, c=0.15, gamma=0.101,
                      on_before_step=None):
    """Same as spsa_gibbs.run_spsa_adam (same rng stream) but both evals of a step share one shot seed."""
    from . import spsa_gibbs as sg
    lr = sg.ADAM_LR if lr is None else lr
    b1 = float(sg.ADAM_BETA1 if beta1 is None else beta1)
    b2 = float(sg.ADAM_BETA2 if beta2 is None else beta2)
    eps = sg.ADAM_EPS if eps is None else eps
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x); v = np.zeros_like(x); nfev = 0
    for k in range(1, int(maxiter) + 1):
        if on_before_step is not None:
            on_before_step(k, x)
        ck = c / k**gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        STATE["crn_seed"] = int(STATE["rng"].integers(2**62))
        try:
            yp = float(fun(x + ck * delta)); ym = float(fun(x - ck * delta))
        finally:
            STATE["crn_seed"] = None
        nfev += 2
        ghat = (yp - ym) / (2.0 * ck) * delta
        m = b1 * m + (1.0 - b1) * ghat
        v = b2 * v + (1.0 - b2) * ghat * ghat
        x = x - float(lr) * (m / (1.0 - b1**k)) / (np.sqrt(v / (1.0 - b2**k)) + float(eps))
    return x, float(fun(x)), nfev + 1


def patch(cls) -> None:
    if S <= 0 or getattr(cls, "_shots_patched", False):
        return
    cls._shots_patched = True
    from .spsa_gibbs import beta_regularizer, betas_from_x
    ev0 = cls.evaluate

    def cost(self, x, eta=None):
        emp = sample(self, self.probs_from_x(x))
        if COST == "cvar":
            g = cvar_support(emp, self.energies_flat)
        else:
            g = gibbs_support(emp, self.energies_flat, self._current_eta if eta is None else float(eta))
        if float(getattr(self, "lambda1", 0.0)) == 0.0 and float(getattr(self, "lam", 0.0)) == 0.0:
            return g
        return float(g + beta_regularizer(betas_from_x(x, self.n_layers), lambda1=self.lambda1, lam=self.lam, beta_max=self.beta_max))

    def refresh_eta(self, x, step):
        emp = sample(self, self.probs_from_x(x))
        idx = np.flatnonzero(emp)
        eta = self.eta_ctrl.update(np.asarray(self.energies_flat)[idx], emp[idx], step)
        self._current_eta = eta * float(self.eta_scale)
        return self._current_eta

    def evaluate(self, x):
        ev = ev0(self, x)
        ev["probs_exact"] = ev["probs"]
        ev["probs"] = sample(self, ev["probs"])
        return ev

    cls.cost, cls.refresh_eta, cls.evaluate = cost, refresh_eta, evaluate
    if CRN:
        from . import spsa_gibbs as sg
        sg.run_spsa_adam = run_spsa_adam_crn
