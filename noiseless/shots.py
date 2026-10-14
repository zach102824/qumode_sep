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
STATE = {"rng": None, "union": set(), "n_shot_evals": 0}


def reset(seed: int) -> None:
    STATE["rng"] = np.random.default_rng(int(seed))
    STATE["union"] = set()
    STATE["n_shot_evals"] = 0


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
    counts = STATE["rng"].multinomial(S, p)
    idx = np.flatnonzero(counts)
    orig = _orig_map(sim)[idx]
    STATE["union"].update(int(v) for v in orig if v >= 0)
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


def patch(cls) -> None:
    if S <= 0 or getattr(cls, "_shots_patched", False):
        return
    cls._shots_patched = True
    from .spsa_gibbs import beta_regularizer, betas_from_x
    ev0 = cls.evaluate

    def cost(self, x, eta=None):
        emp = sample(self, self.probs_from_x(x))
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
