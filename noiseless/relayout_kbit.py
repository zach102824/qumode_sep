"""Multi-round XOR relabel ("relayout") for the k-bit-per-cavity ECD chip (n = 2 + 2k).

Generalizes the n = 8 headline method (``spsa_gibbs.relayout_trial`` with ``--relayout-init small
--relayout-return best --relayout-guess best --no-relayout-fixed-stop --relayout-target xor_vacuum``,
RELAYOUT_LOWBUDGET_SUMMARY.md) to any k, on top of ``noiseless/ecd_kbit.py``:

* Layout: transmons d, e hold logical bits 0, 1; cavity A holds bits 2..k+1 and cavity B bits
  k+2..2k+1 as the binary digits of the Fock number (MSB first), exactly the n = 8 map for k = 3.
  Each cavity is simulated with ``nf >= 2^k`` levels; levels >= 2^k are leaked states with energy
  E_max + 1 (see ecd_kbit.py). ``nf = 2^k`` (no margin) is the n = 8 legacy truncation.
* Cavity XOR relabel (``xor_vacuum``): with masks (x_A, x_B) the codeword held by Fock level n_A is
  n_A ^ x_A (same for B); transmon bits are unchanged. After each round the best-so-far round's
  most-likely bitstring is polished (lowest energy among it and its n one-bit flips, n + 1 classical
  lookups) and the masks are set to its cavity codewords, so the guess sits at Fock (0, 0).
  This is exactly ``encoding.corner_spec_for_bitstring`` for k = 3 (identity perm).
* Round 0: tuned growth L = 1 -> L_max (``grow_trial``), s steps per stage, lr schedule.
  Rounds 1..R: fresh small-beta init (|beta| ~ U(0, 0.1), same rng draws as a random init) at L_max,
  ``r`` SPSA-Adam steps at lr ``relayout_lr`` (headline 0.1), c = first-stage c.
* Returned round = lowest Gibbs cost at the common eta (largest end-of-round eta), no GS knowledge.

RNG stream and arithmetic are identical to ``spsa_gibbs.relayout_trial`` for k = 3, nf = 8,
method = "legacy" (tested bit-for-bit against the stored headline records).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit_local_ecd import n_parameters
from .ecd_kbit import KbitEcdSimulator, KbitLayout, bitstring_from_logical
from .spsa_gibbs import (
    ADAM_LR,
    GROW_KICK_SIGMA,
    gibbs_objective,
    grow_trial,
    optimize_trial,
    scale_spsa_a,
    small_beta_parameters,
)

KBIT_TARGETS = ("xor_vacuum", "none")


def default_nf(k: int, margin_frac: float = 0.25, min_margin: int = 16) -> int:
    """Fock truncation used in the scaling study: 2^k + max(2^k/4, 16) levels per cavity
    (k = 3..7: 24, 32, 48, 80, 160). A 2^k/4-only margin leaked up to 0.23 at n = 10 (nf = 20)."""
    return int((1 << k) + max(int(round(margin_frac * (1 << k))), int(min_margin)))


def polish_logical(v0: int, energies: np.ndarray, n: int, radius: int = 1) -> int:
    """Lowest-energy logical index within Hamming 1 of v0 (ties -> smaller index).

    Same rule as encoding.polish_bitstring (radius 1 only; radius 0 = no-op).
    """
    if radius <= 0:
        return int(v0)
    if radius != 1:
        raise NotImplementedError("only polish radius 0/1")
    best = int(v0)
    for k in range(n):
        v = int(v0) ^ (1 << (n - 1 - k))
        if (energies[v], v) < (energies[best], best):
            best = v
    return best


def cavity_masks_for(v: int, k: int) -> tuple[int, int]:
    m = (1 << k) - 1
    return (int(v) >> k) & m, int(v) & m


@dataclass
class XorKbitSim(KbitEcdSimulator):
    """KbitEcdSimulator whose cavity codewords are XOR-relabelled by (xa, xb)."""

    xa: int = 0
    xb: int = 0

    def __post_init__(self):
        lay = self.layout
        k = lay.k
        E = np.asarray(self.energies_logical, dtype=float)
        self._mask = (int(self.xa) << k) | int(self.xb)
        leak = float(E.max()) + 1.0 if self.leak_energy is None else float(self.leak_energy)
        flat0 = lay.flat_of_logical()  # physical flat of logical index under mask 0
        self._flat0 = flat0
        # physical state flat0[j] holds logical j ^ mask
        ef = np.full(int(np.prod(lay.dims)), leak)
        ef[flat0] = E[np.arange(E.size) ^ self._mask]
        self.energies_flat = ef
        self.valid = lay.valid_mask()
        self.encoding = lay.encoding
        gl = int(self.ground_bitstring, 2)
        self.ground_flat_index = int(flat0[gl ^ self._mask])
        self._current_eta = 1.0

    def evaluate(self, x) -> dict:
        ev = super().evaluate(x)
        probs = ev["probs"]
        idx = int(np.argmax(probs))
        lg = self.layout.logical_of_flat(idx)
        ml = "LEAKED" if lg is None else bitstring_from_logical(lg ^ self._mask, self.layout.n_qubits)
        ev["most_likely_bitstring"] = ml
        ev["success"] = ml == self.ground_bitstring
        return ev


def relayout_trial_kbit(
    energies_logical: np.ndarray,
    ground_bitstring: str,
    *,
    k: int,
    nf: int,
    final_layers: int,
    rng: np.random.Generator,
    relayout_rounds: int = 4,
    relayout_steps: int = 200,
    relayout_lr: float = 0.1,
    steps_per_stage: int = 10,
    lr_schedule: list[float] | None = None,
    start_layers: int = 1,
    kick_sigma: float = GROW_KICK_SIGMA,
    c: float = 0.15,
    A: float = 10.0,
    adam_lr: float = ADAM_LR,
    target: str = "xor_vacuum",
    polish_radius: int = 1,
    method: str = "fast",
    u_name: str = "jp",
    want_probs: bool = False,
) -> dict:
    """Headline relayout protocol on the k-bit layout. Returns a plain dict record."""
    if target not in KBIT_TARGETS:
        raise ValueError(f"target must be one of {KBIT_TARGETS}")
    n = 2 + 2 * int(k)
    E = np.asarray(energies_logical, dtype=float).reshape(-1)
    if E.size != (1 << n):
        raise ValueError("energies size mismatch")
    layout = KbitLayout(int(k), int(nf))
    L = int(final_layers)
    n_st = L - int(start_layers) + 1
    if lr_schedule is None:
        lr_schedule = [0.5, 0.2, 0.05, 0.02] if n_st == 4 else None
    if lr_schedule is not None and len(lr_schedule) != n_st:
        raise ValueError(f"lr_schedule needs {n_st} entries")
    extra_c = float(c)
    extra_a = scale_spsa_a(n_parameters(L))
    gl = int(ground_bitstring, 2)
    xa = xb = 0
    rounds: list[dict] = []
    res_list, sim_list, prob_list = [], [], []
    total_nfev = 0
    for rnd in range(int(relayout_rounds) + 1):
        sim = XorKbitSim(layout, E, L, ground_bitstring, u_name=u_name, method=method, xa=xa, xb=xb)
        if rnd == 0:
            res = grow_trial(sim, final_layers=L, rng=rng, start_layers=int(start_layers),
                             kick_sigma=float(kick_sigma), c=float(c), A=float(A), optimizer="spsa_adam",
                             steps_per_stage=int(steps_per_stage), adam_lr=float(adam_lr),
                             lr_schedule=lr_schedule)
        else:
            x0 = small_beta_parameters(L, rng)
            res = optimize_trial(sim, maxiter=int(relayout_steps), rng=rng, x0=x0, a=extra_a, c=extra_c,
                                 A=float(A), optimizer="spsa_adam", adam_lr=float(relayout_lr))
        total_nfev += int(res.nfev)
        ev = sim.evaluate(res.x)
        res_list.append(res)
        sim_list.append(sim)
        prob_list.append(np.asarray(ev["probs"], dtype=float))
        eta_now = max(float(r.eta) for r in res_list)
        funs = [float(gibbs_objective(pp, ss.energies_flat, eta_now)) for pp, ss in zip(prob_list, sim_list)]
        bsf = min(range(len(funs)), key=lambda j: (funs[j], j))
        cand = res.most_likely_bitstring
        cand_i = None if cand == "LEAKED" else int(cand, 2)
        pol_i = None if cand_i is None else polish_logical(cand_i, E, n, polish_radius)
        src = res_list[bsf].most_likely_bitstring
        guess_i = None if src == "LEAKED" else polish_logical(int(src, 2), E, n, polish_radius)
        rounds.append({
            "round": rnd,
            "xa": int(xa), "xb": int(xb),
            "n_layers": L,
            "steps": int(sum(s["steps"] for s in res.stages)) if (rnd == 0 and res.stages) else int(relayout_steps),
            "success": bool(res.success),
            "p_gs": float(res.p_gs),
            "most_likely_bitstring": cand,
            "polished_is_ground": pol_i == gl,
            "fun": float(res.fun),
            "eta": float(res.eta),
            "energy_mean": float(ev["energy_mean"]),
            "leakage": float(ev["leakage"]),
            "mean_n_a": float(ev["mean_n_a"]), "mean_n_b": float(ev["mean_n_b"]),
            "max_abs_beta": float(res.max_abs_beta),
            "nfev": int(res.nfev),
            "cum_nfev": int(total_nfev),
            "next_guess": None if guess_i is None else bitstring_from_logical(guess_i, n),
            "next_guess_is_ground": guess_i == gl,
            "bsf_round": int(bsf),
            "bsf_success": bool(res_list[bsf].success),
            "bsf_p_gs": float(res_list[bsf].p_gs),
            "stages": [{kk: s[kk] for kk in ("n_layers", "steps", "p_gs", "success", "most_likely_bitstring", "nfev")}
                       for s in res.stages] if (rnd == 0 and res.stages) else None,
        })
        if target == "xor_vacuum" and guess_i is not None:
            xa, xb = cavity_masks_for(guess_i, int(k))
    eta_ref = max(float(r.eta) for r in res_list)
    for j, (pp, ss) in enumerate(zip(prob_list, sim_list)):
        rounds[j]["fun_common_eta"] = float(gibbs_objective(pp, ss.energies_flat, eta_ref))
    sel = min(range(len(rounds)), key=lambda j: (rounds[j]["fun_common_eta"], j))
    for j in range(len(rounds)):
        rounds[j]["selected"] = j == sel
    best = res_list[sel]
    out = {
        "success": bool(best.success),
        "p_gs": float(best.p_gs),
        "most_likely_bitstring": best.most_likely_bitstring,
        "ground_bitstring": ground_bitstring,
        "selected_round": int(sel),
        "nfev": int(total_nfev),
        "leakage": float(rounds[sel]["leakage"]),
        "x": np.asarray(best.x, dtype=float).tolist(),
        "sel_xa": int(rounds[sel]["xa"]), "sel_xb": int(rounds[sel]["xb"]),
        "rounds": rounds,
    }
    if want_probs:
        out["_probs"] = prob_list
        out["_results"] = res_list
    return out
