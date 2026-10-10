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
    code: str = "gray",
) -> dict:
    """Headline relayout protocol on the k-bit layout. Returns a plain dict record."""
    if target not in KBIT_TARGETS:
        raise ValueError(f"target must be one of {KBIT_TARGETS}")
    n = 2 + 2 * int(k)
    E = np.asarray(energies_logical, dtype=float).reshape(-1)
    if E.size != (1 << n):
        raise ValueError("energies size mismatch")
    layout = KbitLayout(int(k), int(nf), encoding=str(code))
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
        "sel_layers": int(rounds[sel]["n_layers"]),
        "rounds": rounds,
    }
    if want_probs:
        out["_probs"] = prob_list
        out["_results"] = res_list
    return out


def growth_lr_schedule(L: int) -> list[float]:
    """Same as run_relayout_kbit.lr_schedule_for: 0.5/0.2/0.05/0.02 at L = 4, log-linear otherwise."""
    head = (0.5, 0.2, 0.05, 0.02)
    L = int(L)
    if L == 4:
        return list(head)
    if L == 1:
        return [head[0]]
    pos = np.linspace(0.0, 3.0, L)
    return [float(np.exp(np.interp(p, np.arange(4), np.log(head)))) for p in pos]


def bit_perm_map(perm, n: int) -> np.ndarray:
    """Pm[v'] = logical index whose bit perm[i] equals bit i of v' (MSB-first positions)."""
    idx = np.arange(1 << n, dtype=np.int64)
    out = np.zeros_like(idx)
    for i in range(n):
        out |= ((idx >> (n - 1 - i)) & 1) << (n - 1 - int(perm[i]))
    return out


def explore_exploit_trial_kbit(
    energies_logical: np.ndarray,
    ground_bitstring: str,
    *,
    k: int,
    nf: int,
    final_layers: int,
    rng: np.random.Generator,
    explore_rounds: int = 5,
    explore_mask: str = "random",
    explore_perm=None,
    topk: int = 4,
    relayout_rounds: int = 4,
    relayout_steps: int = 200,
    relayout_lr: float = 0.1,
    steps_per_stage: int = 50,
    lr_schedule: list[float] | None = None,
    start_layers: int = 1,
    kick_sigma: float = GROW_KICK_SIGMA,
    c: float = 0.15,
    A: float = 10.0,
    adam_lr: float = ADAM_LR,
    polish_radius: int = 1,
    method: str = "fast",
    u_name: str = "jp",
    explore_eta_scale: float = 1.0,
    exploit_eta_scale: float = 1.0,
    explore_layers: int | None = None,
    explore_steps: int | None = None,
    cool_steps: int = 0,
    cool_eta_scale: float = 64.0,
    cool_lr: float = 0.05,
    raw_k: int = 0,
    polish_t: int = 1,
    init_beta_max: float | None = None,
    kick_sigma_x: float | None = None,
    code: str = "gray",
    sa_budget: int = 0,
    sa_seed: int | None = None,
    sa_noise: float = 0.15,
    sa_mode: str = "uncertain_low",
    xstop: int = -1,
    xstop_q: float = 0.5,
    thr_eps: float = -1.0,
    xwarm_steps: int = 0,
    xmut: int = 2,
    lmut_swaps: int = 0,
    xagree: int = 0,
) -> dict:
    """Explore-then-exploit relayout (n = 10 cost fix, RELAYOUT_N10_COST_DIAGNOSIS.md).

    * Explore: ``explore_rounds`` independent layer-growth runs (L = 1 -> L_max, s steps per stage,
      the round-0 recipe). Run 0 uses mask (0, 0); later runs use a fresh uniformly random cavity XOR
      mask (``explore_mask='random'``; ``'zero'`` keeps (0, 0)), which changes which Fock states the
      circuit is biased toward. ``explore_mask='perm'``: random mask plus a uniformly random permutation
      of the n logical bits over the physical slots (transmons d, e and the cavity Fock digits), i.e. a
      fresh qubit-to-hardware assignment per explore run. After every round (explore or exploit) the ``topk`` most probable
      code states are polished (radius 1) and added to a candidate pool.
    * Guess = lowest classical energy in the pool (ties -> smaller index); ``topk * (n + 1)`` classical
      lookups per round, no GS knowledge.
    * Exploit: ``relayout_rounds`` rounds of ``relayout_steps`` SPSA-Adam steps from small beta with the
      cavity masks set to the current guess (guess at Fock (0, 0)), exactly the old relabel round.
    * Returned round: lowest Gibbs cost at the common eta over all rounds (same rule as before).
    * ``explore_eta_scale`` / ``exploit_eta_scale``: multiplier on the sampled-tail η controller in the
      explore growth stages / exploit rounds (η is an inverse temperature; > 1 = colder, weights the
      low-energy tail more). ``explore_layers`` (default L) and ``explore_steps`` (default s) set the
      explore growth target depth and steps per stage independently of the exploit depth L.
    * ``cool_steps`` > 0: after each explore growth, a cold "concentration" stage of that many SPSA-Adam
      steps from the grown parameters at η × ``cool_eta_scale`` and lr ``cool_lr``; the explore candidate
      is the top-K of the cooled state (counted in evals).
    * ``raw_k`` > 0 (energy-ranked pool): after every round the ``raw_k`` most probable code states are added to a
      raw pool (1 classical energy lookup each); the guess is the best radius-``polish_radius`` fix-up of the
      ``polish_t`` lowest-energy raw-pool members (``topk`` is then ignored). ``init_beta_max`` rescales the random
      |beta| of the first growth stage (default 3); ``kick_sigma_x`` overrides the growth kick sigma.
    * ``xwarm_steps`` > 0 (warm-started explore): explore runs after the first start from the parameters and layout of the
      lowest-<E> earlier explore run (layout mutated by ``xmut`` random slot swaps, same centre state) and run a single
      stage of ``xwarm_steps`` SPSA-Adam steps at depth Lx instead of a full growth (cheaper; counted in evals).
    * ``lmut_swaps`` > 0 (layout mutation, cold): explore runs after the first use the lowest-<E> earlier explore run's
      layout mutated by ``lmut_swaps`` random slot swaps (same centre), but a full cold growth from scratch (no warm
      parameters). Distinct from ``xwarm``; when both are set, ``xwarm`` wins. Tag suffix ``_lmN``.
    * ``xagree`` > 0 (agreement early stop, adaptive number of explore runs): after each explore run, a run "votes" for the
      current guess if its top-``raw_k`` probable states contain a state within Hamming distance 1 of the guess (pure bit test,
      no lookups). Remaining explore runs are skipped once ``xagree`` runs agree (explore_rounds then acts as a maximum E).
      Skipped runs cost no evals. Tag suffix ``_xagN``.
    """
    n = 2 + 2 * int(k)
    E = np.asarray(energies_logical, dtype=float).reshape(-1)
    layout = KbitLayout(int(k), int(nf), encoding=str(code))
    L = int(final_layers)
    n_st = L - int(start_layers) + 1
    if lr_schedule is None:
        lr_schedule = [0.5, 0.2, 0.05, 0.02] if n_st == 4 else None
    extra_a = scale_spsa_a(n_parameters(L))
    Lx = int(explore_layers) if explore_layers else L
    sx = int(explore_steps) if explore_steps else int(steps_per_stage)
    if Lx == L:
        lr_x = list(lr_schedule)
    else:
        lr_x = growth_lr_schedule(Lx)
    gl = int(ground_bitstring, 2)
    flat0 = layout.flat_of_logical()
    pool: set[int] = set()
    rounds: list[dict] = []
    res_list, sim_list, prob_list = [], [], []
    total_nfev = 0
    guess = None
    n_lookups = 0
    seen: set[int] = set()  # distinct energy-table entries read (cache)
    raw_pool: set[int] = set()
    polished_cache: dict[int, int] = {}
    n_total = int(explore_rounds) + int(relayout_rounds)
    sa_centers = sa_m = None
    sa_rng = None
    n_sa = 0
    if explore_mask == "sa":
        from noiseless.classical_layout import sa_sample, marginals, layout_for_run
        sa_rng = np.random.default_rng(int(sa_seed if sa_seed is not None else 12345))
        sa_seen = sa_sample(E, n, int(sa_budget), sa_rng)
        seen.update(sa_seen.keys())
        n_sa = len(sa_seen)
        n_lookups += n_sa
        sa_centers, sa_m = marginals(sa_seen, n)
        pool.add(int(sa_centers[0]))  # classical best (already looked up) enters the pool
    best_x = best_perm = None  # warm-started explore: lowest-<E> earlier explore run
    best_cen = None
    best_em = np.inf
    stage_emeans: list[float] = []  # explore early-stop: stage-xstop <E> of earlier runs (quantum, no lookups)
    n_stopped = 0
    run_raws: list[list[int]] = []
    agree_stop = False
    n_expl_run = 0
    for rnd in range(n_total):
        exploring = rnd < int(explore_rounds)
        if exploring and agree_stop:
            continue
        n_expl_run += int(exploring)
        Pm = None
        perm = cen = None  # set in sa / warm / lmut branches for best-layout tracking
        if exploring:
            warm = int(xwarm_steps) > 0 and rnd > 0 and best_x is not None
            lmut = (not warm) and int(lmut_swaps) > 0 and rnd > 0 and best_perm is not None
            if warm:
                perm = np.array(best_perm, dtype=int)
                for _ in range(int(xmut)):
                    i, j = sa_rng.choice(n, 2, replace=False) if sa_rng is not None else rng.choice(n, 2, replace=False)
                    perm[i], perm[j] = perm[j], perm[i]
                cen = int(best_cen)
                Pm = bit_perm_map(perm, n)
                pc = int(np.flatnonzero(Pm == cen)[0])
                xa, xb = cavity_masks_for(pc, int(k))
            elif lmut:
                # cold growth on a mutation of the best earlier layout (keeps layout diversity without warm params)
                perm = np.array(best_perm, dtype=int)
                _rng = sa_rng if sa_rng is not None else rng
                for _ in range(int(lmut_swaps)):
                    i, j = _rng.choice(n, 2, replace=False)
                    perm[i], perm[j] = perm[j], perm[i]
                cen = int(best_cen)
                Pm = bit_perm_map(perm, n)
                pc = int(np.flatnonzero(Pm == cen)[0])
                xa, xb = cavity_masks_for(pc, int(k))
            elif explore_mask == "sa":
                base_mode = sa_mode.replace("_adapt", "")
                perm, cen = layout_for_run(rnd, sa_centers, sa_m, n, int(k), sa_rng, float(sa_noise), base_mode)
                if sa_mode.endswith("_adapt") and guess is not None:
                    cen = int(guess)  # adaptive: center the next layout on the best state found so far
                Pm = bit_perm_map(perm, n)
                pc = int(np.flatnonzero(Pm == cen)[0])
                xa, xb = cavity_masks_for(pc, int(k))
            elif rnd == 0 and explore_perm is not None:
                # explicit layout for explore run 1 (physical slot i holds logical bit explore_perm[i]), zero mask
                xa = xb = 0
                Pm = bit_perm_map(np.asarray(explore_perm, dtype=int), n)
            elif rnd == 0 and explore_mask == "perm0":
                # layout-only arm: random bit permutation, zero mask; drawn from a copy of rng so init draws are unchanged
                import copy as _copy
                xa = xb = 0
                Pm = bit_perm_map(_copy.deepcopy(rng).permutation(n), n)
            elif rnd == 0 or explore_mask == "zero":
                xa = xb = 0
            elif explore_mask in ("random", "perm", "perm0"):
                xa, xb = int(rng.integers(1 << k)), int(rng.integers(1 << k))
                if explore_mask in ("perm", "perm0"):
                    Pm = bit_perm_map(rng.permutation(n), n)
            else:
                raise ValueError(f"unknown explore_mask {explore_mask!r}")
        else:
            xa, xb = cavity_masks_for(guess, int(k)) if guess is not None else (0, 0)
        Lr = Lx if exploring else L
        if Pm is None:
            sim = XorKbitSim(layout, E, Lr, ground_bitstring, u_name=u_name, method=method, xa=xa, xb=xb)
        else:
            gs_p = bitstring_from_logical(int(np.flatnonzero(Pm == gl)[0]), n)
            sim = XorKbitSim(layout, E[Pm], Lr, gs_p, u_name=u_name, method=method, xa=xa, xb=xb)
        if exploring:
            cb = None
            if int(xstop) >= 0:
                def cb(si, r_, _hist=stage_emeans):
                    if si != int(xstop):
                        return False
                    em = float(r_.energy_mean)
                    stop = len(_hist) >= 2 and em > float(np.quantile(_hist, float(xstop_q)))
                    _hist.append(em)
                    return stop
            if warm:
                res = grow_trial(sim, final_layers=Lx, start_layers=Lx, rng=rng, c=float(c), A=float(A), optimizer="spsa_adam",
                                 steps_per_stage=int(xwarm_steps), adam_lr=float(adam_lr), lr_schedule=[lr_x[-1]],
                                 eta_scale_schedule=[float(explore_eta_scale)], init_x=best_x)
            else:
              res = grow_trial(sim, final_layers=Lx, stage_callback=cb, rng=rng, start_layers=int(start_layers),
                             kick_sigma=float(kick_sigma if kick_sigma_x is None else kick_sigma_x), c=float(c), A=float(A), optimizer="spsa_adam",
                             steps_per_stage=sx, adam_lr=float(adam_lr), lr_schedule=lr_x,
                             eta_scale_schedule=[float(explore_eta_scale)] * len(lr_x),
                             init_beta_max=init_beta_max)
            if int(cool_steps) > 0:
                sim.eta_scale = float(cool_eta_scale)
                res_c = optimize_trial(sim, maxiter=int(cool_steps), rng=rng, x0=np.asarray(res.x, dtype=float),
                                       a=scale_spsa_a(n_parameters(Lx)), c=float(c), A=float(A),
                                       optimizer="spsa_adam", adam_lr=float(cool_lr))
                res_c.nfev = int(res_c.nfev) + int(res.nfev)
                res = res_c
            if not warm and int(cool_steps) == 0 and len(getattr(res, "stages", None) or []) < len(lr_x):
                n_stopped += 1
                from noiseless.spsa_gibbs import transparent_layer_params
                while np.asarray(res.x).size < n_parameters(Lx):  # pad with identity layers (same state)
                    res.x = np.concatenate([np.asarray(res.x, dtype=float), transparent_layer_params()])
                sim.n_layers = Lx
        else:
            sim_opt = sim
            if float(thr_eps) >= 0 and guess is not None:
                # threshold-shifted cost: only states at or below E(guess) score; guess gets -eps*scale
                Es = np.asarray(sorted(E[list(seen)])) if seen else E
                scale = float(np.std(Es)) or 1.0
                Ethr = np.minimum(E - float(E[guess]) - float(thr_eps) * scale, 0.0)
                if Pm is None:
                    sim_opt = XorKbitSim(layout, Ethr, Lr, ground_bitstring, u_name=u_name, method=method, xa=xa, xb=xb)
            sim_opt.eta_scale = float(exploit_eta_scale)
            x0 = small_beta_parameters(L, rng)
            res = optimize_trial(sim_opt, maxiter=int(relayout_steps), rng=rng, x0=x0, a=extra_a, c=float(c),
                                 A=float(A), optimizer="spsa_adam", adam_lr=float(relayout_lr))
            if sim_opt is not sim:
                ev_o = sim.evaluate(res.x)
                res.success = bool(ev_o["success"]) if "success" in ev_o else res.success
                res.p_gs = float(ev_o["p_gs"])
        total_nfev += int(res.nfev)
        ev = sim.evaluate(res.x)
        if exploring and float(ev["energy_mean"]) < best_em and perm is not None and cen is not None:
            # track best layout for xwarm / lmut (perm/cen set in sa / warm / lmut branches)
            best_em, best_x, best_perm, best_cen = float(ev["energy_mean"]), np.asarray(res.x, dtype=float).copy(), np.array(perm), int(cen)
        probs = np.asarray(ev["probs"], dtype=float)
        res_list.append(res)
        sim_list.append(sim)
        prob_list.append(probs)
        mask = (int(xa) << k) | int(xb)
        pl = probs[flat0]  # pl[j] = prob of logical j ^ mask
        top = np.argsort(-pl, kind="stable")[: max(1, int(topk))]
        raw = [int(j) ^ mask for j in top]
        if Pm is not None:
            raw = [int(Pm[v]) for v in raw]
        if int(raw_k) > 0:
            top = np.argsort(-pl, kind="stable")[: int(raw_k)]
            raw = [int(j) ^ mask for j in top]
            if Pm is not None:
                raw = [int(Pm[v]) for v in raw]
            raw_pool.update(raw)
            seen.update(raw)
            lowest = sorted(raw_pool, key=lambda v: (E[v], v))[: max(1, int(polish_t))]
            for v in lowest:
                if v not in polished_cache:
                    polished_cache[v] = polish_logical(v, E, n, polish_radius)
                    seen.add(v)
                    seen.update(v ^ (1 << b) for b in range(n))
            new = [polished_cache[v] for v in lowest]
            pool.update(new)
            n_lookups = len(seen)
        else:
            new = [polish_logical(v, E, n, polish_radius) for v in raw]
            n_lookups += len(new) * (n + 1)
            for v in raw:
                seen.add(v)
                seen.update(v ^ (1 << b) for b in range(n))
            pool.update(new)
        guess = min(pool, key=lambda v: (E[v], v))
        if exploring and int(xagree) > 0:
            run_raws.append(list(raw))
            votes = sum(1 for rr in run_raws if any(bin(int(v) ^ int(guess)).count("1") <= 1 for v in rr))
            if votes >= int(xagree):
                agree_stop = True
        eta_now = max(float(r.eta) for r in res_list)
        funs = [float(gibbs_objective(pp, ss.energies_flat, eta_now)) for pp, ss in zip(prob_list, sim_list)]
        bsf = min(range(len(funs)), key=lambda j: (funs[j], j))
        rounds.append({
            "round": rnd, "phase": "explore" if exploring else "exploit", "perm": Pm is not None,
            "xa": int(xa), "xb": int(xb), "n_layers": int(Lx if exploring else L),
            "success": bool(res.success), "p_gs": float(res.p_gs),
            "most_likely_bitstring": res.most_likely_bitstring,
            "polished_is_ground": gl in new,
            "eta": float(res.eta), "energy_mean": float(ev["energy_mean"]),
            "leakage": float(ev["leakage"]), "max_abs_beta": float(res.max_abs_beta),
            "nfev": int(res.nfev), "cum_nfev": int(total_nfev),
            "next_guess": bitstring_from_logical(guess, n), "next_guess_is_ground": guess == gl,
            "next_guess_energy": float(E[guess]),
            "bsf_round": int(bsf), "bsf_success": bool(res_list[bsf].success), "bsf_p_gs": float(res_list[bsf].p_gs),
        })
    eta_ref = max(float(r.eta) for r in res_list)
    for j, (pp, ss) in enumerate(zip(prob_list, sim_list)):
        rounds[j]["fun_common_eta"] = float(gibbs_objective(pp, ss.energies_flat, eta_ref))
    sel = min(range(len(rounds)), key=lambda j: (rounds[j]["fun_common_eta"], j))
    for j in range(len(rounds)):
        rounds[j]["selected"] = j == sel
    best = res_list[sel]
    return {
        "success": bool(best.success), "p_gs": float(best.p_gs),
        "most_likely_bitstring": best.most_likely_bitstring, "ground_bitstring": ground_bitstring,
        "selected_round": int(sel), "nfev": int(total_nfev), "n_lookups": int(n_lookups),
        "n_lookups_distinct": int(len(seen)), "n_lookups_sa": int(n_sa), "n_explore_stopped": int(n_stopped), "n_explore_run": int(n_expl_run),
        "sa_best_is_ground": bool(sa_centers is not None and int(sa_centers[0]) == gl),
        "leakage": float(rounds[sel]["leakage"]),
        "x": np.asarray(best.x, dtype=float).tolist(),
        "sel_xa": int(rounds[sel]["xa"]), "sel_xb": int(rounds[sel]["xb"]),
        "sel_layers": int(rounds[sel]["n_layers"]),
        "rounds": rounds,
    }
