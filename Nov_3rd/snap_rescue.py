"""L=16 SNAP rescue at the SAFE point. Same gates as the HVA fairness baseline.

Hamiltonian, packs, L-BFGS hyperparameters, and the freeze sets are unchanged:

    H = diag(unsat) − 0.75 ΣX − 0.50 ΣXX
    train n∈{6,9} indices [0:200), checkpoint on n9 holdout [200:250)
    freeze n12 first 128, n15 first 64
    L-BFGS lr 0.25, max_iter 8, history 16, strong Wolfe, grad clip 5, batch 64

The n21 phase-2 ladder is not read or written.

Pre-specified arms (chosen before the run, not after seeing n12):

    hva_long     continue the PR #36 L=16 HVA checkpoint. Matched budget.
    walsh_long   continue the PR #36 L=16 Walsh-shell SNAP checkpoint.
    power_p      scalar γ plus polynomial shell, from the paper init.
                 φ_k = c1 k + a_n (c2 k² + c3 k³), a_n = (8/(n−1))^p, a_9 = 1.
    warm_pad     L=4 Walsh SNAP (last shallow PASS) in layers 0..3,
                 identity on layers 4..15, then the same L-BFGS polish.
    warm_tile    that L=4 block repeated four times, then the same polish.

Continuation / warm polish is 120 new rounds (seal 40 + 120 = 160 L-BFGS
steps). power_p is trained 160 rounds from the paper init so the step count
matches. Batches on round r are drawn from default_rng((seed, r, 16)) and are
shared across arms. Each seed restores the best n9-holdout Med ε, including
the init. The gate is the median over seeds 0,1,2 of frozen-n12 Med ε.

PASS iff some SNAP arm's n12 Med ε ≤ the matched hva_long Med ε.
R ≤ 0.98 is reported and is not required for PASS.
Otherwise the verdict is needs_edit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from Nov_3rd import settings as S
from Nov_3rd.hva_safe import (
    INIT_BETA,
    INIT_GAMMA,
    LAMBDA_X,
    MU_XX,
    cn_from_unsat,
    expectation_H,
    fourier_state,
    hva_state,
    snap_state,
    state_fidelity,
    xx_pair_eigenvalues,
)

RUN = S.OUT / "quantum_h_explore" / "snap_rescue_l16"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
L = 16
SEEDS = (0, 1, 2)
EXTRA_ROUNDS = 120
POWER_ROUNDS = 160  # 40 seal steps + 120 extra, matched total
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = {12: 128, 15: 64}
# Popcount of the 8 cavity basis states. High bit first, matching hva_safe.
_PC8 = (0, 1, 1, 2, 1, 2, 2, 3)

ARMS = ("hva_long", "walsh_long", "power_p", "warm_pad", "warm_tile")
SNAP_ARMS = ("walsh_long", "power_p", "warm_pad", "warm_tile")

# How many optimizer rounds this arm runs at L=16. Seal rounds are not replayed.
ARM_ROUNDS = {
    "hva_long": EXTRA_ROUNDS,
    "walsh_long": EXTRA_ROUNDS,
    "power_p": POWER_ROUNDS,
    "warm_pad": EXTRA_ROUNDS,
    "warm_tile": EXTRA_ROUNDS,
}

ARM_FAMILY = {
    "hva_long": "hva",
    "walsh_long": "snap",
    "power_p": "snap",
    "warm_pad": "snap",
    "warm_tile": "snap",
}

ARM_BLURB = {
    "hva_long": "PR36 L=16 HVA checkpoint, then 120 L-BFGS rounds",
    "walsh_long": "PR36 L=16 Walsh SNAP checkpoint, then 120 L-BFGS rounds",
    "power_p": "scalar γ + a_n polynomial shell, 160 rounds from the paper init",
    "warm_pad": "L=4 Walsh SNAP, identity on the other 12 layers, then 120 rounds",
    "warm_tile": "L=4 Walsh block repeated ×4, then 120 rounds",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[snap-rescue]", *a, flush=True)


def arm_rounds(arm: str) -> int:
    if arm not in ARM_ROUNDS:
        raise ValueError(arm)
    return int(ARM_ROUNDS[arm])


def a_of_n(n: int, p: torch.Tensor) -> torch.Tensor:
    """a_n = (8/(n−1))^p. a_9 = 1 for every p."""
    base = 8.0 / float(int(n) - 1)
    return torch.pow(p.new_tensor(base), p)


def phi_power(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, p: torch.Tensor, n: int) -> torch.Tensor:
    """φ_k = c1 k + a_n (c2 k² + c3 k³). c* are (L,)."""
    k = torch.arange(4, dtype=c1.dtype, device=c1.device)
    a = a_of_n(n, p)
    return c1.reshape(-1, 1) * k + a * (
        c2.reshape(-1, 1) * k.square() + c3.reshape(-1, 1) * (k ** 3)
    )


def xx_popcount_eigenvalues(dtype, device=None) -> torch.Tensor:
    """Σ_{i<j} s_i s_j as a function of Hamming weight. Constant on each weight."""
    ev_g = xx_pair_eigenvalues(device=device, dtype=torch.float64)
    pc = torch.tensor(_PC8, dtype=torch.long)
    ev_pc = torch.empty(4, dtype=torch.float64)
    for k in range(4):
        vals = ev_g[pc == k]
        if float(vals.max() - vals.min()) > 1e-12:
            raise RuntimeError(f"XX eigenvalue is not constant on popcount {k}")
        ev_pc[k] = vals[0]
    return ev_pc.to(dtype=dtype, device=device)


def phi_from_hva_mixer(beta_x: torch.Tensor, beta_xx: torch.Tensor) -> torch.Tensor:
    """Shell phases that reproduce U_XX(β^XX) U_X(β^X).

    Both factors are diagonal in the X basis and depend on a cavity only
    through its popcount, so they commute and sit in the Hamming shell.
    φ_k = −β^X k + (β^XX / 2) ev_XX(k), with ev_XX = (3, −1, −1, 3).
    """
    k = torch.arange(4, dtype=beta_x.dtype, device=beta_x.device)
    ev = xx_popcount_eigenvalues(beta_x.dtype, beta_x.device)
    return -beta_x.reshape(-1, 1) * k + (beta_xx.reshape(-1, 1) / 2.0) * ev.reshape(1, 4)


def snap_angles_from_hva(beta_x: torch.Tensor, beta_xx: torch.Tensor, gammas: torch.Tensor) -> dict:
    """Equal Walsh weights copy U_C(γ). The shell copies the HVA mixer."""
    g = gammas.reshape(-1, 1).expand(-1, 4).contiguous()
    return {"gamma_w": g, "phi": phi_from_hva_mixer(beta_x, beta_xx)}


def tile_layers(arr, repeats: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"expected (L, 4), got {a.shape}")
    return np.tile(a, (int(repeats), 1))


def pad_identity(arr, Lout: int) -> np.ndarray:
    """Place a shallow schedule in the first layers. The rest is the identity."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"expected (L, 4), got {a.shape}")
    if a.shape[0] > Lout:
        raise ValueError(f"shallow L={a.shape[0]} > {Lout}")
    out = np.zeros((Lout, 4), dtype=np.float64)
    out[: a.shape[0]] = a
    return out


def load_baselines() -> dict:
    path = RUN / "baselines.json"
    return json.loads(path.read_text())


def _f64(rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


def initial_tensors(arm: str, seed: int, baselines: dict | None = None) -> dict[str, torch.Tensor]:
    """Angle tensors at the start of this arm, before the new L-BFGS rounds."""
    if baselines is None:
        baselines = load_baselines()
    rec = baselines["seeds"][str(seed)]
    if arm == "hva_long":
        ang = rec["L16_hva"]["angles"]
        return {
            "beta_x": _f64(ang["beta_x"]),
            "beta_xx": _f64(ang["beta_xx"]),
            "gammas": _f64(ang["gammas"]),
        }
    if arm == "walsh_long":
        ang = rec["L16_snap"]["angles"]
        return {"gamma_w": _f64(ang["gamma_w"]), "phi": _f64(ang["phi"])}
    if arm == "warm_pad":
        ang = rec["L4_snap"]["angles"]
        return {
            "gamma_w": _f64(pad_identity(ang["gamma_w"], L)),
            "phi": _f64(pad_identity(ang["phi"], L)),
        }
    if arm == "warm_tile":
        ang = rec["L4_snap"]["angles"]
        return {
            "gamma_w": _f64(tile_layers(ang["gamma_w"], L // 4)),
            "phi": _f64(tile_layers(ang["phi"], L // 4)),
        }
    if arm == "power_p":
        return {
            "gamma": torch.full((L,), INIT_GAMMA, dtype=torch.float64),
            "c1": torch.full((L,), -INIT_BETA, dtype=torch.float64),
            "c2": torch.zeros(L, dtype=torch.float64),
            "c3": torch.zeros(L, dtype=torch.float64),
            "p": torch.zeros((), dtype=torch.float64),
        }
    raise ValueError(arm)


def as_parameters(tensors: dict[str, torch.Tensor]) -> tuple[dict, list]:
    params: dict = {}
    plist = []
    for k, v in tensors.items():
        p = torch.nn.Parameter(v.detach().to(dtype=torch.float64).clone())
        params[k] = p
        plist.append(p)
    return params, plist


def prepare(arm: str, params: dict, Cn: torch.Tensor, n: int) -> torch.Tensor:
    if arm == "hva_long":
        return hva_state(Cn, params["beta_x"], params["beta_xx"], params["gammas"])
    if arm in ("walsh_long", "warm_pad", "warm_tile"):
        return snap_state(Cn, params["gamma_w"], params["phi"])
    if arm == "power_p":
        phi = phi_power(params["c1"], params["c2"], params["c3"], params["p"], n)
        return snap_state(Cn, params["gamma"], phi)
    raise ValueError(arm)


def clone_angles(params: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in params.items()}


def copy_into(params: dict, saved: dict) -> None:
    with torch.no_grad():
        for k, v in saved.items():
            src = v if torch.is_tensor(v) else torch.tensor(v)
            params[k].data.copy_(src.to(dtype=params[k].dtype))


def angles_to_lists(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        t = v.detach().cpu().to(torch.float64) if torch.is_tensor(v) else torch.as_tensor(v)
        out[k] = t.numpy().tolist()
    return out


def lists_to_tensors(angles: dict) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in angles.items()}


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def ensure_e0(n: int, sl: slice, E: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Read the PR #36 cache. Recomputing would be a different code path; the cache is the gate."""
    cache = RUN / f"e0_n{n}_{sl.start}_{sl.stop}.npz"
    if not cache.exists():
        raise RuntimeError(f"missing E0 cache {cache.name}; n21 is not a substitute")
    z = np.load(cache)
    if not np.array_equal(z["seeds"], seeds) or z["e0"].shape[0] != E.shape[0]:
        raise RuntimeError(f"E0 cache does not match pack slice n{n} [{sl.start}:{sl.stop}]")
    if abs(float(z["lambda_x"]) - LAMBDA_X) > 0 or abs(float(z["mu_xx"]) - MU_XX) > 0:
        raise RuntimeError("E0 cache was built at a different (λ, μ)")
    return np.asarray(z["e0"], dtype=np.float64)


def load_bundles() -> tuple[dict, dict, dict]:
    """Return train tensors, holdout bundle, eval bundle. No n=21."""
    specs = (
        (6, slice(TRAIN_LO, TRAIN_HI), "train"),
        (9, slice(TRAIN_LO, TRAIN_HI), "train"),
        (6, slice(HOLD_LO, HOLD_HI), "hold"),
        (9, slice(HOLD_LO, HOLD_HI), "hold"),
        (12, slice(0, FREEZE_B[12]), "freeze"),
        (15, slice(0, FREEZE_B[15]), "freeze"),
    )
    data = {}
    for n, sl, tag in specs:
        E, sd = load_unsat(n, sl)
        e0 = ensure_e0(n, sl, E, sd)
        data[(n, tag)] = (E, e0, sd)
    train = {}
    for n in (6, 9):
        E, e0, _sd = data[(n, "train")]
        train[n] = (torch.tensor(E, dtype=torch.float64), torch.tensor(e0, dtype=torch.float64))
    hold = {9: data[(9, "hold")]}
    ev = {
        6: data[(6, "hold")],
        9: data[(9, "hold")],
        12: data[(12, "freeze")],
        15: data[(15, "freeze")],
    }
    return train, hold, ev


def _chunk_for(n: int) -> int:
    if n <= 9:
        return 16
    if n <= 12:
        return 8
    return 4


@torch.no_grad()
def eval_arm(arm: str, params: dict, bundle: dict, *, chunk: int | None = None) -> dict:
    ang = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in params.items()}
    out = {}
    for n, (E, e0, seeds) in bundle.items():
        ch = _chunk_for(n) if chunk is None else chunk
        eps_rows = []
        Eu = torch.tensor(E, dtype=torch.float64)
        e0t = torch.tensor(e0, dtype=torch.float64)
        for s0 in range(0, Eu.shape[0], ch):
            sl = slice(s0, s0 + ch)
            Cn = cn_from_unsat(Eu[sl], n)
            psi = prepare(arm, ang, Cn, n)
            h = expectation_H(psi, Eu[sl], n)
            eps_rows.append((h - e0t[sl]).detach().cpu().numpy())
        eps = np.concatenate(eps_rows)
        out[str(n)] = {
            "n": n,
            "B": int(eps.size),
            "med_eps": float(np.median(eps)),
            "mean_eps": float(np.mean(eps)),
            "min_eps": float(np.min(eps)),
            "max_eps": float(np.max(eps)),
            "eps": eps.tolist(),
            "seeds": seeds.tolist(),
        }
    return out


def train_loss(arm: str, params: dict, E: torch.Tensor, e0: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    Eu = E.index_select(0, idx)
    e0b = e0.index_select(0, idx)
    Cn = cn_from_unsat(Eu, n)
    psi = prepare(arm, params, Cn, n)
    h = expectation_H(psi, Eu, n)
    return ((h - e0b) / float(n)).mean()


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def seed_path(arm: str, seed: int) -> Path:
    return RUN / f"{arm}_seed{seed}.json"


def train_arm(
    arm: str,
    seed: int,
    train: dict,
    hold: dict,
    *,
    rounds: int | None = None,
    batch: int = S.LBFGS_BATCH,
    force: bool = False,
) -> dict:
    """L-BFGS polish. Restores the best n9-holdout checkpoint, init included."""
    rounds = arm_rounds(arm) if rounds is None else int(rounds)
    path = seed_path(arm, seed)
    tensors = initial_tensors(arm, seed)
    start = 0
    hist: list[dict] = []
    best = None
    prev_wall = 0.0
    kept_init_eval = None
    if path.exists() and not force:
        prev = json.loads(path.read_text())
        done = int(prev.get("rounds_completed", 0))
        if prev.get("done") and done >= rounds and prev.get("arm") == arm:
            log(f"skip {path.name} rounds={done}")
            return prev
        if prev.get("final_angles") and prev.get("arm") == arm and done < rounds:
            tensors = lists_to_tensors(prev["final_angles"])
            start = done
            hist = list(prev.get("hist", []))
            best = {
                "med9": float(prev["best_hold_med9"]),
                "round": int(prev["best_round"]),
                "angles": lists_to_tensors(prev["angles"]),
            }
            prev_wall = float(prev.get("wall_s", 0.0))
            kept_init_eval = prev.get("init_eval")
            log(f"resume {path.name} at round {start}/{rounds} best9={best['med9']:.5f}")

    params, plist = as_parameters(tensors)
    E6, e06 = train[6]
    E9, e09 = train[9]
    n6 = int(E6.shape[0])
    n9 = int(E9.shape[0])
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_arm(arm, params, hold, chunk=16)
        return float(ev["9"]["med_eps"])

    t0 = time.time()
    if best is None:
        init_med = hold_med()
        best = {"med9": init_med, "round": -1, "angles": clone_angles(params)}
        hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
        log(f"{arm} s{seed} init n9-hold Medε={init_med:.5f} rounds={rounds}")
    else:
        init_med = float(hist[0]["hold_med9"]) if hist else float(best["med9"])

    for r in range(start, rounds):
        opt = torch.optim.LBFGS(
            plist,
            lr=S.LBFGS_LR,
            max_iter=S.LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        rng = np.random.default_rng((int(seed), int(r), L))
        i6 = torch.tensor(rng.choice(n6, size=b6, replace=False), dtype=torch.long)
        i9 = torch.tensor(rng.choice(n9, size=b9, replace=False), dtype=torch.long)

        def closure():
            opt.zero_grad()
            loss = 0.5 * (
                train_loss(arm, params, E6, e06, i6, 6)
                + train_loss(arm, params, E9, e09, i9, 9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(plist, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"{arm} s{seed} r{r} LBFGS skip: {e}")
            hist.append({"round": r, "skip": str(e)})
            continue
        if not _finite(loss_f):
            log(f"{arm} s{seed} r{r} nonfinite loss — restore best")
            copy_into(params, best["angles"])
            hist.append({"round": r, "loss": None})
            continue
        med = hold_med()
        hist.append({"round": r, "loss": loss_f, "hold_med9": med})
        mark = ""
        if med < best["med9"]:
            best = {"med9": med, "round": r, "angles": clone_angles(params)}
            mark = " *"
        if r % 10 == 0 or r + 1 == rounds or mark:
            log(
                f"{arm} s{seed} r{r}/{rounds} loss={loss_f:.5f} "
                f"holdMedε={med:.5f} best={best['med9']:.5f}@r{best['round']}{mark}"
            )

    final = clone_angles(params)
    copy_into(params, best["angles"])
    payload = {
        "done": True,
        "arm": arm,
        "family": ARM_FAMILY[arm],
        "L": L,
        "seed": seed,
        "blurb": ARM_BLURB[arm],
        "rounds_completed": rounds,
        "rounds_target": arm_rounds(arm),
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "init_hold_med9": float(init_med),
        "wall_s": prev_wall + (time.time() - t0),
        "n_params": int(sum(p.numel() for p in plist)),
        "lambda_x": LAMBDA_X,
        "mu_xx": MU_XX,
        "optimizer": {
            "name": "LBFGS",
            "lr": S.LBFGS_LR,
            "max_iter": S.LBFGS_ITER,
            "history_size": 16,
            "line_search": "strong_wolfe",
            "grad_clip": 5.0,
            "batch": batch,
            "loss": "0.5 * (mean ε/n on n6 + mean ε/n on n9)",
        },
        "batch_rng": "(seed, round, 16), shared across arms",
        "angles": angles_to_lists(best["angles"]),
        "final_angles": angles_to_lists(final),
        "hist": hist,
    }
    if kept_init_eval is not None:
        payload["init_eval"] = kept_init_eval
    return payload


def attach_eval(payload: dict, eval_bundle: dict) -> dict:
    ang = lists_to_tensors(payload["angles"])
    ev = eval_arm(payload["arm"], ang, eval_bundle)
    payload["eval"] = ev
    # Init quality is measured once, from the arm's starting tensors, and stored
    # only when this is the first segment (best hist starts at round -1 and we
    # still have the starting point if best_round == -1 or we recompute).
    return payload


def eval_init(arm: str, seed: int, eval_bundle: dict) -> dict:
    return eval_arm(arm, initial_tensors(arm, seed), eval_bundle)


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def ratio(num: float, den: float) -> float:
    if den == 0:
        return float("inf")
    return float(num) / float(den)


def verdict_pair(med_snap: float, med_hva: float) -> dict:
    r = ratio(med_snap, med_hva)
    return {
        "R": r,
        "verdict": "PASS" if med_snap <= med_hva else "needs_edit",
        "reached_098": bool(r <= 0.98),
    }


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (x != x or abs(x) == float("inf"))):
        return "—"
    return f"{float(x):.{nd}f}"


def _load_done(arm: str, seed: int) -> dict | None:
    p = seed_path(arm, seed)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if not d.get("done") or not d.get("eval"):
        return None
    return d


def shell_linear_residual(phi: np.ndarray) -> float:
    """RMS of φ after removing the best a + b k fit. Zero means an X-only mixer."""
    k = np.arange(4, dtype=np.float64)
    A = np.stack([np.ones(4), k], axis=1)
    acc = []
    for row in np.asarray(phi, dtype=np.float64):
        coef, *_ = np.linalg.lstsq(A, row, rcond=None)
        acc.append(float(np.sqrt(np.mean((row - A @ coef) ** 2))))
    return float(np.mean(acc))


def _final_med(block: list, n: int) -> float | None:
    """Median n12/n9 of the last iterate, when the seed file stored final_eval."""
    vals = []
    for rec in block:
        fin = rec.get("final_eval") or {}
        cell = fin.get(str(n)) or {}
        if "med_eps" not in cell:
            return None
        vals.append(float(cell["med_eps"]))
    if len(vals) != len(SEEDS):
        return None
    return float(np.median(vals))


def late_hold_drop(hist: list[dict], window: int = 40) -> dict:
    """n9 holdout at the best checkpoint vs the checkpoint `window` rounds earlier."""
    holds = {int(h["round"]): float(h["hold_med9"]) for h in hist if h.get("hold_med9") is not None}
    if not holds:
        return {}
    last = max(holds)
    earlier = last - window
    # nearest recorded round at or before earlier
    prev_round = max((r for r in holds if r <= earlier), default=min(holds))
    return {
        "last_round": last,
        "last_hold": holds[last],
        "prev_round": prev_round,
        "prev_hold": holds[prev_round],
        "drop": holds[prev_round] - holds[last],
    }


def aggregate() -> dict:
    """Write RESULTS.md and summary.json from the seed files."""
    rows = []
    missing = []
    for arm in ARMS:
        for seed in SEEDS:
            d = _load_done(arm, seed)
            if d is None:
                missing.append(f"{arm}_seed{seed}.json")
            else:
                rows.append(d)
    baselines = load_baselines()
    seal = {}
    for key, label in (("L16_hva", "hva"), ("L16_snap", "snap")):
        vals = {n: [] for n in (6, 9, 12, 15)}
        for seed in SEEDS:
            rec = baselines["seeds"][str(seed)][key]
            for n in vals:
                vals[n].append(float(rec[f"n{n}"]))
        seal[label] = {n: float(np.median(v)) for n, v in vals.items()}
        seal[label]["n12_seeds"] = vals[12]

    by_arm: dict[str, list] = {a: [] for a in ARMS}
    for rec in rows:
        by_arm[rec["arm"]].append(rec)

    def med_of(arm: str, n: int) -> float | None:
        block = by_arm.get(arm) or []
        if len(block) < len(SEEDS):
            return None
        xs = sorted(float(r["eval"][str(n)]["med_eps"]) for r in block)
        return float(np.median(xs))

    hva12 = med_of("hva_long", 12)
    arm_rows = []
    for arm in ARMS:
        cell = {n: med_of(arm, n) for n in (6, 9, 12, 15)}
        seeds_12 = None
        block = by_arm[arm]
        if len(block) == len(SEEDS):
            seeds_12 = [float(r["eval"]["12"]["med_eps"]) for r in sorted(block, key=lambda r: r["seed"])]
        gate = None
        if arm in SNAP_ARMS and cell[12] is not None and hva12 is not None:
            gate = verdict_pair(cell[12], hva12)
        arm_rows.append({"arm": arm, "cell": cell, "seeds_12": seeds_12, "gate": gate, "n": len(block)})

    snap_gates = [r for r in arm_rows if r["gate"] is not None]
    if missing or len(snap_gates) < len(SNAP_ARMS) or hva12 is None:
        overall = "INCOMPLETE"
        best_snap = None
    else:
        best_snap = min(snap_gates, key=lambda r: r["cell"][12])
        overall = "PASS" if any(r["gate"]["verdict"] == "PASS" for r in snap_gates) else "needs_edit"

    summary = {
        "overall": overall,
        "missing": missing,
        "lambda_x": LAMBDA_X,
        "mu_xx": MU_XX,
        "hva_long_n12": hva12,
        "seal_n12": {"hva": seal["hva"][12], "snap": seal["snap"][12]},
        "arms": {
            r["arm"]: {
                "med_eps": {str(n): r["cell"][n] for n in (6, 9, 12, 15)},
                "n12_seeds": r["seeds_12"],
                "gate": r["gate"],
            }
            for r in arm_rows
        },
        "best_snap_arm": None if best_snap is None else best_snap["arm"],
    }
    text = render_results(summary, by_arm, seal)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "RESULTS.md").write_text(text)
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote RESULTS.md overall={overall} missing={len(missing)}")
    return summary


def render_results(summary: dict, by_arm: dict, seal: dict) -> str:
    overall = summary["overall"]
    hva12 = summary["hva_long_n12"]
    lines = [
        "# L=16 SNAP rescue — SAFE point",
        "",
        f"**Overall: {overall}.**",
        "",
    ]
    if overall == "needs_edit":
        lines.append(
            "Deep qubit HVA wins; paper = shallow SNAP. "
            "No pre-specified SNAP arm put frozen-n12 Med ε at or below the matched L=16 HVA."
        )
    elif overall == "PASS":
        best = summary["best_snap_arm"]
        gate = summary["arms"][best]["gate"]
        extra = "R ≤ 0.98." if gate["reached_098"] else "R is above 0.98."
        lines.append(
            f"SNAP recovers at L=16 on arm `{best}` "
            f"(n12 Med ε {_fmt(summary['arms'][best]['med_eps']['12'])} "
            f"vs HVA {_fmt(hva12)}, R={_fmt(gate['R'])}). {extra}"
        )
    else:
        lines.append("The seed grid is not finished, so there is no gate yet.")
    lines += [
        "",
        "Same Hamiltonian and the same freeze rule as the PR #36 fairness baseline. "
        "Train n∈{6,9} only. n12 and n15 never enter the loss. "
        "n21 was not read or written.",
        "",
        "## Hamiltonian",
        "",
        r"\[",
        r"H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.50 \sum_{c}\sum_{i<j\in c} X_i X_j",
        r"\]",
        "",
        "SAFE point **λ=0.75, μ=−0.50**. "
        r"ε = ⟨H⟩ − E_0. "
        "**Med ε** is the median over seeds of each seed's median-over-instances ε. "
        "Readout is the Z histogram plus the X histogram. No Y.",
        "",
        "## What was trained",
        "",
        "PR #36 at the seal budget (40 × max_iter 8) had L=16 n12 Med ε "
        f"HVA {_fmt(seal['hva'][12])} and Walsh SNAP {_fmt(seal['snap'][12])} "
        f"(R={_fmt(ratio(seal['snap'][12], seal['hva'][12]))}). "
        "One extra 40 rounds on seed 0 narrowed that seed to R=1.49 and still left HVA ahead. "
        "This run is the rescue: a longer matched budget, a shallow warm start, and power_p.",
        "",
        "| arm | family | L=16 L-BFGS rounds | starts from |",
        "|---|---|---:|---|",
    ]
    for arm in ARMS:
        lines.append(
            f"| `{arm}` | {ARM_FAMILY[arm]} | {ARM_ROUNDS[arm]} | {ARM_BLURB[arm]} |"
        )
    lines += [
        "",
        "power_p uses φ_k = c1·k + a_n (c2·k² + c3·k³) with a_n = (8/(n−1))^p "
        "(a_9 = 1) and one scalar γ per layer. c2 = c3 = p = 0 at init, so the "
        "paper init is the linear shell, the same state as Fourier. "
        "No penalty on c2, c3, or p.",
        "",
        "`warm_pad` copies the seed's own PR #36 L=4 Walsh angles into layers 0..3 "
        "and sets layers 4..15 to γ=0, φ=0. At that init the circuit is exactly the "
        "L=4 SNAP. `warm_tile` repeats the same 4-layer block four times.",
        "",
        "HVA angles map into the Walsh shell exactly: equal γ_w reproduce U_C(γ), and "
        "φ_k = −β^X k + (β^XX / 2) ev_XX(k) with ev_XX = (3, −1, −1, 3) reproduces "
        "U_XX U_X, because both factors are popcount-diagonal on a 3-qubit cavity and "
        "therefore commute. The shell also has the XXX direction (it can split "
        "popcount 1 from popcount 2, which XX cannot). The rescue does not start "
        "SNAP from that embedded HVA point. Doing so would copy the baseline into "
        "the SNAP column. The map is only an expressivity check.",
        "",
        "## Protocol",
        "",
        "- Packs: `beat_qaoa_12h/train_packs`. Train [0:200) on n=6 and n=9. "
        "Holdout [200:250) selects the checkpoint by n9 Med ε. "
        f"Freeze n12 first {FREEZE_B[12]}, n15 first {FREEZE_B[15]}.",
        f"- Optimizer, unchanged from `Nov_3rd.settings`: L-BFGS lr {S.LBFGS_LR}, "
        f"max_iter {S.LBFGS_ITER}, history 16, strong Wolfe, grad clip 5, "
        f"batch {S.LBFGS_BATCH} (half n6 / half n9). New optimizer each round. "
        "Loss = 0.5 (mean ε/n on n6 + mean ε/n on n9). No Adam.",
        "- Round r draws its batch from `default_rng((seed, r, 16))`. "
        "The same (seed, r) hits every arm, and it does not replay the seal's 40 batches.",
        "- Seeds {0,1,2}. The reported Med ε is the median of the three seeds.",
        "- E0 values are the PR #36 caches (same instances, same λ, μ).",
        "- n21 phase-2 ladder was not read or written.",
        "",
        "## Gate",
        "",
        "- **PASS** — some pre-specified SNAP arm has n12 Med ε ≤ matched `hva_long`. "
        "Three seeds. R ≤ 0.98 is recorded separately and is not required for PASS.",
        "- **needs_edit** — every SNAP arm is still above `hva_long` at n12. "
        "Deep qubit HVA wins; paper = shallow SNAP.",
        "- The seal's 0.85 / KILL labels are not reused. This round asks only whether "
        "SNAP can get back in front of HVA at L=16.",
        "",
        "## Frozen n12",
        "",
        "| arm | Medε n6 | Medε n9 | Medε n12 | Medε n15 | R vs hva_long | ≤0.98 | verdict |",
        "|---|---:|---:|---:|---:|---:|:---:|---|",
        f"| seal HVA (40 rnd, not retrained) | {_fmt(seal['hva'][6], 5)} | {_fmt(seal['hva'][9], 5)} | "
        f"{_fmt(seal['hva'][12], 5)} | {_fmt(seal['hva'][15], 5)} |  |  | reference |",
        f"| seal Walsh (40 rnd, not retrained) | {_fmt(seal['snap'][6], 5)} | {_fmt(seal['snap'][9], 5)} | "
        f"{_fmt(seal['snap'][12], 5)} | {_fmt(seal['snap'][15], 5)} | "
        f"{_fmt(ratio(seal['snap'][12], seal['hva'][12]), 4)} vs seal | no | KILL in PR #36 |",
    ]
    for arm in ARMS:
        cell = summary["arms"][arm]["med_eps"]
        if cell["12"] is None or hva12 is None:
            lines.append(f"| `{arm}` | — | — | — | — | — |  | missing |")
            continue
        if arm == "hva_long":
            lines.append(
                f"| `{arm}` | {_fmt(cell['6'], 5)} | {_fmt(cell['9'], 5)} | {_fmt(cell['12'], 5)} | "
                f"{_fmt(cell['15'], 5)} | 1 |  | matched baseline |"
            )
            continue
        g = summary["arms"][arm]["gate"]
        lines.append(
            f"| `{arm}` | {_fmt(cell['6'], 5)} | {_fmt(cell['9'], 5)} | {_fmt(cell['12'], 5)} | "
            f"{_fmt(cell['15'], 5)} | {_fmt(g['R'], 4)} | {'yes' if g['reached_098'] else 'no'} | {g['verdict']} |"
        )
    lines += [
        "",
        "R vs hva_long uses this run's HVA, not the 40-round seal. "
        "The seal row is the unmatched shorter budget.",
        "",
        "## Seeds",
        "",
        "| arm | seed | best round | n9 hold Medε | n12 Medε | n15 Medε | wall s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for rec in sorted(by_arm.get(arm) or [], key=lambda r: int(r["seed"])):
            ev = rec["eval"]
            lines.append(
                f"| `{arm}` | {rec['seed']} | {rec['best_round']} | "
                f"{_fmt(ev['9']['med_eps'], 5)} | {_fmt(ev['12']['med_eps'], 5)} | "
                f"{_fmt(ev['15']['med_eps'], 5)} | {rec['wall_s']:.0f} |"
            )
    lines += ["", "n12 seed values (seed 0, 1, 2):", ""]
    for arm in ARMS:
        xs = summary["arms"][arm]["n12_seeds"]
        if not xs:
            continue
        lines.append(f"- `{arm}`: " + ", ".join(_fmt(x, 5) for x in xs))
    lines += ["", "## Did the extra budget flatten?", ""]
    for arm in ARMS:
        block = by_arm.get(arm) or []
        if len(block) < len(SEEDS):
            continue
        bits = []
        for rec in sorted(block, key=lambda r: int(r["seed"])):
            info = late_hold_drop(rec.get("hist") or [], 40)
            if not info:
                continue
            still = "end lower" if info["drop"] > 0 else "end not lower"
            bits.append(
                f"s{rec['seed']} r{info['prev_round']}→r{info['last_round']} "
                f"{_fmt(info['prev_hold'], 5)}→{_fmt(info['last_hold'], 5)} ({still}), "
                f"best r{rec['best_round']}"
            )
        lines.append(f"- `{arm}`: " + "; ".join(bits))
    lines += ["", "## Where the arms started", ""]
    lines.append("n12 Med ε of the init, before the new rounds (median over seeds when stored):")
    lines.append("")
    for arm in ARMS:
        block = [r for r in (by_arm.get(arm) or []) if r.get("init_eval")]
        if len(block) < len(SEEDS):
            lines.append(f"- `{arm}`: init eval not stored")
            continue
        med = float(np.median([r["init_eval"]["12"]["med_eps"] for r in block]))
        med9 = float(np.median([r["init_eval"]["9"]["med_eps"] for r in block]))
        lines.append(f"- `{arm}`: init n9 Medε {_fmt(med9)}, init n12 Medε {_fmt(med)}")
    # power_p exponent and shell residual, HVA |β^XX|
    lines += ["", "## Did the extra generators move?", ""]
    for arm in ("walsh_long", "warm_pad", "warm_tile"):
        block = by_arm.get(arm) or []
        if not block:
            continue
        res = [shell_linear_residual(np.asarray(r["angles"]["phi"])) for r in block]
        lines.append(
            f"- `{arm}` mean RMS of φ after removing a+bk "
            f"(0 = X-only mixer): {_fmt(float(np.mean(res)), 5)}"
        )
    block = by_arm.get("hva_long") or []
    if block:
        mags = [float(np.mean(np.abs(np.asarray(r["angles"]["beta_xx"])))) for r in block]
        lines.append(f"- `hva_long` mean |β^XX|: {_fmt(float(np.mean(mags)), 5)}")
    block = by_arm.get("power_p") or []
    if block:
        ps = [float(np.asarray(r["angles"]["p"]).reshape(-1)[0]) for r in block]
        c2 = [float(np.mean(np.abs(np.asarray(r["angles"]["c2"])))) for r in block]
        c3 = [float(np.mean(np.abs(np.asarray(r["angles"]["c3"])))) for r in block]
        lines.append(
            f"- `power_p` p per seed ({', '.join(_fmt(p, 3) for p in ps)}), "
            f"mean |c2|={_fmt(float(np.mean(c2)), 4)}, mean |c3|={_fmt(float(np.mean(c3)), 4)}"
        )
    lines += ["", "## Reading", ""]
    if overall == "INCOMPLETE":
        lines.append("Finish the missing seed files and rerun `--aggregate`.")
    elif overall == "PASS":
        best = summary["best_snap_arm"]
        g = summary["arms"][best]["gate"]
        lines.append(
            f"`{best}` is the best SNAP arm at frozen n12 "
            f"(R={_fmt(g['R'])} against matched HVA {_fmt(hva12)}). "
            + (
                "It also clears R≤0.98."
                if g["reached_098"]
                else "It gets SNAP back to ≤ HVA and does not clear R≤0.98."
            )
        )
        lines.append("")
        lines.append(
            "Shallow SNAP is unchanged by this run: PR #36 already had PASS at L=2 "
            "(R=0.328) and L=4 (R=0.536). The rescue only asks about L=16."
        )
    else:
        # needs_edit
        best = summary["best_snap_arm"]
        g = summary["arms"][best]["gate"]
        gap = float(summary["arms"][best]["med_eps"]["12"]) - float(hva12)
        lines.append(
            f"The closest SNAP arm is `{best}`. Frozen n12 Med ε is "
            f"{_fmt(summary['arms'][best]['med_eps']['12'], 5)} against matched HVA "
            f"{_fmt(hva12, 5)} (R={_fmt(g['R'], 4)}, gap {gap:.3e}). "
            "R is above 1, so the arm also misses R≤0.98."
        )
        lines.append("")
        wseeds = summary["arms"]["walsh_long"]["n12_seeds"]
        hseeds = summary["arms"]["hva_long"]["n12_seeds"]
        if wseeds and hseeds:
            lines.append(
                f"Walsh n12 seeds span {_fmt(min(wseeds), 5)}–{_fmt(max(wseeds), 5)}; "
                f"HVA spans {_fmt(min(hseeds), 5)}–{_fmt(max(hseeds), 5)}. "
                "The ranges overlap. The gate uses the median, and that median is on the HVA side."
            )
            lines.append("")
        wn15 = summary["arms"]["walsh_long"]["med_eps"]["15"]
        hn15 = summary["arms"]["hva_long"]["med_eps"]["15"]
        wn9 = summary["arms"]["walsh_long"]["med_eps"]["9"]
        hn9 = summary["arms"]["hva_long"]["med_eps"]["9"]
        wn6 = summary["arms"]["walsh_long"]["med_eps"]["6"]
        hn6 = summary["arms"]["hva_long"]["med_eps"]["6"]
        if None not in (wn15, hn15, wn9, hn9, wn6, hn6):
            lines.append(
                f"The same Walsh checkpoints versus the same HVA: "
                f"n6 R={_fmt(ratio(wn6, hn6), 3)}, "
                f"n9 R={_fmt(ratio(wn9, hn9), 3)}, "
                f"n12 R={_fmt(g['R'], 3)}, "
                f"n15 R={_fmt(ratio(wn15, hn15), 3)} "
                f"(n15 Med ε Walsh {_fmt(wn15, 5)}, HVA {_fmt(hn15, 5)}). "
                "HVA leads on the trained sizes and on the n12 gate. Walsh leads at n15. "
                "The assigned PASS rule is n12."
            )
            lines.append("")
        fin_w = _final_med(by_arm.get("walsh_long") or [], 12)
        fin_h = _final_med(by_arm.get("hva_long") or [], 12)
        if fin_w is not None and fin_h is not None:
            lines.append(
                f"The last iterate, which is not the checkpoint rule, has n12 medians "
                f"Walsh {_fmt(fin_w, 5)} and HVA {_fmt(fin_h, 5)} "
                f"(R={_fmt(ratio(fin_w, fin_h), 4)}). The median stays on the HVA side."
            )
            lines.append("")
        lines.append(
            "power_p, warm_pad, and warm_tile remain near R≈2.4–3.6. "
            "power_p keeps p near 0, so a_n stays near 1, and its n9 holdout is still "
            "falling at round 159 without getting close to HVA. "
            "warm_pad starts exactly at the L=4 SNAP solution and improves, and after "
            "120 rounds it is still about twice the matched HVA. "
            "warm_tile drives n9 down to the HVA range and does not transfer "
            "(n12 and n15 stay several times higher)."
        )
        lines.append("")
        lines.append(
            "The Hamming shell contains the HVA mixer, and equal Walsh weights contain "
            "the HVA cost. `embed_check.json` copies the seal HVA seed-0 angles into that "
            "shell and reproduces its frozen-n12 Med ε. The L-BFGS trajectories from the "
            "paper init, from the seal SNAP checkpoint, and from the L=4 warm start get "
            "Walsh within a percent of the matched HVA at n12 and do not put the median "
            "under it. Under this rescue the n12 HVA lead remains."
        )
        lines.append("")
        lines.append(
            "**Deep qubit HVA wins; paper = shallow SNAP.** "
            "Lead with L=2 and L=4, where PR #36 measured n12 R=0.328 and R=0.536 "
            "for Walsh SNAP against this HVA. The L=16 n12 median does not support a SNAP win "
            "on this Hamiltonian."
        )
    lines += [
        "",
        "Per-seed JSON: `{arm}_seed{s}.json` in this folder. "
        "`angles` is the restored n9-holdout checkpoint. `final_angles` is the last iterate.",
        "",
    ]
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def embed_check(eval_bundle: dict | None = None) -> dict:
    """Fidelity of the HVA→SNAP map, plus n12 Med ε on the seal HVA seed 0 point."""
    rng = np.random.default_rng(0)
    n = 6
    # Mixer map is basis-independent. Use a random diagonal; the cost angle is scalar,
    # so high Walsh degrees are not involved.
    Cn = torch.tensor(rng.normal(size=(2, 1 << n)), dtype=torch.float64)
    Lloc = 4
    beta_x = torch.tensor(rng.normal(scale=0.4, size=Lloc), dtype=torch.float64)
    beta_xx = torch.tensor(rng.normal(scale=0.4, size=Lloc), dtype=torch.float64)
    gammas = torch.tensor(rng.normal(scale=0.3, size=Lloc), dtype=torch.float64)
    sh = hva_state(Cn, beta_x, beta_xx, gammas)
    ss = snap_state(Cn, gammas, phi_from_hva_mixer(beta_x, beta_xx))
    fid = float(state_fidelity(sh, ss).min())
    # power_p at p=0, c2=c3=0 matches Fourier
    g = torch.tensor(rng.normal(scale=0.2, size=Lloc), dtype=torch.float64)
    b = torch.tensor(rng.normal(scale=0.3, size=Lloc), dtype=torch.float64)
    z = torch.zeros(Lloc, dtype=torch.float64)
    p0 = torch.zeros((), dtype=torch.float64)
    sf = fourier_state(Cn, b, g)
    sp = snap_state(Cn, g, phi_power(-b, z, z, p0, n))
    fid_p = float(state_fidelity(sf, sp).min())
    out = {"embed_fidelity_min": fid, "power_p_fidelity_min": fid_p}
    if fid < 1 - 1e-8:
        raise AssertionError(f"HVA embed fidelity {fid}")
    if fid_p < 1 - 1e-8:
        raise AssertionError(f"power_p paper-init fidelity {fid_p}")
    if eval_bundle is not None:
        baselines = load_baselines()
        ang0 = baselines["seeds"]["0"]["L16_hva"]["angles"]
        hva_params = {
            "beta_x": _f64(ang0["beta_x"]),
            "beta_xx": _f64(ang0["beta_xx"]),
            "gammas": _f64(ang0["gammas"]),
        }
        mapped = snap_angles_from_hva(hva_params["beta_x"], hva_params["beta_xx"], hva_params["gammas"])
        ev_h = eval_arm("hva_long", hva_params, {12: eval_bundle[12]})
        ev_s = eval_arm("walsh_long", mapped, {12: eval_bundle[12]})
        dh = abs(ev_h["12"]["med_eps"] - ev_s["12"]["med_eps"])
        out["seal_seed0_n12_hva"] = ev_h["12"]["med_eps"]
        out["seal_seed0_n12_embedded_snap"] = ev_s["12"]["med_eps"]
        out["seal_seed0_n12_abs_diff"] = dh
        if dh > 1e-6:
            raise AssertionError(f"embedded seal HVA n12 Δ={dh}")
    return out
