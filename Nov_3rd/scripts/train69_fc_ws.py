#!/usr/bin/env python3
"""FC-WS smoke: footprint-class Walsh × commutator-matched shell (Opus Idea 1).

Graded H_c. Train only n∈{6,9}. Eval CORE {6,9,12,15}. Does not touch power_p
or Walsh-degree champ paths.

Arms
  champ       frozen schedule, δ=0, κ=0 (recovery check)
  walsh       published Walsh-degree γ_w, δ=0, κ=0
  footprint   learn δ, κ fixed at 0
  joint       learn δ and κ

Warm starts (schedule frozen; only the globals move)
  champ       L=2 pareto a0.80 s1 (scalar γ + shell; p=0 ≡ power_p at p=0)
              L=16 sealed power_p seed0
  walsh       L=2 Idea1 γ_w seed1. Base γ is (L,4); δ=0 recovers that circuit.

δ=0, κ=0 recovers the warm schedule. δ_C = δ_degree(C), κ=0 is the
degree-tied Walsh special case of a scalar base γ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCR = Path(__file__).resolve().parent
for p in (str(_ROOT), str(_SCR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, load_split_E, log
from Nov_3rd.fc_ws import CLASS_NAME, effective_delta
from beat_params import phi_from_c
from train69_ucx_walsh_degree import load_train_n, precompute_qaoa_inv

OUT_ROOT = S.OUT / "beat_qaoa_12h" / "train69" / "fc_ws"
TRAIN_NS = [6, 9]
EVAL_NS = [6, 9, 12, 15]
L2_CHAMP = S.OUT / "beat_qaoa_12h" / "train69" / "champion" / "params_L2_train69_pareto_a080_s1.pt"
L2_WALSH = (
    S.OUT / "beat_qaoa_12h" / "train69" / "ucx_bit_a" / "walsh_degree" / "params_L2_seed1_B100.pt"
)
L16_CHAMP = S.OUT / "beat_qaoa_12h" / "champion" / "params_L16_train69_power_p.pt"
WALSH_DIR = S.OUT / "beat_qaoa_12h" / "train69" / "ucx_bit_a" / "walsh_degree"

DELTA_LO, DELTA_HI = -0.9, 3.0
KAPPA_LO, KAPPA_HI = -3.0, 3.0


def a_of_n(n: int, p) -> torch.Tensor | float:
    if p is None:
        return 1.0
    base = 8.0 / float(n - 1)
    if torch.is_tensor(p):
        return torch.pow(torch.tensor(base, device=p.device, dtype=p.dtype), p)
    return float(base) ** float(p)


def phi_for_n(c1, c2, c3, p, n: int) -> torch.Tensor:
    a = a_of_n(n, p)
    if torch.is_tensor(a):
        return phi_from_c(c1, c2 * a, c3 * a)
    return phi_from_c(c1, c2 * float(a), c3 * float(a))


def load_schedule(path: Path, device):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "gamma_w" in d and torch.is_tensor(d["gamma_w"]):
        gamma = d["gamma_w"].float()
        graded = True
    else:
        gamma = d["gamma"].float()
        graded = False
    c1 = d["c1"].float()
    c2 = d["c2"].float()
    c3 = d["c3"].float()
    p = d.get("p", None)
    if p is not None:
        p = float(p)
    return {
        "gamma": gamma.to(device),
        "c1": c1.to(device),
        "c2": c2.to(device),
        "c3": c3.to(device),
        "p": p,
        "graded_base": graded,
        "path": str(path),
        "idea": d.get("idea"),
        "raw": d,
    }


def resolve_warm(L: int, warm: str) -> Path:
    if L == 16:
        if warm != "champ":
            raise SystemExit("L=16 smoke freezes the sealed power_p champ only")
        return L16_CHAMP
    if warm == "champ":
        return L2_CHAMP
    if warm == "walsh":
        return L2_WALSH
    raise SystemExit(f"unknown warm {warm}")


def run_dir(root: Path, L: int, warm: str, arm: str, pool: bool, seed) -> Path:
    return root / f"L{L}" / f"warm_{warm}" / f"arm_{arm}" / f"pool{int(pool)}" / f"seed{seed}"


def fcws_dict(delta, kappa, pool: bool, parts=None) -> dict:
    d = {"delta": delta, "kappa": kappa, "pool_111": pool}
    if parts is not None:
        d["parts"] = parts
    return d


def precompute_parts(sim, E: torch.Tensor) -> list:
    out = []
    for i in range(len(E)):
        out.append(sim.footprint_parts(E[i]))
    return out


def mean_mass(parts_list) -> list[float]:
    w = torch.stack([p["w"] for p in parts_list], dim=0)
    return w.mean(dim=0).tolist()


@torch.no_grad()
def eval_n_B(n, L, gamma, phi, delta, kappa, pool, device, *, max_count: int, save_inv: bool):
    E, key = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
    sim = make_sim(n, L, device)
    invs, leaks = [], []
    t0 = time.time()
    zero_fc = float(delta.detach().abs().sum()) == 0.0 and float(kappa.detach().abs()) == 0.0
    for i in range(len(E)):
        if zero_fc and gamma.ndim == 1:
            out = sim.forward_nov3rd(E[i], gamma, phi)
        elif zero_fc and gamma.ndim == 2:
            out = sim.forward_nov3rd(E[i], gamma, phi)
        else:
            out = sim.forward_nov3rd(
                E[i], gamma, phi, fcws=fcws_dict(delta, kappa, pool)
            )
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs, dtype=np.float64)
    row = {
        "n": n,
        "split": key,
        "B": int(len(E)),
        "median_inv_p": float(np.median(invs_a)),
        "B90_inv_p": float(np.quantile(invs_a, 0.90)),
        "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0,
    }
    if save_inv:
        row["inv"] = invs_a.tolist()
    return row


@torch.no_grad()
def eval_qaoa_med(n, L, betas, gammas, device, max_count: int) -> float:
    E, _ = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
    inv = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
    return float(inv.median().cpu())


def instance_logs(sim, E, inv_q, gamma, phi, delta, kappa, pool, parts, idx):
    ls = []
    for j in idx:
        out = sim.forward_nov3rd(
            E[int(j)], gamma, phi, fcws=fcws_dict(delta, kappa, pool, parts[int(j)])
        )
        ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
        ls.append(torch.log(ratio.clamp(min=1e-12)))
    return torch.stack(ls)


def batch_loss(bundles, gamma, phis, delta, kappa, pool, idx_map, weights):
    terms = []
    wsum = 0.0
    for n, bun in bundles.items():
        E, inv_q, sim, parts = bun
        logs = instance_logs(sim, E, inv_q, gamma, phis[n], delta, kappa, pool, parts, idx_map[n])
        w = float(weights.get(n, 1.0))
        terms.append(w * logs.mean())
        wsum += w
    return torch.stack(terms).sum() / max(wsum, 1e-12)


def holdout_score(bun9, gamma, phi9, delta, kappa, pool, hold_idx) -> float:
    E, inv_q, sim, parts = bun9
    logs = instance_logs(sim, E, inv_q, gamma, phi9, delta, kappa, pool, parts, hold_idx)
    return float(logs.median().detach())


def clip_globals(delta, kappa):
    hit = False
    with torch.no_grad():
        if bool((delta < DELTA_LO).any() or (delta > DELTA_HI).any()):
            hit = True
        delta.clamp_(DELTA_LO, DELTA_HI)
        if kappa is not None and torch.is_tensor(kappa) and kappa.requires_grad:
            if float(kappa) < KAPPA_LO or float(kappa) > KAPPA_HI:
                hit = True
            kappa.clamp_(KAPPA_LO, KAPPA_HI)
    return hit


def train_one(args, device):
    L = int(args.L)
    warm_path = resolve_warm(L, args.warm)
    sched = load_schedule(warm_path, device)
    gamma = sched["gamma"]
    c1, c2, c3 = sched["c1"], sched["c2"], sched["c3"]
    p = sched["p"]
    if tuple(gamma.shape)[0] != L:
        raise SystemExit(f"schedule L {gamma.shape} != {L}")

    pool = bool(args.pool_111)
    n_delta = 5 if pool else 6
    delta = torch.nn.Parameter(torch.zeros(n_delta, device=device))
    learn_k = args.arm == "joint"
    if learn_k:
        kappa = torch.nn.Parameter(torch.zeros((), device=device))
        trainables = [delta, kappa]
    else:
        kappa = torch.zeros((), device=device)
        trainables = [delta]
    if args.arm in ("champ", "walsh"):
        raise SystemExit("champ/walsh are eval-only")

    betas, gammas = load_qaoa(L, )
    # pareto champ has no stored betas; angles file is the QAOA reference
    phis = {n: phi_for_n(c1, c2, c3, p, n).detach() for n in TRAIN_NS}
    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(n, max_count=args.max_train).to(device)
        log(f"fc_ws L{L} {args.arm} warm={args.warm} s{args.seed}: train n{n} B={len(E)}")
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device).to(device)
        sim = make_sim(n, L, device)
        parts = precompute_parts(sim, E)
        bundles[n] = (E, inv_q, sim, parts)
        log(f"  mean w_C n{n}: " + ", ".join(f"{CLASS_NAME[i]}={m:.4f}" for i, m in enumerate(mean_mass(parts))))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    n9_B = len(bundles[9][0])
    perm = rng.permutation(n9_B)
    n_hold = max(1, int(round(args.holdout_frac * n9_B)))
    hold_idx = perm[:n_hold]
    train_idx9 = perm[n_hold:]
    weights = {6: float(args.w6), 9: float(args.w9)}

    def snap():
        return {
            "delta": delta.detach().cpu().clone(),
            "kappa": float(kappa.detach().cpu()) if learn_k else 0.0,
        }

    best = {"score": float("inf"), "snap": snap(), "step": "init"}
    hist = []
    t0 = time.time()
    clipped = False

    def maybe_best(tag):
        sc = holdout_score(bundles[9], gamma, phis[9], delta, kappa, pool, hold_idx)
        hist.append({"phase": "holdout", "tag": tag, "hold_med_logR": sc})
        if sc < best["score"]:
            best["score"] = sc
            best["snap"] = snap()
            best["step"] = tag
            log(f"fc_ws L{L} {args.arm} s{args.seed} NEW BEST hold={sc:.5f} @ {tag} κ={float(kappa.detach()):.4f}")

    opt = torch.optim.Adam(trainables, lr=args.adam_lr)
    for step in range(args.adam_steps):
        idx_map = {}
        for n in TRAIN_NS:
            pool_idx = train_idx9 if n == 9 else np.arange(len(bundles[n][0]))
            take = min(args.batch_per_n, len(pool_idx))
            idx_map[n] = rng.choice(pool_idx, size=take, replace=False)
        opt.zero_grad()
        loss = batch_loss(bundles, gamma, phis, delta, kappa, pool, idx_map, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        clipped = clip_globals(delta, kappa) or clipped
        if step % 25 == 0 or step == args.adam_steps - 1:
            de = effective_delta(delta.detach(), pool).tolist()
            log(
                f"fc_ws L{L} {args.arm} warm={args.warm} s{args.seed} adam {step}/{args.adam_steps} "
                f"loss={float(loss.detach()):.5f} κ={float(kappa.detach()):.4f} δ={['%.3f'%x for x in de]}"
            )
            maybe_best(f"adam{step}")
        hist.append({"phase": "adam", "step": step, "loss": float(loss.detach())})

    for r in range(args.lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            trainables, lr=0.5, max_iter=args.lbfgs_iter, history_size=10, line_search_fn="strong_wolfe"
        )
        idx_map = {}
        for n in TRAIN_NS:
            pool_idx = train_idx9 if n == 9 else np.arange(len(bundles[n][0]))
            take = min(args.batch_per_n, len(pool_idx))
            idx_map[n] = rng.choice(pool_idx, size=take, replace=False)

        def closure():
            opt_l.zero_grad()
            lt = batch_loss(bundles, gamma, phis, delta, kappa, pool, idx_map, weights)
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt

        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"fc_ws LBFGS r{r} skip: {e}")
            continue
        clipped = clip_globals(delta, kappa) or clipped
        hist.append({"phase": "lbfgs", "round": r, "loss": loss_f})
        log(f"fc_ws L{L} {args.arm} s{args.seed} lbfgs r{r} loss={loss_f:.5f} κ={float(kappa.detach()):.4f}")
        maybe_best(f"lbfgs{r}")

    with torch.no_grad():
        delta.copy_(best["snap"]["delta"].to(device))
        if learn_k:
            kappa.fill_(best["snap"]["kappa"])

    state = {
        "idea": "fc_ws",
        "label": "graded_Hc_footprint_class",
        "arm": args.arm,
        "warm": args.warm,
        "warm_path": str(warm_path),
        "L": L,
        "seed": int(args.seed),
        "pool_111": pool,
        "delta": delta.detach().cpu(),
        "kappa": float(kappa.detach().cpu()),
        "gamma": gamma.detach().cpu(),
        "c1": c1.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "p": p,
        "graded_base": bool(sched["graded_base"]),
        "holdout_best": best["score"],
        "holdout_best_step": best["step"],
        "clipped": clipped,
        "wall_s": time.time() - t0,
        "max_train": args.max_train,
        "weights": weights,
        "class_names": list(CLASS_NAME),
        "hist_tail": hist[-30:],
        "betas_qaoa": betas.cpu(),
        "gammas_qaoa": gammas.cpu(),
    }
    out = run_dir(Path(args.out_dir), L, args.warm, args.arm, pool, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    pt = out / "params.pt"
    torch.save(state, pt)
    meta = {
        "arm": args.arm,
        "warm": args.warm,
        "L": L,
        "seed": int(args.seed),
        "pool_111": pool,
        "delta": effective_delta(delta.detach().cpu(), pool).tolist(),
        "delta_free": delta.detach().cpu().tolist(),
        "kappa": float(kappa.detach().cpu()),
        "holdout_best": best["score"],
        "holdout_best_step": best["step"],
        "clipped": clipped,
        "wall_s": state["wall_s"],
        "params": str(pt),
        "warm_path": str(warm_path),
        "p": p,
    }
    (out / "train.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"fc_ws saved {pt} wall={state['wall_s']:.0f}s κ={meta['kappa']:.4f}")
    return state, meta, pt


def _verdict(R: float) -> str:
    if R != R:
        return "NO_REF"
    if R < 1.0:
        return "BEAT"  # raw Med < QAOA Med
    if R == 1.0:
        return "TIE"
    return "UNDER"


def eval_state(state, device, eval_B: int, save_inv: bool) -> dict:
    L = int(state["L"])
    gamma = state["gamma"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    p = state.get("p")
    if p is not None and not torch.is_tensor(p):
        p = float(p)
    pool = bool(state.get("pool_111", False))
    delta = state["delta"].to(device)
    kappa = torch.tensor(float(state["kappa"]), device=device)
    betas = state.get("betas_qaoa")
    gammas = state.get("gammas_qaoa")
    if betas is None:
        betas, gammas = load_qaoa(L)
    qref = load_qaoa_scale_ref(L) if eval_B >= 1000 else {}
    rows = {}
    for n in EVAL_NS:
        phi = phi_for_n(c1, c2, c3, p, n)
        m = eval_n_B(
            n, L, gamma, phi, delta, kappa, pool, device,
            max_count=eval_B, save_inv=save_inv,
        )
        if eval_B >= 1000 and n in qref and qref[n]:
            qmed = float(qref[n]["Med"])
        else:
            qmed = eval_qaoa_med(n, L, betas, gammas, device, eval_B)
        med = float(m["median_inv_p"])
        R = med / qmed if qmed > 0 else float("nan")
        rows[n] = {
            "Med": med,
            "B90": float(m["B90_inv_p"]),
            "QAOA_Med": qmed,
            "R": R,
            "verdict": _verdict(R),
            "mean_leak": float(m["mean_leak"]),
            "B": int(m["B"]),
            "eval_wall_s": float(m["wall_s"]),
        }
        if "inv" in m:
            rows[n]["inv"] = m["inv"]
        log(
            f"eval fc_ws L{L} n{n} B={m['B']}: Med={med:.4f} Q={qmed:.4f} R={R:.4f} {rows[n]['verdict']}"
        )
    return rows


def eval_only_schedule(args, device):
    """champ or walsh arm: δ=0, κ=0 on the named schedule."""
    L = int(args.L)
    if args.arm == "walsh":
        if L != 2:
            raise SystemExit("published Walsh-degree checkpoint is L=2")
        path = WALSH_DIR / f"params_L2_seed{args.seed}_B100.pt"
        if args.warm == "walsh" and int(args.seed) == 1 and not path.exists():
            path = L2_WALSH
    else:
        path = resolve_warm(L, "champ" if args.arm == "champ" else args.warm)
        if args.arm == "champ":
            path = resolve_warm(L, "champ")
    sched = load_schedule(path, device)
    betas, gammas = load_qaoa(L)
    state = {
        "idea": "fc_ws",
        "arm": args.arm,
        "warm": args.warm,
        "L": L,
        "seed": int(args.seed),
        "pool_111": bool(args.pool_111),
        "delta": torch.zeros(6),
        "kappa": 0.0,
        "gamma": sched["gamma"].cpu(),
        "c1": sched["c1"].cpu(),
        "c2": sched["c2"].cpu(),
        "c3": sched["c3"].cpu(),
        "p": sched["p"],
        "graded_base": sched["graded_base"],
        "warm_path": str(path),
        "betas_qaoa": betas.cpu(),
        "gammas_qaoa": gammas.cpu(),
    }
    return state


def write_eval(out: Path, state: dict, rows: dict, tag: str):
    # Drop bulky inv vectors from the markdown; keep them in json.
    payload = {
        "idea": "fc_ws",
        "arm": state.get("arm"),
        "warm": state.get("warm"),
        "L": state.get("L"),
        "seed": state.get("seed"),
        "pool_111": state.get("pool_111"),
        "kappa": state.get("kappa"),
        "delta": effective_delta(
            state["delta"] if torch.is_tensor(state["delta"]) else torch.tensor(state["delta"]),
            bool(state.get("pool_111", False)),
        ).tolist() if "delta" in state else None,
        "p": state.get("p"),
        "per_n": {},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    for n, r in rows.items():
        payload["per_n"][str(n)] = {k: v for k, v in r.items() if k != "inv"}
        if "inv" in r:
            payload["per_n"][str(n)]["inv"] = r["inv"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{tag}.json"
    path.write_text(json.dumps(payload) + "\n")
    lines = [
        f"# FC-WS arm={state.get('arm')} warm={state.get('warm')} L={state.get('L')} seed={state.get('seed')} {tag}",
        "",
        "| n | QAOA Med | Med | R | verdict | B |",
        "|---|----------|-----|---|---------|---|",
    ]
    for n in EVAL_NS:
        r = rows[n]
        lines.append(
            f"| {n} | {r['QAOA_Med']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | {r['verdict']} | {r['B']} |"
        )
    all_beat = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
    lines += ["", f"**ALL_CORE_BEAT** = {all_beat}", ""]
    (out / f"eval_{tag}.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {path} ALL_CORE_BEAT={all_beat}")
    return all_beat


def bootstrap_med_diff(inv_b, inv_c, n_boot=2000, seed=0):
    """Paired bootstrap of Med_b - Med_c. Positive ⇒ c has lower Med."""
    rng = np.random.default_rng(seed)
    b = np.asarray(inv_b, dtype=np.float64)
    c = np.asarray(inv_c, dtype=np.float64)
    n = len(b)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.median(b[idx]) - np.median(c[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "mean_diff": float(diffs.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "c_better_95": bool(lo > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "baseline"], default="train")
    ap.add_argument("--arm", choices=["footprint", "joint", "champ", "walsh"], default="joint")
    ap.add_argument("--warm", choices=["champ", "walsh"], default="champ")
    ap.add_argument("--L", type=int, default=2, choices=[2, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_111", action="store_true")
    ap.add_argument("--max_train", type=int, default=100)
    ap.add_argument("--eval_B", type=int, default=100)
    ap.add_argument("--adam_steps", type=int, default=400)
    ap.add_argument("--adam_lr", type=float, default=1e-2)
    ap.add_argument("--lbfgs_rounds", type=int, default=8)
    ap.add_argument("--lbfgs_iter", type=int, default=8)
    ap.add_argument("--batch_per_n", type=int, default=32)
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--save_inv", action="store_true")
    ap.add_argument("--out_dir", type=str, default=str(OUT_ROOT))
    ap.add_argument("--eval_pt", type=str, default="")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    if args.mode == "baseline" or args.arm in ("champ", "walsh"):
        state = eval_only_schedule(args, device)
        rows = eval_state(state, device, args.eval_B, args.save_inv)
        out = run_dir(Path(args.out_dir), args.L, args.warm if args.arm != "champ" else "champ", args.arm, False, args.seed)
        if args.arm == "walsh":
            out = Path(args.out_dir) / f"L{args.L}" / "baseline_walsh" / f"seed{args.seed}"
        write_eval(out, state, rows, tag=f"B{args.eval_B}")
        return 0

    if args.mode == "eval":
        pt = args.eval_pt
        if not pt:
            pt = str(run_dir(Path(args.out_dir), args.L, args.warm, args.arm, args.pool_111, args.seed) / "params.pt")
        state = torch.load(pt, map_location="cpu", weights_only=False)
        rows = eval_state(state, device, args.eval_B, args.save_inv)
        write_eval(Path(pt).parent, state, rows, tag=f"B{args.eval_B}")
        return 0

    state, meta, pt = train_one(args, device)
    rows = eval_state(state, device, args.eval_B, args.save_inv)
    write_eval(Path(pt).parent, state, rows, tag=f"B{args.eval_B}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
