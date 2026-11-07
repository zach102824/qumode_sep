#!/usr/bin/env python3
"""Train FM-WS (Opus Idea 3) on train{6,9} and eval CORE.

Warm start is the full DST/DCT of the power_p champ, truncated to Q, with
λ=0 and p_2=p_3=p_champ. INTERP polish: load those coefficients and rebuild
the basis at --L (modes q>Q stay 0).

Loss matches power_p on the fresh250 ladder: mean log(inv/inv_QAOA), w6=1,
w9=2, λ_c annealed on the scaled c2,c3, hold out 20% of n9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))


def main():
    ap = argparse.ArgumentParser(description="FM-WS train / INTERP polish / CORE eval")
    ap.add_argument("--Q", type=int, required=True)
    ap.add_argument("--L", type=int, required=True, help="circuit depth (basis length)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--champ", type=str, required=True, help="power_p params_L16 .pt")
    ap.add_argument("--init-ckpt", type=str, default="", help="INTERP: FM-WS params to keep (A,B,p,λ)")
    ap.add_argument("--packs", type=str, required=True)
    ap.add_argument("--qaoa-dir", type=str, default="", help="free QAOA angles/results root (L{L}/)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--adam-steps", type=int, default=600)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs-rounds", type=int, default=12)
    ap.add_argument("--lbfgs-iter", type=int, default=8)
    ap.add_argument("--batch-per-n", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--lam0", type=float, default=1e-1)
    ap.add_argument("--lam1", type=float, default=1e-3)
    ap.add_argument("--lam-p", type=float, default=1e-2)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--eval-ns", type=str, default="6,9,12,15")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)

    from Nov_3rd import settings as S
    from Nov_3rd.ansatz import make_sim
    from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, load_split_E, log
    from Nov_3rd.fm_ws import (
        clamp_exponents_,
        degree_mad,
        gamma_phi_from_modes,
        param_count,
        phi_from_c,
        schedules_from_modes,
        truncation_rel_l2,
        warm_modes_from_champ,
    )
    from train_beat_train69 import precompute_qaoa_inv

    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.pt"
    train_path = out_dir / "train.json"
    Q = int(args.Q)
    L = int(args.L)
    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    weights = {6: float(args.w6), 9: float(args.w9)}

    champ = torch.load(args.champ, map_location="cpu", weights_only=False)
    g_c = champ["gamma"].float().reshape(-1)
    log(
        f"FM-WS Q={Q} L={L}: champ L={int(g_c.numel())} p={float(champ['p']):.4f} "
        f"trunc γ={truncation_rel_l2(g_c, min(Q, int(g_c.numel())), 'dst'):.4f} "
        f"c1={truncation_rel_l2(champ['c1'], min(Q, int(g_c.numel())), 'dct'):.4f} "
        f"c2={truncation_rel_l2(champ['c2'], min(Q, int(g_c.numel())), 'dct'):.4f}"
    )

    def load_init():
        if args.init_ckpt:
            prev = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            if int(prev["Q"]) != Q:
                raise SystemExit(f"init ckpt Q={prev['Q']} != --Q {Q}")
            log(f"FM-WS INTERP init from {args.init_ckpt} (basis was L={prev.get('basis_L')})")
            return {
                "A": prev["A"].float(),
                "B": prev["B"].float(),
                "p2": float(prev["p2"]),
                "p3": float(prev["p3"]),
                "lam": float(prev["lam"]),
                "init": "interp",
                "init_from": str(args.init_ckpt),
            }
        warm = warm_modes_from_champ(champ["gamma"], champ["c1"], champ["c2"], champ["c3"], float(champ["p"]), Q)
        log(f"FM-WS warm DCT/DST of champ, truncate Q={Q}, λ=0, p2=p3={warm['p2']:.4f}")
        warm["init"] = "champ_dct"
        warm["init_from"] = str(args.champ)
        return warm

    def load_train_n(n: int) -> torch.Tensor:
        d = np.load(Path(args.packs) / f"train_n{n}.npz", allow_pickle=True)
        E = np.asarray(d["train_E"])
        if args.max_train is not None:
            E = E[: int(args.max_train)]
        return torch.as_tensor(E, dtype=torch.float32)

    def qaoa_pair():
        if not args.qaoa_dir:
            return None
        path = Path(args.qaoa_dir) / f"L{L}" / "angles.json"
        if not path.exists():
            log(f"FM-WS: no free QAOA angles at {path}; train loss uses inv_Q=1")
            return None
        return load_qaoa(L, qaoa_dir=Path(args.qaoa_dir))

    if not args.skip_train:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        init = load_init()
        A = torch.nn.Parameter(init["A"].to(device).float())
        B = torch.nn.Parameter(init["B"].to(device).float())
        p2 = torch.nn.Parameter(torch.tensor(float(init["p2"]), device=device))
        p3 = torch.nn.Parameter(torch.tensor(float(init["p3"]), device=device))
        lam = torch.nn.Parameter(torch.tensor(float(init["lam"]), device=device))
        trainables = [A, B, p2, p3, lam]
        if sum(t.numel() for t in trainables) != param_count(Q):
            raise SystemExit(f"param count {sum(t.numel() for t in trainables)} != {param_count(Q)}")

        qpair = qaoa_pair()
        bundles = {}
        for n in (6, 9):
            E = load_train_n(n).to(device)
            sim = make_sim(n, L, device)
            if qpair is None:
                inv_q = torch.ones(len(E), device=device)
            else:
                betas, gammas = qpair
                inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device).to(device)
            bundles[n] = (E, inv_q, sim)
            log(f"FM-WS Q{Q} L{L}: train n{n} B={len(E)} QAOA_Med={float(inv_q.median()):.4f}")

        n9_B = len(bundles[9][0])
        perm = rng.permutation(n9_B)
        n_hold = max(1, int(round(args.holdout_frac * n9_B))) if args.holdout_frac > 0 else 0
        hold_idx = perm[:n_hold] if n_hold else np.array([], dtype=int)
        train_pool = perm[n_hold:] if n_hold else perm

        def angles(n: int):
            gamma, phi = gamma_phi_from_modes(A, B, p2, p3, lam, L, n)
            return gamma, phi

        def log_ratios(n: int, idx) -> torch.Tensor:
            E, inv_q, sim = bundles[n]
            gamma, c1, c2, c3 = schedules_from_modes(A, B, p2, p3, lam, L, n)
            phi = phi_from_c(c1, c2, c3)
            idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=E.device)
            inv = sim.forward_nov3rd_batch(E.index_select(0, idx_t), gamma, phi)["inv_p"]
            ref = inv_q.index_select(0, idx_t).clamp(min=1e-12)
            return torch.log((inv / ref).clamp(min=1e-12)), c2, c3

        def batch_loss(idx_map, lam_c: float):
            terms = []
            wsum = 0.0
            for n in bundles:
                logs, c2, c3 = log_ratios(n, idx_map[n])
                w = float(weights[n])
                terms.append(w * logs.mean())
                wsum += w
                if lam_c > 0:
                    terms.append(lam_c * w * (c2.pow(2).mean() + c3.pow(2).mean()))
                    wsum += w
            loss = torch.stack(terms).sum() / max(wsum, 1e-12)
            if args.lam_p > 0:
                loss = loss + float(args.lam_p) * (p2.pow(2) + p3.pow(2) + lam.pow(2))
            return loss

        @torch.no_grad()
        def holdout():
            if n_hold == 0:
                return float("nan")
            logs, _, _ = log_ratios(9, hold_idx)
            return float(logs.median().detach())

        def snapshot():
            return {
                "A": A.detach().cpu().clone(),
                "B": B.detach().cpu().clone(),
                "p2": float(p2.detach().cpu()),
                "p3": float(p3.detach().cpu()),
                "lam": float(lam.detach().cpu()),
            }

        def restore(snap):
            with torch.no_grad():
                A.copy_(snap["A"].to(device))
                B.copy_(snap["B"].to(device))
                p2.fill_(snap["p2"])
                p3.fill_(snap["p3"])
                lam.fill_(snap["lam"])

        hist = []
        t0 = time.time()
        best = {"score": float("inf"), "snap": snapshot(), "step": "init"}
        sc0 = holdout()
        best["score"] = sc0
        log(f"FM-WS Q{Q} L{L}: init holdout med logR={sc0:.5f} params={param_count(Q)}")

        opt = torch.optim.Adam(trainables, lr=args.adam_lr)
        for step in range(args.adam_steps):
            frac = step / max(args.adam_steps - 1, 1)
            lam_c = args.lam0 * (1 - frac) + args.lam1 * frac
            idx_map = {}
            for n in (6, 9):
                pool = train_pool if n == 9 else np.arange(len(bundles[n][0]))
                take = min(args.batch_per_n, len(pool))
                idx_map[n] = rng.choice(pool, size=take, replace=False)
            opt.zero_grad()
            loss = batch_loss(idx_map, lam_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            opt.step()
            clamp_exponents_(p2, p3, lam)
            hist.append({"phase": "adam", "step": step, "loss": float(loss.detach()), "lam_c": lam_c})
            if step % 25 == 0 or step == args.adam_steps - 1:
                sc = holdout()
                mark = ""
                if sc < best["score"]:
                    best = {"score": sc, "snap": snapshot(), "step": f"adam{step}"}
                    mark = " BEST"
                log(
                    f"FM-WS Q{Q} L{L} adam {step}/{args.adam_steps} loss={hist[-1]['loss']:.5f} "
                    f"hold={sc:.5f} p2={float(p2.detach()):.3f} p3={float(p3.detach()):.3f} "
                    f"λ={float(lam.detach()):.3f}{mark}"
                )

        for r in range(args.lbfgs_rounds):
            opt_l = torch.optim.LBFGS(
                trainables, lr=0.25, max_iter=args.lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
            )
            idx_map = {}
            for n in (6, 9):
                pool = train_pool if n == 9 else np.arange(len(bundles[n][0]))
                take = min(args.batch_per_n, len(pool))
                idx_map[n] = rng.choice(pool, size=take, replace=False)
            lam_c = float(args.lam1)

            def closure():
                opt_l.zero_grad()
                lt = batch_loss(idx_map, lam_c)
                lt.backward()
                torch.nn.utils.clip_grad_norm_(trainables, 5.0)
                return lt

            try:
                loss = opt_l.step(closure)
                loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            except Exception as e:
                log(f"FM-WS Q{Q} L{L} LBFGS r{r} skip: {e}")
                continue
            clamp_exponents_(p2, p3, lam)
            hist.append({"phase": "lbfgs", "round": r, "loss": loss_f})
            sc = holdout()
            mark = ""
            if sc < best["score"]:
                best = {"score": sc, "snap": snapshot(), "step": f"lbfgs{r}"}
                mark = " BEST"
            log(f"FM-WS Q{Q} L{L} lbfgs {r}/{args.lbfgs_rounds} loss={loss_f:.5f} hold={sc:.5f}{mark}")

        if best["snap"] is not None and best["score"] == best["score"]:
            log(f"FM-WS Q{Q} L{L}: restore best holdout {best['score']:.5f} @ {best['step']}")
            restore(best["snap"])

        with torch.no_grad():
            g_ref, _, _, _ = schedules_from_modes(A, B, p2, p3, lam, L, 9)
            mad = degree_mad(g_ref)
        state = {
            "idea": "fm_ws",
            "Q": Q,
            "L": L,
            "basis_L": L,
            "seed": args.seed,
            "A": A.detach().cpu(),
            "B": B.detach().cpu(),
            "p2": float(p2.detach().cpu()),
            "p3": float(p3.detach().cpu()),
            "lam": float(lam.detach().cpu()),
            "p1": 0.0,
            "n_params": param_count(Q),
            "init": init["init"],
            "init_from": init["init_from"],
            "champ": str(args.champ),
            "degree_mad_n9": mad,
            "holdout_best": best["score"],
            "holdout_best_step": best["step"],
            "wall_s": time.time() - t0,
            "hist_tail": hist[-30:],
            "formula": "γ_d = s_n^(λ p_d) Σ A_{q,d} u_q ; c_k = s_n^(p_k) Σ B_{q,k} v_q ; p1=0",
        }
        torch.save(state, params_path)
        meta = {
            "done": True,
            "idea": "fm_ws",
            "Q": Q,
            "L": L,
            "seed": args.seed,
            "n_params": param_count(Q),
            "p2": state["p2"],
            "p3": state["p3"],
            "lam": state["lam"],
            "degree_mad_n9": mad,
            "holdout_best": best["score"],
            "holdout_best_step": best["step"],
            "wall_s": state["wall_s"],
            "init": init["init"],
            "init_from": init["init_from"],
            "final_loss": hist[-1]["loss"] if hist else None,
            "params": str(params_path),
        }
        train_path.write_text(json.dumps(meta, indent=2) + "\n")
        log(
            f"FM-WS Q{Q} L{L} saved {params_path} wall={state['wall_s']:.0f}s "
            f"p2={state['p2']:.4f} p3={state['p3']:.4f} λ={state['lam']:.4f} mad={mad:.4f}"
        )
    elif not params_path.exists():
        raise SystemExit(f"--skip-train but no {params_path}")

    if args.skip_eval:
        return 0

    state = torch.load(params_path, map_location="cpu", weights_only=False)
    A = state["A"].to(device)
    B = state["B"].to(device)
    p2 = torch.tensor(float(state["p2"]), device=device)
    p3 = torch.tensor(float(state["p3"]), device=device)
    lam = torch.tensor(float(state["lam"]), device=device)
    qref = load_qaoa_scale_ref(L, qaoa_dir=Path(args.qaoa_dir)) if args.qaoa_dir else {}

    def _eval_batch_size(n: int) -> int:
        if n <= 6:
            return 64
        if n <= 9:
            return 32
        if n <= 12:
            return 8
        return 4

    @torch.no_grad()
    def eval_batched(n: int, gamma, phi) -> dict:
        E, key = load_split_E(n, "scale1000", args.eval_B, device, data_dir=S.SCALE1000_DIR)
        sim = make_sim(n, L, device)
        bs = _eval_batch_size(n)
        inv_parts, leak_parts = [], []
        t0 = time.time()
        B = len(E)
        for s in range(0, B, bs):
            out = sim.forward_nov3rd_batch(E[s : s + bs], gamma, phi)
            inv_parts.append(out["inv_p"].detach().cpu())
            leak_parts.append(out["p_leak"].detach().cpu())
            done = min(s + bs, B)
            if done == B or (done // bs) % 10 == 0:
                med_now = float(torch.cat(inv_parts).median())
                rate = done / max(time.time() - t0, 1e-9)
                log(f"eval n{n} {done}/{B} Med≈{med_now:.3f} ({rate:.1f}/s)")
        inv = torch.cat(inv_parts).numpy()
        leak = torch.cat(leak_parts).numpy()
        return {
            "median_inv_p": float(np.median(inv)),
            "B90_inv_p": float(np.quantile(inv, 0.90)),
            "mean_leak": float(leak.mean()),
            "B": int(inv.shape[0]),
            "wall_s": time.time() - t0,
            "split": key,
        }

    rows = {}
    for n in eval_ns:
        gamma, phi = gamma_phi_from_modes(A, B, p2, p3, lam, L, n)
        m = eval_batched(n, gamma, phi)
        q = qref.get(n) or {}
        qmed = float(q.get("Med", float("nan")))
        qb90 = float(q.get("B90", float("nan")))
        med = float(m["median_inv_p"])
        b90 = float(m["B90_inv_p"])
        if qmed == qmed and med < qmed:
            mark = "BEAT"
        elif qmed == qmed and med == qmed:
            mark = "TIE"
        elif qmed == qmed:
            mark = "UNDER"
        else:
            mark = "NO_REF"
        rows[str(n)] = {
            "Med": med,
            "B90": b90,
            "QAOA_Med": qmed,
            "QAOA_B90": qb90,
            "verdict": mark,
            "B": int(m["B"]),
            "mean_leak": float(m["mean_leak"]),
            "wall_s": float(m["wall_s"]),
        }
        log(f"FM-WS Q{Q} L{L} n{n}: Med={med:.4f} Q={qmed:.4f} {mark} B90={b90:.4f}")
    payload = {
        "idea": "fm_ws",
        "Q": Q,
        "L": L,
        "seed": args.seed,
        "eval_B": args.eval_B,
        "p2": float(state["p2"]),
        "p3": float(state["p3"]),
        "lam": float(state["lam"]),
        "degree_mad_n9": state.get("degree_mad_n9"),
        "init": state.get("init"),
        "per_n": rows,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    eval_path = out_dir / "eval.json"
    eval_path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"# FM-WS Q={Q} L={L}",
        "",
        f"p2={state['p2']:.4f} p3={state['p3']:.4f} λ={state['lam']:.4f}",
        "",
        "| n | FM-WS Med | free QAOA Med | mark | FM-WS B90 | free QAOA B90 |",
        "|---|-----------|---------------|------|-----------|---------------|",
    ]
    for n in eval_ns:
        r = rows[str(n)]
        lines.append(
            f"| {n} | {r['Med']:.4f} | {r['QAOA_Med']:.4f} | {r['verdict']} | {r['B90']:.4f} | {r['QAOA_B90']:.4f} |"
        )
    lines.append("")
    (out_dir / "eval.md").write_text("\n".join(lines) + "\n")
    log(f"FM-WS wrote {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
