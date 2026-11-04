#!/usr/bin/env python3
"""Fourier QAOA on a Walsh-graded cost with a frozen shared tilt τ.

    β_ℓ = Σ_{q=1..Q} B_q v_q(ℓ)     v = DCT-IV cosine
    γ_ℓ = Σ_{q=1..Q} A_q u_q(ℓ)     u = DST-IV sine
    U_C = exp(-i γ_ℓ Σ_{w=1..4} τ_w f_w)

τ is an input, not a parameter. τ = 1 is ordinary Fourier QAOA (degree 0 is a
global phase). Optimizer budget matches the FM-WS Fourier QAOA trainer:
Adam 600 at lr 0.05, then 12 L-BFGS rounds, batch 32+32 on train {6, 9}.
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.data_io import log
from Nov_3rd.ensure_qaoa import eval_scale_ladder
from Nov_3rd.wt import qaoa_angles_from_modes, qaoa_modes_from_angles, qaoa_psucc_graded, tau_full
from explore_OCT29.lib.qaoa import inv_p_from_psucc, summarize_inv


def _tau4(spec: str, device) -> torch.Tensor:
    if not spec:
        return torch.ones(4, device=device)
    vals = [float(x) for x in spec.split(",")]
    if len(vals) == 3:
        return tau_full(torch.tensor(vals, device=device))
    if len(vals) == 4:
        if abs(vals[0] - 1.0) > 1e-8:
            raise SystemExit(f"τ_1 must be 1, got {vals[0]}")
        return torch.tensor(vals, dtype=torch.float32, device=device)
    raise SystemExit(f"--tau needs 3 or 4 numbers, got {spec!r}")


def main():
    ap = argparse.ArgumentParser(description="Graded Fourier QAOA with frozen τ")
    ap.add_argument("--Q", type=int, default=6)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--packs", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--angles", type=str, default="", help="free QAOA angles.json, DCT/DST warm start")
    ap.add_argument("--init-ckpt", type=str, default="", help="Fourier params.pt (A, B) warm start")
    ap.add_argument("--tau", type=str, default="", help="τ2,τ3,τ4 or τ1..τ4. Empty means τ=1")
    ap.add_argument("--adam-steps", type=int, default=600)
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--lbfgs-rounds", type=int, default=12)
    ap.add_argument("--lbfgs-iter", type=int, default=8)
    ap.add_argument("--batch-per-n", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--eval-ns", type=str, default="6,9,12,15")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.pt"
    Q = int(args.Q)
    L = int(args.L)
    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    tau4 = _tau4(args.tau, device)

    def load_pack(n: int) -> torch.Tensor:
        d = np.load(Path(args.packs) / f"train_n{n}.npz", allow_pickle=True)
        E = np.asarray(d["train_E"])
        if args.max_train is not None:
            E = E[: int(args.max_train)]
        return torch.as_tensor(E, dtype=torch.float32)

    def load_init():
        if args.init_ckpt:
            prev = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            if int(prev["Q"]) != Q:
                raise SystemExit(f"init ckpt Q={prev['Q']} != --Q {Q}")
            log(f"graded Fourier warm from {args.init_ckpt}")
            return prev["A"].float(), prev["B"].float(), "fourier_ckpt", str(args.init_ckpt)
        if not args.angles:
            raise SystemExit("need --init-ckpt or --angles")
        blob = json.loads(Path(args.angles).read_text())
        best = blob["best"]
        betas = torch.tensor(best["betas"], dtype=torch.float32)
        gammas = torch.tensor(best["gammas"], dtype=torch.float32)
        modes = qaoa_modes_from_angles(betas, gammas, Q)
        log(f"graded Fourier warm DCT/DST of {args.angles} → Q={Q}")
        return modes["A"], modes["B"], "free_qaoa_dct", str(args.angles)

    if not args.skip_train:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        A0, B0, how, src = load_init()
        A = torch.nn.Parameter(A0.to(device).float())
        B = torch.nn.Parameter(B0.to(device).float())
        E6 = load_pack(6)
        E9 = load_pack(9)
        log(
            f"graded Fourier Q{Q} L{L}: τ={tau4.tolist()} n6={len(E6)} n9={len(E9)} "
            f"steps={args.adam_steps}"
        )

        def loss_on(idx6, idx9):
            betas, gammas = qaoa_angles_from_modes(A, B, L)
            ps6 = qaoa_psucc_graded(betas, gammas, E6[idx6].to(device), 6, tau4)
            ps9 = qaoa_psucc_graded(betas, gammas, E9[idx9].to(device), 9, tau4)
            loss6 = -torch.log(ps6.clamp_min(1e-12)).mean()
            loss9 = -torch.log(ps9.clamp_min(1e-12)).mean()
            return 0.5 * (loss6 + loss9)

        opt = torch.optim.Adam([A, B], lr=args.adam_lr)
        hist = []
        t0 = time.time()
        for step in range(args.adam_steps):
            i6 = rng.choice(len(E6), size=min(args.batch_per_n, len(E6)), replace=False)
            i9 = rng.choice(len(E9), size=min(args.batch_per_n, len(E9)), replace=False)
            opt.zero_grad()
            loss = loss_on(i6, i9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([A, B], 5.0)
            opt.step()
            hist.append(float(loss.detach().cpu()))
            if step % 25 == 0 or step == args.adam_steps - 1:
                log(
                    f"graded Fourier Q{Q} L{L} adam {step}/{args.adam_steps} "
                    f"loss={hist[-1]:.4f} wall={time.time()-t0:.0f}s"
                )
        for r in range(args.lbfgs_rounds):
            opt_l = torch.optim.LBFGS(
                [A, B], lr=0.25, max_iter=args.lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
            )
            i6 = rng.choice(len(E6), size=min(args.batch_per_n, len(E6)), replace=False)
            i9 = rng.choice(len(E9), size=min(args.batch_per_n, len(E9)), replace=False)

            def closure():
                opt_l.zero_grad()
                lt = loss_on(i6, i9)
                lt.backward()
                torch.nn.utils.clip_grad_norm_([A, B], 5.0)
                return lt

            try:
                loss = opt_l.step(closure)
                loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            except Exception as e:
                log(f"graded Fourier LBFGS r{r} skip: {e}")
                continue
            hist.append(loss_f)
            log(f"graded Fourier Q{Q} L{L} lbfgs {r}/{args.lbfgs_rounds} loss={loss_f:.4f}")

        with torch.no_grad():
            betas, gammas = qaoa_angles_from_modes(A, B, L)
        state = {
            "idea": "qaoa_fourier_graded",
            "Q": Q,
            "L": L,
            "seed": args.seed,
            "A": A.detach().cpu(),
            "B": B.detach().cpu(),
            "betas": betas.detach().cpu(),
            "gammas": gammas.detach().cpu(),
            "tau": tau4.detach().cpu(),
            "init": how,
            "init_from": src,
            "n_params": int(2 * Q),
            "wall_s": time.time() - t0,
            "final_loss": hist[-1] if hist else None,
            "hist_tail": hist[-20:],
        }
        torch.save(state, params_path)
        (out_dir / "train.json").write_text(
            json.dumps(
                {
                    "done": True,
                    "Q": Q,
                    "L": L,
                    "seed": args.seed,
                    "tau": tau4.detach().cpu().tolist(),
                    "final_loss": state["final_loss"],
                    "wall_s": state["wall_s"],
                    "init": how,
                    "init_from": src,
                },
                indent=2,
            )
            + "\n"
        )
        log(f"graded Fourier saved {params_path} wall={state['wall_s']:.0f}s")
    elif not params_path.exists():
        raise SystemExit(f"--skip-train but no {params_path}")

    if args.skip_eval:
        return 0

    state = torch.load(params_path, map_location="cpu", weights_only=False)
    if args.tau:
        tau_eval = _tau4(args.tau, device)
    elif "tau" in state:
        tau_eval = state["tau"].to(device=device, dtype=torch.float32)
    else:
        tau_eval = torch.ones(4, device=device)
    betas, gammas = qaoa_angles_from_modes(state["A"].to(device), state["B"].to(device), L)
    # τ=1 uses the ungraded ladder (identical probabilities). A non-trivial tilt
    # has to go through the graded cost.
    if torch.allclose(tau_eval, torch.ones_like(tau_eval)):
        ladder = eval_scale_ladder(L, betas, gammas, device=device, ns=eval_ns, max_count=args.eval_B)
        per_n = {
            str(n): {
                "Med": float(row["median_inv_p"]),
                "B90": float(row["B90_inv_p"]),
                "B": int(row["B"]),
                "wall_s": float(row.get("wall_s", float("nan"))),
            }
            for n, row in ladder.items()
        }
    else:
        per_n = _eval_graded(L, betas, gammas, tau_eval, eval_ns, args.eval_B, device)
    payload = {
        "idea": "qaoa_fourier_graded",
        "Q": Q,
        "L": L,
        "seed": int(state.get("seed", args.seed)),
        "tau": [float(x) for x in tau_eval.detach().cpu().reshape(-1)],
        "eval_B": args.eval_B,
        "final_loss": state.get("final_loss"),
        "per_n": per_n,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"# graded Fourier QAOA Q={Q} L={L} τ={payload['tau']}",
        "",
        "| n | Med | B90 | B |",
        "|---|-----|-----|---|",
    ]
    for n in eval_ns:
        r = per_n[str(n)]
        lines.append(f"| {n} | {r['Med']:.4f} | {r['B90']:.4f} | {r['B']} |")
    (out_dir / "eval.md").write_text("\n".join(lines) + "\n")
    log(f"graded Fourier wrote {out_dir / 'eval.json'}")
    return 0


@torch.no_grad()
def _eval_graded(L, betas, gammas, tau4, ns, max_count, device) -> dict:
    from Nov_3rd.ensure_qaoa import _load_E

    per_n = {}
    for n in ns:
        pack = S.SCALE1000_DIR / f"data_n{n}.npz"
        if not pack.exists():
            log(f"graded Fourier skip n{n}: missing pack")
            continue
        _seeds, E = _load_E(n, S.SCALE1000_DIR, "scale1000", max_count)
        B = len(_seeds)
        chunk = 64 if n <= 9 else (16 if n <= 12 else (4 if n <= 15 else 1))
        invs = []
        t0 = time.time()
        for s0 in range(0, B, chunk):
            Eb = torch.as_tensor(E[s0 : s0 + chunk], dtype=torch.float32, device=device)
            ps = qaoa_psucc_graded(betas, gammas, Eb, n, tau4).cpu().numpy()
            invs.extend(inv_p_from_psucc(ps).tolist())
            done = min(s0 + chunk, B)
            if done == B or done % 200 < chunk:
                log(f"graded Fourier L{L} n{n} {done}/{B} Med≈{float(np.median(invs)):.3f}")
        summ = summarize_inv(invs)
        per_n[str(n)] = {
            "Med": float(summ["median_inv_p"]),
            "B90": float(summ["B90_inv_p"]),
            "B": int(summ["B"]),
            "wall_s": time.time() - t0,
        }
        log(f"graded Fourier L{L} n{n}: Med={per_n[str(n)]['Med']:.4f}")
    return per_n


if __name__ == "__main__":
    raise SystemExit(main())
