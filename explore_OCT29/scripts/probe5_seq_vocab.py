#!/usr/bin/env python3
"""Opus probe 5 — seq-evolve with W/SNAP0/PROJ atoms; promote on fitted b.

Train {n9,n12}; eval n9/12/15; fit b_Med; discard if worse than projector
fixed-angle baseline (probe2) or Fourier L12 ceiling reference.

Stop: touch runs/STOP_PROBE5
Forever unless --max_trials N.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.ansatz import AnsatzConfig  # noqa: E402
from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.seq_w_ansatz import (  # noqa: E402
    SEED_W_MOTIFS,
    WMotif,
    MultiCavityWSeqAnsatz,
    init_w_params,
    load_W,
    mutate_w_motif,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _PKG / "runs" / "probe5_seq_vocab"
STOP = _PKG / "runs" / "STOP_PROBE5"
P2_PARAMS = _PKG / "runs" / "probe2_train912" / "projector_912_params.pt"
P3_JSON = _PKG / "runs" / "probe3_fourier" / "results.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[probe5]", *a, flush=True)


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def _loss(out, leakpen=0.1):
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


@torch.no_grad()
def eval_wmotif(n, motif, params, W, nf, max_count, device, split="val"):
    pack = load_pack(n)
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    cfg = AnsatzConfig(n=n, L=motif.L, nf=nf)
    sim = MultiCavityWSeqAnsatz(cfg, motif, W, device=device).to(device)
    p = {k: v.detach().to(device) for k, v in params.items()}
    invs, leaks = [], []
    t0 = time.time()
    for b in range(len(seeds)):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_wmotif(Eb, p)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n, "B": len(invs), "median_inv_p": med, "B90_inv_p": b90,
        "S": _S(med, b90), "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0, "motif": motif.name, "L": motif.L,
    }


def eval_ladder(motif, params, W, nf, device, max_counts):
    out = {}
    for n, mc in max_counts.items():
        if not (_PKG / "data" / f"data_n{n}.npz").exists():
            continue
        out[n] = eval_wmotif(n, motif, params, W, nf, mc, device)
    return out


def fit_b_from_ladder(ladder):
    ns, meds, b90s = [], [], []
    for n in (9, 12, 15):
        m = ladder.get(n)
        if m and "median_inv_p" in m:
            ns.append(n); meds.append(m["median_inv_p"]); b90s.append(m["B90_inv_p"])
    return {"Med": fit_b(ns, meds), "B90": fit_b(ns, b90s)}


def short_train(
    motif, W, steps, max_train, max_val, nf, seed, device,
    warm_g=None, warm_b=None, lr=4e-3,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = init_w_params(motif, seed=seed, device=device, warm_gamma=warm_g, warm_beta=warm_b)
    sims, Es = {}, {}
    for n in (9, 12):
        pack = load_pack(n)
        _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        Es[n] = E
        sims[n] = MultiCavityWSeqAnsatz(
            AnsatzConfig(n=n, L=motif.L, nf=nf), motif, W, device=device
        ).to(device)

    opt = torch.optim.Adam(list(params.values()), lr=lr)
    best = {"score": float("inf"), "params": None, "val": None, "step": -1}
    cut = max(steps - 15, steps * 2 // 3)

    for step in range(steps):
        if step == cut and any(p.numel() <= 16 for p in params.values()):
            # LBFGS only when few params (projector-like)
            n_p = sum(p.numel() for p in params.values())
            if n_p <= 24:
                opt = torch.optim.LBFGS(
                    list(params.values()), lr=0.15, max_iter=3,
                    history_size=15, line_search_fn="strong_wolfe",
                )

        def closure():
            opt.zero_grad(set_to_none=True)
            losses = []
            for n, sim in sims.items():
                idx = rng.choice(len(Es[n]), size=min(5, len(Es[n])), replace=False)
                for i in idx:
                    Eb = torch.as_tensor(Es[n][i], dtype=torch.float32, device=device)
                    out = sim.forward_wmotif(Eb, params)
                    losses.append(_loss(out))
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        if isinstance(opt, torch.optim.LBFGS):
            try:
                loss = opt.step(closure)
            except Exception:
                loss = closure()
                opt_a = torch.optim.Adam(list(params.values()), lr=lr)
                opt_a.step()
                opt = opt_a
        else:
            loss = closure()
            torch.nn.utils.clip_grad_norm_(list(params.values()), 5.0)
            opt.step()

        if not ((step % 5 == 0) or step < 2 or step + 1 == steps):
            continue
        vals = {
            n: eval_wmotif(n, motif, params, W, nf, max_val, device)
            for n in (9, 12)
        }
        sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
        if sc < best["score"]:
            best = {
                "score": sc,
                "params": {k: v.detach().cpu().clone() for k, v in params.items()},
                "val": vals,
                "step": step,
            }
            log(
                f"  step {step}: Scomb={sc:.3f} "
                + " ".join(f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}" for n in (9, 12))
            )
    if best["params"] is None:
        best["params"] = {k: v.detach().cpu().clone() for k, v in params.items()}
        best["val"] = {n: eval_wmotif(n, motif, params, W, nf, max_val, device) for n in (9, 12)}
    return best


def load_baselines():
    """Projector probe2 b and Fourier L12 n15 Med as ceilings."""
    base = {
        "b_med_gate": 1.42,  # probe2 ~1.405; must beat meaningfully or match + better Med15
        "med15_proj_L4": 30.9,
        "med15_fourier_L12": 9.67,
        "med12_proj": 17.5,
    }
    if P2_PARAMS.exists():
        blob = torch.load(P2_PARAMS, map_location="cpu", weights_only=False)
        base["warm_gamma"] = blob["gamma"].tolist() if torch.is_tensor(blob["gamma"]) else blob["gamma"]
        base["warm_beta"] = blob["beta"].tolist() if torch.is_tensor(blob["beta"]) else blob["beta"]
    else:
        base["warm_gamma"] = None
        base["warm_beta"] = None
    if P3_JSON.exists():
        js = json.loads(P3_JSON.read_text())
        try:
            base["med15_fourier_L12"] = js["sweep"]["15"]["12"]["median_inv_p"]
        except Exception:
            pass
        try:
            base["b_fourier_L12"] = js["fits_by_L"]["12"]["Med"]["b"]
        except Exception:
            base["b_fourier_L12"] = 1.44
    return base


def write_status(out_dir, champ, trials_done, baselines):
    lines = [
        "# Opus probe 5 — W-atom seq vocab evolve\n\n",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_\n\n",
        f"Trials done: {trials_done}\n\n",
        "## Baselines to beat\n\n",
        f"- Probe2 projector L4 train{{9,12}}: n15 Med≈{baselines['med15_proj_L4']}, b≈1.41\n",
        f"- Probe3 Fourier L12: n15 Med≈{baselines['med15_fourier_L12']:.3g}\n\n",
        "## Current champion\n\n",
    ]
    if champ is None:
        lines.append("_none yet_\n")
    else:
        lines.append(f"- motif: `{champ['motif']['name']}`\n")
        lines.append(f"- b_Med={champ['b_med']:.4f} b_B90={champ.get('b_b90', float('nan')):.4f}\n")
        lad = champ.get("ladder", {})
        for n in (9, 12, 15):
            m = lad.get(n) or lad.get(str(n)) or {}
            if "median_inv_p" in m:
                lines.append(f"- n{n}: {m['median_inv_p']:.4g}/{m['B90_inv_p']:.4g}\n")
        m15 = lad.get(15) or lad.get("15") or {}
        lines.append(f"- vs Fourier L12 n15: champ={m15.get('median_inv_p','?')} "
                     f"ceiling={baselines['med15_fourier_L12']:.3g}\n")
    lines.append("\n## Note\n\nPromote on fitted b over {9,12,15}; reject if Med15 worse than probe2 "
                 "projector unless b clearly better.\n")
    (out_dir / "RESULTS.md").write_text("".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_trials", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--confirm_steps", type=int, default=120)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--max_val15", type=int, default=6)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_mutate", type=float, default=0.7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = load_W(nf=args.nf, device=device)
    baselines = load_baselines()
    log(f"baselines={ {k:v for k,v in baselines.items() if k not in ('warm_gamma','warm_beta')} }")

    max_counts = {9: args.max_val, 12: args.max_val, 15: args.max_val15}
    csv_path = OUT / "trials.csv"
    fieldnames = [
        "trial", "mode", "keep", "motif", "L", "n_params", "b_med", "b_b90",
        "Med9", "B90_9", "Med12", "B90_12", "Med15", "B90_15", "wall_s", "best_b",
    ]
    new = not csv_path.exists()
    rng = np.random.default_rng(args.seed)
    champ = None
    best_b = float(baselines["b_med_gate"])  # must beat probe2 b to promote as "better transfer"
    # but allow promote if Med15 beats probe2 even if b similar
    best_med15 = float(baselines["med15_proj_L4"])
    parent = SEED_W_MOTIFS[0]
    seed_i = 0
    trial = 0

    # load prior champ if any
    champ_path = OUT / "champ.json"
    if champ_path.exists():
        try:
            champ = json.loads(champ_path.read_text())
            best_b = float(champ["b_med"])
            best_med15 = float(champ["ladder"]["15"]["median_inv_p"])
            parent = WMotif.from_dict(champ["motif"])
            log(f"resumed champ {parent.name} b={best_b:.4f} Med15={best_med15:.3g}")
        except Exception as e:
            log(f"champ load failed: {e}")

    write_status(OUT, champ, trial, baselines)

    with open(csv_path, "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if new:
            w.writeheader(); fcsv.flush()

        while True:
            if STOP.exists():
                log("STOP_PROBE5 — exit")
                break
            if args.max_trials and trial >= args.max_trials:
                log("max_trials — exit")
                break
            trial += 1
            t0 = time.time()

            if seed_i < len(SEED_W_MOTIFS):
                cand = SEED_W_MOTIFS[seed_i]
                mode = "seed"
                seed_i += 1
            elif rng.random() < args.eps_mutate:
                cand = mutate_w_motif(parent, rng)
                mode = "mutate"
            else:
                cand = SEED_W_MOTIFS[int(rng.integers(0, len(SEED_W_MOTIFS)))]
                L = int(np.clip(cand.L + int(rng.choice([-1, 0, 1, 2])), 3, 10))
                cand = WMotif(name=f"{cand.name}_rL{L}", prep=cand.prep, layer=cand.layer, L=L)
                mode = "reseed"

            log(f"=== trial {trial} {mode} {cand.name} prep={cand.prep} layer={cand.layer} "
                f"L={cand.L} n_params={cand.n_params()}")

            # warm-start γ/β only for projector-like motifs
            use_warm = any(o in ("PROJ", "SNAP0") for o in cand.layer) and (("W" in cand.prep) or ("FLAT_PREP" in cand.prep))
            try:
                best = short_train(
                    cand, W, args.steps, args.max_train, args.max_val, args.nf,
                    seed=args.seed + trial, device=device,
                    warm_g=baselines["warm_gamma"] if use_warm else None,
                    warm_b=baselines["warm_beta"] if use_warm else None,
                )
                ladder = eval_ladder(cand, best["params"], W, args.nf, device, max_counts)
                fits = fit_b_from_ladder(ladder)
                b_med = float(fits["Med"]["b"])
                b_b90 = float(fits["B90"]["b"])
            except Exception as e:
                log(f"trial {trial} FAILED: {e}")
                w.writerow({
                    "trial": trial, "mode": mode, "keep": "error", "motif": cand.name,
                    "L": cand.L, "n_params": cand.n_params(), "b_med": "", "b_b90": "",
                    "Med9": "", "B90_9": "", "Med12": "", "B90_12": "", "Med15": "", "B90_15": "",
                    "wall_s": f"{time.time()-t0:.1f}", "best_b": f"{best_b:.4f}",
                })
                fcsv.flush()
                continue

            m9, m12, m15 = ladder.get(9, {}), ladder.get(12, {}), ladder.get(15, {})
            med15 = m15.get("median_inv_p", float("inf"))
            med12 = m12.get("median_inv_p", float("inf"))

            # promote rules:
            # 1) b_med < best_b * 0.98 AND med15 not worse than 1.15× probe2
            # 2) OR med15 < best_med15 * 0.95 AND b_med < 1.55 (still transferring)
            # 3) never promote if med15 > 2× probe2 or med12 > 1e4
            keep = False
            reason = "no"
            if med12 > 1e4 or med15 > 2.5 * baselines["med15_proj_L4"]:
                reason = "reject_explode"
            elif b_med < best_b * 0.98 and med15 <= baselines["med15_proj_L4"] * 1.15:
                keep = True
                reason = "promote_b"
            elif med15 < best_med15 * 0.95 and b_med < 1.55:
                keep = True
                reason = "promote_med15"
            elif cand.name.startswith("proj_") and trial <= len(SEED_W_MOTIFS) and champ is None:
                # first projector seed always becomes initial champ if sane
                if med15 < baselines["med15_proj_L4"] * 1.3 and b_med < 1.6:
                    keep = True
                    reason = "promote_seed_init"

            row = {
                "trial": trial, "mode": mode, "keep": reason, "motif": cand.name,
                "L": cand.L, "n_params": cand.n_params(),
                "b_med": f"{b_med:.4f}", "b_b90": f"{b_b90:.4f}",
                "Med9": f"{m9.get('median_inv_p', float('nan')):.4f}",
                "B90_9": f"{m9.get('B90_inv_p', float('nan')):.4f}",
                "Med12": f"{m12.get('median_inv_p', float('nan')):.4f}",
                "B90_12": f"{m12.get('B90_inv_p', float('nan')):.4f}",
                "Med15": f"{m15.get('median_inv_p', float('nan')):.4f}",
                "B90_15": f"{m15.get('B90_inv_p', float('nan')):.4f}",
                "wall_s": f"{time.time()-t0:.1f}", "best_b": f"{best_b:.4f}",
            }
            w.writerow(row); fcsv.flush()
            log(
                f"trial {trial} {reason} b={b_med:.4f} Med9/12/15="
                f"{m9.get('median_inv_p', float('nan')):.3g}/"
                f"{m12.get('median_inv_p', float('nan')):.3g}/"
                f"{m15.get('median_inv_p', float('nan')):.3g}"
            )

            if keep:
                log(f"CONFIRM {args.confirm_steps} steps...")
                conf = short_train(
                    cand, W, args.confirm_steps, args.max_train, max(args.max_val, 16),
                    args.nf, seed=args.seed + trial + 9999, device=device,
                    warm_g=baselines["warm_gamma"] if use_warm else None,
                    warm_b=baselines["warm_beta"] if use_warm else None,
                )
                lad_c = eval_ladder(cand, conf["params"], W, args.nf, device, max_counts)
                fits_c = fit_b_from_ladder(lad_c)
                b_c = float(fits_c["Med"]["b"])
                med15_c = lad_c.get(15, {}).get("median_inv_p", float("inf"))
                ok = (b_c < best_b * 1.01 and med15_c <= baselines["med15_proj_L4"] * 1.2) or (
                    med15_c < best_med15 and b_c < 1.55
                )
                if ok or (champ is None and b_c < 1.55):
                    champ = {
                        "motif": cand.to_dict(),
                        "b_med": b_c,
                        "b_b90": float(fits_c["B90"]["b"]),
                        "ladder": {str(k): v for k, v in lad_c.items()},
                        "trial": trial,
                        "reason": reason,
                        "vs_fourier_L12_med15": baselines["med15_fourier_L12"],
                    }
                    best_b = min(best_b, b_c)
                    best_med15 = min(best_med15, med15_c)
                    parent = cand
                    torch.save(conf["params"], OUT / "champ_params.pt")
                    champ_path.write_text(json.dumps(champ, indent=2) + "\n")
                    (OUT / "results.json").write_text(json.dumps({
                        "champ": champ,
                        "baselines": {k: v for k, v in baselines.items() if k not in ("warm_gamma", "warm_beta")},
                        "trials": trial,
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, indent=2) + "\n")
                    write_status(OUT, champ, trial, baselines)
                    log(f"PROMOTED {cand.name} b={b_c:.4f} Med15={med15_c:.3g}")
                else:
                    log(f"confirm failed b={b_c:.4f} Med15={med15_c:.3g}")
                    if b_med < best_b * 1.05:
                        parent = cand
            else:
                if med12 < 500 and b_med < best_b * 1.1:
                    parent = cand

            write_status(OUT, champ, trial, baselines)

    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
