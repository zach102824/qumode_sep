#!/usr/bin/env python3
"""Clean A/B: learned q vs q≡1 freeze on qtimes_cost.

Recipe matched to box BEST hot0003:
  L=10, leakpen=0.1, lr=5e-5, learned alpha0,
  early-stop on S_comb, 8 CPU threads.

Arms
  learned  — q = 1 + 9*sigmoid(logit), q head trained
  freeze1  — q_from_logit forced to 1, logits zeroed, q head frozen

Selection uses val[:max_val_select] (default 20). The official comparison is a
reload reval of the best checkpoint on val[:40]. A win is claimed only from
that reval, and only when the disjoint val[20:40] slice agrees.

Arch is the baseline factor GNN (H=128, rounds=4, mean+degree) with lr overridden.
No gnn_best snapshot is loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Thread recipe before numpy/torch spin a pool.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from qtimes_cost.lib.gnn import BASELINE_ARCH, GNNArch
from qtimes_cost.lib.train_loop import (
    combined_score,
    eval_pack_metrics,
    load_pack,
    short_train,
)
from qtimes_cost.lib.walksat import aggregate_bars, walksat_instance
from qtimes_cost.scripts.gen_teb_packs import build_dataset

# Pre-registered decision rule (applied only to reval40).
REL_WIN = 0.01
LEAK_MARGIN = 0.05


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _s_of(inv) -> float:
    inv = np.asarray(inv, float)
    if inv.size == 0:
        return float("nan")
    med = float(np.median(inv))
    b90 = float(np.quantile(inv, 0.9))
    if med <= 0 or b90 <= 0:
        return float("nan")
    return float((math.log(med) + math.log(b90)) / 2.0)


def _summarize_instances(inv, leak) -> dict:
    inv = np.asarray(inv, float)
    leak = np.asarray(leak, float)
    med = float(np.median(inv)) if inv.size else float("nan")
    b90 = float(np.quantile(inv, 0.9)) if inv.size else float("nan")
    return {
        "B": int(inv.size),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _s_of(inv),
        "mean_leak": float(np.mean(leak)) if leak.size else float("nan"),
    }


def ensure_packs(data_dir: Path, ntrain: int, nval: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for n in (6, 9):
        path = data_dir / f"data_n{n}.npz"
        ok = False
        if path.exists():
            z = np.load(path)
            ok = (
                "val_E" in z.files
                and "train_E" in z.files
                and len(z["val_seeds"]) >= nval
                and len(z["train_seeds"]) >= ntrain
            )
            z.close()
        if ok:
            print(f"[data] reuse {path}", flush=True)
            continue
        print(f"[data] building n={n} train={ntrain} val={nval}", flush=True)
        build_dataset(n, data_dir, ntrain=ntrain, nval=nval, ntest=0, store_E=True, overwrite=True)


def walksat_val(n: int, data_dir: Path, max_count: int, R: int, p: float) -> dict:
    pack = load_pack(n, data_dir=data_dir)
    m = int(pack["m"])
    seeds = pack["val_seeds"][:max_count]
    cls_v = pack["val_cls_v"][:max_count]
    cls_s = pack["val_cls_s"][:max_count]
    nsol = pack["val_nsol"][:max_count]
    per_med = []
    per_b90 = []
    t0 = time.time()
    for i in range(len(seeds)):
        cl = [(cls_v[i, c].tolist(), cls_s[i, c].tolist()) for c in range(m)]
        res = walksat_instance(n, cl, R=R, p=p, seed=int(seeds[i]))
        per_med.append(res["flips"]["median"])
        per_b90.append(res["flips"]["B90"])
        if (i + 1) % 10 == 0 or i + 1 == len(seeds):
            print(f"[ws] n={n} {i+1}/{len(seeds)} {time.time()-t0:.1f}s", flush=True)
    uniform = (float(2 ** n) / np.maximum(nsol.astype(float), 1.0)).tolist()
    return {
        "n": n,
        "R": R,
        "p": p,
        "per_inst_median_flips": per_med,
        "per_inst_B90_flips": per_b90,
        "flips_median_of_medians": aggregate_bars(per_med),
        "uniform_inv_p": uniform,
        "uniform_median": float(np.median(uniform)),
        "wall_s": time.time() - t0,
    }


def _ratio(med_q, med_ref) -> float | None:
    if med_ref is None or med_q is None:
        return None
    if not (math.isfinite(med_q) and math.isfinite(med_ref)) or med_ref <= 0:
        return None
    return float(med_q) / float(med_ref)


def _attach_ratio(met: dict, ws_med, ws_b90, uniform_med) -> dict:
    out = dict(met)
    out["ratio"] = _ratio(met.get("median_inv_p"), ws_med)
    out["ratio_b90"] = _ratio(met.get("B90_inv_p"), ws_b90)
    out["ratio_uniform"] = _ratio(met.get("median_inv_p"), uniform_med)
    out["ws_median_flips"] = ws_med
    out["ws_B90_flips"] = ws_b90
    out["uniform_median_inv_p"] = uniform_med
    return out


def _pack_arm_view(by_n: dict) -> dict:
    """by_n: str -> metrics with S."""
    keyed = {int(n): by_n[n] for n in by_n}
    return {
        "by_n": by_n,
        "S_comb": combined_score(keyed),
    }


def decide(learned_reval, freeze_reval, learned_hold, freeze_hold) -> dict:
    """Winner only from reval40, and only if the disjoint holdout slice agrees.

    Lower S_comb is better. Relative gap must be at least REL_WIN.
    A leak increase above LEAK_MARGIN on either n blocks the claim.
    """
    sl = float(learned_reval["S_comb"])
    sf = float(freeze_reval["S_comb"])
    base = {
        "rule": (
            f"reval40 S_comb lower by >= {REL_WIN:.0%} relative, "
            f"mean leak not worse by > {LEAK_MARGIN} on either n, "
            "and disjoint val[20:40] S_comb agrees"
        ),
        "rel_gap": None,
        "lower_S_reval40": None,
        "leak_blocks": False,
        "holdout_agrees": None,
        "winner": None,
    }
    if not (math.isfinite(sl) and math.isfinite(sf)):
        base["reason"] = "non-finite reval40 S_comb"
        return base
    denom = max(abs(sl), abs(sf), 1e-12)
    rel = abs(sl - sf) / denom
    base["rel_gap"] = rel
    if sl < sf:
        lower = "learned"
    elif sf < sl:
        lower = "freeze1"
    else:
        lower = None
    base["lower_S_reval40"] = lower
    if lower is None or rel < REL_WIN:
        base["reason"] = (
            f"no clear reval40 win (relative |ΔS|/max|S|={rel:.4f}, threshold {REL_WIN})"
        )
        return base
    better = learned_reval if lower == "learned" else freeze_reval
    other = freeze_reval if lower == "learned" else learned_reval
    leak_blocks = False
    for n in ("6", "9"):
        if better["by_n"][n]["mean_leak"] > other["by_n"][n]["mean_leak"] + LEAK_MARGIN:
            leak_blocks = True
    base["leak_blocks"] = leak_blocks
    if leak_blocks:
        base["reason"] = (
            f"reval40 S_comb favors {lower} by {rel:.2%}, but mean leak is worse "
            f"by more than {LEAK_MARGIN} on at least one n; no win claimed"
        )
        return base
    sh_l = float(learned_hold["S_comb"])
    sh_f = float(freeze_hold["S_comb"])
    if not (math.isfinite(sh_l) and math.isfinite(sh_f)):
        base["holdout_agrees"] = False
        base["reason"] = "reval40 favors a side but holdout S_comb is non-finite; no win claimed"
        return base
    hold_lower = "learned" if sh_l < sh_f else "freeze1" if sh_f < sh_l else None
    agrees = hold_lower == lower
    base["holdout_agrees"] = agrees
    base["holdout_lower_S"] = hold_lower
    if not agrees:
        base["reason"] = (
            f"reval40 S_comb favors {lower} by {rel:.2%}, but disjoint val[20:40] "
            f"favors {hold_lower}; not a robust win"
        )
        return base
    base["winner"] = lower
    base["reason"] = (
        f"reval40 S_comb favors {lower} by {rel:.2%} and disjoint val[20:40] agrees"
    )
    return base


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _row(arm, n, met) -> str:
    return (
        f"| {arm} | {n} | {_fmt(met['median_inv_p'], 3)} | {_fmt(met['B90_inv_p'], 3)} | "
        f"{_fmt(met['mean_leak'], 4)} | {_fmt(met.get('ratio'), 3)} |"
    )


def write_report(path: Path, payload: dict) -> None:
    r = payload["recipe"]
    dec = payload["decision"]
    lines = []
    lines.append("# q≡1 freeze vs learned q")
    lines.append("")
    lines.append("Clean paired A/B on `qtimes_cost`, recipe matched to box BEST **hot0003**.")
    lines.append("")
    lines.append("## Recipe")
    lines.append("")
    lines.append(f"- L = {r['L']}, leakpen = {r['leakpen']}, lr = {r['lr']}")
    lines.append("- learned α0 (both arms; `head_alpha0` stays in Adam)")
    lines.append("- early-stop on S_comb = 0.35·S(n=6) + 0.65·S(n=9), S = (log Med + log B90) / 2")
    lines.append(f"- patience = {r['patience']} evals, max steps = {r['max_steps']}, min_delta = {r['min_delta']}")
    lines.append(f"- CPU threads requested = {r['threads']} (intra-op). Host CPUs = {r['host_cpus']}, interop threads = {r['interop_threads']}")
    lines.append(f"- arch = baseline factor GNN {json.dumps(r['arch'])}")
    lines.append(f"- seed = {r['seed']}, max_train = {r['max_train']}, batch = 4 per n")
    lines.append(f"- select val = first {r['max_val_select']} val instances; official reval = first {r['max_val_reval']}")
    lines.append("- learned q: `1 + 9·sigmoid(u)`")
    lines.append("- freeze q: `q_from_logit(..., freeze_q1=True)` returns exact 1 (logit × 0), logits zeroed, `head_q` removed from Adam")
    lines.append(f"- ratio = Med(1/p) / Med(WalkSAT flips) on the same instances (R={r['ws_R']}, p={r['ws_p']})")
    lines.append("- init checksum of every parameter except `head_q` matched across arms")
    lines.append("")
    lines.append("## Decision rule (fixed before the run)")
    lines.append("")
    lines.append(dec["rule"] + ".")
    lines.append("Select-val S_comb is not used to name a winner.")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    winner = dec.get("winner")
    if winner:
        lines.append(f"**Reval40 win: `{winner}`.** {dec['reason']}.")
    else:
        lines.append(f"**No win claimed.** {dec['reason']}.")
    lines.append("")
    lines.append(
        f"Reval40 S_comb learned = {_fmt(payload['reval40']['learned']['S_comb'], 4)}, "
        f"freeze1 = {_fmt(payload['reval40']['freeze1']['S_comb'], 4)} "
        f"(relative gap {_fmt(dec.get('rel_gap'), 4)})."
    )
    lines.append(
        f"Disjoint val[20:40] S_comb learned = {_fmt(payload['holdout20']['learned']['S_comb'], 4)}, "
        f"freeze1 = {_fmt(payload['holdout20']['freeze1']['S_comb'], 4)}."
    )
    lines.append("")
    lines.append("### Select val (early-stop objective, not the claim)")
    lines.append("")
    lines.append("| arm | n | Med | B90 | leak | ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ("learned", "freeze1"):
        view = payload["select_val"][arm]
        for n in ("6", "9"):
            lines.append(_row(arm, n, view["by_n"][n]))
        lines.append(f"| {arm} | S_comb | {_fmt(view['S_comb'], 4)} |  |  |  |")
    lines.append("")
    lines.append("### Reval max_val=40 (official)")
    lines.append("")
    lines.append("| arm | n | Med | B90 | leak | ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ("learned", "freeze1"):
        view = payload["reval40"][arm]
        for n in ("6", "9"):
            lines.append(_row(arm, n, view["by_n"][n]))
        lines.append(f"| {arm} | S_comb | {_fmt(view['S_comb'], 4)} |  |  |  |")
    lines.append("")
    lines.append("### Disjoint slice val[20:40] (robustness, not a second claim)")
    lines.append("")
    lines.append("| arm | n | Med | B90 | leak | ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ("learned", "freeze1"):
        view = payload["holdout20"][arm]
        for n in ("6", "9"):
            lines.append(_row(arm, n, view["by_n"][n]))
        lines.append(f"| {arm} | S_comb | {_fmt(view['S_comb'], 4)} |  |  |  |")
    lines.append("")
    lines.append("## Training")
    lines.append("")
    lines.append("| arm | best step | last step | early-stopped | hit step cap | select S_comb | mean q (reval n=6 / n=9) |")
    lines.append("|---|---:|---:|---|---|---:|---|")
    for arm in ("learned", "freeze1"):
        info = payload["arms"][arm]["info"]
        rq = payload["reval40"][arm]["by_n"]
        lines.append(
            f"| {arm} | {info['best_step']} | {info['last_step']} | {info['early_stopped']} | "
            f"{info['hit_step_cap']} | {_fmt(info.get('best_score'), 4)} | "
            f"{_fmt(rq['6']['mean_q'], 3)} / {_fmt(rq['9']['mean_q'], 3)} |"
        )
    lines.append("")
    lines.append("Learned-q per-layer means on reval40 are in `results.json` (`q_per_layer`).")
    lines.append("Freeze arm mean q is 1.0 on every layer.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Single seed. This is a paired point estimate, not a significance test.")
    lines.append("- Select val[:20] is a prefix of reval val[:40], so reval is not a pure holdout. The disjoint slice is val[20:40].")
    lines.append("- Train loss is the unweighted mean of the n=6 and n=9 mini-batch losses. The 0.35/0.65 weights apply only to S_comb.")
    lines.append("- WalkSAT is pure-Python SKC (p=0.5) on the val clauses, median flips per instance, then the median across instances.")
    lines.append("- `ratio_uniform` in the json is Med(1/p) / median(2^n / n_solutions) and is not the headline ratio.")
    lines.append("- No `runs/gnn_best` weights were loaded. If hot0003 used a different evolved arch, this run matches the listed train knobs on the baseline arch only.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run(args) -> dict:
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        print(f"[threads] interop left unchanged: {exc}", flush=True)
    torch.set_num_threads(args.threads)
    print(
        f"[threads] intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()} "
        f"host_cpus={os.cpu_count()}",
        flush=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data) if args.data else _PKG / "data"
    ensure_packs(data_dir, ntrain=max(args.max_train, 64), nval=max(args.max_val_reval, 40))

    arch_d = BASELINE_ARCH.to_dict()
    arch_d["lr"] = args.lr
    arch = GNNArch.from_dict(arch_d)

    ws = {}
    for n in (6, 9):
        ws[str(n)] = walksat_val(n, data_dir, args.max_val_reval, R=args.ws_R, p=args.ws_p)

    log_path = out_dir / "train.log"
    log_f = open(log_path, "w")

    def make_log(tag):
        def _log(*parts):
            msg = time.strftime("%H:%M:%S") + f" [{tag}] " + " ".join(str(p) for p in parts)
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()
        return _log

    arms_out = {}
    states = {}
    for mode in ("learned", "freeze1"):
        t0 = time.time()
        val, state, info = short_train(
            arch,
            steps=args.steps,
            L=args.L,
            max_val=args.max_val_select,
            max_train=args.max_train,
            seed=args.seed,
            data_dir=data_dir,
            device="cpu",
            leakpen=args.leakpen,
            q_mode=mode,
            patience=args.patience,
            num_threads=args.threads,
            log_fn=make_log(mode),
        )
        info = dict(info)
        info["wall_s"] = time.time() - t0
        if not state:
            raise RuntimeError(f"{mode} produced no checkpoint")
        wdir = out_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        for n, sd in state.items():
            torch.save(sd, wdir / f"{mode}_n{n}.pt")
        states[mode] = state
        arms_out[mode] = {"select_metrics": {str(k): v for k, v in val.items()}, "info": info}
        log_f.write(f"[{mode}] done wall_s={info['wall_s']:.1f} best_step={info['best_step']}\n")
        log_f.flush()

    if arms_out["learned"]["info"]["init_checksum"] != arms_out["freeze1"]["info"]["init_checksum"]:
        raise RuntimeError("init checksum mismatch — A/B is not paired")

    def reval(mode: str, max_count: int) -> dict:
        by_n = {}
        for n in (6, 9):
            # Reload the checkpoint that was written, so reval is not the in-memory object.
            sd = torch.load(out_dir / "weights" / f"{mode}_n{n}.pt", map_location="cpu", weights_only=True)
            met = eval_pack_metrics(
                n, arch, sd, split="val", L=args.L, max_count=max_count, seed=args.seed,
                data_dir=data_dir, device="cpu", q_mode=mode, return_instances=True,
            )
            w = ws[str(n)]
            # Same val prefix as this eval. WS Med/B90 are the median and 90th
            # percentile of per-instance median flips (aggregate_bars).
            k = met["B"]
            flips = w["per_inst_median_flips"][:k]
            ws_med = float(np.median(flips))
            ws_b90 = float(np.quantile(flips, 0.9))
            uni = float(np.median(w["uniform_inv_p"][:k]))
            by_n[str(n)] = _attach_ratio(met, ws_med, ws_b90, uni)
        return _pack_arm_view(by_n)

    def holdout_from_reval(reval_view: dict) -> dict:
        by_n = {}
        for n in ("6", "9"):
            met = reval_view["by_n"][n]
            inv = met["inv_p"][20:40]
            leak = met["p_leak"][20:40]
            summary = _summarize_instances(inv, leak)
            w = ws[n]
            flips = w["per_inst_median_flips"][20:40]
            uni = w["uniform_inv_p"][20:40]
            summary["mean_q"] = met["mean_q"]
            summary["q_per_layer"] = met["q_per_layer"]
            by_n[n] = _attach_ratio(
                summary,
                float(np.median(flips)) if flips else None,
                float(np.quantile(flips, 0.9)) if flips else None,
                float(np.median(uni)) if uni else None,
            )
        return _pack_arm_view(by_n)

    reval40 = {mode: reval(mode, args.max_val_reval) for mode in ("learned", "freeze1")}
    holdout = {mode: holdout_from_reval(reval40[mode]) for mode in ("learned", "freeze1")}

    # Select-val ratios on the prefix used for early stop.
    select_val = {}
    for mode in ("learned", "freeze1"):
        by_n = {}
        raw = arms_out[mode]["select_metrics"]
        for n in ("6", "9"):
            w = ws[n]
            k = int(raw[n]["B"])
            flips = w["per_inst_median_flips"][:k]
            by_n[n] = _attach_ratio(
                raw[n],
                float(np.median(flips)),
                float(np.quantile(flips, 0.9)),
                float(np.median(w["uniform_inv_p"][:k])),
            )
        select_val[mode] = _pack_arm_view(by_n)

    # Freeze must be exact 1 on the official reval.
    for n in ("6", "9"):
        mq = reval40["freeze1"]["by_n"][n]["mean_q"]
        if abs(mq - 1.0) > 1e-6:
            raise RuntimeError(f"freeze1 reval mean_q n={n} is {mq}, expected 1")

    decision = decide(reval40["learned"], reval40["freeze1"], holdout["learned"], holdout["freeze1"])

    payload = {
        "recipe": {
            "name": "hot0003",
            "L": args.L,
            "leakpen": args.leakpen,
            "lr": args.lr,
            "alpha0": "learned",
            "early_stop": "S_comb",
            "threads": args.threads,
            "host_cpus": os.cpu_count(),
            "intra_threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
            "patience": args.patience,
            "max_steps": args.steps,
            "min_delta": 1e-6,
            "max_val_select": args.max_val_select,
            "max_val_reval": args.max_val_reval,
            "max_train": args.max_train,
            "batch_per_n": 4,
            "seed": args.seed,
            "ws_R": args.ws_R,
            "ws_p": args.ws_p,
            "arch": arch.to_dict(),
            "q_learned": "1+9*sigmoid",
            "q_freeze": "q_from_logit freeze_q1 exact 1, logits zeroed",
            "S_comb_weights": {"6": 0.35, "9": 0.65},
        },
        "walksat_val": ws,
        "arms": {
            mode: {
                "info": arms_out[mode]["info"],
            }
            for mode in ("learned", "freeze1")
        },
        "select_val": select_val,
        "reval40": reval40,
        "holdout20": holdout,
        "decision": decision,
        "init_checksum_match": True,
    }
    payload = _clean(payload)
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(out_dir / "REPORT.md", payload)
    log_f.write("wrote " + str(json_path) + "\n")
    log_f.close()
    print("Wrote", json_path, flush=True)
    print("Wrote", out_dir / "REPORT.md", flush=True)
    print("DECISION", decision["reason"], flush=True)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_PKG / "runs" / "cloud_q1_ab"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--leakpen", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--max-val-select", type=int, default=20)
    ap.add_argument("--max-val-reval", type=int, default=40)
    ap.add_argument("--max-train", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ws-R", type=int, default=20)
    ap.add_argument("--ws-p", type=float, default=0.5)
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
