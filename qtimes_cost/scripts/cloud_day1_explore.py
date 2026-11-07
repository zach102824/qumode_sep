#!/usr/bin/env python3
"""Complementary CPU explore for qtimes_cost (cloud day 1).

Box NEW BEST is L=10, leakpen=0.1, lr=5e-5 (small val n6≈10.2 / n9≈18,
reval40 ≈14.5 / 19.5). This farm asks whether L∈{12,14} at a similar lr and
leakpen can beat that reval40 bar. α0 stays the learned per-cavity head.
A same-budget L=10 control is included so depth is compared under one protocol.
Search-val scores do not count as a win.

Search scores use a small val and early-stop on S_comb. BEST is mid-saved on
each promote. After the farm, the search champion and nearby candidates are
re-evaluated on a larger val and on the test split. Test does not select.

  python -m qtimes_cost.scripts.cloud_day1_explore --workers 4 --steps 400
"""
from __future__ import annotations

import os

# One intra-op thread per worker. The pool width is the parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import multiprocessing as mp
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from qtimes_cost.lib.ansatz import AnsatzConfig, MultiCavityAnsatz
from qtimes_cost.lib.cloud_worker import eval_nets as _eval_nets, run_trial as _worker
from qtimes_cost.lib.gnn import GNNArch, FactorGNN
from qtimes_cost.lib.promote import BOX_NEW, BOX_R014, beats_box_reval, promotion_decision, ratio_9_6
from qtimes_cost.lib.train_loop import PKG, _split_arrays, combined_score, load_pack

# Equal step budget. Depth trials are not killed for trailing mid-run:
# L=14 learns slower than L=10 and needs the full budget.
SEARCH = {
    "max_steps": 1000,
    "eval_every": 40,
    "patience_evals": 5,
    "min_steps": 480,
    "batch": 4,
    "max_train": 128,
    "max_val": 16,
    "hopeless_after": 280,
    "hopeless_med9": 400.0,
    "hopeless_S": 6.2,
    "kill_after": 10**9,
    "kill_margin": 2.0,
}
REVAL_VAL = 40
REVAL_TEST = 100

CSV_FIELDS = [
    "trial_id", "tag", "L", "leakpen", "lr", "seed", "H", "rounds",
    "steps_ran", "best_step", "wall_s", "status",
    "S_comb", "med6", "b90_6", "S6", "leak6",
    "med9", "b90_9", "S9", "leak9",
    "ratio_9_6", "q_med", "keep", "reason", "best_S_after", "alpha0",
]


def _cell(L, lr, lp, seed=0, H=128, rounds=4, tag="grid") -> dict:
    return {
        "L": L, "lr": lr, "leakpen": lp, "seed": seed,
        "H": H, "rounds": rounds, "tag": tag,
    }


def build_grid() -> list[dict]:
    """Deeper L around the box lr/leakpen, plus an L=10 same-budget control.

    Listed best-first so the first wave is the comparison that matters.
    """
    rows = [
        _cell(12, 5e-5, 0.1, tag="core"),
        _cell(14, 5e-5, 0.1, tag="core"),
        _cell(10, 5e-5, 0.1, tag="l10_control"),
        _cell(12, 5e-5, 0.2, tag="core"),
        _cell(14, 5e-5, 0.2, tag="core"),
        _cell(12, 5e-5, 0.05, tag="core"),
        _cell(14, 5e-5, 0.05, tag="core"),
        _cell(12, 1e-4, 0.1, tag="grid"),
        _cell(14, 1e-4, 0.1, tag="grid"),
        _cell(12, 3e-5, 0.1, tag="grid"),
        _cell(14, 3e-5, 0.1, tag="grid"),
        _cell(12, 5e-5, 0.3, tag="grid"),
        _cell(14, 5e-5, 0.3, tag="grid"),
        _cell(12, 5e-5, 0.1, seed=1, tag="seed"),
        _cell(14, 5e-5, 0.1, seed=1, tag="seed"),
        _cell(10, 5e-5, 0.1, seed=1, tag="l10_control"),
        _cell(12, 1e-4, 0.2, tag="grid"),
        _cell(14, 1e-4, 0.2, tag="grid"),
        _cell(12, 5e-5, 0.1, H=192, tag="gnn_H192"),
        _cell(14, 5e-5, 0.1, rounds=6, tag="gnn_rounds6"),
        _cell(12, 3e-5, 0.2, tag="grid"),
        _cell(14, 3e-5, 0.2, tag="grid"),
    ]
    for i, row in enumerate(rows):
        row["trial_id"] = i
        row["alpha0"] = "learned"
    return rows



def _gate_view(sc: float, metrics: dict) -> dict:
    return {
        "s_comb": float(sc),
        "med6": float(metrics[6]["median_inv_p"]),
        "med9": float(metrics[9]["median_inv_p"]),
        "leak9": float(metrics[9]["mean_leak"]),
    }


def _row_from_result(result: dict, keep: bool, reason: str, best_s_after: float) -> dict:
    cfg = result["cfg"]
    m = result.get("metrics") or {}
    m6 = m.get(6) or m.get("6") or {}
    m9 = m.get(9) or m.get("9") or {}
    # metrics keys are ints inside the worker; after any json roundtrip they are str.
    # The in-process result uses int keys.
    if not m6 and 6 in m:
        m6 = m[6]
    if not m9 and 9 in m:
        m9 = m[9]
    med6 = float(m6.get("median_inv_p", float("nan")))
    med9 = float(m9.get("median_inv_p", float("nan")))
    return {
        "trial_id": result["trial_id"],
        "tag": cfg.get("tag", ""),
        "L": cfg.get("L", ""),
        "leakpen": cfg.get("leakpen", ""),
        "lr": cfg.get("lr", ""),
        "seed": cfg.get("seed", ""),
        "H": cfg.get("H", ""),
        "rounds": cfg.get("rounds", ""),
        "steps_ran": result.get("steps_ran", 0),
        "best_step": result.get("best_step", -1),
        "wall_s": f"{float(result.get('wall_s', 0.0)):.2f}",
        "status": result.get("status", ""),
        "S_comb": "" if not m6 else f"{float(result['s_comb']):.6f}",
        "med6": "" if not m6 else f"{med6:.6f}",
        "b90_6": "" if not m6 else f"{float(m6['B90_inv_p']):.6f}",
        "S6": "" if not m6 else f"{float(m6['S']):.6f}",
        "leak6": "" if not m6 else f"{float(m6['mean_leak']):.6f}",
        "med9": "" if not m9 else f"{med9:.6f}",
        "b90_9": "" if not m9 else f"{float(m9['B90_inv_p']):.6f}",
        "S9": "" if not m9 else f"{float(m9['S']):.6f}",
        "leak9": "" if not m9 else f"{float(m9['mean_leak']):.6f}",
        "ratio_9_6": "" if not m6 else f"{ratio_9_6(med6, med9):.6f}",
        "q_med": "" if not m9 else f"{float(m9.get('q_med', float('nan'))):.4f}",
        "keep": int(bool(keep)),
        "reason": reason,
        "best_S_after": f"{best_s_after:.6f}" if math.isfinite(best_s_after) else "",
        "alpha0": "learned",
    }


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        f.flush()


def _save_best(root: Path, result: dict, reason: str, kind: str = "search") -> None:
    dest = root / "BEST"
    tmp = root / f".BEST_tmp_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    arch = result.get("arch") or {}
    payload = {
        "kind": kind,
        "reason": reason,
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trial_id": result["trial_id"],
        "cfg": result["cfg"],
        "arch": arch,
        "status": result.get("status"),
        "steps_ran": result.get("steps_ran"),
        "best_step": result.get("best_step"),
        "wall_s": result.get("wall_s"),
        "s_comb": result.get("s_comb"),
        "metrics": _json_metrics(result.get("metrics") or {}),
        "alpha0": "learned",
        "box_r014": BOX_R014,
    }
    (tmp / "best.json").write_text(json.dumps(payload, indent=2))
    (tmp / "arch.json").write_text(json.dumps({"arch": arch, "backend": "torch"}, indent=2))
    for n, state in (result.get("states") or {}).items():
        torch.save(state, tmp / f"weights_n{n}.pt")
    prev = root / ".BEST_prev"
    if prev.exists():
        shutil.rmtree(prev)
    if dest.exists():
        dest.rename(prev)
    tmp.rename(dest)
    if prev.exists():
        shutil.rmtree(prev)


def _json_metrics(metrics: dict) -> dict:
    out = {}
    for n, m in metrics.items():
        out[str(n)] = {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in m.items()}
    return out


def _build_loaded(arch_d: dict, L: int, states: dict):
    arch = GNNArch.from_dict(arch_d)
    nets, sims, packs = {}, {}, {}
    for n in (6, 9):
        pack = load_pack(n)
        gnn = FactorGNN(arch, n=n, m=int(pack["m"]), L=L, seed=0)
        gnn.load_state_dict(states[n])
        gnn.eval()
        nets[n] = gnn
        sims[n] = MultiCavityAnsatz(AnsatzConfig(n=n, L=L))
        packs[n] = pack
    return nets, sims, packs


def reval_states(arch_d: dict, L: int, states: dict, max_val: int, max_test: int) -> dict:
    nets, sims, packs = _build_loaded(arch_d, L, states)
    out = {}
    for split, count in (("val", max_val), ("test", max_test)):
        arrays = {}
        for n, pack in packs.items():
            arrays[n] = _split_arrays(pack, split, max_count=count)
        metrics = _eval_nets(nets, sims, arrays, count)
        sc = combined_score(metrics)
        out[split] = {
            "S_comb": sc,
            "by_n": _json_metrics(metrics),
            "ratio_9_6": ratio_9_6(metrics[6]["median_inv_p"], metrics[9]["median_inv_p"]),
            "gate": _gate_view(sc, metrics),
        }
    return out


def _fmt(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def _vs_r014(med6: float, med9: float) -> dict:
    ratio = ratio_9_6(med6, med9)
    return {
        "med6": med6,
        "med9": med9,
        "delta_med6_vs_18": med6 - BOX_R014["med6"],
        "delta_med9_vs_23": med9 - BOX_R014["med9"],
        "beats_n6_med": bool(med6 < BOX_R014["med6"]),
        "beats_n9_med": bool(med9 < BOX_R014["med9"]),
        "ratio_9_6": ratio,
        "r014_ratio_9_6": BOX_R014["ratio_9_6"],
        "ratio_healthy": bool(0.75 <= ratio <= 2.0),
    }


def _val_pair(item: dict) -> tuple[float, float]:
    vn = item["reval"]["val"]["by_n"]
    return float(vn["6"]["median_inv_p"]), float(vn["9"]["median_inv_p"])


def depth_verdict(also: list[dict]) -> dict:
    """Judge L=12/14 on val-40 against the box reval bar. Search val is ignored."""
    def pack(item):
        m6, m9 = _val_pair(item)
        info = beats_box_reval(m6, m9)
        cfg = item["cfg"]
        info["trial_id"] = item["trial_id"]
        info["cfg"] = {
            "tag": cfg.get("tag"), "L": cfg.get("L"), "leakpen": cfg.get("leakpen"),
            "lr": cfg.get("lr"), "seed": cfg.get("seed"), "H": cfg.get("H"),
            "rounds": cfg.get("rounds"),
        }
        info["S_comb"] = item["reval"]["val"]["S_comb"]
        return info

    deep = [pack(it) for it in also if int(it["cfg"]["L"]) in (12, 14)]
    ctrl = [pack(it) for it in also if int(it["cfg"]["L"]) == 10]
    deep_by_s = sorted(deep, key=lambda d: d["S_comb"])
    deep_by_n9 = sorted(deep, key=lambda d: d["med9"])
    ctrl_by_s = sorted(ctrl, key=lambda d: d["S_comb"])
    return {
        "box_reval40": {"med6": BOX_NEW["reval40_med6"], "med9": BOX_NEW["reval40_med9"]},
        "best_deep_by_S": deep_by_s[0] if deep_by_s else None,
        "best_deep_by_n9": deep_by_n9[0] if deep_by_n9 else None,
        "l10_control": ctrl_by_s[0] if ctrl_by_s else None,
        "any_deep_full_win": any(d["full_win"] for d in deep),
    }


def _verdict_lines(v: dict) -> list[str]:
    if not v:
        return ["No val-40 reval was recorded."]
    lines = []
    deep = v.get("best_deep_by_S")
    deep9 = v.get("best_deep_by_n9")
    ctrl = v.get("l10_control")
    if not deep:
        lines.append("No L=12 or L=14 trial has a val-40 reval.")
    else:
        c = deep["cfg"]
        lines.append(
            f"Best L∈{{12,14}} by val-40 S_comb is trial {deep['trial_id']} "
            f"(L={c['L']}, leakpen={c['leakpen']}, lr={c['lr']}, tag={c['tag']}, seed={c['seed']}): "
            f"n6={deep['med6']:.2f}, n9={deep['med9']:.2f}, ratio={deep['ratio_9_6']:.3f}."
        )
        if deep.get("full_win"):
            lines.append(
                "This beats the box reval40 bar of 14.5 / 19.5 on both medians, with a healthy ratio."
            )
        elif deep.get("beats_n9") and not deep.get("beats_n6"):
            lines.append("n9 median beats 19.5, and n6 does not beat 14.5. Not a full win.")
        elif deep.get("beats_n6") and not deep.get("beats_n9"):
            lines.append(
                "n6 median beats 14.5, and n9 does not beat 19.5. Not a full win. "
                "Extra depth did not beat the box n9 median."
            )
        else:
            lines.append(
                "This does not beat the box reval40 bar of 14.5 / 19.5. "
                "Deeper L did not clear that bar under this CPU step budget."
            )
        if not deep.get("ratio_healthy"):
            lines.append("The n9/n6 ratio is outside [0.75, 2.0], so ratio health fails too.")
    if deep and deep9 and deep9["trial_id"] != deep["trial_id"]:
        lines.append(
            f"Best deep trial by n9 median alone is trial {deep9['trial_id']}: "
            f"n6={deep9['med6']:.2f}, n9={deep9['med9']:.2f}, full_win={deep9['full_win']}."
        )
    if ctrl:
        lines.append(
            f"Same-budget L=10 control trial {ctrl['trial_id']}: val-40 n6={ctrl['med6']:.2f}, "
            f"n9={ctrl['med9']:.2f}, ratio={ctrl['ratio_9_6']:.3f}. "
            "This is what the step budget does at the box depth. It is not a new box result."
        )
        if deep:
            if deep["med9"] + 1e-9 < ctrl["med9"]:
                lines.append("The best deep trial has a lower n9 median than that L=10 control.")
            else:
                lines.append(
                    "The L=10 control has an n9 median at least as good as the best deep trial. "
                    "Extra depth did not help n9 at equal steps."
                )
    if v.get("any_deep_full_win"):
        lines.append("At least one L=12 or L=14 val-40 reval is a full win against 14.5 / 19.5.")
    elif deep:
        lines.append("No L=12 or L=14 val-40 reval is a full win against 14.5 / 19.5.")
    return lines


def render_report(rows: list[dict], final: dict, protocol: dict) -> str:
    ranked = [r for r in rows if r.get("S_comb")]
    ranked.sort(key=lambda r: float(r["S_comb"]))
    lines = []
    lines.append("# cloud_day1 CPU explore")
    lines.append("")
    lines.append(
        "Target is the box **NEW BEST**: L=10, leakpen=0.1, lr=5e-5, learned per-cavity α0. "
        "Small val n6 Med≈10.2 / n9≈18. **reval40 ≈ 14.5 / 19.5** is the bar. "
        "A result is a win only when a val-40 reval beats both of those medians and the "
        "n9/n6 ratio stays in [0.75, 2.0]. Search-val numbers are not a win."
    )
    lines.append("")
    lines.append(
        "The question for this farm is whether **L∈{12,14}** at a similar lr and leakpen "
        "can beat that bar. An L=10 run at the same lr, leakpen, and step count is a control, "
        "not a new box config. Learned α0 only. Earlier r014 (L=10, lr=1e-4, small-val 18/23) "
        "is an easier, superseded reference."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"- Workers: {protocol['workers']} processes × 1 torch thread "
        f"(host CPUs: {protocol['cpus']})."
    )
    lines.append(
        f"- Search: {protocol['steps']} Adam steps, batch {SEARCH['batch']} per n, "
        f"train pool {SEARCH['max_train']}, val {SEARCH['max_val']} "
        f"(early-stop on S_comb, patience {SEARCH['patience_evals']} evals)."
    )
    lines.append(
        "- S_comb = 0.35·S6 + 0.65·S9, S = (log Med + log B90) / 2. Lower is better."
    )
    lines.append(
        "- Promote only when S_comb improves **and** n9 Med does not regress. "
        "A healthier n6→n9 ratio with a worse n9 Med is rejected (`ratio_only`)."
    )
    lines.append(
        f"- Reval: val n={protocol['max_val']} and test n={protocol['max_test']} "
        "(test is reported, not used to pick the winner). nf=24 (library default)."
    )
    lines.append(
        "- Grad clip 5.0, same as `short_train`. nf=24. No inter-trial kill: "
        "deeper L is slower early and keeps its full step budget."
    )
    lines.append("")
    lines.append("## Box reference")
    lines.append("")
    lines.append("| | L | leakpen | lr | α0 | n6 Med | n9 Med | what it is |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---|")
    lines.append("| box NEW BEST, small val | 10 | 0.1 | 5e-5 | learned | 10.2 | 18 | optimistic |")
    lines.append("| **box NEW BEST, reval40** | 10 | 0.1 | 5e-5 | learned | **14.5** | **19.5** | win bar |")
    lines.append("| old r014, small val | 10 | 0.1 | 1e-4 | learned | 18 | 23 | superseded |")
    lines.append("")
    lines.append("## Search trials")
    lines.append("")
    lines.append(
        "Med/B90 are 1/p on the **search val** (16 formulas). "
        "Lower is better. keep=1 means that trial promoted when it finished."
    )
    lines.append("")
    lines.append("| id | tag | L | lp | lr | H | r | steps | status | S_comb | n6 Med/B90 | n9 Med/B90 | ratio | leak9 | keep |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|")
    for r in ranked:
        lines.append(
            "| {trial_id} | {tag} | {L} | {leakpen} | {lr} | {H} | {rounds} | {steps_ran} | {status} | "
            "{S} | {m6}/{b6} | {m9}/{b9} | {ratio} | {lk} | {keep} |".format(
                trial_id=r["trial_id"],
                tag=r["tag"],
                L=r["L"],
                leakpen=r["leakpen"],
                lr=r["lr"],
                H=r["H"],
                rounds=r["rounds"],
                steps_ran=r["steps_ran"],
                status=r["status"],
                S=_fmt(r["S_comb"], 3),
                m6=_fmt(r["med6"], 1),
                b6=_fmt(r["b90_6"], 1),
                m9=_fmt(r["med9"], 1),
                b9=_fmt(r["b90_9"], 1),
                ratio=_fmt(r["ratio_9_6"], 2),
                lk=_fmt(r["leak9"], 3),
                keep=r["keep"],
            )
        )
    lines.append("")
    lines.append(f"Trials logged: **{len(rows)}** (finite scores: {len(ranked)}).")
    lines.append("")
    fb = final.get("search_best") or {}
    rv = final.get("reval_best") or {}
    lines.append("## Winner")
    lines.append("")
    if not rv:
        lines.append("No finite trial finished.")
        lines.append("")
        return "\n".join(lines)

    cfg = rv.get("cfg") or {}
    val = (rv.get("reval") or {}).get("val") or {}
    test = (rv.get("reval") or {}).get("test") or {}
    lines.append(
        f"Search champion was trial **{fb.get('trial_id')}** "
        f"(S_comb={_fmt(fb.get('s_comb'), 3)} on val {SEARCH['max_val']}). "
        f"Final BEST is trial **{rv.get('trial_id')}** "
        f"(`{cfg.get('tag')}`, L={cfg.get('L')}, leakpen={cfg.get('leakpen')}, "
        f"lr={cfg.get('lr')}, H={cfg.get('H')}, rounds={cfg.get('rounds')}, seed={cfg.get('seed')}), "
        f"chosen on the val-{protocol['max_val']} reval"
        + (" (same trial)." if rv.get("trial_id") == fb.get("trial_id") else " (replaced the search champion).")
    )
    lines.append("")
    lines.append("| split | n | B | Med | B90 | S | mean leak |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split_name, block in (("val", val), ("test", test)):
        by_n = block.get("by_n") or {}
        for n in ("6", "9"):
            m = by_n.get(n) or {}
            if not m:
                continue
            lines.append(
                f"| {split_name} | {n} | {m.get('B', '')} | {_fmt(m.get('median_inv_p'), 2)} | "
                f"{_fmt(m.get('B90_inv_p'), 2)} | {_fmt(m.get('S'), 3)} | {_fmt(m.get('mean_leak'), 4)} |"
            )
    lines.append("")
    verdict = final.get("depth_verdict") or {}
    lines.append("### Versus box reval40 (14.5 / 19.5)")
    lines.append("")
    lines.extend(_verdict_lines(verdict))
    lines.append("")
    others = final.get("also_reval") or []
    if others:
        lines.append(f"### Other revals (val {protocol['max_val']} / test)")
        lines.append("")
        lines.append("| id | tag | L | lp | lr | H | r | val n6 | val n9 | val ratio | val S | test n6 | test n9 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in others:
            c = item["cfg"]
            v = item["reval"]["val"]
            te = item["reval"]["test"]
            vn = v["by_n"]
            tn = te["by_n"]
            lines.append(
                f"| {item['trial_id']} | {c['tag']} | {c['L']} | {c['leakpen']} | {c['lr']} | {c['H']} | {c['rounds']} | "
                f"{_fmt(vn['6']['median_inv_p'], 1)} | {_fmt(vn['9']['median_inv_p'], 1)} | "
                f"{_fmt(v['ratio_9_6'], 2)} | {_fmt(v['S_comb'], 3)} | "
                f"{_fmt(tn['6']['median_inv_p'], 1)} | {_fmt(tn['9']['median_inv_p'], 1)} |"
            )
        lines.append("")
    lines.append("## Forks")
    lines.append("")
    forks = [r for r in ranked if str(r["tag"]).startswith("gnn_")]
    if not forks:
        lines.append("No GNN fork completed.")
    else:
        for r in forks:
            lines.append(
                f"- `{r['tag']}` trial {r['trial_id']}: L={r['L']} lp={r['leakpen']} lr={r['lr']} "
                f"H={r['H']} rounds={r['rounds']} → search S={_fmt(r['S_comb'], 3)} "
                f"n6={_fmt(r['med6'], 1)} n9={_fmt(r['med9'], 1)} ratio={_fmt(r['ratio_9_6'], 2)} "
                f"keep={r['keep']} ({r['reason']})."
            )
    lines.append("")
    lines.append(
        "A stronger H or extra rounds is a thin fork on the same ansatz "
        "(learned α0, exact cost phase). It is not an OCT20 calibrated SNAP–D–SNAP mixer; "
        "that stack is a different layer primitive and was not ported."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `trials.csv` — every trial, including early-stops.")
    lines.append("- `BEST/` — final weights (`weights_n6.pt`, `weights_n9.pt`), `best.json`, `arch.json`. Mid-saved on each search promote, then refreshed if the val-40 reval picks a different trial.")
    lines.append("- `BEST_reval.json` — val-40 and test metrics.")
    lines.append("- `BEST_search/` — present only if the reval winner differs from the search champion.")
    lines.append("")
    return "\n".join(lines)


def _select_reval_ids(rows: list[dict], champ_id) -> list[int]:
    finite = []
    for r in rows:
        if not r.get("S_comb"):
            continue
        finite.append(r)
    finite.sort(key=lambda r: float(r["S_comb"]))
    if not finite:
        return []
    best_s = float(finite[0]["S_comb"])
    ids = []
    if champ_id is not None:
        ids.append(int(champ_id))
    for r in finite:
        tid = int(r["trial_id"])
        if tid in ids:
            continue
        if len(ids) >= 5:
            break
        if float(r["S_comb"]) <= best_s + 0.25 or len(ids) < 3:
            ids.append(tid)
    return ids


def _choose_final(search_id: int, revals: list[dict]) -> dict:
    by_id = {int(item["trial_id"]): item for item in revals}
    if search_id in by_id:
        incumbent = by_id[search_id]
        why = "search_champion"
    else:
        incumbent = min(revals, key=lambda item: item["reval"]["val"]["S_comb"])
        why = "best_reval_S"
    for item in sorted(revals, key=lambda it: it["reval"]["val"]["S_comb"]):
        if int(item["trial_id"]) == int(incumbent["trial_id"]):
            continue
        ok, reason = promotion_decision(item["reval"]["val"]["gate"], incumbent["reval"]["val"]["gate"])
        if ok:
            incumbent = item
            why = reason
    incumbent = dict(incumbent)
    incumbent["final_reason"] = why
    return incumbent


def _run_farm(args, root: Path) -> tuple[list[dict], dict]:
    grid = build_grid()
    if args.limit:
        grid = grid[: args.limit]
    search = dict(SEARCH)
    if args.max_val_search:
        search["max_val"] = args.max_val_search
    cpus = os.cpu_count() or 1
    workers = args.workers or min(8, cpus)
    workers = max(1, min(workers, len(grid), cpus))
    print(
        f"cloud_day1: {len(grid)} trials, workers={workers}, cpus={cpus}, steps={args.steps}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    csv_path = root / "trials.csv"
    if csv_path.exists() and not args.append:
        csv_path.unlink()

    states_by_id: dict[int, dict] = {}
    results_by_id: dict[int, dict] = {}
    rows: list[dict] = []
    champ = None
    champ_result = None

    # Manager proxies pickle cleanly. A raw Value does not travel in a task arg.
    with ctx.Manager() as manager, ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        best_s = manager.dict()
        best_s["s"] = 1e300
        best_lock = manager.Lock()
        payloads = [
            {
                "trial": trial,
                "best_s": best_s,
                "best_lock": best_lock,
                "steps": int(args.steps),
                "search": search,
            }
            for trial in grid
        ]
        futs = {pool.submit(_worker, p): p["trial"]["trial_id"] for p in payloads}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                result = fut.result()
            except Exception:
                result = {
                    "trial_id": tid,
                    "cfg": next(t for t in grid if t["trial_id"] == tid),
                    "arch": {},
                    "status": "error",
                    "steps_ran": 0,
                    "best_step": -1,
                    "wall_s": 0.0,
                    "metrics": {},
                    "s_comb": float("nan"),
                    "states": {},
                    "error": traceback.format_exc(limit=6),
                }
            results_by_id[int(result["trial_id"])] = result
            if result.get("states"):
                states_by_id[int(result["trial_id"])] = result["states"]
            keep = False
            reason = result.get("status") or ""
            if result.get("metrics") and math.isfinite(float(result["s_comb"])):
                view = _gate_view(result["s_comb"], result["metrics"])
                keep, reason = promotion_decision(view, champ)
                if keep:
                    champ = view
                    champ_result = result
                    with best_lock:
                        best_s["s"] = float(result["s_comb"])
                    _save_best(root, result, reason, kind="search")
                    print(
                        f"PROMOTE t{result['trial_id']} S={result['s_comb']:.4f} reason={reason}",
                        flush=True,
                    )
            elif result.get("error"):
                reason = "error"
                err_path = root / "errors" / f"t{tid}.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(result["error"])
            best_after = float(champ["s_comb"]) if champ else float("nan")
            row = _row_from_result(result, keep, reason, best_after)
            rows.append(row)
            _append_csv(csv_path, row)
            print(
                f"done t{tid} status={result.get('status')} keep={int(keep)} reason={reason} "
                f"wall={float(result.get('wall_s') or 0):.1f}s",
                flush=True,
            )

    rows.sort(key=lambda r: int(r["trial_id"]))
    # Appends above are the crash-safe mid-run log. Rewrite in trial order once complete.
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    tmp_csv.replace(csv_path)

    protocol = {
        "workers": workers,
        "cpus": cpus,
        "steps": int(args.steps),
        "max_val": int(args.max_val),
        "max_test": int(args.max_test),
        "search": search,
        "alpha0": "learned",
        "grid": "L={12,14} x lr~5e-5 x leakpen~0.1, plus L=10 control and a thin GNN fork",
    }
    final_meta = {"protocol": protocol, "search_best": None, "reval_best": None, "also_reval": []}
    if champ_result is None:
        return rows, final_meta

    search_id = int(champ_result["trial_id"])
    final_meta["search_best"] = {
        "trial_id": search_id,
        "s_comb": champ_result["s_comb"],
        "cfg": champ_result["cfg"],
        "metrics": _json_metrics(champ_result["metrics"]),
        "status": champ_result["status"],
        "best_step": champ_result["best_step"],
    }
    # Drop tensors before the reval report blob; weights are on disk and in states_by_id.
    reval_ids = _select_reval_ids(rows, search_id)
    for Lfocus in (12, 14, 10):
        cands = [r for r in rows if str(r.get("L")) == str(Lfocus) and r.get("S_comb")]
        if not cands:
            continue
        tid = int(min(cands, key=lambda r: float(r["S_comb"]))["trial_id"])
        if tid not in reval_ids:
            reval_ids.append(tid)
    revals = []
    for tid in reval_ids:
        result = results_by_id[tid]
        if tid not in states_by_id:
            continue
        print(f"reval t{tid} val={args.max_val} test={args.max_test}", flush=True)
        rev = reval_states(
            result["arch"], int(result["cfg"]["L"]), states_by_id[tid],
            int(args.max_val), int(args.max_test),
        )
        item = {
            "trial_id": tid,
            "cfg": result["cfg"],
            "arch": result["arch"],
            "search_s": result["s_comb"],
            "reval": rev,
        }
        revals.append(item)
    if not revals:
        return rows, final_meta

    winner = _choose_final(search_id, revals)
    wid = int(winner["trial_id"])
    if wid != search_id:
        search_dir = root / "BEST_search"
        if search_dir.exists():
            shutil.rmtree(search_dir)
        shutil.copytree(root / "BEST", search_dir)
        _save_best(root, results_by_id[wid], winner.get("final_reason", "reval"), kind="reval")
    else:
        # Refresh best.json kind but keep the same weights.
        best_json = json.loads((root / "BEST" / "best.json").read_text())
        best_json["confirmed_by_reval"] = True
        best_json["final_reason"] = winner.get("final_reason", "search_champion")
        (root / "BEST" / "best.json").write_text(json.dumps(best_json, indent=2))

    val_gate = winner["reval"]["val"]["gate"]
    winner_out = {
        "trial_id": wid,
        "cfg": winner["cfg"],
        "final_reason": winner.get("final_reason"),
        "reval": {
            "val": {k: v for k, v in winner["reval"]["val"].items() if k != "gate"},
            "test": {k: v for k, v in winner["reval"]["test"].items() if k != "gate"},
        },
        "vs_r014_val": _vs_r014(val_gate["med6"], val_gate["med9"]),
        "vs_r014_test": _vs_r014(
            winner["reval"]["test"]["gate"]["med6"],
            winner["reval"]["test"]["gate"]["med9"],
        ),
    }
    also = []
    for item in revals:
        also.append({
            "trial_id": item["trial_id"],
            "cfg": item["cfg"],
            "search_s": item["search_s"],
            "reval": {
                "val": {k: v for k, v in item["reval"]["val"].items() if k != "gate"},
                "test": {k: v for k, v in item["reval"]["test"].items() if k != "gate"},
            },
            "vs_r014_val": _vs_r014(item["reval"]["val"]["gate"]["med6"], item["reval"]["val"]["gate"]["med9"]),
            "vs_box_reval40": beats_box_reval(
                item["reval"]["val"]["gate"]["med6"], item["reval"]["val"]["gate"]["med9"],
            ),
        })
    verdict = depth_verdict(also)
    final_meta["reval_best"] = winner_out
    final_meta["also_reval"] = also
    final_meta["depth_verdict"] = verdict
    winner_out["vs_box_reval40"] = beats_box_reval(val_gate["med6"], val_gate["med9"])
    (root / "BEST_reval.json").write_text(json.dumps({
        "protocol": protocol,
        "box_r014": BOX_R014,
        "box_new": BOX_NEW,
        "search_best": final_meta["search_best"],
        "reval_best": winner_out,
        "depth_verdict": verdict,
        "also_reval": also,
        "alpha0": "learned",
        "win_rule": "val-40 med6<14.5 and med9<19.5 and ratio in [0.75, 2]; test does not select",
    }, indent=2))
    (root / "REPORT.md").write_text(render_report(rows, final_meta, protocol))
    # Free large state dicts.
    states_by_id.clear()
    return rows, final_meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=SEARCH["max_steps"])
    ap.add_argument("--workers", type=int, default=0, help="0 = min(8, nproc)")
    ap.add_argument("--limit", type=int, default=0, help="optional cap for a smoke run")
    ap.add_argument("--out", type=str, default=str(PKG / "runs" / "cloud_day1"))
    ap.add_argument("--max-val-search", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=REVAL_VAL)
    ap.add_argument("--max-test", type=int, default=REVAL_TEST)
    ap.add_argument("--append", action="store_true", help="append trials.csv instead of rewriting at start")
    args = ap.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, final_meta = _run_farm(args, root)
    n_keep = sum(int(r["keep"]) for r in rows)
    print(
        f"finished {len(rows)} trials, promotions={n_keep}, wall={time.time()-t0:.1f}s, out={root}",
        flush=True,
    )
    verdict = final_meta.get("depth_verdict") or {}
    deep = verdict.get("best_deep_by_S")
    if deep:
        print(
            f"best deep val40 n6={deep['med6']:.2f} n9={deep['med9']:.2f} "
            f"full_win={deep['full_win']} trial={deep['trial_id']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
