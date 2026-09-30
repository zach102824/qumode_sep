#!/usr/bin/env python3
"""ECD (jp, k=7 bits/cavity) vs RY-only / HEA L=1 at n = 16 (F1 instances).

Reads ECD checkpoints noiseless/results/ecd_kbit_runs/main/ecd_jp_L{2,4}_n16.jsonl and the existing
scaling checkpoints noiseless/results/scaling_runs/{main,steps400,steps600,steps800}/{ry0,hea1}_n16.jsonl.
Writes noiseless/results/ecd_vs_hea_n16_summary.json (table rows + ECD leakage/|β|/photon diagnostics)
and prints markdown tables.  Photon diagnostics re-simulate each ECD trial's final x (nf = 160).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from noiseless.ecd_kbit import apply_circuit, born  # noqa: E402

RES = _REPO / "noiseless" / "results"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def row(label, recs, steps=None) -> dict:
    pg = np.array([r["p_gs"] for r in recs])
    by = {}
    for r in recs:
        by.setdefault(r["inst"], []).append(r)
    return {
        "arm": label, "steps": steps, "n_params": recs[0]["n_params"], "evals": recs[0]["nfev"],
        "trials": len(recs), "instances": len(by),
        "success": float(np.mean([r["success"] for r in recs])),
        "mean_p_gs": float(pg.mean()), "median_p_gs": float(np.median(pg)),
        "best_of_trials_mean": float(np.mean([max(x["p_gs"] for x in v) for v in by.values()])),
        "inst_ge1_success": int(sum(any(x["success"] for x in v) for v in by.values())),
        "bin_gt_0p5": float(np.mean(pg > 0.5)), "bin_0p1_0p5": float(np.mean((pg > 0.1) & (pg <= 0.5))),
        "bin_0p01_0p1": float(np.mean((pg > 0.01) & (pg <= 0.1))), "bin_lt_0p01": float(np.mean(pg <= 0.01)),
        "mean_H": float(np.mean([r["energy_mean"] for r in recs])),
        "s_per_trial": float(np.mean([r["wall_s"] for r in recs])),
    }


def photon_diag(recs: list[dict]) -> dict:
    out = []
    for r in recs:
        L, nf = r["L"], r["nf"]
        p = born(apply_circuit(np.asarray(r["x"]), L, nf)).reshape(2, 2, nf, nf)
        pa, pb = p.sum(axis=(0, 1, 3)), p.sum(axis=(0, 1, 2))
        n = np.arange(nf)
        gs = r["ground_bitstring"]
        na_t, nb_t = int(gs[2:9], 2), int(gs[9:16], 2)
        sa = float(np.sqrt(max(pa @ n**2 - (pa @ n) ** 2, 0)))
        sb = float(np.sqrt(max(pb @ n**2 - (pb @ n) ** 2, 0)))
        out.append(dict(inst=r["inst"], na_target=na_t, nb_target=nb_t, mean_na=float(pa @ n), mean_nb=float(pb @ n),
                        std_na=sa, std_nb=sb, pa_target=float(pa[na_t]), pb_target=float(pb[nb_t]),
                        pmax_a=float(pa.max()), pmax_b=float(pb.max()),
                        n_eff_a=float(1 / np.sum(pa**2)), n_eff_b=float(1 / np.sum(pb**2)),
                        p_gs=r["p_gs"], bound=float(min(pa[na_t], pb[nb_t]))))
    return out


def main() -> int:
    rows, diag = [], {}
    for L in (2, 4):
        p = RES / "ecd_kbit_runs" / "main" / f"ecd_jp_L{L}_n16.jsonl"
        if not p.exists():
            continue
        recs = load(p)
        rw = row(f"ECD jp L={L}", recs, "400+400" if L == 2 else "200x4")
        lk = np.array([r["leakage"] for r in recs])
        mb = np.array([r["max_abs_beta"] for r in recs])
        ab = np.concatenate([r["abs_betas"] for r in recs])
        rw.update(leakage_mean=float(lk.mean()), leakage_median=float(np.median(lk)), leakage_max=float(lk.max()),
                  frac_leak_gt_1e3=float(np.mean(lk > 1e-3)), max_abs_beta_mean=float(mb.mean()),
                  max_abs_beta_median=float(np.median(mb)), max_abs_beta_max=float(mb.max()),
                  abs_beta_all_mean=float(ab.mean()),
                  max_abs_dp_check=float(max(r["max_abs_dp_check"] for r in recs)),
                  success_check_agree=float(np.mean([r["success_check"] == r["success"] for r in recs])),
                  most_likely_leaked=int(sum(r["most_likely_bitstring"] == "LEAKED" for r in recs)),
                  p_gs_max=float(max(r["p_gs"] for r in recs)))
        rows.append(rw)
        d = photon_diag(recs)
        keys = [k for k in d[0] if k != "inst"]
        per = {}
        for x, r in zip(d, recs):
            per.setdefault(r["inst"], []).append((x, r))
        per_inst = [{"inst": i, "na_target": v[0][0]["na_target"], "nb_target": v[0][0]["nb_target"],
                     "success": float(np.mean([r["success"] for _, r in v])),
                     "mean_p_gs": float(np.mean([r["p_gs"] for _, r in v])),
                     "max_p_gs": float(max(r["p_gs"] for _, r in v)),
                     "median_pa_target": float(np.median([x["pa_target"] for x, _ in v])),
                     "median_pb_target": float(np.median([x["pb_target"] for x, _ in v])),
                     "median_mean_na": float(np.median([x["mean_na"] for x, _ in v])),
                     "median_mean_nb": float(np.median([x["mean_nb"] for x, _ in v])),
                     "median_std_na": float(np.median([x["std_na"] for x, _ in v])),
                     "median_std_nb": float(np.median([x["std_nb"] for x, _ in v])),
                     "mean_H": float(np.mean([r["energy_mean"] for _, r in v]))} for i, v in sorted(per.items())]
        diag[f"L{L}"] = {"per_instance": per_inst,"median": {k: float(np.median([x[k] for x in d])) for k in keys},
                         "mean": {k: float(np.mean([x[k] for x in d])) for k in keys},
                         "max_bound": float(max(x["bound"] for x in d)),
                         "frac_pa_target_gt_0p01": float(np.mean([x["pa_target"] > 0.01 for x in d])),
                         "frac_pb_target_gt_0p01": float(np.mean([x["pb_target"] > 0.01 for x in d])),
                         "frac_both_gt_0p01": float(np.mean([(x["pa_target"] > 0.01) and (x["pb_target"] > 0.01) for x in d]))}
    for arm, name in (("ry0", "RY-only"), ("hea1", "HEA L=1")):
        for tag, st in (("main", 200), ("steps400", 400), ("steps600", 600), ("steps800", 800)):
            rows.append(row(name, load(RES / "scaling_runs" / tag / f"{arm}_n16.jsonl"), st))
    # GS Fock targets over the 20 instances
    from noiseless.run_ecd_kbit import instance_paths
    import numpy as _np
    tg = []
    for pth in instance_paths(16):
        gs = str(_np.load(pth)["ground_bitstring"])
        tg.append((int(gs[2:9], 2), int(gs[9:16], 2)))
    out = {"rows": rows, "photon_diag": diag, "gs_fock_targets": tg}
    (RES / "ecd_vs_hea_n16_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    f = lambda v: f"{v:.3g}"  # noqa: E731
    print("| arm | SPSA steps | params | evals | trials | success | mean p(GS) | median p(GS) | best-of-25 mean | inst ≥1 succ | >0.5 | 0.1–0.5 | 0.01–0.1 | <0.01 | mean <H> | s/trial |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['arm']} | {r['steps']} | {r['n_params']} | {r['evals']} | {r['trials']} | {r['success']:.3f} | {f(r['mean_p_gs'])} | "
              f"{f(r['median_p_gs'])} | {f(r['best_of_trials_mean'])} | {r['inst_ge1_success']}/{r['instances']} | {r['bin_gt_0p5']:.3f} | "
              f"{r['bin_0p1_0p5']:.3f} | {r['bin_0p01_0p1']:.3f} | {r['bin_lt_0p01']:.3f} | {r['mean_H']:.3f} | {r['s_per_trial']:.1f} |")
    for key, dg in diag.items():
        print(f"\nper-instance {key}: inst | GS (n_A*, n_B*) | success | mean p(GS) | max p(GS) | med P_A(n_A*) | med P_B(n_B*) | med <n_A>±sd | med <n_B>±sd | <H>")
        for q in dg["per_instance"]:
            print(f"| {q['inst']} | ({q['na_target']}, {q['nb_target']}) | {q['success']:.2f} | {q['mean_p_gs']:.2e} | {q['max_p_gs']:.2e} | "
                  f"{q['median_pa_target']:.1e} | {q['median_pb_target']:.1e} | {q['median_mean_na']:.1f}±{q['median_std_na']:.1f} | "
                  f"{q['median_mean_nb']:.1f}±{q['median_std_nb']:.1f} | {q['mean_H']:.2f} |")
        print(json.dumps({k: v for k, v in dg.items() if k != "per_instance"}, indent=1))
    print(json.dumps([{k: v for k, v in r.items() if k.startswith(("leak", "max_abs", "frac", "abs", "succ", "most", "p_gs_max"))} for r in rows[:2]], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
