#!/usr/bin/env python3
"""Analysis for ENTANGLEMENT_CONTROL_SUMMARY.md.

Reads the latest full JSON per tag from noiseless/results/ent_control_runs/ (gitignored):
``ent_legacy`` (fixed L=4 plain SPSA, 200 steps, 401 evals) and ``ent_tuned`` (growth
L=1→4 + SPSA-Adam, 1604 evals), each 7 arms × 3 layouts × 20 H × 25 trials, seeds paired
across arms, layouts and protocols. Optional extra tags (e.g. 100-trial reruns) via argv.

Writes noiseless/results/entanglement_control_analysis_summary.json and figures under
noiseless/results/ent_control_figs/.

Statistics: for every (protocol, layout, entangling arm, product control) the paired
per-trial Δp(GS) (sign test on wins/losses, Wilcoxon signed-rank), Δsuccess (exact McNemar
on discordant pairs), and a conservative Hamiltonian-level sign test (per-H mean Δp, 20
clusters). Holm correction is applied within each (protocol, layout) family of 12 tests
(4 entangling arms × 3 controls) on the Wilcoxon p-values.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
RUNS = _REPO / "noiseless" / "results" / "ent_control_runs"
OUT_JSON = _REPO / "noiseless" / "results" / "entanglement_control_analysis_summary.json"
FIG_DIR = _REPO / "noiseless" / "results" / "ent_control_figs"
ENT_ARMS = ("g0.0625", "g0.125", "g0.1875", "jp")
PROD_ARMS = ("identity", "jp_local", "g0.5")
ALL_ARMS = ("identity", "jp_local", "g0.5", "g0.0625", "g0.125", "g0.1875", "jp")
LAYOUTS = ("identity", "rule_best", "rule_bad")
GAMMA = {"identity": 0.0, "g0.0625": 0.0625, "g0.125": 0.125, "g0.1875": 0.1875, "jp": 0.25, "g0.5": 0.5}


def load(tag: str) -> list[dict] | None:
    files = [f for f in sorted(glob.glob(str(RUNS / f"{tag}_*Z.json")))
             if re.fullmatch(rf".*/{re.escape(tag)}_\d{{8}}T\d{{6}}Z\.json", f)]
    if not files:
        return None
    recs = json.load(open(files[-1]))["records"]
    bad = [r for r in recs if not r.get("ok")]
    if bad:
        print(f"WARNING {tag}: {len(bad)} failed", file=sys.stderr)
    return [r for r in recs if r.get("ok")]


def holm(ps: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i])
    m = len(order)
    out: list[float | None] = [None] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * ps[i])
        run = max(run, adj)
        out[i] = run
    return out


def btest(w: int, l: int) -> float | None:
    return float(stats.binomtest(w, w + l, 0.5).pvalue) if w + l else None


def wil(d) -> float | None:
    d = np.asarray(d, float)
    if d.size == 0 or np.all(np.abs(d) < 1e-12):
        return None
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return None


def index(recs: list[dict]) -> dict:
    return {(r["layout"], r["arm"], r["ham_file"], int(r["trial"])): r for r in recs}


def arm_rows(recs: list[dict], protocol: str) -> list[dict]:
    rows = []
    for lay in LAYOUTS:
        for arm in ALL_ARMS:
            R = [r for r in recs if r["layout"] == lay and r["arm"] == arm]
            if not R:
                continue
            pg = np.array([r["p_gs"] for r in R])
            peff = np.array([r["peak_entropy_effective"] for r in R])
            rows.append({
                "protocol": protocol, "layout": lay, "arm": arm, "n": len(R),
                "success_rate": float(np.mean([r["success"] for r in R])),
                "mean_p_gs": float(pg.mean()), "median_p_gs": float(np.median(pg)),
                "mean_final_entropy": float(np.mean([r["final_entropy"] for r in R])),
                "mean_final_entropy_pre_u": float(np.mean([r["final_entropy_pre_u"] for r in R])),
                "mean_peak_entropy_effective": float(peff.mean()),
                "spearman_peak_eff_vs_p_gs": (float(stats.spearmanr(peff, pg)[0])
                                              if np.std(peff) > 0 else None),
            })
    return rows


def paired_block(recs: list[dict], protocol: str, *, stage_L: int | None = None) -> list[dict]:
    """Paired Δ for every (layout, entangling arm, product control). stage_L picks a growth stage."""
    idx = index(recs)

    def pv(r):
        if stage_L is None:
            return float(r["p_gs"]), bool(r["success"])
        st = r["stages"][stage_L - 1]
        return float(st["p_gs"]), bool(st["success"])

    out = []
    for lay in LAYOUTS:
        fam = []
        for arm in ENT_ARMS:
            for ctl in PROD_ARMS:
                d, sg, sl, perh = [], 0, 0, {}
                for (l2, a2, h, t), r in idx.items():
                    if l2 != lay or a2 != arm:
                        continue
                    c = idx.get((lay, ctl, h, t))
                    if c is None:
                        continue
                    (pa, sa), (pc, sc) = pv(r), pv(c)
                    d.append(pa - pc)
                    perh.setdefault(h, []).append(pa - pc)
                    sg += int(sa and not sc)
                    sl += int(sc and not sa)
                if not d:
                    continue
                d = np.array(d)
                w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
                hm = np.array([np.mean(v) for v in perh.values()])
                hw, hl = int((hm > 0).sum()), int((hm < 0).sum())
                fam.append({
                    "protocol": protocol, "stage_L": stage_L or 4, "layout": lay, "arm": arm,
                    "control": ctl, "n_pairs": int(d.size), "mean_delta_p_gs": float(d.mean()),
                    "median_delta_p_gs": float(np.median(d)),
                    "ci95_mean_delta": [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(d.size)),
                                        float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(d.size))],
                    "win_tie_loss": [w, int(d.size - w - l), l], "sign_p": btest(w, l),
                    "wilcoxon_p": wil(d), "success_gained_lost": [sg, sl], "mcnemar_p": btest(sg, sl),
                    "h_level_win_loss": [hw, hl], "h_level_sign_p": btest(hw, hl),
                })
        adj = holm([f["wilcoxon_p"] for f in fam])
        for f, a in zip(fam, adj):
            f["wilcoxon_p_holm"] = a
        out.extend(fam)
    return out


def control_vs_control(recs: list[dict], protocol: str) -> list[dict]:
    idx = index(recs)
    out = []
    for lay in LAYOUTS:
        for a, b in (("jp_local", "identity"), ("g0.5", "identity"), ("g0.5", "jp_local")):
            d = []
            for (l2, a2, h, t), r in idx.items():
                if l2 == lay and a2 == a and (lay, b, h, t) in idx:
                    d.append(r["p_gs"] - idx[(lay, b, h, t)]["p_gs"])
            if d:
                d = np.array(d)
                w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
                out.append({"protocol": protocol, "layout": lay, "a": a, "b": b, "n": int(d.size),
                            "mean_delta_p_gs": float(d.mean()), "win_tie_loss": [w, int(d.size - w - l), l],
                            "sign_p": btest(w, l), "wilcoxon_p": wil(d)})
    return out


def layout_effect(recs: list[dict], protocol: str) -> list[dict]:
    idx = index(recs)
    out = []
    for arm in ALL_ARMS:
        for a, b in (("rule_best", "identity"), ("identity", "rule_bad")):
            d = [r["p_gs"] - idx[(b, arm, h, t)]["p_gs"] for (l2, a2, h, t), r in idx.items()
                 if l2 == a and a2 == arm and (b, arm, h, t) in idx]
            if d:
                out.append({"protocol": protocol, "arm": arm, "a": a, "b": b,
                            "mean_delta_p_gs": float(np.mean(d)), "wilcoxon_p": wil(d)})
    return out


def main() -> int:
    extra = sys.argv[1:]
    out: dict = {"arms": [], "paired": [], "paired_stages": [], "controls": [], "layout_effect": []}
    data = {}
    for tag, proto in [("ent_legacy", "legacy"), ("ent_tuned", "tuned")] + [(t, t) for t in extra]:
        R = load(tag)
        if R is None:
            print("missing", tag)
            continue
        data[proto] = R
        out["arms"] += arm_rows(R, proto)
        out["paired"] += paired_block(R, proto)
        out["controls"] += control_vs_control(R, proto)
        out["layout_effect"] += layout_effect(R, proto)
        if R and R[0].get("stages"):
            for L in (1, 2, 3):
                out["paired_stages"] += paired_block(R, proto, stage_L=L)
            # per-stage arm means
            for lay in LAYOUTS:
                for arm in ALL_ARMS:
                    S = [r for r in R if r["layout"] == lay and r["arm"] == arm]
                    if S:
                        out.setdefault("stage_means", []).append({
                            "protocol": proto, "layout": lay, "arm": arm,
                            "mean_p_gs_by_L": [float(np.mean([r["stages"][k]["p_gs"] for r in S])) for k in range(4)],
                            "success_by_L": [float(np.mean([r["stages"][k]["success"] for r in S])) for k in range(4)],
                            "peak_eff_entropy_by_L": [float(np.mean([r["stages"][k]["peak_entropy_effective"] for r in S])) for k in range(4)],
                        })
    OUT_JSON.write_text(json.dumps(out, indent=2))
    for r in out["arms"]:
        print(f"{r['protocol']:>7} {r['layout']:>9} {r['arm']:>8} succ={r['success_rate']:.3f} p={r['mean_p_gs']:.3f} "
              f"med={r['median_p_gs']:.3f} Sfin={r['mean_final_entropy']:.2f} Spre={r['mean_final_entropy_pre_u']:.2f} "
              f"Speak={r['mean_peak_entropy_effective']:.2f} rho={r['spearman_peak_eff_vs_p_gs']}")
    for r in out["paired"]:
        print(f"{r['protocol']:>7} {r['layout']:>9} {r['arm']:>8} vs {r['control']:>8}: d={r['mean_delta_p_gs']:+.3f} "
              f"W/T/L={r['win_tie_loss']} wil={r['wilcoxon_p']:.2g} holm={r['wilcoxon_p_holm']:.2g} "
              f"succ+/-={r['success_gained_lost']} H={r['h_level_win_loss']}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        protos = [p for p in ("legacy", "tuned") if p in data]
        fig, axes = plt.subplots(1, len(protos), figsize=(5.5 * len(protos), 4.2), squeeze=False)
        for ax, proto in zip(axes[0], protos):
            for lay, col in zip(LAYOUTS, ("C0", "C1", "C2")):
                rows = {r["arm"]: r for r in out["arms"] if r["protocol"] == proto and r["layout"] == lay}
                g = ["identity", "g0.0625", "g0.125", "g0.1875", "jp", "g0.5"]
                ax.plot([GAMMA[a] for a in g if a in rows], [rows[a]["mean_p_gs"] for a in g if a in rows],
                        "-o", color=col, label=f"{lay}")
                if "jp_local" in rows:
                    ax.scatter([0.25], [rows["jp_local"]["mean_p_gs"]], marker="x", s=70, color=col)
            ax.set_xlabel("γ/π  (bus gate exp(iγ Π_A Π_B); 0 and 0.5 are product, 0.25 = jp)")
            ax.set_ylabel("mean p(GS)")
            ax.set_title(f"{proto}: dose-response (x = jp_local)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "dose_response.png", dpi=130)
        if "tuned" in data:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, lay in zip(axes, LAYOUTS):
                for arm in ALL_ARMS:
                    S = [r for r in data["tuned"] if r["layout"] == lay and r["arm"] == arm]
                    ax.scatter([r["peak_entropy_effective"] for r in S], [r["p_gs"] for r in S], s=4, alpha=0.35, label=arm)
                ax.set_title(f"tuned, {lay}")
                ax.set_xlabel("peak effective entropy (bits)")
                ax.set_ylabel("p(GS)")
            axes[0].legend(fontsize=7, markerscale=3)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "entropy_vs_pgs_tuned.png", dpi=110)
        print("figs in", FIG_DIR)
    except Exception as exc:  # noqa: BLE001
        print("figure failed:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
