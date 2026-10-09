"""Analysis for the low-budget multi-round relayout sweep (RELAYOUT_LOWBUDGET_SUMMARY.md).

Reads noiseless/results/relayout_runs/lowbudget/lb_<target>_<r0>_r<steps>_*.json (latest per tag)
and lbgrow_<n>_*.json (plain tuned growth, matched evals). Per round k (0..4) reports the
best-so-far (lowest common-η Gibbs cost over rounds 0..k) success rate and mean p(GS), mean
cumulative circuit evals, and the hit rate of the polished guess that seeds round k+1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "noiseless" / "results" / "relayout_runs" / "lowbudget"
OUT_JSON = REPO / "noiseless" / "results" / "relayout_lowbudget_summary.json"
FIGS = REPO / "noiseless" / "results" / "relayout_lowbudget_figs"
SLOTS = ("d", "e", "A2", "A1", "A0", "B2", "B1", "B0")  # identity layout: variable i -> slot i
POLISH_LOOKUPS = 9  # radius-1 classical energy lookups per round (not circuit evals)


def latest_by_tag(prefix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(RUNS.glob(f"{prefix}*.json")):
        if p.name.endswith("_summary.json"):
            continue
        tag = re.sub(r"_\d{8}T\d{6}Z\.json$", "", p.name)
        out[tag] = p  # sorted → latest stamp wins
    return out


def load(p: Path) -> list[dict]:
    recs = json.loads(p.read_text())["records"]
    bad = [r for r in recs if not r.get("ok")]
    if bad:
        raise SystemExit(f"{p.name}: {len(bad)} failed trials")
    return recs


def per_round(recs: list[dict]) -> list[dict]:
    nr = len(recs[0]["rounds"])
    rows = []
    for k in range(nr):
        rr = [r["rounds"][k] for r in recs]
        rows.append({
            "round": k,
            "success": float(np.mean([x["bsf_success"] for x in rr])),
            "mean_p_gs": float(np.mean([x["bsf_p_gs"] for x in rr])),
            "cum_nfev": float(np.mean([x["cum_nfev"] for x in rr])),
            "polish_lookups": POLISH_LOOKUPS * (k + 1),
            "round_success": float(np.mean([x["success"] for x in rr])),
            "round_mean_p_gs": float(np.mean([x["p_gs"] for x in rr])),
            "guess_hit": float(np.mean([x["next_guess_is_ground"] for x in rr])),
            "round_polished_hit": float(np.mean([x["polished_is_ground"] for x in rr])),
        })
    return rows


def slot_misses(recs: list[dict]) -> dict:
    """Initially wrong polished guesses (round 0): which slots differ from the true GS, and
    whether the final best-so-far round succeeded / a later guess was right.

    Per slot: n_missed = # wrong-0 trials whose guess differs from the GS in that slot;
    n_fixed = of those, # with final best-so-far success. Also the number-of-wrong-bits
    distribution and single-slot-miss breakdown (identity layout, so slot = variable index)."""
    per = {sl: {"n_missed": 0, "n_fixed": 0} for sl in SLOTS}
    nbits: dict[int, list[int]] = {}
    single = {sl: {"n": 0, "n_fixed": 0} for sl in SLOTS}
    for r in recs:
        g0 = r["rounds"][0]["next_guess"]
        gs = r["ground_bitstring"]
        if g0 == gs:
            continue
        fixed = bool(r["rounds"][-1]["bsf_success"])
        diff = [i for i in range(8) if g0[i] != gs[i]]
        nbits.setdefault(len(diff), [0, 0])
        nbits[len(diff)][0] += 1
        nbits[len(diff)][1] += int(fixed)
        for i in diff:
            per[SLOTS[i]]["n_missed"] += 1
            per[SLOTS[i]]["n_fixed"] += int(fixed)
        if len(diff) == 1:
            single[SLOTS[diff[0]]]["n"] += 1
            single[SLOTS[diff[0]]]["n_fixed"] += int(fixed)
    for d in list(per.values()) + list(single.values()):
        n = d.get("n_missed", d.get("n"))
        d["fix_rate"] = d["n_fixed"] / n if n else None
    return {"per_slot": per, "single_slot_misses": single,
            "n_wrong_bits": {str(k): {"n": v[0], "n_fixed": v[1]} for k, v in sorted(nbits.items())}}


def criterion(rows: list[dict], weak=(0.35, 0.65), top=0.985) -> dict:
    s = [r["success"] for r in rows]
    p = [r["mean_p_gs"] for r in rows]
    weak_ok = weak[0] <= s[0] <= weak[1]
    mono = all((s[k] > s[k - 1] + 1e-12) or (p[k] > p[k - 1] + 1e-12) for k in range(1, len(rows)))
    final_ok = s[-1] >= top and p[-1] >= top
    only_last = all(not (s[k] >= top and p[k] >= top) for k in range(len(rows) - 1))
    return {"round0_weak": weak_ok, "each_round_improves": mono, "final_near_099": final_ok,
            "only_final_near_099": only_last,
            "meets_all": weak_ok and mono and final_ok and only_last}


def per_h(recs: list[dict], k: int | None = None) -> dict[str, tuple[float, float]]:
    out: dict[str, list] = {}
    for r in recs:
        if k is None:
            s, p = r["success"], r["p_gs"]
        else:
            x = r["rounds"][k]
            s, p = x["bsf_success"], x["bsf_p_gs"]
        out.setdefault(r["ham_file"], []).append((s, p))
    return {h: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))
            for h, v in sorted(out.items())}


def main() -> None:
    main_runs = latest_by_tag("lb_")
    grow_runs = latest_by_tag("lbgrow_")
    settings = {}
    recs_by_tag = {}
    for tag, p in sorted(main_runs.items()):
        recs = load(p)
        recs_by_tag[tag] = recs
        rows = per_round(recs)
        m = re.match(r"lb_(\w+?)_(L\ds\d+)_r(\d+)$", tag)
        target, r0, rs = m.group(1), m.group(2), int(m.group(3))
        # corrected-guess fraction: round-0 guess wrong → final best-so-far success
        wrong0 = [r for r in recs if not r["rounds"][0]["next_guess_is_ground"]]
        settings[tag] = {
            "target": target, "round0": r0, "relabel_steps": rs, "n_trials": len(recs),
            "rows": rows, "criterion": criterion(rows),
            "n_round0_guess_wrong": len(wrong0),
            "frac_wrong0_final_success": float(np.mean([r["rounds"][-1]["bsf_success"] for r in wrong0])) if wrong0 else None,
            "frac_wrong0_later_guess_right": float(np.mean([any(x["next_guess_is_ground"] for x in r["rounds"][1:]) for r in wrong0])) if wrong0 else None,
            "slot_misses": slot_misses(recs),
            "frac_right0_final_success": float(np.mean([r["rounds"][-1]["bsf_success"] for r in recs if r["rounds"][0]["next_guess_is_ground"]])) if len(wrong0) < len(recs) else None,
        }
    grow = {}
    for tag, p in sorted(grow_runs.items()):
        recs = load(p)
        recs_by_tag[tag] = recs
        grow[tag] = {"steps_per_stage": int(recs[0]["stages"][0]["steps"]) if recs[0].get("stages") and "steps" in recs[0]["stages"][0] else None,
                     "success": float(np.mean([r["success"] for r in recs])),
                     "mean_p_gs": float(np.mean([r["p_gs"] for r in recs])),
                     "nfev": float(np.mean([r["nfev"] for r in recs])), "n_trials": len(recs)}
    # per-H comparisons: relabel vs none for same r0/steps; relabel vs matched growth
    perh = {}
    for tag, st in settings.items():
        if st["target"] != "xor_vacuum":
            continue
        ctl = f"lb_none_{st['round0']}_r{st['relabel_steps']}"
        a = per_h(recs_by_tag[tag], -1)
        comp = {}
        if ctl in recs_by_tag:
            b = per_h(recs_by_tag[ctl], -1)
            comp["vs_norelabel"] = _wins(a, b)
        gtag = f"lbgrow_{st['round0']}_r{st['relabel_steps']}"
        if gtag in recs_by_tag:
            comp["vs_matched_growth"] = _wins(a, per_h(recs_by_tag[gtag]))
        if comp:
            perh[tag] = comp
    OUT_JSON.write_text(json.dumps({"settings": settings, "matched_growth": grow,
                                    "per_h": perh}, indent=2))
    _figs(settings, grow)
    _print(settings, grow, perh)


def _wins(a, b):
    ws = sum(1 for h in a if a[h][0] > b[h][0])
    ls = sum(1 for h in a if a[h][0] < b[h][0])
    wp = sum(1 for h in a if a[h][1] > b[h][1])
    lp = sum(1 for h in a if a[h][1] < b[h][1])
    return {"success_win_tie_loss": [ws, len(a) - ws - ls, ls],
            "mean_p_win_loss": [wp, lp],
            "per_h": {h: {"relabel": a[h], "control": b[h]} for h in a}}


def _figs(settings, grow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    r0s = sorted({s["round0"] for s in settings.values()}, key=lambda x: (x[1], int(x[3:])))
    fig, axes = plt.subplots(2, len(r0s), figsize=(4.2 * len(r0s), 7), squeeze=False)
    for j, r0 in enumerate(r0s):
        for tag, st in sorted(settings.items(), key=lambda kv: kv[1]["relabel_steps"]):
            if st["round0"] != r0:
                continue
            x = [r["cum_nfev"] for r in st["rows"]]
            ls = "-" if st["target"] == "xor_vacuum" else ":"
            lab = f"{'relabel' if st['target']=='xor_vacuum' else 'no relabel'} {st['relabel_steps']} st"
            axes[0][j].plot(x, [r["success"] for r in st["rows"]], ls, marker="o", label=lab)
            axes[1][j].plot(x, [r["mean_p_gs"] for r in st["rows"]], ls, marker="o", label=lab)
        for g, gv in grow.items():
            if g.startswith(f"lbgrow_{r0}_"):
                axes[0][j].plot([gv["nfev"]], [gv["success"]], "k*", ms=12)
                axes[1][j].plot([gv["nfev"]], [gv["mean_p_gs"]], "k*", ms=12)
        axes[0][j].set_title(f"round 0 = {r0}")
        for i in range(2):
            axes[i][j].axhline(0.99, color="gray", lw=0.6)
            axes[i][j].set_xlabel("cumulative circuit evals")
            axes[i][j].set_ylim(0, 1.02)
        axes[0][j].set_ylabel("success (best-so-far)")
        axes[1][j].set_ylabel("mean p(GS) (best-so-far)")
        axes[0][j].legend(fontsize=7)
    fig.suptitle("Low-budget relayout: per round 0..4 (star = plain tuned growth, matched evals)")
    fig.tight_layout()
    fig.savefig(FIGS / "per_round_vs_evals.png", dpi=120)
    plt.close(fig)


def _print(settings, grow, perh):
    for tag, st in settings.items():
        c = st["criterion"]
        print(f"\n{tag}  meets_all={c['meets_all']} {c}")
        print("  rnd  success  mean_p  evals  guess_hit")
        for r in st["rows"]:
            print(f"  {r['round']}    {r['success']:.3f}   {r['mean_p_gs']:.3f}  {r['cum_nfev']:.0f}   {r['guess_hit']:.3f}")
        sm = st["slot_misses"]["per_slot"]
        print("  slot miss/fixed: " + " ".join(f"{k}:{v['n_missed']}/{v['n_fixed']}" for k, v in sm.items()))
        print(f"  wrong0={st['n_round0_guess_wrong']} final_succ_among_wrong0={st['frac_wrong0_final_success']} later_guess_right={st['frac_wrong0_later_guess_right']}")
    for g, gv in grow.items():
        print(g, gv)
    for t, c in perh.items():
        print(t, {k: (v["success_win_tie_loss"], v["mean_p_win_loss"]) for k, v in c.items()})


if __name__ == "__main__":
    main()


def markdown_tables() -> str:
    """Markdown tables (absolute values) from the summary JSON, for the report."""
    d = json.loads(OUT_JSON.read_text())
    st = d["settings"]
    order = sorted(st, key=lambda t: (st[t]["target"] != "xor_vacuum", st[t]["round0"][1],
                                      int(st[t]["round0"][3:]), st[t]["relabel_steps"]))
    lines = ["| setting | round 0 | round 1 | round 2 | round 3 | round 4 |",
             "|---|---|---|---|---|---|"]
    for t in order:
        s = st[t]
        name = f"{'relabel' if s['target']=='xor_vacuum' else 'NO relabel'} {s['round0']} r{s['relabel_steps']}"
        cells = [f"{r['success']:.3f} / {r['mean_p_gs']:.3f} / {r['cum_nfev']:.0f}" for r in s["rows"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| setting | guess hit after r0 | r1 | r2 | r3 | r4 |", "|---|---|---|---|---|---|"]
    for t in order:
        s = st[t]
        name = f"{'relabel' if s['target']=='xor_vacuum' else 'NO relabel'} {s['round0']} r{s['relabel_steps']}"
        lines.append(f"| {name} | " + " | ".join(f"{r['guess_hit']:.3f}" for r in s["rows"]) + " |")
    return "\n".join(lines)
