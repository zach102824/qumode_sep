#!/usr/bin/env python3
"""Adaptive budget search: for each n, the cheapest (L, s, r, R) with final mean p(GS) >= 0.90.

Method = headline relayout (noiseless/relayout_kbit.py). Rules (approved plan, 2026-10-09):
  * n = 8 starts from the headline L4 s10 r200 R4; every later n starts from the previous n's
    final (cheapest passing) setting.
  * Screen = 5 trials per H (20 H). Pass = screen mean p(GS) >= 0.90.
  * Escalation from the per-round curves (one notch per knob; two notches if mean p < 0.3):
      - guess right but p low (mean p(GS) over successful trials < 0.92)          -> raise r
      - final success < 0.93 and p still climbing at the last round (+0.02)      -> raise R
      - final success < 0.93, not climbing, low round-0 polished-guess hit (<0.5) -> raise s
        (s at its cap -> raise L)
      - final success < 0.93, not climbing, round-0 hit ok                        -> raise L
        (L at cap -> R)
    (the r rule and one success rule can fire together).
  * Trim after passing: greedily try each knob one notch down (largest eval saving first), accept a
    passing trim and repeat until no single-notch trim passes.
  * Confirm the final setting and its next-cheaper neighbour (the failed trim with the most evals,
    i.e. the closest cheaper setting) at 25 trials per H (fewer if the time estimate is too long).
  * Baseline: plain tuned growth (R = 0) at L_max with steps/stage matched to the chosen setting's
    evals, 10 trials per H, run right after each n's confirm if it fits (else at the end).

Status: logs/kbit_scaling/status.json + status.md (rewritten after every config); search path:
noiseless/results/relayout_scaling_search.json (committable summary).
"""

from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from noiseless.run_relayout_kbit import RUN_ROOT, config_tag, run_config, summarize  # noqa: E402

LADDER = {
    "L": [2, 4, 6, 8, 10, 12],
    "s": [5, 10, 25, 50, 100],
    "r": [100, 200, 400, 800, 1600],
    "R": [2, 4, 6, 8, 12, 16, 24, 32],
}
TARGET = 0.90
LOG_DIR = _REPO / "logs" / "kbit_scaling"
SEARCH_JSON = _REPO / "noiseless" / "results" / "relayout_scaling_search.json"
FMT = "%Y-%m-%d %H:%M"


def evals_of(c: dict) -> int:
    return c["L"] * (2 * c["s"] + 1) + c["R"] * (2 * c["r"] + 1)


def key(c):
    return (c["L"], c["s"], c["r"], c["R"])


def cname(c):
    return f"L{c['L']}s{c['s']}-r{c['r']}-R{c['R']}"


def notch(c: dict, knob: str, d: int) -> dict | None:
    lad = LADDER[knob]
    v = c[knob]
    if v in lad:
        i = lad.index(v) + d
    else:  # off-ladder value: next ladder value in that direction
        i = next((j for j, x in enumerate(lad) if x > v), len(lad)) if d > 0 else \
            max((j for j, x in enumerate(lad) if x < v), default=-1)
        i = i + (d - 1 if d > 0 else d + 1)
    if i < 0 or i >= len(lad):
        return None
    out = dict(c)
    out[knob] = lad[i]
    return out


class Driver:
    def __init__(self, a):
        self.a = a
        self.deadline = datetime.strptime(a.deadline, FMT)
        self.ns = [int(x) for x in a.ns.split(",")]
        self.soft = {int(kv.split(":")[0]): datetime.strptime(kv.split(":", 1)[1], FMT)
                     for kv in a.soft_deadlines.split(",")} if a.soft_deadlines else {}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logf = open(LOG_DIR / "driver.log", "a")
        self.state = {"started": datetime.now().strftime(FMT), "deadline": a.deadline, "target": TARGET,
                      "ladder": LADDER, "per_n": {}, "current": None, "phase": "init", "notes": []}
        if SEARCH_JSON.exists() and a.resume:
            self.state = json.loads(SEARCH_JSON.read_text())
            self.state["deadline"] = a.deadline
        self.eval_ms = {}  # n -> ms per (eval * layer), from finished runs

    # ------------------------------------------------------------------ io
    def log(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def save(self):
        self.state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        SEARCH_JSON.write_text(json.dumps(self.state, indent=1) + "\n")
        (LOG_DIR / "status.json").write_text(json.dumps(self.state, indent=1) + "\n")
        lines = [f"# kbit relayout scaling search status (updated {self.state['updated']} CST)", "",
                 f"phase: {self.state['phase']}  current: {self.state.get('current')}",
                 f"deadline: {self.state['deadline']} CST", ""]
        lines.append("| n | config | trials | success | mean p(GS) | evals | round0 hit | p|succ | s/trial | role |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for n, st in self.state["per_n"].items():
            for step in st.get("path", []):
                m = step["metrics"]
                lines.append(f"| {n} | {step['config']} | {m.get('n_trials')} | {m.get('success', 0):.3f} | "
                             f"{m.get('mean_p_gs', 0):.3f} | {m.get('evals', 0):.0f} | {m.get('round0_polished_hit', 0):.3f} | "
                             f"{m.get('mean_p_gs_given_success', 0):.3f} | {m.get('wall_s_mean', 0):.1f} | {step['role']} |")
            if st.get("final"):
                lines.append(f"| {n} | **final {st['final']}** | | | | | | | | {st.get('status')} |")
        (LOG_DIR / "status.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ timing
    def est_trial_s(self, n, c):
        ms = self.eval_ms.get(n)
        if ms is None:
            ms = {8: 0.15, 10: 0.25, 12: 0.4, 14: 1.6, 16: 11.0}.get(n, 11.0)  # ms per eval per layer, 8 workers
        return ms * 1e-3 * (c["L"] * (2 * c["s"] + 1) * (c["L"] + 1) / 2 + c["R"] * (2 * c["r"] + 1) * c["L"])

    def est_config_s(self, n, c, trials, done=0):
        tot = 20 * trials - done
        return max(0, tot) * self.est_trial_s(n, c) / self.a.workers * 1.1

    def update_timing(self, n, recs):
        vals = []
        for x in recs:
            L, s, r, R = x["L"], x["s"], x["r"], x["R"]
            work = L * (2 * s + 1) * (L + 1) / 2 + R * (2 * r + 1) * L
            vals.append(x["wall_s"] / work * 1e3)
        if vals:
            self.eval_ms[n] = float(sorted(vals)[len(vals) // 2])

    def time_left(self, n=None):
        lim = self.deadline
        if n is not None and n in self.soft:
            lim = min(lim, self.soft[n])
        return (lim - datetime.now()).total_seconds()

    # ------------------------------------------------------------- runs
    def run(self, n, c, trials, role):
        st = self.state["per_n"].setdefault(str(n), {"path": [], "screened": {}})
        self.state["current"] = f"n={n} {cname(c)} x{trials} ({role})"
        self.state["phase"] = f"n={n} {role}"
        self.save()
        t0 = time.time()
        tag, recs = run_config(n=n, L=c["L"], s=c["s"], r=c["r"], R=c["R"], trials=trials,
                               workers=self.a.workers, log=self.log)
        self.update_timing(n, recs)
        m = summarize(recs)
        m["tag"] = tag
        m["config"] = dict(c)
        m["evals_formula"] = evals_of(c)
        m["run_wall_s"] = time.time() - t0
        st["path"].append({"config": cname(c), "role": role, "time": datetime.now().strftime(FMT),
                           "metrics": {kk: v for kk, v in m.items() if kk != "per_h"},
                           "per_h": m.get("per_h")})
        if trials == self.a.screen_trials:
            st["screened"][cname(c)] = {"mean_p_gs": m["mean_p_gs"], "success": m["success"],
                                        "evals": m["evals"], "config": dict(c)}
        self.log(f"n={n} {cname(c)} x{trials} [{role}]: success={m['success']:.3f} mean_p={m['mean_p_gs']:.3f} "
                 f"evals={m['evals']:.0f} r0hit={m['round0_polished_hit']:.3f} p|succ={m['mean_p_gs_given_success']:.3f} "
                 f"curve={[round(q['mean_p_gs'], 3) for q in m['curve']]} wall={m['run_wall_s']:.0f}s")
        self.save()
        return m

    def screen(self, n, c, role):
        st = self.state["per_n"].setdefault(str(n), {"path": [], "screened": {}})
        if cname(c) in st["screened"]:
            return st["screened"][cname(c)] | {"cached": True}
        return self.run(n, c, self.a.screen_trials, role)

    # ---------------------------------------------------------- escalate
    def escalate(self, c, m):
        reasons, new = [], dict(c)
        jump = 2 if m["mean_p_gs"] < 0.3 else 1
        curve = m["curve"]
        climbing = len(curve) >= 2 and (curve[-1]["mean_p_gs"] - curve[-2]["mean_p_gs"] >= 0.02
                                         or curve[-1]["success"] - curve[-2]["success"] >= 0.02)

        def bump(knob, alts=()):
            for kb in (knob, *alts):
                nx = notch(new, kb, jump) or notch(new, kb, 1)
                if nx is not None:
                    new.update(nx)
                    return kb
            return None

        if m["mean_p_gs_given_success"] < 0.92:
            kb = bump("r", ("L",))
            reasons.append(f"guess right but p low (p|succ={m['mean_p_gs_given_success']:.3f}) -> raise {kb}")
        if m["success"] < 0.93:
            if climbing:
                kb = bump("R", ("s", "L"))
                reasons.append(f"success {m['success']:.3f}, still climbing at last round -> raise {kb}")
            elif m["round0_polished_hit"] < 0.5:
                kb = bump("s", ("L", "R"))
                reasons.append(f"success {m['success']:.3f}, flat, low round-0 hit {m['round0_polished_hit']:.3f} -> raise {kb}")
            else:
                kb = bump("L", ("R",))
                reasons.append(f"success {m['success']:.3f}, flat, round-0 hit ok {m['round0_polished_hit']:.3f} -> raise {kb}")
        if key(new) == key(c):
            # success fine and p|succ fine but mean below target (should not happen) -> raise r
            kb = bump("r", ("R", "L", "s"))
            reasons.append(f"fallback -> raise {kb}")
        if key(new) == key(c):
            return None, "; ".join(reasons + ["all knobs at cap"])
        return new, "; ".join(reasons)

    # ------------------------------------------------------------ per n
    def do_n(self, n, start):
        st = self.state["per_n"].setdefault(str(n), {"path": [], "screened": {}})
        st["start"] = cname(start)
        c = dict(start)
        passed = None
        best = None
        for it in range(self.a.max_escalations + 1):
            need = self.est_config_s(n, c, self.a.screen_trials)
            if need > self.time_left(n):
                self.log(f"n={n}: screen {cname(c)} needs ~{need/60:.0f} min > time left {self.time_left(n)/60:.0f} min; stop search")
                st["status"] = "time_limit"
                break
            m = self.screen(n, c, "screen" if it == 0 else "escalate")
            if best is None or m["mean_p_gs"] > best[1]["mean_p_gs"]:
                best = (dict(c), m)
            if m["mean_p_gs"] >= TARGET:
                passed = dict(c)
                break
            if m.get("cached"):
                m = self._full_metrics(n, c)
            nxt, why = self.escalate(c, m)
            st.setdefault("decisions", []).append({"from": cname(c), "to": None if nxt is None else cname(nxt), "why": why})
            self.log(f"n={n}: escalate {cname(c)} -> {None if nxt is None else cname(nxt)}: {why}")
            self.save()
            if nxt is None:
                st["status"] = "ladder_cap"
                break
            c = nxt
        if passed is None:
            st["final"] = cname(best[0]) if best else None
            st["final_config"] = best[0] if best else None
            st.setdefault("status", "not_passed")
            self.save()
            return best[0] if best else start, False
        # ---- trim
        failed_trims = []
        improved = True
        while improved:
            improved = False
            cands = []
            for kb in ("R", "r", "L", "s"):
                t = notch(passed, kb, -1)
                if t is not None:
                    cands.append(t)
            cands.sort(key=lambda t: evals_of(t))  # biggest saving first
            for t in cands:
                need = self.est_config_s(n, t, self.a.screen_trials)
                if need > self.time_left(n):
                    continue
                m = self.screen(n, t, "trim")
                if m["mean_p_gs"] >= TARGET:
                    st.setdefault("decisions", []).append({"from": cname(passed), "to": cname(t), "why": "trim passes"})
                    self.log(f"n={n}: trim {cname(passed)} -> {cname(t)} passes ({m['mean_p_gs']:.3f})")
                    passed = t
                    improved = True
                    break
                failed_trims.append(t)
        neigh = [t for t in failed_trims if evals_of(t) < evals_of(passed)]
        neighbor = max(neigh, key=lambda t: (evals_of(t), st["screened"].get(cname(t), {}).get("mean_p_gs", 0))) if neigh else None
        st["final"] = cname(passed)
        st["final_config"] = passed
        st["neighbor"] = None if neighbor is None else cname(neighbor)
        st["neighbor_config"] = neighbor
        st["status"] = "passed_screen"
        self.save()
        # ---- confirm
        for role, cc in (("confirm", passed), ("confirm_neighbor", neighbor)):
            if cc is None:
                continue
            trials = self.a.confirm_trials
            while trials > self.a.screen_trials and self.est_config_s(n, cc, trials, 20 * self.a.screen_trials) > \
                    min(self.a.confirm_max_s, self.time_left(n) + (1800 if role == "confirm" else 0)):
                trials -= 5
            if trials <= self.a.screen_trials:
                self.log(f"n={n}: skip {role} {cname(cc)} (no time)")
                continue
            m = self.run(n, cc, trials, role)
            st[role] = {"config": cname(cc), "trials_per_h": trials, "success": m["success"],
                        "mean_p_gs": m["mean_p_gs"], "evals": m["evals"]}
        if st.get("confirm") and st["confirm"]["mean_p_gs"] < TARGET:
            st["status"] = "passed_screen_failed_confirm"
            st["notes"] = "confirm below 0.90; next n starts from this setting anyway"
        elif st.get("confirm"):
            st["status"] = "passed"
        self.save()
        if self.a.baseline:
            self.baseline_n(n, min(0.15 * max(self.time_left(n), 0) + 600, 3600))
        return passed, True

    def _full_metrics(self, n, c):
        from noiseless.run_relayout_kbit import load_records
        tag = config_tag(n, c["L"], c["s"], c["r"], c["R"])
        recs = load_records(RUN_ROOT / f"{tag}.jsonl", self.a.screen_trials)
        return summarize(recs)

    def baseline_n(self, n, budget_s):
        st = self.state["per_n"].get(str(n))
        if not st or not st.get("final_config") or st.get("baseline"):
            return
        c = st["final_config"]
        target = evals_of(c)
        sg = max(5, round((target / c["L"] - 1) / 2))
        b = {"L": c["L"], "s": sg, "r": 100, "R": 0}
        trials = self.a.baseline_trials
        while trials > 2 and self.est_config_s(n, b, trials) > budget_s:
            trials -= 1
        if trials <= 2:
            self.log(f"baseline n={n}: no time")
            return
        m = self.run(n, b, trials, "baseline_growth_matched")
        st["baseline"] = {"config": f"L{b['L']}s{sg} (plain growth, no relabel)", "trials_per_h": trials,
                          "success": m["success"], "mean_p_gs": m["mean_p_gs"], "evals": m["evals"]}
        self.save()

    def baseline(self):
        for n in self.ns:
            self.baseline_n(n, self.time_left())

    def main(self):
        c = dict(L=4, s=10, r=200, R=4)
        self.log(f"driver start ns={self.ns} deadline={self.deadline}")
        for n in self.ns:
            if self.time_left() < 600:
                self.log("global deadline reached")
                break
            st = self.state["per_n"].get(str(n), {})
            if st.get("status") in ("passed", "passed_screen_failed_confirm") and self.a.resume:
                c = st["final_config"]
                continue
            try:
                c, ok = self.do_n(n, c)
            except Exception:  # noqa: BLE001
                self.log(f"n={n} crashed:\n{traceback.format_exc()}")
                self.state["notes"].append(f"n={n} crashed; see driver.log")
                self.save()
        if self.a.baseline and self.time_left() > 1200:
            self.state["phase"] = "baseline"
            self.save()
            self.baseline()
        self.state["phase"] = "done"
        self.state["current"] = None
        self.save()
        self.log("driver done")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", default="8,10,12,14,16")
    p.add_argument("--deadline", default="2026-10-10 09:40")
    p.add_argument("--soft-deadlines", default="8:2026-10-09 19:30,10:2026-10-09 21:00,12:2026-10-09 23:30,14:2026-10-10 03:30")
    p.add_argument("--screen-trials", type=int, default=5)
    p.add_argument("--confirm-trials", type=int, default=25)
    p.add_argument("--confirm-max-s", type=float, default=2.5 * 3600)
    p.add_argument("--baseline-trials", type=int, default=10)
    p.add_argument("--no-baseline", dest="baseline", action="store_false")
    p.add_argument("--max-escalations", type=int, default=12)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    Driver(a).main()


if __name__ == "__main__":
    main()
