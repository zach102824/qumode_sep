"""OpenEvolve evaluator for gnn_tiecost: evolve adaptive rule(L) for tied-cost GNN.
Stage 1 (screen): train n=6 at OE_L with OE_SCREEN_STEPS; cut weak early.
Stage 2: full train_rule(L, hypers) on train_ns (6+8); fitness = TEB Med/B90 score + growth proxy.
Hard: cost_tied must be True.
"""
from __future__ import annotations
import os, sys, json, time, re, importlib.util, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiecost_lib as TL

RUN = os.environ.get("OE_RUN_DIR", os.path.join(HERE, "..", "runs", "gnn_tiecost"))
L = int(os.environ.get("OE_L", "5"))
S1_MARGIN = float(os.environ.get("OE_S1_MARGIN", "0.25"))
S1_STATE = os.environ.get("OE_S1_STATE", os.path.join(RUN, "s1_state.json"))
MEM_MB = int(os.environ.get("OE_MEM_MB", "4500"))

ALLOWED = {"numpy", "math", "jax", "jax.numpy"}

def static_check(src: str):
    bad = re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w\.]*)", src, re.M)
    bad = [x for x in bad if x.split(".")[0] not in ALLOWED]
    for tok in ("open(", "__import__", "subprocess", "eval(", "exec(", "os.", "sys.", "importlib", "socket", "requests"):
        if tok in src: bad.append(tok)
    return bad

def load_rule(program_path):
    src = open(program_path).read()
    bad = static_check(src)
    if bad: raise ValueError(f"forbidden: {sorted(set(bad))}")
    spec = importlib.util.spec_from_file_location("cand_rule", program_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not hasattr(mod, "rule"): raise ValueError("missing rule(L)")
    h = mod.rule(L)
    if not isinstance(h, dict): raise ValueError("rule(L) must return dict")
    # normalize
    h = dict(h)
    tn = h.get("train_ns", (6, 8))
    if isinstance(tn, (list, tuple)):
        h["train_ns"] = tuple(int(x) for x in tn)
    else:
        h["train_ns"] = (6, 8)
    if 6 not in h["train_ns"] or 8 not in h["train_ns"]:
        # force include 6,8
        h["train_ns"] = tuple(sorted(set(h["train_ns"]) | {6, 8}))
    return h, src

def _s1_best():
    try: return float(json.load(open(S1_STATE)).get("best_s1", -1e9))
    except Exception: return -1e9

def _s1_update(s1):
    os.makedirs(RUN, exist_ok=True)
    cur = _s1_best()
    if s1 > cur:
        json.dump({"best_s1": s1, "t": time.time()}, open(S1_STATE, "w"))

def _evaluate_inner(program_path):
    t0 = time.time()
    try:
        hypers, src = load_rule(program_path)
    except Exception as e:
        return {"combined_score": -20.0, "error": f"load: {e}"}
    # Stage 1 screen
    try:
        r1 = TL.train_rule(L, hypers, screen=True)
        s1 = float(r1.get("combined_score", -20))
        if s1 <= -19:
            return {**{k: v for k, v in r1.items() if k != "results"}, "cut_at_stage1": 0, "s1": s1,
                    "eval_s": time.time()-t0, "error": r1.get("error", "screen fail"),
                    "artifacts_note": "screen failed"}
        best = _s1_best()
        if best > -1e8 and s1 < best - S1_MARGIN:
            return {"combined_score": -12.0 + 0.1 * s1, "s1": s1, "cut_at_stage1": 1,
                    "eval_s": time.time()-t0, "hypers_H": hypers.get("H"), "hypers_lr": hypers.get("lr"),
                    "train_ns": list(hypers.get("train_ns", [])), "L": L}
        _s1_update(s1)
    except Exception as e:
        return {"combined_score": -20.0, "error": f"screen: {type(e).__name__}: {e}", "trace": traceback.format_exc()[-500:], "eval_s": time.time()-t0}
    # Stage 2 full
    try:
        r2 = TL.train_rule(L, hypers, screen=False)
    except Exception as e:
        return {"combined_score": -20.0, "error": f"full: {type(e).__name__}: {e}", "s1": s1, "trace": traceback.format_exc()[-500:], "eval_s": time.time()-t0}
    out = {
        "combined_score": float(r2.get("combined_score", -20)),
        "s1": s1, "cut_at_stage1": 0,
        "growth_proxy": r2.get("growth_proxy"),
        "L": L,
        "hypers_H": hypers.get("H"), "hypers_rounds": hypers.get("rounds"),
        "hypers_lr": hypers.get("lr"), "hypers_steps": hypers.get("steps"),
        "hypers_bs": hypers.get("bs"), "hypers_wd": hypers.get("wd"),
        "hypers_leakpen": hypers.get("leakpen"), "hypers_ntrain": hypers.get("ntrain"),
        "hypers_arch": 0 if hypers.get("arch","mp")=="mp" else 1,
        "train_ns": list(hypers.get("train_ns", [])),
        "requested_extra_ns": r2.get("requested_extra_ns"),
        "eval_s": time.time() - t0,
    }
    # flatten per-n metrics
    for nk, rr in (r2.get("results") or {}).items():
        te = rr.get("test") or {}
        out[f"n{nk}_med"] = te.get("median_inv_p")
        out[f"n{nk}_b90"] = te.get("B90_inv_p")
        out[f"n{nk}_leak"] = te.get("mean_leak")
        out[f"n{nk}_tied"] = 1.0 if rr.get("cost_tied") else 0.0
    # artifact summary for LLM
    art = {
        "hypers": hypers,
        "s1": s1,
        "growth_proxy": r2.get("growth_proxy"),
        "per_n": {nk: {"med": (rr.get("test") or {}).get("median_inv_p"),
                       "b90": (rr.get("test") or {}).get("B90_inv_p"),
                       "leak": (rr.get("test") or {}).get("mean_leak"),
                       "tied": rr.get("cost_tied")} for nk, rr in (r2.get("results") or {}).items()},
    }
    out["_artifact_json"] = json.dumps(art)[:4000]
    return out

def evaluate(program_path):
    """Child-process RSS guard."""
    import subprocess, tempfile
    out = tempfile.mktemp(suffix=".json")
    env = os.environ.copy()
    env["OE_RUN_DIR"] = os.path.abspath(RUN)
    pr = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--inner", program_path, out], env=env)
    peak = 0; t0 = time.time()
    while pr.poll() is None:
        try: rss = int(open(f"/proc/{pr.pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024
        except Exception: rss = 0
        peak = max(peak, rss)
        if rss > MEM_MB:
            pr.kill(); pr.wait()
            return {"combined_score": -20.0, "error": f"memory {rss}MB > {MEM_MB}MB"}
        if time.time() - t0 > 7200:
            pr.kill(); pr.wait()
            return {"combined_score": -20.0, "error": "wall timeout"}
        time.sleep(2.0)
    try:
        r = json.load(open(out)); os.remove(out)
    except Exception:
        return {"combined_score": -20.0, "error": f"child failed rc={pr.returncode}"}
    r["peak_rss_mb"] = peak
    # OpenEvolve wants metrics dict; optional artifacts via EvaluationResult if used by CLI
    return r

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--inner":
    r = _evaluate_inner(sys.argv[2])
    # strip non-numeric for OE metrics dump companion
    json.dump(r, open(sys.argv[3], "w"))
    sys.exit(0)
if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "initial_gnn_tiecost.py")
    print(json.dumps(evaluate(p), indent=1, default=str))
