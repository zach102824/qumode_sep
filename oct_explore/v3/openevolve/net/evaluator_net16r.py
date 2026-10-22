"""net16r evaluator: 3-stage screen around the FROZEN evaluator_net (imported unchanged; gates/compiler/simulator/readout locked).
Stage 1 (~OE_S1_BUDGET=60 s): train on n=8+12, score the n=12 SCREEN set (30 unseen formulas, base OE_SCREEN_BASE, make_screen12.py).
  s1 = -(ln med + ln B90)/2 (round-robin). If s1 < best_s1 - OE_S1_MARGIN (state file OE_S1_STATE) -> early return
  combined_score = -12 + 0.1*s1 (always below any stage-2 score, above hard failures at -20/-30).
Stage 2: EXACT frozen evaluator_net._evaluate_inner (full train n=8+12, OE_BUDGET s; 60-formula n=16 VAB selection set;
  combined_score = -(ln med + ln B90)/2 round-robin; pv_acc>0.85 -> -20). TEB (test) is NEVER touched here.
Diagnostics (from the stage-1 model, returned as metrics + artifact 'diagnostics' shown to the LLM): chain p vs global-cost p
(glob=True) at n=8 (train slice) and n=12 (screen set), 5 worst n=12 screen formulas with their dropped-cost fraction
(share of the per-layer cost-phase variance removed by the chain projection, placement 0)."""
import os, sys, json, time, numpy as np
if len(sys.argv) > 1 and sys.argv[1] == "--inner":   # persistent XLA cache: the frozen circuit/readout jits are identical across candidates
    _cd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "jaxcache16r"); os.makedirs(_cd, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cd); os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evaluator_net as EV
import frozen_circuit as FC, jax, jax.numpy as jnp
RUN = os.environ.get("OE_RUN_DIR", os.path.join(HERE, "..", "runs", "net16r"))
S1_BUDGET = float(os.environ.get("OE_S1_BUDGET", "60")); S1_MARGIN = float(os.environ.get("OE_S1_MARGIN", "0.35"))
S1_STATE = os.environ.get("OE_S1_STATE", os.path.join(RUN, "s1_state.json"))
class _FreezeClock:
    """Training-clock shim for the evaluator's OWN timing only (train_s / over-budget check). A daemon thread ticks every 0.5 s;
    any wall gap > 5 s between ticks means the whole box was paused/frozen (seen 2026-10-22: 15-25 min host stalls freeze every
    process at once), so that gap is not training. The candidate's own clock (its budget loop) is untouched, budget unchanged."""
    def __init__(self):
        import threading
        self.frozen = 0.0; self._last = time.time(); self._lk = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while True:
            time.sleep(0.5); now = time.time()
            with self._lk:
                g = now - self._last
                if g > 5.0: self.frozen += g - 0.5
                self._last = now
    def time(self):
        with self._lk: return time.time() - self.frozen
    def __getattr__(self, k): return getattr(time, k)
_clk = time
def screen_base():
    try: return float(json.load(open(os.path.join(RUN, "screen.json")))["base"])
    except Exception: return float(os.environ.get("OE_SCREEN_BASE", "34e9"))
def summ(p): ip = 1 / np.maximum(p, 1e-12); return float(np.median(ip)), float(np.quantile(ip, .9))
def p_both(predict, D):
    """chain p and global-cost p (F,R) for every formula of D (+ outputs)."""
    import inspect
    n, F, m = D["n"], D["V"].shape[0], D["m"]
    kw = dict(place=EV.place_of(D, np.arange(F))) if "place" in inspect.signature(predict).parameters else {}
    o = EV.check_outs(predict(n, D["V"].copy(), D["SG"].copy(), m, **kw), F, n, m)
    f = jax.jit(lambda g, b_, w, pv, VP, BB, PERM, S, dS, dB, gl: FC.circuit_p(n, g, b_, w, pv, VP, BB, PERM, S, n / m, dS, dB, gl), static_argnums=10)
    Pc, Pg = [], []
    for s in range(0, F, 10):
        idx = np.arange(s, min(F, s + 10)); args = FC.entries(D, idx, {k: np.asarray(o[k])[idx] for k in ("gamma", "beta", "w", "pv")})
        dS, dB = EV._corr({k: (o[k][idx] if k in o else None) for k in ("dS", "dB")}, n)
        Pc.append(np.asarray(f(*args[:8], dS, dB, False)).reshape(-1, EV.R)); Pg.append(np.asarray(f(*args[:8], None, None, True)).reshape(-1, EV.R))
    return np.concatenate(Pc), np.concatenate(Pg), o
def dropped_frac(D, o, i):
    """share of cost-phase variance (uniform measure, mean over layers) lost by the chain projection, placement 0."""
    n = D["n"]; args = FC.entries(D, np.array([i]), {k: np.asarray(o[k])[i:i + 1] for k in ("gamma", "beta", "w", "pv")})
    g, b_, w, pv, VP, BB, PERM, S, nm = args; VP, BB, PERM, w, g, pv = VP[:1], BB[:1], PERM[:1], w[:1], g[:1], pv[:1]
    pvc = jnp.take_along_axis(pv, PERM.astype(jnp.int32), axis=1); xb = jnp.asarray(FC.bits(n))
    viol = jnp.prod((xb[:, VP] == BB[None].astype(jnp.int8)).astype(jnp.float32), -1); fr = []
    for l in range(FC.L):
        Ew = nm * jnp.einsum("xbm,bm->bx", viol, g[:, l][:, None] + w[:, :, l]); Pj = FC.project_product_j(Ew, pvc, n, FC.CP.LAYOUTS[n])
        e = Ew - Ew.mean(); fr.append(float(jnp.sum((Ew - Pj - (Ew - Pj).mean()) ** 2) / (jnp.sum(e * e) + 1e-12)))
    return float(np.mean(fr))
def _stage1_inner(prog):
    t0 = time.time()
    try: EV.verify_frozen()
    except Exception as e: return {"combined_score": -30.0, "error": str(e)}
    src = open(prog).read(); bad = EV.static_check(src)
    if bad: return {"combined_score": -20.0, "error": f"forbidden: {sorted(set(bad))}"}
    import importlib.util
    sets = {n: EV.get_set(n, EV.NTR[n], FC.TRB, EV.R) for n in EV.TRAIN_N}
    spec = importlib.util.spec_from_file_location("cand", prog); mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod); tt = _clk.time(); predict = mod.train(EV.API(sets), S1_BUDGET); ttr = _clk.time() - tt
        if ttr > S1_BUDGET * 2.0: return {"combined_score": -20.0, "error": f"stage-1 train over budget {ttr:.0f}s"}
        D12 = EV.get_set(12, 30, screen_base(), EV.R); Pc, Pg, o = p_both(predict, D12)
        D8 = EV.get_set(8, EV.NTR[8], FC.TRB, EV.R); D8 = {**D8, **{k: D8[k][:30] for k in ("V", "SG", "PERM", "S", "seed")}, "sol": D8["sol"][:30]}
        Pc8, Pg8, _ = p_both(predict, D8)
    except Exception as e: return {"combined_score": -20.0, "error": f"stage1 {type(e).__name__}: {str(e)[:300]}"}
    pv = np.asarray(o["pv"]); acc = float(np.mean([max(np.mean((pv[i] > .5) == (s_ == 1)) for s_ in D12["sol"][i]) for i in range(len(pv))]))
    med, b90 = summ(Pc.mean(1)); s1 = -(np.log(med) + 2.0 * np.log(b90)) / 3   # v2: held-out n=12 B90 weighted 2:1 - (20.0 if acc > EV.PVACC_MAX else 0.0)
    worst = np.argsort(Pc.mean(1))[:5]
    W = [dict(idx=int(i), seed=int(D12["seed"][i]), chain_p=float(Pc[i].mean()), glob_p=float(Pg[i].mean()), dropped_frac=dropped_frac(D12, o, int(i))) for i in worst]
    gm = lambda P: float(np.exp(np.mean(np.log(np.maximum(P.mean(1), 1e-12)))))
    return dict(s1=float(s1), s1_med=med, s1_B90=b90, s1_pv_acc=acc, s1_train_s=ttr, s1_s=time.time() - t0,
                n8_chain_p=gm(Pc8), n8_glob_p=gm(Pg8), n12_chain_p=gm(Pc), n12_glob_p=gm(Pg), worst=W)
def _child(stage, prog, mem_mb, wall):
    import subprocess, tempfile
    out = tempfile.mktemp(suffix=".json")
    pr = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--inner", stage, prog, out], env=os.environ.copy())
    peak = 0; t0 = time.time(); act = 0.0   # wall timeout counts only time the child is not SIGSTOPped by the governor
    while pr.poll() is None:
        try: act += 0.0 if open(f"/proc/{pr.pid}/stat").read().split()[2] == "T" else 1.0
        except Exception: pass
        try: rss = int(open(f"/proc/{pr.pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024
        except Exception: rss = 0
        peak = max(peak, rss)
        if rss > mem_mb: pr.kill(); pr.wait(); return {"combined_score": -20.0, "error": f"memory {rss}MB > {mem_mb}MB (reduce batch sizes)"}, peak
        if act > wall: pr.kill(); pr.wait(); return {"combined_score": -20.0, "error": f"{stage} wall timeout"}, peak
        time.sleep(1.0)
    try: r = json.load(open(out)); os.remove(out); return r, peak
    except Exception: return {"combined_score": -20.0, "error": f"{stage} child failed rc={pr.returncode}"}, peak
def _child_retry(stage, prog, mem, wall):
    """crash (child died / OOM / wall) -> retry once, then mark failed."""
    r, pk = _child(stage, prog, mem, wall)
    if "error" in r and any(x in str(r["error"]) for x in ("child failed", "memory", "wall timeout")):
        r2, pk2 = _child(stage, prog, mem, wall); r2["retried"] = 1; return r2, max(pk, pk2)
    return r, pk
def s1_margin():
    """live-tunable stage-1 cut margin (file runs/net16r/s1_margin overrides env). Calibrated 2026-10-22 on 46 stage-2 programs: 0.20."""
    try: return float(open(os.path.join(RUN, "s1_margin")).read().strip())
    except Exception: return S1_MARGIN
def _state():
    try: return json.load(open(S1_STATE))
    except Exception: return {"best": None, "n": 0, "cut": 0}
def _save_state(st):
    os.makedirs(os.path.dirname(S1_STATE), exist_ok=True); import threading; tmp = S1_STATE + f".{os.getpid()}.{threading.get_ident()}.tmp"; json.dump(st, open(tmp, "w")); os.replace(tmp, S1_STATE)
def diag_text(r1, r2=None):
    L_ = [f"STAGE-1 (60 s train, 30 unseen n=12 screen formulas; s1 = -(ln med + 2 ln B90)/3): s1={r1['s1']:.3f} (med 1/p {r1['s1_med']:.1f}, B90 {r1['s1_B90']:.1f}), pv_acc {r1['s1_pv_acc']:.2f}",
          f"geo-mean p  CHAIN vs GLOBAL-COST (same outputs): n=8 {r1['n8_chain_p']:.4f} vs {r1['n8_glob_p']:.4f};  n=12 {r1['n12_chain_p']:.4f} vs {r1['n12_glob_p']:.4f}",
          "5 WORST n=12 screen formulas (chain p | global p | dropped-cost fraction):"]
    L_ += [f"  #{w['idx']}: {w['chain_p']:.4f} | {w['glob_p']:.4f} | {w['dropped_frac']:.2f}" for w in r1["worst"]]
    tgt = r1["worst"]; mean_w = np.mean([w["chain_p"] for w in tgt])
    L_.append(f"CHECKABLE TARGET: raise mean chain p on screen formulas {[w['idx'] for w in tgt]} from {mean_w:.4f} to >= {1.5 * mean_w:.4f} "
              f"(global-cost reaches {np.mean([w['glob_p'] for w in tgt]):.4f}) without lowering the n=12 median.")
    if r2: L_.append(f"STAGE-2 n=16 selection: score {r2.get('combined_score')}, rr med {r2.get('rr_median_inv_p')}, B90 {r2.get('rr_B90_inv_p')}, pv_acc {r2.get('pv_acc')}")
    return "\n".join(L_)
import re as _re
GUARD = [r"\b1\s*<<\s*n\b", r"\b2\s*\*\*\s*n\b", r"\*\*\s*n\b", r"<<\s*n\s*\)", r"itertools\.product", r"\bargmin\b", r"65536", r"\b1\s*<<\s*1[6-9]\b",
         r"\(\s*256\s*,\s*256", r"\b256\s*\*\s*256\b", r"\bproduct\s*\(\s*\[?\s*\(?\s*0\s*,\s*1"]
def guard_check(src):
    """v2 anti-classical-solving guard: no full-assignment enumeration / argmin, no tables larger than one 256-entry pair table at once."""
    return [g for g in GUARD if _re.search(g, src)]
def evaluate(program_path):
    gb = guard_check(open(program_path).read())
    if gb:
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(metrics={"combined_score": -20.0, "error": f"guard: full-assignment enumeration / argmin / >1 pair table at once: {gb}"}, artifacts={"diagnostics": "rejected by guard " + str(gb)})
    from openevolve.evaluation_result import EvaluationResult
    mem = int(os.environ.get("OE_MEM_MB", "4500")); t0 = time.time()
    r1, pk1 = _child_retry("s1", program_path, mem, S1_BUDGET * 2 + 400)
    if "s1" not in r1:
        r1["stage"] = 1; r1["peak_rss_mb"] = pk1; return EvaluationResult(metrics=r1, artifacts={"diagnostics": "stage 1 failed: " + str(r1.get("error"))})
    import fcntl; os.makedirs(os.path.dirname(S1_STATE), exist_ok=True); lk = open(S1_STATE + ".lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
    st = _state(); best = st.get("best"); st["n"] = st.get("n", 0) + 1
    flat = {k: v for k, v in r1.items() if k != "worst"}; flat["s1_best_before"] = best if best is not None else -99.0
    M = s1_margin(); flat["s1_margin"] = M
    if best is not None and r1["s1"] < best - M:
        st["cut"] = st.get("cut", 0) + 1; _save_state(st); lk.close()
        m = dict(flat, combined_score=-12.0 + 0.1 * max(r1["s1"], -30.0), stage=1, cut_at_stage1=1, peak_rss_mb=pk1, eval_s=time.time() - t0)
        return EvaluationResult(metrics=m, artifacts={"diagnostics": diag_text(r1) + f"\nCUT AT STAGE 1: s1 {r1['s1']:.3f} < best {best:.3f} - {M}"})
    if best is None or r1["s1"] > best: st["best"] = r1["s1"]
    _save_state(st); lk.close()
    r2, pk2 = _child_retry("s2", program_path, mem, EV.BUDGET * 2 + 400)
    m = dict(flat); m.update(r2); m.update(stage=2, cut_at_stage1=0, peak_rss_mb=max(pk1, pk2), eval_s=time.time() - t0)
    return EvaluationResult(metrics=m, artifacts={"diagnostics": diag_text(r1, r2)})
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--inner":
    stage, prog, out = sys.argv[2:5]
    _clk = _FreezeClock(); EV.time = _clk   # freeze-aware evaluator clock (see _FreezeClock); EV's file itself is unchanged
    _r = _stage1_inner(prog) if stage == "s1" else EV._evaluate_inner(prog); _r["frozen_s"] = float(_clk.frozen)
    json.dump(_r, open(out, "w")); sys.exit(0)
if __name__ == "__main__":
    r = evaluate(sys.argv[1]); print(json.dumps({"metrics": r.metrics, "artifacts": r.artifacts}, indent=1, default=str))
