"""net16x checkpoint handler (copy of net16r_checker.py; pair-unitary TEST checker chunked_check16x.py, n_pair_unitaries recorded).
Original doc: net16r checkpoint handler (one pass; net16r_checker.sh loops it). For every new checkpoint_k (k>0, every 10 iterations):
fair TEB test (100 n=16, chunked checker; trains the best program once, TEB never feeds back), append to NATIVE_TEMPLATE.md, commit+push.
Also: best_s2.json for the adapter; stall counter (3 stalled -> reseed.pending + Opus batch of 5 techniques); every 5 checkpoints a new
n=12 screen set; big jump (fair score up > 0.3) -> recheck with frozen score_set (verify) before it is reported. Events -> events.log."""
import os, sys, json, time, subprocess, glob, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); OUT = "runs/net16x"; CH = f"{OUT}/check"; os.makedirs(CH, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..")); MD = os.path.join(REPO, "oct_explore", "NATIVE_TEMPLATE.md"); STF = f"{OUT}/checker_state.json"
FLOOR, NEED = 0, 2600   # 14:57 monitor: RAM floor removed per Zach (was 3200); only wait for the check's own need
def ev(s): open(f"{OUT}/events.log", "a").write(time.strftime("%F %T ") + s + "\n")
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
def fair(r): return -(np.log(r["n20_rr_median"]) + np.log(r["n20_rr_B90"])) / 2
def git(paths, msg):
    try:
        subprocess.run(["git", "-C", REPO, "add", "-f"] + paths, check=True, capture_output=True)
        subprocess.run(["git", "-C", REPO, "commit", "-m", msg, "--"] + paths, check=True, capture_output=True)
        for _ in range(3):
            if subprocess.run(["git", "-C", REPO, "pull", "--rebase", "--autostash", "-q", "origin", "main"], capture_output=True).returncode == 0 and \
               subprocess.run(["git", "-C", REPO, "push", "-q", "origin", "HEAD:main"], capture_output=True).returncode == 0: return True
            time.sleep(30)
        ev("PUSH FAILED (commit kept locally): " + msg)
    except subprocess.CalledProcessError as e: ev(f"git error: {e.stderr.decode()[-200:]}")
    return False
def run_check(prog, out, verify=0):
    while avail() < FLOOR + NEED: time.sleep(30)
    env = dict(os.environ, OMP_NUM_THREADS="1", OE_N20="1", OE_NTEST="100")
    with open(out, "w") as fo, open(out[:-5] + ".err", "w") as fe:
        subprocess.run(["python3.13", "net/chunked_check16x.py", prog] + ([str(verify)] if verify else []), stdout=fo, stderr=fe, env=env)
    return json.load(open(out))
def main():
    st = json.load(open(STF)) if os.path.exists(STF) else {"fair_best": None, "stall": 0, "n": 0, "screen_cycle": 0}
    cks = sorted(glob.glob(f"{OUT}/checkpoints/checkpoint_*"), key=lambda c: int(c.rsplit("_", 1)[1]))
    for ck in cks:
        k = int(ck.rsplit("_", 1)[1]); jf = f"{CH}/checkpoint_{k}.json"
        if k == 0 or os.path.exists(jf) or not os.path.exists(f"{ck}/best_program.py"): continue
        info = json.load(open(f"{ck}/best_program_info.json")); sel = info.get("metrics", {})
        json.dump({"combined_score": sel.get("combined_score"), "id": info.get("id"), "ckpt": k}, open(f"{OUT}/best_s2.json", "w"))
        t = time.time(); r = run_check(f"{ck}/best_program.py", jf); dt = time.time() - t
        if "n20_rr_median" not in r: ev(f"ckpt {k}: fair check failed {r.get('error')}"); continue
        f = fair(r); prev = st["fair_best"]; st["n"] += 1; tag = ""
        if prev is not None and f > prev + 0.3:   # big jump: recheck scorer with the frozen score_set on the same TEST set
            v = run_check(f"{ck}/best_program.py", f"{CH}/checkpoint_{k}_verify.json", verify=4)
            ok = "verify_match_rr" in v and max(v["verify_match_rr"]) < 0.25 and abs(fair(v) - f) < 0.3
            tag = " RECHECKED-OK" if ok else " RECHECK-MISMATCH(not counted)"; ev(f"ckpt {k}: big jump {prev:.3f}->{f:.3f}{tag}")
            if not ok: f = prev
        if prev is None or f > prev + 1e-9:
            if prev is not None: ev(f"IMPROVED fair n=16 best: ckpt {k} {r['n20_rr_median']:.1f}/{r['n20_rr_B90']:.1f} (was score {prev:.3f}, now {f:.3f}){tag}")
            st["fair_best"] = f; st["stall"] = 0
        else: st["stall"] += 1
        line = (f"net16x ckpt_{k} ({time.strftime('%F %H:%M')} UTC+8) fair TEST n=16 rr {r['n20_rr_median']:.1f}/{r['n20_rr_B90']:.1f} single "
                f"{r['n20_single_median']:.1f}/{r['n20_single_B90']:.1f} pv_acc {r['n20_pv_acc']:.2f} pair_unitaries {r.get('n_pair_unitaries')} | selection {sel.get('rr_median_inv_p', float('nan')):.1f}/"
                f"{sel.get('rr_B90_inv_p', float('nan')):.1f} | stall {st['stall']}{tag} (QAOA L4 34.82/98.20; net16r best ckpt_10 83.9/2607)")
        open(MD, "a").write(line + "\n"); ev(line + f" [check {dt:.0f}s]")
        if st["stall"] and st["stall"] % 3 == 0:
            try: ev("Opus new island families: " + urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8766/v1/new_families", method="POST"), timeout=900).read().decode()[:800])
            except Exception as e: ev(f"new_families failed: {e}")
            open(f"{OUT}/reseed.pending", "w").write(str(k)); ev(f"ckpt {k}: {st['stall']} stalled checkpoints -> reseed weakest 2 islands + Opus 5 techniques")
            try: ev("techniques: " + urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8766/v1/ideas_batch", method="POST"), timeout=900).read().decode()[:1500])
            except Exception as e: ev(f"ideas_batch failed: {e}")
        if st["n"] % 5 == 0:
            st["screen_cycle"] += 1; c = st["screen_cycle"]
            p = subprocess.run(["python3.13", "make_screen12.py", str(c)], capture_output=True, text=True)
            if p.returncode == 0:
                json.dump({"base": 34e9 + 1e8 * c, "cycle": c}, open(f"{OUT}/screen.json", "w"))
                if os.path.exists(f"{OUT}/s1_state.json"): os.replace(f"{OUT}/s1_state.json", f"{OUT}/s1_state_cycle{c-1}.json")
                ev(f"new n=12 screen set cycle {c} (base {34e9 + 1e8 * c:.0f}); stage-1 bar reset")
            else: ev("screen regen failed: " + p.stderr[-300:])
        json.dump(st, open(STF, "w"))
        git([MD, os.path.abspath(jf), os.path.abspath(f"{OUT}/events.log"), os.path.abspath(f"{ck}/best_program.py")], f"net16x ckpt_{k}: fair TEST n=16 rr {r['n20_rr_median']:.1f}/{r['n20_rr_B90']:.1f}")
if __name__ == "__main__": main()
