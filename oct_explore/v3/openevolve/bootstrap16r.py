"""Build runs/net16r/checkpoints/checkpoint_0: evaluate the 5 island seeds (net/seeds16r/<family>.py) with evaluator_net16r
(stage-1 cut disabled for seeds) and put seed i on island i. Usage: python3.13 bootstrap16r.py [run_dir]"""
import os, sys, json, time, uuid, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/net16r"
os.environ.setdefault("OE_RUN_DIR", os.path.abspath(RUN)); os.environ["OE_S1_MARGIN"] = "99"
sys.path.insert(0, os.path.join(HERE, "net")); import evaluator_net16r as E
from openevolve.config import load_config
from openevolve.database import ProgramDatabase, Program
FAMS = ["loss", "architecture", "features", "curriculum", "placement"]
cfg = load_config("config_net16r.yaml"); cfg.database.db_path = None
res = {}
from openevolve.evaluation_result import EvaluationResult
def ev(f):
    cf_ = os.path.join(RUN, f"seed_{f}.json")
    if os.path.exists(cf_): d = json.load(open(cf_)); return f, EvaluationResult(metrics=d["metrics"], artifacts=d["artifacts"])
    for a in range(2):   # one retry (e.g. host stall -> train over budget)
        t = time.time(); r = E.evaluate(f"net/seeds16r/{f}.py")
        if r.metrics.get("combined_score", -20) > -19: break
    if r.metrics.get("combined_score", -20) > -19: json.dump({"metrics": r.metrics, "artifacts": r.artifacts}, open(cf_, "w"))
    _ = None; print(time.strftime("%H:%M:%S"), f, f"{time.time()-t:.0f}s", json.dumps({k: r.metrics.get(k) for k in ("combined_score", "s1", "rr_median_inv_p", "rr_B90_inv_p", "pv_acc", "s1_s", "train_s", "eval_s", "peak_rss_mb", "error")}), flush=True)
    return f, r
with cf.ThreadPoolExecutor(int(os.environ.get("BOOT_PAR", "2"))) as ex:
    for f, r in ex.map(ev, FAMS): res[f] = r
db = ProgramDatabase(cfg.database)
for i, f in enumerate(FAMS):
    r = res[f]; m = {k: v for k, v in r.metrics.items() if isinstance(v, (int, float))}
    p = Program(id=str(uuid.uuid4()), code=open(f"net/seeds16r/{f}.py").read(), metrics=m, iteration_found=0, metadata={"island": i, "family": f})
    db.current_island = i; db.add(p, iteration=0, target_island=i)
    if r.artifacts: db.store_artifacts(p.id, r.artifacts) if hasattr(db, "store_artifacts") else None
ck = os.path.join(RUN, "checkpoints", "checkpoint_0"); os.makedirs(ck, exist_ok=True); db.save(ck, iteration=0)
b = db.get_best_program(); open(os.path.join(ck, "best_program.py"), "w").write(b.code)
json.dump({"id": b.id, "metrics": b.metrics, "iteration": 0}, open(os.path.join(ck, "best_program_info.json"), "w"), indent=1)
print("checkpoint_0 saved; islands:", [len(s) for s in db.islands], "best", b.metrics.get("combined_score"))
