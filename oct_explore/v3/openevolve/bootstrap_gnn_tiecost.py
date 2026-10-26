"""Bootstrap gnn_tiecost checkpoint_0 from initial_gnn_tiecost.py (screen-only for speed, then store)."""
import os, sys, json, time, uuid, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/gnn_tiecost"
os.makedirs(os.path.join(RUN, "checkpoints"), exist_ok=True)
os.environ["OE_RUN_DIR"] = os.path.abspath(RUN)
os.environ.setdefault("OE_L", "5")
os.environ.setdefault("OE_S1_MARGIN", "99")  # never cut seed
sys.path.insert(0, os.path.join(HERE, "net"))
import evaluator_gnn_tiecost as E
from openevolve.config import load_config
from openevolve.database import ProgramDatabase, Program

SEED_SRC = open("net/initial_gnn_tiecost.py").read()
open(os.path.join(RUN, "seed_program.py"), "w").write(SEED_SRC)

# Fast bootstrap: only screen path by temporarily forcing screen-only score into DB
# Full evaluate would take ~30-60min for n6+n8; bootstrap uses screen so OE can start, first real eval does full.
print(time.strftime("%H:%M:%S"), "bootstrap: evaluating seed (screen+full may take a while)...", flush=True)
# For bootstrap speed: set env to make full also short? Better: call train_rule screen and synthesize metrics
import tiecost_lib as TL
hypers, _ = E.load_rule(os.path.join(RUN, "seed_program.py"))
L = int(os.environ.get("OE_L", "5"))
t0 = time.time()
# screen on n=6
r_screen = TL.train_rule(L, hypers, screen=True)
print(time.strftime("%H:%M:%S"), "screen done", r_screen.get("combined_score"), f"{time.time()-t0:.0f}s", flush=True)
# lightweight full: train n=6 only at reduced steps for checkpoint_0 (OE will re-eval)
h2 = dict(hypers); h2["steps"] = min(int(hypers.get("steps", 1000)), 400); h2["ntrain"] = min(int(hypers.get("ntrain", 10000)), 3000)
h2["train_ns"] = (6,)  # bootstrap only n=6 to start fast; real OE evals use full rule
r_full = TL.train_rule(L, h2, screen=False)
print(time.strftime("%H:%M:%S"), "bootstrap full-lite done", r_full.get("combined_score"), flush=True)

metrics = {
    "combined_score": float(r_full.get("combined_score", -20)),
    "s1": float(r_screen.get("combined_score", -20)),
    "growth_proxy": r_full.get("growth_proxy") if r_full.get("growth_proxy") is not None else -1.0,
    "L": float(L),
    "hypers_H": float(hypers.get("H", 0)),
    "hypers_lr": float(hypers.get("lr", 0)),
    "hypers_steps": float(h2.get("steps", 0)),
    "cut_at_stage1": 0.0,
    "bootstrap": 1.0,
}
for nk, rr in (r_full.get("results") or {}).items():
    te = rr.get("test") or {}
    if te.get("median_inv_p") is not None:
        metrics[f"n{nk}_med"] = float(te["median_inv_p"])
        metrics[f"n{nk}_b90"] = float(te["B90_inv_p"])
        metrics[f"n{nk}_leak"] = float(te.get("mean_leak", 0))
        metrics[f"n{nk}_tied"] = 1.0 if rr.get("cost_tied") else 0.0

cfg = load_config("config_gnn_tiecost.yaml"); cfg.database.db_path = None
db = ProgramDatabase(cfg.database)
# 3 islands: baseline / capacity / trainmix — same seed code with different FAMILY tags for now
fams = [
    ("baseline", SEED_SRC),
    ("capacity", SEED_SRC.replace("FAMILY: baseline", "FAMILY: capacity — tune H/rounds/steps vs L")),
    ("trainmix", SEED_SRC.replace("FAMILY: baseline", "FAMILY: trainmix — decide +9/+12 and ntrain/bs")),
]
for i, (fam, code) in enumerate(fams):
    p = Program(id=str(uuid.uuid4()), code=code, metrics=dict(metrics), iteration_found=0, metadata={"island": i, "family": fam})
    db.current_island = i; db.add(p, iteration=0, target_island=i)

ck = os.path.join(RUN, "checkpoints", "checkpoint_0"); os.makedirs(ck, exist_ok=True)
db.save(ck, iteration=0)
b = db.get_best_program()
open(os.path.join(ck, "best_program.py"), "w").write(b.code)
json.dump({"id": b.id, "metrics": b.metrics, "iteration": 0}, open(os.path.join(ck, "best_program_info.json"), "w"), indent=1)
json.dump({"metrics": metrics, "sec": time.time()-t0}, open(os.path.join(RUN, "bootstrap_metrics.json"), "w"), indent=1)
print("checkpoint_0 saved; best", metrics.get("combined_score"), "islands", [len(s) for s in db.islands])
# STATUS
open(os.path.join(RUN, "STATUS.md"), "w").write(
    f"# gnn_tiecost STATUS\nupdated: {time.strftime('%F %T %z')}\n\n- bootstrap complete\n- best_score: {metrics.get('combined_score')}\n- n6_med: {metrics.get('n6_med')}\n"
)
