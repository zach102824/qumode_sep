"""Stagnation response: in checkpoint dir CK, reseed the 2 weakest islands with copies of the 2 best programs overall
(FAMILY line rewritten to the island's family so its idea direction is kept). Usage: python3.13 reseed16r.py CK"""
import sys, re, uuid, copy
from openevolve.config import load_config
from openevolve.database import ProgramDatabase, Program
FAMS = ["loss", "architecture", "features", "curriculum", "placement"]
ck = sys.argv[1]; cfg = load_config("config_net16r.yaml"); cfg.database.db_path = None
db = ProgramDatabase(cfg.database); db.load(ck)
sc = lambda p: float(p.metrics.get("combined_score", -99))
best_of = [max([sc(db.programs[i]) for i in isl if i in db.programs] or [-99]) for isl in db.islands]
weak = sorted(range(len(db.islands)), key=lambda i: best_of[i])[:2]
top = sorted(db.programs.values(), key=sc, reverse=True)[:2]
for i in weak:
    for t in top:
        code = re.sub(r"# FAMILY: [^\n]*", f"# FAMILY: {FAMS[i]} - reseeded from best ({t.id[:8]}, score {sc(t):.3f})", t.code, count=1)
        p = Program(id=str(uuid.uuid4()), code=code, metrics=dict(t.metrics), parent_id=None, generation=t.generation, iteration_found=db.last_iteration,
                    metadata={"island": i, "family": FAMS[i], "reseed": True})
        db.add(p, iteration=db.last_iteration, target_island=i)
db.save(ck, iteration=db.last_iteration); print("reseeded islands", weak, "island bests were", [round(b, 3) for b in best_of])
