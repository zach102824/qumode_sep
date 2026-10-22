"""net16x copy of reseed16r.py (only after runs/net16x/wildcard.open; driver enforces). The FAMILY line keeps the ISLAND's net16x direction
name (reencode/grover/nondiag/wildcard) and appends the Opus mechanism, so island mandates survive reseeding.
Original doc: Stagnation response: in checkpoint dir CK, reseed the 2 weakest islands with copies of the 2 best programs overall
(FAMILY line rewritten to the island's family so its idea direction is kept). Usage: python3.13 reseed16r.py CK"""
import sys, re, uuid, copy
from openevolve.config import load_config
from openevolve.database import ProgramDatabase, Program
FAMS = ["reencode", "grover", "nondiag", "wildcard"]
ck = sys.argv[1]; cfg = load_config("config_net16x.yaml"); cfg.database.db_path = None
db = ProgramDatabase(cfg.database); db.load(ck)
sc = lambda p: float(p.metrics.get("combined_score", -99))
best_of = [max([sc(db.programs[i]) for i in isl if i in db.programs] or [-99]) for isl in db.islands]
weak = sorted(range(len(db.islands)), key=lambda i: best_of[i])[:2]
top = sorted(db.programs.values(), key=sc, reverse=True)[:2]
import json, os
OUTD = os.path.dirname(os.path.dirname(ck.rstrip("/"))); FJ = os.path.join(OUTD, "opus_families.json"); PJ = os.path.join(OUTD, "reseed.pending")
if os.path.exists(PJ) and (not os.path.exists(FJ) or os.path.getmtime(FJ) < os.path.getmtime(PJ)):   # stale families: ask Opus now (adapter retries)
    import urllib.request, time as _t
    for k in range(3):
        try: print("fresh Opus families:", urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8766/v1/new_families", method="POST"), timeout=1200).read().decode()[:600]); break
        except Exception as e: print("new_families attempt", k + 1, "failed:", e); _t.sleep(120)
try: newf = json.load(open(os.path.join(os.path.dirname(os.path.dirname(ck.rstrip("/"))), "opus_families.json")))
except Exception: newf = []
for wi, i in enumerate(weak):
    fam_name, fam_desc = (FAMS[i], newf[wi]["name"] + ": " + newf[wi]["desc"]) if wi < len(newf) else (FAMS[i], "reseeded")
    for t in top:
        code = re.sub(r"# FAMILY: [^\n]*", f"# FAMILY: {fam_name} - {fam_desc} (Opus-chosen at stagnation; body reseeded from best {t.id[:8]}, score {sc(t):.3f})", t.code, count=1)
        p = Program(id=str(uuid.uuid4()), code=code, metrics=dict(t.metrics), parent_id=None, generation=t.generation, iteration_found=db.last_iteration,
                    metadata={"island": i, "family": fam_name, "reseed": True})
        db.add(p, iteration=db.last_iteration, target_island=i)
db.save(ck, iteration=db.last_iteration); print("reseeded islands", weak, "island bests were", [round(b, 3) for b in best_of])
