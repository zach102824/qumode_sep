import json,glob,sys,os
root=sys.argv[1] if len(sys.argv)>1 else 'noiseless/results/relayout_runs/qbenefit'
for d in sorted(glob.glob(root+'/*')):
  for f in sorted(glob.glob(d+'/n*.jsonl')):
    R=[json.loads(l) for l in open(f)]
    s=sum(r['success'] for r in R); p=sum(r['p_gs'] for r in R)/len(R)
    cg=sum(bool(r['sa_best_is_ground']) for r in R)
    print(f"{os.path.basename(d):10s} n={R[0]['n']:2d} succ {s}/{len(R)} p={p:.3f} lookups={R[0]['n_lookups_fixed']} evals={R[0]['nfev']} clsGS={cg}")
