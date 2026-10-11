import json,glob,os,sys,time
since=float(sys.argv[1]) if len(sys.argv)>1 else 0
rows=[]
for f in glob.glob('logs/improve/*.json'):
    t=open(f).read(); i=t.find('\n{')
    if i<0 and not t.startswith('{'): continue
    try: d=json.loads(t[i+1:] if i>=0 else t)
    except Exception: continue
    if 'curve' not in d: continue
    rows.append((os.path.getmtime(f),os.path.basename(f)[:-5],d['success'],d['mean_p_gs'],d['evals']))
rows.sort()
for t,nm,s,p,e in rows:
    if t>=since: print(time.strftime('%H:%M',time.localtime(t)),nm,'succ %.2f p %.3f evals %d'%(s,p,e))
