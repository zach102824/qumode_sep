import sys
name=sys.argv[1]; rows=[l.split("|") for l in open("noiseless/results/OVERNIGHT_IMPROVEMENTS.md") if l.startswith(f"| {name} |")]
ok=bool(rows) and all(len(r)>7 and r[5].strip()=="1.00" and float(r[6])>=0.9 for r in rows)
sys.exit(0 if ok else 1)
