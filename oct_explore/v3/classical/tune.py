from runcls import *
key = sys.argv[1]; n = int(key.split("_")[1]); tr = I[key]; R = 50
res = {"walksat": {}, "sa": {}}
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    f = [run(i, n, R, 0, p) for i in tr]; res["walksat"][p] = [score([x[0] for x in f]), score([x[1] for x in f])]
    print("ws", p, res["walksat"][p], flush=True)
for b0 in [0.2, 0.5, 1.0]:
  for b1 in [2.0, 3.0, 5.0]:
    for s in [5, 20, 50, 200]:
        f = [run(i, n, R, 1, b0, b1, s * n) for i in tr]; k = f"{b0},{b1},{s}"; res["sa"][k] = score([x[1] for x in f])
        print("sa", k, res["sa"][k], flush=True)
json.dump(res, open(os.path.join(HERE, f"tune_{key}.json"), "w"))
