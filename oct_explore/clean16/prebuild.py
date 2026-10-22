import os, sys; os.environ.update(OE_TRAIN_N="8,12", OE_NTR="600")
import common as CM, evaluator_net as EV, time
n, cnt, base = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); t = time.time()
D = EV.get_set(n, cnt, base, EV.R); print(n, cnt, base, D["V"].shape, f"{time.time()-t:.0f}s", flush=True)
